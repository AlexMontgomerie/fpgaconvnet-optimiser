from models.modules.SlidingWindow import SlidingWindow
from models.modules.Conv import Conv
from models.modules.Fork import Fork
from models.modules.Accum import Accum
from models.modules.Glue import Glue
from models.layers.Layer import Layer

import numpy as np
import math
import tools.third_party.lmdb_io
import tools.third_party.prototxt
import tempfile
import caffe
import pydot

class ConvolutionLayer(Layer):
    def __init__(
            self,
            dim,
            filters,
            k_size      =3,
            stride      =1,
            groups      =1,
            pad         =0,
            coarse_in   =1,
            coarse_out  =1,
            fine        =1,
            data_width  =16,
            sa          =0.5,
            sa_out      =0.5
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width)

        # update flags
        self.flags['channel_dependant'] = True
        self.flags['transformable']     = True

        # weight width
        self.weight_width = 8

        # init variables
        self.k_size     = k_size
        self.stride     = stride
        self.groups     = groups
        self.pad        = pad
        self.pad_top    = pad - (self.rows - k_size + 2*pad) % stride
        self.pad_right  = pad - (self.cols - k_size + 2*pad) % stride
        self.pad_bottom = pad
        self.pad_left   = pad
        self.fine       = fine
        self.filters    = filters

        dim_out = [
            self.filters,
            self.rows_out(),
            self.cols_out()
        ]
        # init modules
        self.modules = {
            "sliding_window" : SlidingWindow(dim, k_size, stride, self.pad_top, self.pad_right, self.pad_bottom, self.pad_left, data_width),
            "fork"           : Fork(dim_out,k_size,coarse_out),
            "conv"           : Conv(dim_out,filters,fine,k_size,groups),
            "accum"          : Accum(dim_out,filters,groups),
            "glue"           : Glue(dim_out,filters,coarse_in,coarse_out)
        }
        self.update()
        #self.load_coef()

        # switching activity
        self.sa     = sa
        self.sa_out = sa_out

    def rows_out(self):
        return int(math.floor((self.rows_in()-self.k_size+2*self.pad)/self.stride)+1)

    def cols_out(self):
        return int(math.floor((self.cols_in()-self.k_size+2*self.pad)/self.stride)+1)

    def channels_out(self):
        return self.filters

    def rate_in(self,index):
        return abs(self.balance_module_rates(self.rates_graph())[0,0])

    def rate_out(self,index):
        return abs(self.balance_module_rates(self.rates_graph())[4,5])

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1):
        parameters.batch_size   = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        parameters.coarse_in    = self.coarse_in
        parameters.coarse_out   = self.coarse_out
        parameters.fine         = self.fine
        parameters.filters      = self.filters
        parameters.kernel_size  = self.k_size
        parameters.stride       = self.stride
        parameters.groups       = self.groups
        parameters.pad          = self.pad
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left

    """
        return {
            'type'          : 'CONVOLUTION',
            'buffer_depth'  : self.buffer_depth,
            'rows_in'       : self.rows_in(),
            'cols_in'       : self.cols_in(),
            'channels_in'   : self.channels_in(),
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out(),
            'filters'       : self.filters,
            'kernel_size'   : self.k_size,
            'stride'        : self.stride,
            'groups'        : self.groups,
            'pad'           : self.pad,
            'pad_top'       : self.pad_top,
            'pad_right'     : self.pad_right,
            'pad_bottom'    : self.pad_bottom,
            'pad_left'      : self.pad_left,
            'coarse_in'     : self.coarse_in,
            'coarse_out'    : self.coarse_out,
            'fine'          : self.fine,
            'size_in'       : int(self.rows*self.cols*self.channels),
            'size_out'      : int(self.rows_out()*self.cols_out()*self.channels_out()),
            'wr_factor'     : wr_factor
        }
    """

    ## UPDATE MODULES ##
    def update(self): # TODO: update all parameters
        # sliding window
        self.modules['sliding_window'].rows     = self.rows_in()
        self.modules['sliding_window'].cols     = self.cols_in()
        self.modules['sliding_window'].channels = int(self.channels/self.coarse_in)
        # fork
        self.modules['fork'].rows     = self.rows_out()
        self.modules['fork'].cols     = self.cols_out()
        self.modules['fork'].channels = int(self.channels/self.coarse_in)
        self.modules['fork'].coarse   = self.coarse_out
        # conv
        self.modules['conv'].rows     = self.rows_out()
        self.modules['conv'].cols     = self.cols_out()
        self.modules['conv'].channels = int(self.channels/self.coarse_in)
        self.modules['conv'].filters  = int(self.filters/(self.coarse_out*self.groups))
        self.modules['conv'].fine     = self.fine
        # accum
        self.modules['accum'].rows     = self.rows_out()
        self.modules['accum'].cols     = self.cols_out()
        self.modules['accum'].channels = int(self.channels/(self.coarse_in))
        self.modules['accum'].filters  = int(self.filters/(self.coarse_out))
        self.modules['accum'].groups   = self.groups
        # glue
        self.modules['glue'].rows       = self.rows_out()
        self.modules['glue'].cols       = self.cols_out()
        self.modules['glue'].filters    = self.filters
        self.modules['glue'].coarse_in  = self.coarse_in
        self.modules['glue'].coarse_out = self.coarse_out


    ### RATES ### TODO
    def rates_graph(self):
        rates_graph = np.zeros( shape=(5,6) , dtype=float )
        # sliding_window
        if self.k_size == 1:
            rates_graph[0,0] = 1
            rates_graph[0,1] = 1
        else:
            rates_graph[0,0] = self.modules['sliding_window'].rate_in()
            rates_graph[0,1] = self.modules['sliding_window'].rate_out()
        # fork
        rates_graph[1,1] = self.modules['fork'].rate_in()
        rates_graph[1,2] = self.modules['fork'].rate_out()
        # conv
        rates_graph[2,2] = self.modules['conv'].rate_in()
        rates_graph[2,3] = self.modules['conv'].rate_out()
        # accum
        rates_graph[3,3] = self.modules['accum'].rate_in()
        rates_graph[3,4] = self.modules['accum'].rate_out()
        # glue 
        rates_graph[4,4] = self.modules['glue'].rate_in()
        rates_graph[4,5] = self.modules['glue'].rate_out()

        return rates_graph

    def get_coarse_in_feasible(self,wr_factor=1):
        return self.get_factors(int(self.channels_in()/(self.groups*wr_factor)))

    def get_coarse_out_feasible(self,wr_factor=1):
        return self.get_factors(int(self.channels_out()/(self.groups*wr_factor)))

    def get_fine_feasible(self):
        #return self.get_factors(int(self.k_size*self.k_size))
        return [ 1, self.k_size, self.k_size*self.k_size ]

    def get_weights_reloading_feasible(self):
        return self.get_factors(int(self.filters/(self.groups*self.coarse_out)))

    def get_parameters_size(self):
        weights_size = self.channels * int( self.filters / self.groups ) * self.k_size * self.k_size
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def resource(self):

        sw_rsc      = self.modules['sliding_window'].rsc()
        fork_rsc    = self.modules['fork'].rsc()
        conv_rsc    = self.modules['conv'].rsc()
        accum_rsc   = self.modules['accum'].rsc()
        glue_rsc    = self.modules['glue'].rsc()

        if self.k_size == 1:
            sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.coarse_out == 1:
            fork_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if int(self.channels/self.coarse_in) == 1:
            accum_rsc   = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.coarse_in == 1:
            glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        # weight usage
        n_filters = float(self.filters*self.channels*self.k_size*self.k_size)/float(self.fine*self.groups*self.coarse_in*self.coarse_out)
        weights_bram_usage = int(math.ceil((self.weight_width*n_filters)/18000))*self.coarse_in*self.coarse_out*self.fine

        # Total
        return {
            "LUT"  :  sw_rsc['LUT']*self.coarse_in +
                      fork_rsc['LUT']*self.coarse_in +
                      conv_rsc['LUT']*self.coarse_in*self.coarse_out +
                      accum_rsc['LUT']*self.coarse_in*self.coarse_out +
                      glue_rsc['LUT'],
            "FF"   :  sw_rsc['FF']*self.coarse_in +
                      fork_rsc['FF']*self.coarse_in +
                      conv_rsc['FF']*self.coarse_in*self.coarse_out +
                      accum_rsc['FF']*self.coarse_in*self.coarse_out +
                      glue_rsc['FF'],
            "BRAM" :  sw_rsc['BRAM']*self.coarse_in +
                      fork_rsc['BRAM']*self.coarse_in +
                      conv_rsc['BRAM']*self.coarse_in*self.coarse_out +
                      accum_rsc['BRAM']*self.coarse_out +
                      glue_rsc['BRAM'] +
                      weights_bram_usage,
            "DSP" :   sw_rsc['DSP']*self.coarse_in +
                      fork_rsc['DSP']*self.coarse_in +
                      conv_rsc['DSP']*self.coarse_in*self.coarse_out +
                      accum_rsc['DSP']*self.coarse_in*self.coarse_out +
                      glue_rsc['DSP']
        }

    """
    def static_power(self):

        static_power = 0

        static_power += self.coarse_in*self.modules['sliding_window'].static_power()
        static_power += self.coarse_in*self.modules['fork'].static_power()
        static_power += self.coarse_in*self.coarse_out*self.modules['conv'].static_power()
        if self.channels != 1:
            static_power += self.coarse_in*self.coarse_out*self.modules['accum'].static_power()
        if self.channels != 1:
            static_power += self.modules['glue'].static_power()

        # Total
        return static_power


    def dynamic_power(self, freq, rate): # TODO: get models

        freq = freq/1000
        dynamic_power = 0

        dynamic_power += self.coarse_in*self.modules['sliding_window'].dynamic_power(freq,rate,self.sa,self.sa_out)
        dynamic_power += self.coarse_in*self.modules['fork'].dynamic_power(freq,rate,self.sa,self.sa_out)
        dynamic_power += self.coarse_in*self.coarse_out*self.modules['conv'].dynamic_power(freq,rate,self.sa,self.sa_out)

        # update rate
        rate = rate*self.fine/float(self.k_size*self.k_size)

        if self.channels != 1:
            dynamic_power += self.coarse_in*self.coarse_out*self.modules['accum'].dynamic_power(freq,rate,self.sa,self.sa_out)
        if self.channels != 1:
            dynamic_power += self.modules['glue'].dynamic_power(freq,rate,self.sa,self.sa_out)

        # Total
        return dynamic_power
    """

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"sw",str(i)]), label="sw" ))

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"fork",str(i)]), label="fork" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"sw",str(i)]) , "_".join([name,"fork",str(i)]) ))

        for i in range(self.coarse_in):
            for j in range(self.coarse_out):
                cluster.add_node(pydot.Node( "_".join([name,"conv",str(i),str(j)]), label="conv" ))
                cluster.add_edge(pydot.Edge( "_".join([name,"fork",str(i)]) , "_".join([name,"conv",str(i),str(j)]) ))

        for i in range(self.coarse_in):
            for j in range(self.coarse_out):
                cluster.add_node(pydot.Node( "_".join([name,"glue",str(j)]), label="+" ))
                cluster.add_node(pydot.Node( "_".join([name,"accum",str(i),str(j)]), label="accum" ))
                cluster.add_edge(pydot.Edge( "_".join([name,"conv" ,str(i),str(j)]), "_".join([name,"accum",str(i),str(j)]) ))
                cluster.add_edge(pydot.Edge( "_".join([name,"accum",str(i),str(j)]), "_".join([name,"glue",str(j)]) ))

        return cluster, "_".join([name,"sw"]), "_".join([name,"glue"])

    def functional_model(self,data,weights,bias,batch_size=1):

        assert data.shape[0] == self.rows    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters , "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == int(self.channels/self.groups), "ERROR (weights): invalid channel dimension"
        assert weights.shape[2] == self.k_size  , "ERROR (weights): invalid kernel dimension"
        assert weights.shape[3] == self.k_size  , "ERROR (weights): invalid kernel dimension"

        assert bias.shape[0] == self.filters  , "ERROR (bias): invalid filter dimension"

        # create Caffe Layer
        net = caffe.NetSpec()

        lmdb_path = '/tmp/lmdb'
        lmdb = tools.third_party.lmdb_io.LMDB(lmdb_path)

        write_images = [(np.random.rand(self.rows, self.cols, self.channels)*255).astype(np.uint8)]*batch_size
        write_labels = [0]*batch_size
        lmdb.write(write_images,write_labels)

        # create inputs
        net.data, _ = caffe.layers.Data(
            batch_size = batch_size,
            backend = caffe.params.Data.LMDB,
            source = lmdb_path,
            transform_param = dict(scale = 1./255),
            ntop = 2)

        net.conv = caffe.layers.Convolution(
            net.data,
            kernel_size=self.k_size,
            stride=self.stride,
            group=self.groups,
            pad=self.pad,
            num_output=self.filters,
            weight_filler=dict(type='xavier')
        )
        #print(net.to_proto())

        train_prototxt = tempfile.NamedTemporaryFile(prefix='train_',suffix='.prototxt')
        test_prototxt  = tempfile.NamedTemporaryFile(prefix='test_',suffix='.prototxt')

        # write train file
        train_prototxt.write(str(net.to_proto()).encode('utf-8'))
        train_prototxt.seek(0)

        # convert to deploy prototxt
        tools.third_party.prototxt.train2deploy(train_prototxt.name,[batch_size,self.channels,self.rows,self.cols],test_prototxt.name)

        # Load network
        net = caffe.Net(test_prototxt.name,caffe.TEST)

        # Close temporary files
        train_prototxt.close()
        test_prototxt.close()

        # load network inputs
        net.blobs['data'].data[...][0]  = np.moveaxis(data, -1, 0)
        net.params['conv'][0].data[...] = weights
        net.params['conv'][1].data[...] = bias

        # run network
        net.forward()

        return net.blobs['conv'].data[0]
