from models.modules.SlidingWindow import SlidingWindow
from models.modules.Pool import Pool
from models.layers.Layer import Layer

import math
import numpy as np
import tools.third_party.lmdb_io
import tools.third_party.prototxt
import tempfile
import caffe
import pydot

class PoolingLayer(Layer):
    def __init__(
            self,
            dim,
            pool_type   ='max',
            k_size      =2,
            stride      =2,
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
        self.flags['transformable']     = True

        self.k_size     = k_size
        self.stride     = stride
        self.pad        = pad
        self.pad_top    = pad + (self.rows - k_size + 2*pad) % stride
        self.pad_right  = pad + (self.cols - k_size + 2*pad) % stride
        self.pad_bottom = pad
        self.pad_left   = pad
        self.fine       = fine
        self.pool_type  = pool_type
 
        if pool_type == 'max':
            self.fine = self.k_size * self.k_size

        # init modules
        self.modules = {
            "sliding_window" : SlidingWindow(dim, k_size, stride, self.pad_top, self.pad_right, self.pad_bottom, self.pad_left, data_width),
            "pool"           : Pool(dim, k_size)
        }
        self.update()
        #self.load_coef()

        # rows and cols out
        self.rows_out = lambda : int(math.ceil((self.rows_in()-self.k_size+2*self.pad)/self.stride)+1)
        self.cols_out = lambda : int(math.ceil((self.cols_in()-self.k_size+2*self.pad)/self.stride)+1)

        # rates
        self.rate_in  = lambda i : abs(self.balance_module_rates(self.rates_graph())[0,0])
        self.rate_out = lambda i : abs(self.balance_module_rates(self.rates_graph())[1,2])

        # switching activity
        self.sa     = sa
        self.sa_out = sa_out

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1):
        parameters.batch_size   = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in()
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        parameters.coarse       = self.coarse_in
        parameters.coarse_in    = self.coarse_in
        parameters.coarse_out   = self.coarse_out
        parameters.kernel_size  = self.k_size
        parameters.stride       = self.stride
        parameters.pad          = self.pad
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left

  
    """
    return {
            'type'          : 'POOLING',
            'buffer_depth'  : self.buffer_depth,
            'rows'          : self.rows,
            'cols'          : self.cols,
            'channels'      : self.channels,
            'kernel_size'   : self.k_size,
            'stride'        : self.stride,
            'pad'           : self.pad,
            'pad_top'       : self.pad_top,
            'pad_right'     : self.pad_right,
            'pad_bottom'    : self.pad_bottom,
            'pad_left'      : self.pad_left,
            'coarse'        : self.coarse_in,
            'coarse_in'     : self.coarse_in,
            'coarse_out'    : self.coarse_out,
            'fine'          : self.fine,
            'pool_type'     : self.pool_type,
            'size_in'       : int(self.rows*self.cols*self.channels),
            'size_out'      : int(self.rows_out()*self.cols_out()*self.channels_out()),
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
    }   
    """

    ## UPDATE MODULES ##
    def update(self):
        # sliding window
        self.modules['sliding_window'].rows     = self.rows_in()
        self.modules['sliding_window'].cols     = self.cols_in()
        self.modules['sliding_window'].channels = int(self.channels/self.coarse_in)
        # pool
        self.modules['pool'].rows     = self.rows_out()
        self.modules['pool'].cols     = self.cols_out()
        self.modules['pool'].channels = int(self.channels/self.coarse_in)

    ### RATES ### TODO
    def rates_graph(self):
        rates_graph = np.zeros( shape=(2,3) , dtype=float )
        # sliding_window
        rates_graph[0,0] = self.modules['sliding_window'].rate_in()
        rates_graph[0,1] = self.modules['sliding_window'].rate_out()
        # pool
        rates_graph[1,1] = self.modules['pool'].rate_in()
        rates_graph[1,2] = self.modules['pool'].rate_out()

        return rates_graph

    def get_fine_feasible(self):
        return [1] 

    def resource(self):

        sw_rsc      = self.modules['sliding_window'].rsc()
        pool_rsc    = self.modules['pool'].rsc()
        
        # Total
        return {
            "LUT"  :  sw_rsc['LUT']*self.coarse_in +
                      pool_rsc['LUT']*self.coarse_in,
            "FF"   :  sw_rsc['FF']*self.coarse_in +
                      pool_rsc['FF']*self.coarse_in,
            "BRAM" :  sw_rsc['BRAM']*self.coarse_in +
                      pool_rsc['BRAM']*self.coarse_in,
            "DSP" :   sw_rsc['DSP']*self.coarse_in +
                      pool_rsc['DSP']*self.coarse_in
        }

    """
    def static_power(self):

        static_power = 0

        static_power += self.coarse_in*self.modules['sliding_window'].static_power()
        static_power += self.coarse_in*self.modules['pool'].static_power()

        # Total
        return static_power


    def dynamic_power(self, freq, rate):

        freq = freq/1000
        dynamic_power = 0

        dynamic_power += self.coarse_in*self.modules['sliding_window'].dynamic_power(freq,rate,self.sa,self.sa_out)
        dynamic_power += self.coarse_in*self.modules['pool'].dynamic_power(freq,rate,self.sa,self.sa_out)

        # Total
        return dynamic_power

    """
    
    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"sw",str(i)]), label="sw" ))

        for i in range(self.coarse_out):
            cluster.add_node(pydot.Node( "_".join([name,"pool",str(i)]), label="pool" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"sw",str(i)]) , "_".join([name,"pool",str(i)]) ))

        return cluster, "_".join([name,"sw"]), "_".join([name,"pool"])

    def functional_model(self,data,batch_size=1):

        assert data.shape[0] == self.rows    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR (data): invalid channel dimension"

        # create Caffe Layer
        net = caffe.NetSpec()

        #lmdb_path = 'outputs/lmdb'
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

        net.pool = caffe.layers.Pooling(
            net.data,
            pool = caffe.params.Pooling.MAX if self.pool_type == 'max' else caffe.params.Pooling.AVG,
            kernel_size=self.k_size,
            stride=self.stride,
            pad=self.pad
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

        # run network
        net.forward()

        return net.blobs['pool'].data[0]
