import numpy as np
import math
import pydot
import torch

from fpgaconvnet_optimiser.models.modules import SlidingWindow
from fpgaconvnet_optimiser.models.modules import Conv
from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.modules import Accum
from fpgaconvnet_optimiser.models.modules import Glue
from fpgaconvnet_optimiser.models.layers import Layer

class ConvolutionLayer(Layer):
    def __init__(
            self,
            filters: int,
            *args,
            k_size      =3,
            stride      =1,
            groups      =1,
            pad         =0,
            fine        =1,
            sa          =0.5,
            sa_out      =0.5
        ):

        # initialise parent class
        super().__init__(*args)

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
        self.pad_top    = pad - (self.rows_in(0) - k_size + 2*pad) % stride
        self.pad_right  = pad - (self.cols_in(0) - k_size + 2*pad) % stride
        self.pad_bottom = pad
        self.pad_left   = pad
        self.fine       = fine
        self.filters    = filters

        # init modules
        self.modules = {
            "sliding_window" : SlidingWindow(self.rows_in(0), self.cols_in(0), self.channels_in(0), k_size, stride, 
                self.pad_top, self.pad_right, self.pad_bottom, self.pad_left, self.data_width),
            "fork"           : Fork(self.rows_out(0), self.cols_out(0), self.filters, k_size, self.coarse_out),
            "conv"           : Conv(self.rows_out(0), self.cols_out(0), self.filters, filters, fine, k_size, groups),
            "accum"          : Accum(self.rows_out(0), self.cols_out(0), self.filters, filters, groups),
            "glue"           : Glue(self.rows_out(0), self.cols_out(0), self.filters, filters, self.coarse_in[0], self.coarse_out[0])
        }
        self.update()
        #self.load_coef()

        # switching activity
        self.sa     = sa
        self.sa_out = sa_out

    def rows_out(self, port_index):
        return int(math.floor((self.rows_in(0)-self.k_size+2*self.pad)/self.stride)+1)

    def cols_out(self, port_index):
        return int(math.floor((self.cols_in(0)-self.k_size+2*self.pad)/self.stride)+1)

    def channels_out(self, port_index):
        return self.filters

    def rate_in(self,port_index):
        return abs(self.balance_module_rates(self.rates_graph())[0,0])

    def rate_out(self,port_index):
        return abs(self.balance_module_rates(self.rates_graph())[4,5])

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1):
        parameters.batch_size   = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in(0)
        parameters.cols_in      = self.cols_in(0)
        parameters.channels_in  = self.channels_in(0)
        parameters.rows_out     = self.rows_out(0)
        parameters.cols_out     = self.cols_out(0)
        parameters.channels_out = self.channels_out(0)
        parameters.coarse_in    = self.coarse_in[0]
        parameters.coarse_out   = self.coarse_out[0]
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

    ## UPDATE MODULES ##
    def update(self): 
        # sliding window
        self.modules['sliding_window'].rows     = self.rows_in(0)
        self.modules['sliding_window'].cols     = self.cols_in(0)
        self.modules['sliding_window'].channels = int(self.channels[0]/self.coarse_in[0])
        # fork
        self.modules['fork'].rows     = self.rows_out(0)
        self.modules['fork'].cols     = self.cols_out(0)
        self.modules['fork'].channels = int(self.channels[0]/self.coarse_in[0])
        self.modules['fork'].coarse   = self.coarse_out[0]
        # conv
        self.modules['conv'].rows     = self.rows_out(0)
        self.modules['conv'].cols     = self.cols_out(0)
        self.modules['conv'].channels = int(self.channels[0]/self.coarse_in[0])
        self.modules['conv'].filters  = int(self.filters/(self.coarse_out[0]*self.groups))
        self.modules['conv'].fine     = self.fine
        # accum
        self.modules['accum'].rows     = self.rows_out(0)
        self.modules['accum'].cols     = self.cols_out(0)
        self.modules['accum'].channels = int(self.channels[0]/(self.coarse_in[0]))
        self.modules['accum'].filters  = int(self.filters/(self.coarse_out[0]))
        self.modules['accum'].groups   = self.groups
        # glue
        self.modules['glue'].rows       = self.rows_out(0)
        self.modules['glue'].cols       = self.cols_out(0)
        self.modules['glue'].filters    = self.filters
        self.modules['glue'].coarse_in  = self.coarse_in[0]
        self.modules['glue'].coarse_out = self.coarse_out[0]


    ### RATES ### 
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
        return self.get_factors(int(self.channels_in(0)/(self.groups*wr_factor)))

    def get_coarse_out_feasible(self,wr_factor=1):
        return self.get_factors(int(self.channels_out(0)/(self.groups*wr_factor)))

    def update_coarse_in(self, coarse_in):
        self.coarse_in[0]  = coarse_in

    def update_coarse_out(self, coarse_out):
        self.coarse_out[0] = coarse_out

    def get_fine_feasible(self):
        #return self.get_factors(int(self.k_size*self.k_size))
        return [ 1, self.k_size, self.k_size*self.k_size ]

    def get_weights_reloading_feasible(self):
        return self.get_factors(int(self.filters/(self.groups*self.coarse_out[0])))

    def get_parameters_size(self):
        weights_size = self.channels[0] * int( self.filters / self.groups ) * self.k_size * self.k_size
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def get_operations(self):
        return self.k_size*self.k_size*self.channels_in(0)*self.filters*self.rows_out(0)*self.cols_out(0)

    def resource(self):

        sw_rsc      = self.modules['sliding_window'].rsc()
        fork_rsc    = self.modules['fork'].rsc()
        conv_rsc    = self.modules['conv'].rsc()
        accum_rsc   = self.modules['accum'].rsc()
        glue_rsc    = self.modules['glue'].rsc()

        if self.k_size == 1:
            sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.coarse_out[0] == 1:
            fork_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if int(self.channels[0]/self.coarse_in[0]) == 1:
            accum_rsc   = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.coarse_in[0] == 1:
            glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        # weight usage
        n_filters = float(self.filters*self.channels[0]*self.k_size*self.k_size)/float(self.fine*self.groups*self.coarse_in[0]*self.coarse_out[0])
        weights_bram_usage = int(math.ceil((self.weight_width*n_filters)/18000))*self.coarse_in[0]*self.coarse_out[0]*self.fine

        # Total
        return {
            "LUT"  :  sw_rsc['LUT']*self.coarse_in[0] +
                      fork_rsc['LUT']*self.coarse_in[0] +
                      conv_rsc['LUT']*self.coarse_in[0]*self.coarse_out[0] +
                      accum_rsc['LUT']*self.coarse_in[0]*self.coarse_out[0] +
                      glue_rsc['LUT'],
            "FF"   :  sw_rsc['FF']*self.coarse_in[0] +
                      fork_rsc['FF']*self.coarse_in[0] +
                      conv_rsc['FF']*self.coarse_in[0]*self.coarse_out[0] +
                      accum_rsc['FF']*self.coarse_in[0]*self.coarse_out[0] +
                      glue_rsc['FF'],
            "BRAM" :  sw_rsc['BRAM']*self.coarse_in[0] +
                      fork_rsc['BRAM']*self.coarse_in[0] +
                      conv_rsc['BRAM']*self.coarse_in[0]*self.coarse_out[0] +
                      accum_rsc['BRAM']*self.coarse_out[0] +
                      glue_rsc['BRAM'] +
                      weights_bram_usage,
            "DSP" :   sw_rsc['DSP']*self.coarse_in[0] +
                      fork_rsc['DSP']*self.coarse_in[0] +
                      conv_rsc['DSP']*self.coarse_in[0]*self.coarse_out[0] +
                      accum_rsc['DSP']*self.coarse_in[0]*self.coarse_out[0] +
                      glue_rsc['DSP']
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in[0]):
            cluster.add_node(pydot.Node( "_".join([name,"sw",str(i)]), label="sw" ))

        for i in range(self.coarse_in[0]):
            cluster.add_node(pydot.Node( "_".join([name,"fork",str(i)]), label="fork" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"sw",str(i)]) , "_".join([name,"fork",str(i)]) ))

        for i in range(self.coarse_in[0]):
            for j in range(self.coarse_out[0]):
                cluster.add_node(pydot.Node( "_".join([name,"conv",str(i),str(j)]), label="conv" ))
                cluster.add_edge(pydot.Edge( "_".join([name,"fork",str(i)]) , "_".join([name,"conv",str(i),str(j)]) ))

        for i in range(self.coarse_in[0]):
            for j in range(self.coarse_out[0]):
                cluster.add_node(pydot.Node( "_".join([name,"glue",str(j)]), label="+" ))
                cluster.add_node(pydot.Node( "_".join([name,"accum",str(i),str(j)]), label="accum" ))
                cluster.add_edge(pydot.Edge( "_".join([name,"conv" ,str(i),str(j)]), "_".join([name,"accum",str(i),str(j)]) ))
                cluster.add_edge(pydot.Edge( "_".join([name,"accum",str(i),str(j)]), "_".join([name,"glue",str(j)]) ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"sw",str(i)]) for i in range(self.coarse_in[0]) ]
        nodes_out = [ "_".join([name,"glue",str(i)]) for i in range(self.coarse_out[0]) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self,data,weights,bias,batch_size=1):

        assert data.shape[0] == self.rows[0]    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols[0]    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels[0], "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters , "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == int(self.channels[0]/self.groups), "ERROR (weights): invalid channel dimension"
        assert weights.shape[2] == self.k_size  , "ERROR (weights): invalid kernel dimension"
        assert weights.shape[3] == self.k_size  , "ERROR (weights): invalid kernel dimension"

        assert bias.shape[0] == self.filters  , "ERROR (bias): invalid filter dimension"

        # instantiate convolution layer
        convolution_layer = torch.nn.Conv2d(self.channels[0], self.filters, self.k_size, 
                stride=self.stride, padding=self.pad, groups=self.groups)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))
        
        # update bias
        convolution_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))
        
        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0) 
        return convolution_layer(torch.from_numpy(data)).detach().numpy()

if __name__ == "__main__":

    # 
    ConvolutionLayer(
        100,
        [10],
        [20],
        [30],
        2,
        4
    )


