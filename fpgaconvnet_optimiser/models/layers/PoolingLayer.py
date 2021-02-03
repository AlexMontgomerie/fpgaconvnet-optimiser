import torch
import math
import numpy as np
import pydot

from fpgaconvnet_optimiser.models.modules import SlidingWindow
from fpgaconvnet_optimiser.models.modules import Pool
from fpgaconvnet_optimiser.models.layers import Layer

class PoolingLayer(Layer):
    def __init__(
            self,
            rows,
            cols,
            channels,
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
        Layer.__init__(self, [rows], [cols], [channels], [coarse_in], [coarse_out], data_width)

        # update flags
        self.flags['transformable'] = True

        self.k_size     = k_size
        self.stride     = stride
        self.pad        = pad
        self.pad_top    = pad + (self.rows[0] - k_size + 2*pad) % stride
        self.pad_right  = pad + (self.cols[0] - k_size + 2*pad) % stride
        self.pad_bottom = pad
        self.pad_left   = pad
        self.fine       = fine
        self.pool_type  = pool_type
 
        if pool_type == 'max':
            self.fine = self.k_size * self.k_size

        # init modules
        self.modules = {
            "sliding_window" : SlidingWindow(rows, cols, channels, k_size, stride, self.pad_top, self.pad_right, self.pad_bottom, self.pad_left, data_width),
            "pool"           : Pool(rows, cols, channels, k_size)
        }
        self.update()
        #self.load_coef()

        # rows and cols out
        self.rows_out = lambda : int(math.ceil((self.rows_in(0)-self.k_size+2*self.pad)/self.stride)+1)
        self.cols_out = lambda : int(math.ceil((self.cols_in(0)-self.k_size+2*self.pad)/self.stride)+1)

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
        parameters.rows_in      = self.rows_in(0)
        parameters.rows_in      = self.rows_in(0)
        parameters.cols_in      = self.cols_in(0)
        parameters.channels_in  = self.channels_in0()
        parameters.rows_out     = self.rows_out(0)
        parameters.cols_out     = self.cols_out(0)
        parameters.channels_out = self.channels_out(0)
        parameters.coarse       = self.coarse_in[0]
        parameters.coarse_in    = self.coarse_in[0]
        parameters.coarse_out   = self.coarse_out[0]
        parameters.kernel_size  = self.k_size
        parameters.stride       = self.stride
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
        # pool
        self.modules['pool'].rows     = self.rows_out(0)
        self.modules['pool'].cols     = self.cols_out(0)
        self.modules['pool'].channels = int(self.channels[0]/self.coarse_in[0])

    ### RATES ### TODO
    def rates_graph(self):
        rates_graph = np.zeros( shape=(2,3) , dtype=float )
        # sliding_window
        rates_graph[0,0] = self.modules['sliding_window'].rate_in(0)
        rates_graph[0,1] = self.modules['sliding_window'].rate_out(0)
        # pool
        rates_graph[1,1] = self.modules['pool'].rate_in(0)
        rates_graph[1,2] = self.modules['pool'].rate_out(0)

        return rates_graph

    def get_fine_feasible(self):
        return [1] 

    def resource(self):

        sw_rsc      = self.modules['sliding_window'].rsc()
        pool_rsc    = self.modules['pool'].rsc()
        
        # Total
        return {
            "LUT"  :  sw_rsc['LUT']*self.coarse_in[0] +
                      pool_rsc['LUT']*self.coarse_in[0],
            "FF"   :  sw_rsc['FF']*self.coarse_in[0] +
                      pool_rsc['FF']*self.coarse_in[0],
            "BRAM" :  sw_rsc['BRAM']*self.coarse_in[0] +
                      pool_rsc['BRAM']*self.coarse_in[0],
            "DSP" :   sw_rsc['DSP']*self.coarse_in[0] +
                      pool_rsc['DSP']*self.coarse_in[0]
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in[0]):
            cluster.add_node(pydot.Node( "_".join([name,"sw",str(i)]), label="sw" ))

        for i in range(self.coarse_out[0]):
            cluster.add_node(pydot.Node( "_".join([name,"pool",str(i)]), label="pool" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"sw",str(i)]) , "_".join([name,"pool",str(i)]) ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"sw",str(i)]) for i in range(self.coarse_in[0]) ]
        nodes_out = [ "_".join([name,"pool",str(i)]) for i in range(self.coarse_out[0]) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self,data,batch_size=1):

        assert data.shape[0] == self.rows[0]    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols[0]    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels[0], "ERROR (data): invalid channel dimension"

        # instantiate pooling layer
        pooling_layer = torch.nn.MaxPool2d(self.k_size, stride=self.stride, padding=self.pad)
        
        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0) 
        return pooling_layer(torch.from_numpy(data)).detach().numpy()
