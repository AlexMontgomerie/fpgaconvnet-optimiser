import torch
import math
import numpy as np
import pydot

from fpgaconvnet_optimiser.models.modules import SlidingWindow
from fpgaconvnet_optimiser.models.modules import Pool
from fpgaconvnet_optimiser.models.layers import Layer

from fpgaconvnet_optimiser.tools.onnx_helper import _pair, _quadruple

class PoolingLayer(Layer):
    def __init__(
            self,
            dim,
            pool_type   ='max',
            k_size      =[2,2],
            stride      =[2,2],
            pad         =[0,0,0,0],
            coarse_in   =1,
            coarse_out  =1,
            fine        =1,
            data_width  =16,
            sa          =0.5,
            sa_out      =0.5
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width)

        k_size = _pair(k_size)
        stride = _pair(stride)
        pad    = _quadruple(pad)

        # update flags
        self.flags['transformable'] = True

        self.k_size     = k_size
        self.stride     = stride
        self.pad_top    = pad[0]
        self.pad_right  = pad[3]
        self.pad_bottom = pad[2]
        self.pad_left   = pad[1]
        assert(self.pad_top == self.pad_bottom and self.pad_left == self.pad_right)
        self.pad        = [self.pad_top, self.pad_left] 
        self.fine       = fine
        self.pool_type  = pool_type
 
        if pool_type == 'max':
            self.fine = self.k_size[0] * self.k_size[1]

        # init modules
        self.modules = {
            "sliding_window" : SlidingWindow(dim, k_size, stride, self.pad_top, self.pad_right, self.pad_bottom, self.pad_left, data_width),
            "pool"           : Pool(dim, k_size, data_width=data_width)
        }
        self.update()
        #self.load_coef()

        # switching activity
        self.sa     = sa
        self.sa_out = sa_out

    def rows_out(self):
        return int(math.ceil((self.rows_in()-self.k_size[0]+self.pad_top+self.pad_bottom)/self.stride[0])+1)

    def cols_out(self):
        return int(math.ceil((self.cols_in()-self.k_size[1]+self.pad_left+self.pad_right)/self.stride[1])+1)

    def rate_in(self, index):
        return abs(self.balance_module_rates(self.rates_graph())[0,0])
    
    def rate_out(self, index):
        return abs(self.balance_module_rates(self.rates_graph())[1,2])

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
        parameters.kernel_size.extend(self.k_size)
        parameters.stride.extend(self.stride)
        parameters.pad.extend(self.pad)
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left
  
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

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"sw",str(i)]), label="sw" ))

        for i in range(self.coarse_out):
            cluster.add_node(pydot.Node( "_".join([name,"pool",str(i)]), label="pool" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"sw",str(i)]) , "_".join([name,"pool",str(i)]) ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"sw",str(i)]) for i in range(self.coarse_in) ]
        nodes_out = [ "_".join([name,"pool",str(i)]) for i in range(self.coarse_out) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self,data,batch_size=1):

        assert data.shape[0] == self.rows    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR (data): invalid channel dimension"

        # instantiate pooling layer
        pooling_layer = torch.nn.MaxPool2d(self.k_size, stride=self.stride, padding=self.pad)
        
        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0) 
        return pooling_layer(torch.from_numpy(data)).detach().numpy()
