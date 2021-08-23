from fpgaconvnet_optimiser.models.layers import Layer
from fpgaconvnet_optimiser.models.modules import FIFO

import pydot
import numpy as np

class SqueezeLayer(Layer):
    def __init__(
            self,
            dim,
            coarse_in   =1,
            coarse_out  =1,
            data_width  =16,
            sa          =0.5,
            sa_out      =0.5
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width)

    ## UPDATE MODULES ##
    def update(self):
        pass
    
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
        parameters.coarse_in    = self.coarse_in
        parameters.coarse_out   = self.coarse_out

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)
    
        # add squeeze module
        cluster.add_node(pydot.Node( "_".join([name,"squeeze"]), label="squeeze" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"squeeze"]) for i in range(self.coarse_in) ]
        nodes_out = [ "_".join([name,"squeeze"]) for i in range(self.coarse_out) ]

        # return module
        return cluster, nodes_in, nodes_out

    def resource(self):        
        # streams
        channel_cache = FIFO([1,1,1], self.channels_in(), self.buffer_depth, self.data_width)
        channel_cache_rsc = channel_cache.rsc()
        return channel_cache_rsc

