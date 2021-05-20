from fpgaconvnet_optimiser.models.layers import Layer

import pydot
import numpy as np

class SqueezeLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int,
            coarse_out: int,
        ):

        # initialise parent class
        super().__init__([rows],[cols],[channels],[coarse_in],[coarse_out])

    ## UPDATE MODULES ##
    def update(self):
        pass
    
    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1):
        parameters.batch_size   = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in(0)
        parameters.rows_in      = self.rows_in(0)
        parameters.cols_in      = self.cols_in(0)
        parameters.channels_in  = self.channels_in(0)
        parameters.rows_out     = self.rows_out(0)
        parameters.cols_out     = self.cols_out(0)
        parameters.channels_out = self.channels_out(0)
        parameters.coarse_in    = self.coarse_in[0]
        parameters.coarse_out   = self.coarse_out[0]

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)
    
        # add squeeze module
        cluster.add_node(pydot.Node( "_".join([name,"squeeze"]), label="squeeze" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"squeeze"]) for i in range(self.coarse_in[0]) ]
        nodes_out = [ "_".join([name,"squeeze"]) for i in range(self.coarse_out[0]) ]

        # return module
        return cluster, nodes_in, nodes_out


