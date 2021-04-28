from fpgaconvnet_optimiser.models.modules import ReLU
from fpgaconvnet_optimiser.models.layers import Layer

import torch
import numpy as np
import math
import onnx
import pydot

class ReLULayer(Layer):
    def __init__(
            self,
            *args,
        ):

        # initialise parent class
        super().__init__(*args)

        # init modules
        self.modules = {
            "relu" : ReLU(self.rows_in(0), self.cols_in(0), self.channels_in(0))
        }
        self.update()

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
        parameters.coarse       = self.coarse_out[0]

    ## UPDATE MODULES ##
    def update(self):
        self.modules['relu'].rows     = self.rows_in(0)
        self.modules['relu'].cols     = self.cols_in(0)
        self.modules['relu'].channels = int(self.channels_in(0)/self.coarse_in[0])

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"relu",str(i)]), label="relu" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"relu",str(i)]) for i in range(self.coarse_in) ]
        nodes_out = [ "_".join([name,"relu",str(i)]) for i in range(self.coarse_out) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self,data,batch_size=1):

        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        # instantiate relu layer
        relu_layer = torch.nn.ReLU()
        
        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0) 
        return relu_layer(torch.from_numpy(data)).detach().numpy()

