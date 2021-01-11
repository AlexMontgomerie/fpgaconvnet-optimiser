from models.modules.ReLU import ReLU
from models.layers.Layer import Layer

import torch
import numpy as np
import math
import onnx
import pydot

class ReLULayer(Layer):
    def __init__(
            self,
            dim,
            data_width  =16,
            coarse_in   =1,
            coarse_out  =1,
            sa          =0.5,
            sa_out      =0.5
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width)

        # init modules
        self.modules = {
            "relu" : ReLU(dim)
        }
        self.update()

        # switching activity
        self.sa     = sa
        self.sa_out = sa_out

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
        parameters.coarse       = self.coarse_out

    ## UPDATE MODULES ##
    def update(self):
        self.modules['relu'].rows     = self.rows_in()
        self.modules['relu'].cols     = self.cols_in()
        self.modules['relu'].channels = int(self.channels/self.coarse_in)

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"relu",str(i)]), label="relu" ))

        return cluster, "_".join([name,"relu"]), "_".join([name,"relu"])

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

