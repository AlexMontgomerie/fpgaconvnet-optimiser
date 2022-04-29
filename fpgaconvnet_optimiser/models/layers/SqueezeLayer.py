from fpgaconvnet_optimiser.models.layers import Layer
from fpgaconvnet_optimiser.models.modules import FIFO

import pydot
import numpy as np
import math

from fpgaconvnet_optimiser.models.modules import Squeeze

class SqueezeLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int,
            coarse_out: int,
            data_width: int = 16,
        ):

        # initialise parent class
        super().__init__(rows, cols, channels, coarse_in, coarse_out,
                data_width=data_width)

        # initialise modules
        self.modules["squeeze"] = Squeeze(self.rows, self.cols, self.channels, self.coarse_in,
                                          self.coarse_out)

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)

    def update(self):
        self.modules["squeeze"].rows = self.rows
        self.modules["squeeze"].cols = self.cols
        self.modules["squeeze"].channels = self.channels
        self.modules["squeeze"].coarse_in = self.coarse_in
        self.modules["squeeze"].coarse_out = self.coarse_out

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        # add squeeze module
        cluster.add_node(pydot.Node( "_".join([name,"squeeze"]), label="squeeze" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"squeeze"]) for i in range(self.streams_in()) ]
        nodes_out = [ "_".join([name,"squeeze"]) for i in range(self.streams_out()) ]

        # return module
        return cluster, nodes_in, nodes_out

    def lcm(a, b):
        return abs(a*b) // math.gcd(a, b)

    def resource(self):        
        # streams
        channel_cache = FIFO(1, 1, 1, lcm(self.coarse_in, self.coarse_out), self.buffer_depth)
        channel_cache.data_width = self.data_width
        channel_cache_rsc = channel_cache.rsc()
        channel_cache_rsc["URAM"] = 0
        return channel_cache_rsc
        
    def functional_model(self,data,batch_size=1):

        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        return np.repeat(data[np.newaxis,...], batch_size, axis=0)

