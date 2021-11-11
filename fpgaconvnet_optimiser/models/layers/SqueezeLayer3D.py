from fpgaconvnet_optimiser.models.layers import Layer3D

import pydot
import numpy as np

from fpgaconvnet_optimiser.models.modules import Squeeze3D

class SqueezeLayer3D(Layer3D):
    def __init__(
            self,
            depth: int,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int,
            coarse_out: int,
            data_width: int = 16,
        ):

        # initialise parent class
        super().__init__(depth, rows, cols, channels, coarse_in, coarse_out,
                data_width=data_width)

        # initialise modules
        self.modules["squeeze"] = Squeeze3D(self.depth, self.rows, self.cols, self.channels, self.coarse_in,
                                          self.coarse_out)

    def layer_info(self,parameters,batch_size=1):
        Layer3D.layer_info(self, parameters, batch_size)

    def update(self):
        self.modules["squeeze"].depth = self.depth
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

    def functional_model(self,data,batch_size=1):
        
        assert data.shape[0] == self.depth   , "ERROR: invalid depth dimension"
        assert data.shape[1] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[2] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[3] == self.channels, "ERROR: invalid channel dimension"

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        return np.repeat(data[np.newaxis,...], batch_size, axis=0)

