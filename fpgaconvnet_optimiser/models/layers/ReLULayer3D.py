import torch
import numpy as np
import math
import onnx
import pydot

from fpgaconvnet_optimiser.models.modules import ReLU3D
from fpgaconvnet_optimiser.models.layers import Layer3D

class ReLULayer3D(Layer3D):
    def __init__(
            self,
            depth: int,
            rows: int,
            cols: int,
            channels: int,
            coarse: int = 1,
            data_width: int = 16
        ):

        # initialise parent class
        super().__init__(depth, rows, cols, channels, coarse, coarse,
                data_width=data_width)

        # save parameters
        self._coarse = coarse

        # init modules
        self.modules["relu"] = ReLU3D(self.depth_in(), self.rows_in(), self.cols_in(), self.channels_in()/self.coarse)
        self.update()

    @property
    def coarse(self) -> int:
        return self._coarse

    @property
    def coarse_in(self) -> int:
        return self._coarse

    @property
    def coarse_out(self) -> int:
        return self._coarse

    @coarse.setter
    def coarse(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self.coarse_out = val
        self.update()

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    def layer_info(self,parameters,batch_size=1):
        Layer3D.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse

    def update(self):
        self.modules['relu'].depth    = self.depth_in()
        self.modules['relu'].rows     = self.rows_in()
        self.modules['relu'].cols     = self.cols_in()
        self.modules['relu'].channels = int(self.channels_in()/self.coarse)

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse):
            cluster.add_node(pydot.Node( "_".join([name,"relu",str(i)]), label="relu" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"relu",str(i)]) for i in range(self.streams_in()) ]
        nodes_out = [ "_".join([name,"relu",str(i)]) for i in range(self.streams_out()) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self,data,batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR: invalid channel dimension"

        # instantiate relu layer
        relu_layer = torch.nn.ReLU()

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return relu_layer(torch.from_numpy(data)).detach().numpy()

