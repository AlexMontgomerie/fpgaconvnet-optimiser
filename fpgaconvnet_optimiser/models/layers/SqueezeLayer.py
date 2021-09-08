from fpgaconvnet_optimiser.models.layers import Layer

import pydot
import numpy as np

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
        super().__init__([rows],[cols],[channels],[coarse_in],[coarse_out],
                data_width=data_width)

        # initialise parameters
        self.coarse_in = coarse_in
        self.coarse_out = coarse_out

        # initialise modules
        self.modules["squeeze"] = Squeeze(rows, cols, channels, coarse_in,
                                          coarse_out)

    def streams_in(self, port_index=0):
        assert port_index == 0, "squeeze layers are only allowed a single port"
        return self.coarse_in

    def streams_out(self, port_index=0):
        assert port_index == 0, "squeeze layers are only allowed a single port"
        return self.coarse_out

    def update_coarse_in(self, coarse_in, port_index=0):
        assert port_index == 0, "squeeze layers are only allowed a single port"
        self.coarse_in  = coarse_in

    def update_coarse_out(self, coarse_out, port_index=0):
        assert port_index == 0, "squeeze layers are only allowed a single port"
        self.coarse_out = coarse_out

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
        nodes_in  = [ "_".join([name,"squeeze"]) for i in range(self.streams_in()) ]
        nodes_out = [ "_".join([name,"squeeze"]) for i in range(self.streams_out()) ]

        # return module
        return cluster, nodes_in, nodes_out

    def functional_model(self,data,batch_size=1):

        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        return np.repeat(data[np.newaxis,...], batch_size, axis=0)

