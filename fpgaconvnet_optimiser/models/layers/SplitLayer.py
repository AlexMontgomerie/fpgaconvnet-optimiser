"""
The split/fork/branch layer.
Takes one stream input and outputs several streams using the fork module.
"""

from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.layers import Layer

import pydot
import numpy as np
import os
import math
from typing import List

class SplitLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse: int,
            ports_out = 1
        ):
        """
        Parameters
        ----------
        rows: int
            row dimension of input featuremap
        cols: int
            column dimension of input featuremap
        channels: int
            channel dimension of input featuremap

        Attributes
        ----------
        buffer_depth: int, default: 0
            depth of incoming fifo buffers for each stream in.
        rows: list int
            row dimension of input featuremap
        cols: list int
            column dimension of input featuremap
        channels: list int
            channel dimension of input featuremap
        ports_in: int
            number of ports into the layer
        ports_out: int
            number of ports out of the layer
        coarse_in: list int
            number of parallel streams per port into the layer.
        coarse_out: NEED TO DEFINE
           TODO
        data_width: int
            bitwidth of featuremap pixels
        modules: dict
            dictionary of `module` instances that make
            up the layer. These modules are used for the
            resource and performance models of the layer.
        """

        # parameters
        self.coarse = coarse

        # initialise parent class
        super().__init__([rows],[cols],[channels],[coarse],[coarse]*ports_out,ports_out=ports_out)

        # init modules
        #One fork module, fork coarse_out corresponds to number of layer output ports
        self.modules = {
            "fork"          : Fork( self.rows[0], self.cols[0], self.channels[0], 1, self.ports_out)
        }

        self.update()

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1) :
        parameters.batch_size = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in(0)
        parameters.cols_in      = self.cols_in(0)
        parameters.channels_in  = self.channels_in(0)
        parameters.rows_out     = self.rows_out(0)
        parameters.cols_out     = self.cols_out(0)
        parameters.channels_out = self.channels_out(0)
        parameters.coarse_in    = self.coarse
        parameters.coarse_out   = self.coarse
        parameters.ports_out    = self.ports_out

    ## UPDATE MODULES ##
    def update(self):
        # fork
        self.modules['fork'].rows     = self.rows_out(0)
        self.modules['fork'].cols     = self.cols_out(0)
        self.modules['fork'].channels = int(self.channels[0]/self.coarse)
        self.modules['fork'].coarse   = self.ports_out

    def rows_out(self, port_index=0):
        #rows are the same for both outputs
        #currently there isn't a separate rows dim for >1 ports
        return super().rows_out(0)

    def cols_out(self, port_index=0):
        #rows are the same for both outputs
        #currently there isn't a separate rows dim for >1 ports
        return super().cols_out(0)

    def channels_out(self, port_index=0):
        #rows are the same for both outputs
        #currently there isn't a separate rows dim for >1 ports
        return super().channels_out(0)

    def update_coarse_in(self, coarse_in, port_index=0):
        self.coarse = coarse_in
        self.coarse_in[0] = self.coarse
        for i in range(self.ports_out):
            self.coarse_out[0] = self.coarse

    def update_coarse_out(self, coarse_out, port_index=0):
        self.coarse = coarse_out
        self.coarse_in[0] = self.coarse
        for i in range(self.ports_out):
            self.coarse_out[0] = self.coarse

        ### RATES ###
    def rates_graph(self):
        rates_graph = np.zeros( shape=(1,2) , dtype=float)
        # fork
        rates_graph[0,0] = self.modules['fork'].rate_in(0)
        rates_graph[0,1] = self.modules['fork'].rate_out(0)

        return rates_graph

    def resource(self):

        # get module resources
        fork_rsc = self.modules['fork'].rsc()

        #Total
        return {
            "LUT"   :   fork_rsc['LUT']*self.coarse,
            "FF"    :   fork_rsc['FF']*self.coarse,
            "BRAM"  :   fork_rsc['BRAM']*self.coarse,
            "DSP"   :   fork_rsc['DSP']*self.coarse
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse):
            cluster.add_node(pydot.Node( "_".join([name,"fork",str(i)]), label="fork" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"fork",str(i)]) for i in range(self.coarse) ]
        nodes_out = [ "_".join([name,"fork",str(i)]) for i in range(self.ports_out*self.coarse) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self, data, batch_size=1):
        assert data.shape[0] == self.rows_in(0)    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in(0)    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(0), "ERROR (data): invalid channel dimension"


        out = np.ndarray((
            self.rows_in(0),
            self.cols_in(0),
            self.channels_in(0),
            self.ports_out),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[
              index[0],
              index[1],
              index[2]]

        return out

