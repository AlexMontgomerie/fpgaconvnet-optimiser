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

class SplitLayer(Layer):
    def __init__(
            self,
            rows,
            cols,
            channels,
            coarse_in,
            coarse_out,
            ports_in    = 1,
            ports_out   = 2,
            data_width  = 16
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

        Layer.__init__(self, [rows], [cols], [channels], [coarse_in], [coarse_out], ports_in, ports_out, data_width)

        # init modules
        #One fork module, fork coarse_out corresponds to number of layer output ports
        self.modules = {
            "fork"          : Fork( self.rows[0], self.cols[0], self.channels[0], 1, ports_out)
        }

        self.update()

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1) :
        parameters.batchsize = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in(0)
        parameters.cols_in      = self.cols_in(0)
        parameters.channels_in  = self.channels_in(0)
        parameters.rows_out     = self.rows_out(0)
        parameters.cols_out     = self.cols_out(0)
        parameters.channels_out = self.channels_out(0)
        parameters.coarse_in    = self.coarse_in[0]
        parameters.coarse_out   = self.ports_out

    ## UPDATE MODULES ##
    def update(self):
        # fork
        self.modules['fork'].rows     = self.rows_out(0)
        self.modules['fork'].cols     = self.cols_out(0)
        self.modules['fork'].channels = int(self.channels[0]/self.coarse_in[0])
        self.modules['fork'].coarse   = self.ports_out

        ### RATES ###
    def rates_graph(self):
        rates_graph = np.zeros( shape=(5,6) , dtype=float)
        # fork
        rates_graph[1,1] = self.modules['fork'].rate_in(0)
        rates_graph[1,2] = self.modules['fork'].rate_out(0)

        return rates_graph
        
    def resource(self):
        fork_rsc = self.modules['fork'].rsc()

        if self.coarse_out[0] == 1:
            fork_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        #Total
        return {
            "LUT"   :   fork_rsc['LUT']*self.coarse_in[0],
            "FF"    :   fork_rsc['FF']*self.coarse_in[0],
            "BRAM"  :   fork_rsc['BRAM']*self.coarse_in[0],
            "DSP"   :   fork_rsc['DSP']*self.coarse_in[0]
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"split",str(i)]), label="split" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"split",str(i)]) for i in range(self.coarse_in) ]
        nodes_out = [ "_".join([name,"split",str(i)]) for i in range(self.ports_out) ]

        return cluster, nodes_in, nodes_out

    #TODO: workout if there's something in torch that supports this
    #def functional_model(self, data, batch_size=1):

