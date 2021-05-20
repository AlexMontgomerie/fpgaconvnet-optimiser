"""
Exit Selection Layer

This layer merges all exit results into a single point for the output to offchip mem.
The select lines will be driven by the control signal from each exit condition layer.

"""

import numpy as np
import math
import pydot
import torch

from fpgaconvnet_optimiser.models.layers import Layer

class ExitSelectLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int,
            coarse_out: int,
            ctrledge: str,
            data_width  =16,
        ):
        # initialise parent class
        super().__init__([rows],[cols],[channels],[coarse_in],[coarse_out])

        #ctrledge links to exit condition layer
        self.ctrledge = ctrledge

        #init modules
        self.modules = {
        }
        self.update()

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

    ## UPDATE MODULES ##
    def update(self): #TODO
        return

    ### RATES ###
    def rates_graph(self):
        rates_graph = np.zeros( shape=(1,2) , dtype=float )

        #rates_graph[0,0] = self.modules['mod'].rate_in()
        #rates_graph[0,1] = self.modules['mod'].rate_out()
        return rates_graph

    def resource(self):

        mod_rsc    = self.modules['mod'].rsc()

        # Total
        return {
            "LUT"  :  buff_rsc['LUT']*self.coarse_in,
            "FF"   :  buff_rsc['FF']*self.coarse_in,
            "BRAM" :  buff_rsc['BRAM']*self.coarse_in,
            "DSP" :   buff_rsc['DSP']*self.coarse_in,
        }

    def visualise(self,name): #TODO replace 'mod' with actual modules used
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"mod",str(i)]), label="mod" ))

        for i in range(self.coarse_out):
            cluster.add_node(pydot.Node( "_".join([name,"mod",str(i)]), label="mod" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"mod",str(i)]) for i in range(self.coarse_in) ]
        nodes_out = [ "_".join([name,"mod",str(i)]) for i in range(self.coarse_out) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self, EEdata, LEdata, ctrl_pass):
        #Exit select is not an ONNX or pytorch op
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        if ctrl_pass:
            return EEdata
        else:
            return LEdata
