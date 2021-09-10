"""
Exit Selection Layer

This layer merges all exit results into a single point for the output to offchip mem.
The select lines will be driven by the control signal from each exit condition layer.

"""

import numpy as np
import math
import pydot
import torch

from fpgaconvnet_optimiser.models.modules import ExitMerge
from fpgaconvnet_optimiser.models.layers import Layer

class ExitSelectLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int,
            coarse_out: int,
            #early_exit_edge, #edges record where batch ID comes from
            #late_exit_edge,
            ports_in    = 2,
            data_width  =16,
        ):
        # initialise parent class
        #rows, cols, channels will be the same for both inputs
        #coarse will be the same for inputs IF this is used
        super().__init__(   [rows,rows],
                            [cols,cols],
                            [channels,channels],
                            [coarse_in,coarse_in],
                            [coarse_out])

        #index 0 is then_branch, index 1 is else_branch
        #self.early_exit_edge = early_exit_edge
        #self.late_exit_edge  = late_exit_edge
        self.ports_in = ports_in

        #init modules
        self.modules = {
            "emerge" : ExitMerge(rows, cols, channels)#, early_exit_edge, late_exit_edge)
        }
        self.update()

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1):
        #TODO
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
        parameters.ports_in     = self.ports_in

    ## UPDATE MODULES ##
    def update(self): #TODO
        return

    ### RATES ###
    def rates_graph(self): #TODO
        rates_graph = np.zeros( shape=(1,2) , dtype=float )

        rates_graph[0,0] = self.modules['emerge'].rate_in()
        rates_graph[0,1] = self.modules['emerge'].rate_out()
        return rates_graph

    def resource(self): #TODO
        emerge_rsc    = self.modules['emerge'].rsc()

        # Total
        return {
            "LUT"  :  emerge_rsc['LUT']*self.coarse_in(0),
            "FF"   :  emerge_rsc['FF']*self.coarse_in(0),
            "BRAM" :  emerge_rsc['BRAM']*self.coarse_in(0),
            "DSP" :   emerge_rsc['DSP']*self.coarse_in(0),
        }

    def visualise(self,name): #TODO replace 'mod' with actual modules used
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in[0]):
            cluster.add_node(pydot.Node( "_".join([name,"emerge",str(i)]), label="emerge" ))

        for i in range(self.coarse_out[0]):
            cluster.add_node(pydot.Node( "_".join([name,"emerge",str(i)]), label="emerge" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"emerge",str(i)]) for i in range(self.coarse_in[0]) ]
        nodes_out = [ "_".join([name,"emerge",str(i)]) for i in range(self.coarse_out[0]) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self, EEdata, LEdata, EE_ID=None, LE_ID=None):
        #Exit merge is not an ONNX or pytorch op
        # check input dimensionality
        assert EEdata.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert EEdata.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert EEdata.shape[2] == self.channels, "ERROR: invalid channel dimension"

        assert LEdata.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert LEdata.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert LEdata.shape[2] == self.channels, "ERROR: invalid channel dimension"

        if EE_ID is not None:
            return np.concatenate(([EE_ID], EEdata))
        elif LE_ID is not None:
            return np.concatenate(([LE_ID], LEdata))
