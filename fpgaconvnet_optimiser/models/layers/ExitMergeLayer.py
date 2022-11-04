"""
Exit Merge Layer

This layer merges all exit results into a single point for the output to offchip mem.
For now, this just makes sure all data samples for a given batch are kept together/fed out to memory sequentially.

"""

import numpy as np
import math
import pydot
import torch

from fpgaconvnet_optimiser.models.modules import ExitMerge
from fpgaconvnet_optimiser.models.layers import MultiPortLayer

class ExitMergeLayer(MultiPortLayer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse: int = 1,
            #early_exit_edge, #edges record where batch ID comes from
            #late_exit_edge,
            ports_in: int = 2,
            data_width: int =16,
        ):
        # initialise parent class
        #rows, cols, channels will be the same for both inputs
        #coarse will be the same for inputs IF this is used
        super().__init__(   [rows,rows],
                            [cols,cols],
                            [channels,channels],
                            [coarse,coarse],
                            [coarse], ports_in=ports_in)

        #index 0 is then_branch, index 1 is else_branch
        #self.early_exit_edge = early_exit_edge
        #self.late_exit_edge  = late_exit_edge
        self._coarse    = coarse
        self._ports_in  = ports_in

        #init modules
        self.modules["emerge"] = ExitMerge(self.rows_in(), self.cols_in(), self.channels_in(),exits=self._ports_in)
        self.update()

    #@property
    #def ports_in(self) -> int:
    #    return self._ports_in

    @property
    def coarse(self) -> int:
        return self._coarse

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1):
        #TODO
        MultiPortLayer.layer_info(self, parameters, batch_size)
        parameters.coarse   = self.coarse
        parameters.ports_in = self.ports_in

    def update(self): #TODO
        self.modules["emerge"].rows = self.rows_in()
        self.modules["emerge"].cols = self.cols_in()
        #FIXME provision for different coarse rates in exitmerge module
        self.modules["emerge"].channels = int(self.channels_in()/self.coarse_in[0])
            #[int(self.channels_in(chan_i)/crse) for chan_i,crse in enumerate(self.coarse_in)]

    def resource(self): #TODO
        emerge_rsc    = self.modules['emerge'].rsc()

        # Total
        return {
            "LUT"  :  emerge_rsc['LUT']*self.coarse,
            "FF"   :  emerge_rsc['FF']*self.coarse,
            "BRAM" :  emerge_rsc['BRAM']*self.coarse,
            "DSP" :   emerge_rsc['DSP']*self.coarse,
        }

    def latency_in(self):
        # NOTE from multi port but enforcing no coarse parallel for eMerge
        #return max([
        #    abs(self.workload_in(i)/(self.rate_in(i)*self.streams_in(i) )) for
        #    i in range(self.ports_in) ])
        # FIXME latency in
        return abs((self.workload_in(0)*self.ports_in)/(self.rate_in(0)*self.streams_in(0)*self.ports_in))

    def latency_out(self):
        # NOTE from multi port but enforcing no coarse parallel for eMerge
        #return max([
        #    abs(self.workload_out(i)/(self.rate_out(i)*self.streams_out(i)))
        #    for i in range(self.ports_out) ])
        # NOTE latency out should be double in - merging port data (don't cross the streams)
        return abs((self.workload_out(0)*self.ports_in)/(self.rate_out(0)*self.streams_out(0)))

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
