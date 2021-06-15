from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os

class SoftMaxSum(Module):
    def __init__(
            self,
            rows,
            cols,
            channels,
            data_width=30
        ):

        # module name
        self.name = "sm_sum"

        # init module
        Module.__init__(self,rows,cols,channels,data_width)

    def utilisation_model(self):
        #TODO work out what this should be
        #how should the module be laid out?
        return [
            1,
            self.data_width,
            self.data_width*self.channels
        ]

    def pipeline_depth(self):
        #TODO work out if this module can be/needs pipelining
        return 0


    def rsc(self):
        #TODO
        return {
          "LUT"  : 0,
          "BRAM" : 0,
          "DSP"  : 0,
          "FF"   : 0
        }

    def functional_model(self,data):
        # check input dimensionality
        assert data.shape[0] == self.rows      , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols      , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels  , "ERROR: invalid channel dimension"

        out = np.sum(data)

        return out

