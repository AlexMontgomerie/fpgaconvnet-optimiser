"""
ReduceMax Module

Reduces input to the maximum value of that input.

HW TBD.

"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os

class ReduceMax(Module):
    def __init__(
            self,
            rows,
            cols,
            channels,
            data_width=16
        ):
        # module name
        self.name = "RedMx"
        #TODO additional input for denominator

        # init module
        Module.__init__(self,rows,cols,channels,data_width)

        # init variables

        # load resource coefficients
        #TODO resource coefficients file for Cmp module
        #self.rsc_coef = np.load(os.path.join(os.path.dirname(__file__),
        #    "../../coefficients/buffer_rsc_coef.npy"))

    def module_info(self):
        return {
            'type'          : self.__class__.__name__.upper(),
            'rows'          : self.rows_in(),
            'cols'          : self.cols_in(),
            'groups'        : self.groups,
            'channels'      : self.channels_in(),
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

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

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        return torch.max(torch.from_numpy(data)).numpy()
