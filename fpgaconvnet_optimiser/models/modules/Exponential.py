"""
Exponential Module

Calculates the exponent of the input.

HW TBD.

"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os
from dataclasses import dataclass, field

@dataclass
class Exponential(Module):
    def __post_init__(self):
        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/conv_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/conv_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/conv_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/conv_dsp.npy"))

    def utilisation_model(self):
        #TODO work out what this should be for exponential function
        #currently using conv model - getting maths error
        return {
            "LUT"  : np.array([math.log(1,2),math.log(self.cols*self.rows,2),math.log(self.channels,2)]),
            "FF"   : np.array([math.log(1,2),math.log(self.cols*self.rows,2),math.log(self.channels,2)]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1])
        }

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        out = np.exp(data)
        return out
