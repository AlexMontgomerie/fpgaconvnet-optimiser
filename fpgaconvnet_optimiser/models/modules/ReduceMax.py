"""
ReduceMax Module

Reduces input to the maximum value of that input.

HW TBD.

"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field

@dataclass
class ReduceMax(Module):
    def __post_init__(self):
        #FIXME currently using relu but probably more like accum in terms of bram
        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/relu_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/relu_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/relu_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/relu_dsp.npy"))

    def utilisation_model(self):
        return {
            "LUT"  : np.array([self.data_width, math.ceil(math.log(self.channels*self.rows*self.cols,2))]),
            "FF"   : np.array([self.data_width, math.ceil(math.log(self.channels*self.rows*self.cols,2))]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1])
        }

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        return torch.max(torch.from_numpy(data)).numpy()
