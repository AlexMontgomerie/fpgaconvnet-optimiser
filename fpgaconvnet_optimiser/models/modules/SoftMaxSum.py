from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os
from dataclasses import dataclass, field

@dataclass
class SoftMaxSum(Module):
    def __post_init__(self):
        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/softmax_sum_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/softmax_sum_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/softmax_sum_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/softmax_sum_dsp.npy"))

    def utilisation_model(self):
        return {
            "LUT"  : np.array([self.data_width]),
            "FF"   : np.array([self.data_width]),
            "DSP"  : np.array([self.data_width]),
            "BRAM" : np.array([1])
        }

    def functional_model(self,data):
        # check input dimensionality
        assert data.shape[0] == self.rows      , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols      , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels  , "ERROR: invalid channel dimension"

        out = np.sum(data)

        return out

