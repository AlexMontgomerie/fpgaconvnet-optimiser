"""
Comparison Module

Compares input to a constant threshold value.

HW TBD.

Attr: cmp_type
values: gt = greater than
        ge = greater than or equal to
        lt = less than
        le = less than or equal to
"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os
from dataclasses import dataclass, field

@dataclass
class Compare(Module):
    threshold: float
    cmp_type: str = "gt"

    def __post_init__(self):
        #NOTE using pool resources for now
        # load the resource model coefficients
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/pool_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/pool_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/pool_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/pool_dsp.npy"))

    def utilisation_model(self):
        #NOTE using pool resources for now
        return {
            "LUT"  : np.array([1,1,self.cols,self.rows,self.channels,self.data_width]),
            "FF"   : np.array([1,1,self.cols,self.rows,self.channels,self.data_width]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1]),
        }

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["threshold"] = self.threshold
        info["cmp_type"] = 0 if self.cmp_type == 'gt' else 1
        # return the info
        return info

    def functional_model(self, exp_max_set, exp_sum_set):
        #exp_max is the value from reduce max
        #exp_sum is from the sum of exponentials
        # check input dimensionality
        #assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        #assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        #assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        out = []
        thr_set = np.array(exp_sum_set) * self.threshold
        for (exp_max, thr) in zip(exp_max_set, thr_set):
            if self.cmp_type == 'gt':
                if exp_max > thr:
                    out.append( 1.0)
                else:
                    out.append( 0.0)
            elif self.cmp_type == 'ge':
                if exp_max >= thr:
                    out.append( 1.0)
                else:
                    out.append( 0.0)
            elif self.cmp_type == 'lt':
                if exp_max < thr:
                    out.append( 1.0)
                else:
                    out.append( 0.0)
            elif self.cmp_type == 'le':
                if exp_max <= thr:
                    out.append( 1.0)
                else:
                    out.append( 0.0)
        return np.array(out)
