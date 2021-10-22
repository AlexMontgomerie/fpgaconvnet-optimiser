"""
The bias module implements the addition of the
bias term for the convolution (and inner product)
layers when applicable. Each of element of the bias
vector is added to its corresponding output feature
map.

Figure pending.
"""

import numpy as np
import math
import os
import sys
from dataclasses import dataclass, field

from fpgaconvnet_optimiser.models.modules import Module
#from fpgaconvnet_optimiser.tools.resource_model import bram_memory_resource_model
from fpgaconvnet_optimiser.tools.resource_model import dsp_multiplier_resource_model

@dataclass
class Bias(Module):
    filters: int
    groups: int
    weight_width: int = field(default=16, init=False)
    #acc_width: int = field(default=16, init=False)

    def __post_init__(self):
        # load the resource model coefficients
        #TODO add model coefs
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/bias_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/bias_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/bias_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/bias_dsp.npy"))



    def utilisation_model(self):#TODO
        return
    #{
    #        "LUT"   : np.array([self.filters,self.groups,self.data_width,self.cols,self.rows,self.channels]),
    #        "FF"    : np.array([self.filters,self.groups,self.data_width,self.cols,self.rows,self.channels]),
    #        "DSP"   : np.array([self.filters,self.groups,self.data_width,self.cols,self.rows,self.channels]),
    #        "BRAM"  : np.array([self.filters,self.groups,self.data_width,self.cols,self.rows,self.channels]),
    #    }

    def rate_out(self):#TODO
        return

    def pipeline_depth(self):#TODO
        return

    def module_info(self):#TODO
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields

        info['filters'] = self.filters
        info["groups"] = self.groups

        # return the info
        return info

    def rsc(self,coef=None):#TODO
        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef
        # get the BRAM estimate ?
        #TODO
        # get the DSP estimate ?
        #TODO
        # get the linear model estimation
        rsc = Module.rsc(self, coef)
        # add the resource estimation
        #TODO
        # return the resource usage
        return rsc

    def functional_model(self,data,weights):
        # check input dimensionality
        assert data.shape[0] == self.rows                   , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                   , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels               , "ERROR: invalid channel dimension"
        #TODO check filter group thing dimension
        assert data.shape[3] == self.filters//self.groups   , "ERROR: invalid filter  dimension"
        # check weight dimensionality
        assert weights.shape[0] == self.filters, "ERROR: invalid filter dimension"

        #TODO is this required?
        channels_per_group = self.channels//self.groups
        filters_per_group  = self.filters//self.groups

        out = np.zeros((
            self.rows,
            self.cols,
            self.channels,
            self.filters),
            dtype=float)

        #TODO how to do this properly
        for index,_ in np.ndenumerate(out):
            for f in range(self.filters):
                out[index] = data[index[0],index[1],index[2],f] + weights[f]

        return out

