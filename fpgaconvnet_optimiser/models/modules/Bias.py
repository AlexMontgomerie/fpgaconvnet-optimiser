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
    coarse_out: int
    bias_width: int = field(default=16, init=False)
    #acc_width: int = field(default=16, init=False)

    def __post_init__(self):
        # load the resource model coefficients
        #TODO add model coefs FOR BIAS - currently using conv to approx.
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



    def utilisation_model(self):#TODO - copied from conv, FIXME
        return {
            "LUT"  : np.array([math.log(self.filters,2),math.log(self.cols*self.rows,2),math.log(self.channels,2)]),
            "FF"   : np.array([math.log(self.filters,2),math.log(self.cols*self.rows,2),math.log(self.channels,2)]),
            "DSP"  : np.array([1]),
            "BRAM" : np.array([1])
        }

    #def rate_out(self):#TODO
    #    return 1

    #def pipeline_depth(self):#TODO
    #    return 1

    def channels_in(self):
        return self.filters

    def channels_out(self):
        return self.filters

    def module_info(self):#TODO
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields

        info['filters'] = self.filters
        info["groups"] = self.groups

        # return the info
        return info

    #def rsc(self,coef=None):#TODO
    #    # use module resource coefficients if none are given
    #    if coef == None:
    #        coef = self.rsc_coef
    #    # get the BRAM estimate ?
    #    #TODO
    #    # get the DSP estimate ?
    #    #TODO
    #    # get the linear model estimation
    #    rsc = Module.rsc(self, coef)
    #    # add the resource estimation
    #    #TODO
    #    # return the resource usage
    #    return rsc

    def functional_model(self,data,biases):
        f_c_out = int(self.filters/self.coarse_out)

        # check input dimensionality
        assert data.shape[0] == self.rows                   , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                   , "ERROR: invalid column dimension"
        assert data.shape[2] == f_c_out                     , "ERROR: invalid filter dimension"
        assert data.shape[3] == self.coarse_out             , "ERROR: invalid c_out dimension"
        #TODO check filter group thing dimension
        #assert data.shape[3] == self.filters//self.groups   , "ERROR: invalid filter dimension"

        # check bias dimensionality
        assert biases.shape[0] == f_c_out                   , "ERROR: invalid filter dimension"
        assert biases.shape[1] == self.coarse_out           , "ERROR: invalid c_out dimension"

        #TODO is this required?
        #channels_per_group = self.channels//self.groups
        #filters_per_group  = self.filters//self.groups

        out = np.zeros((
            self.rows,
            self.cols,
            f_c_out,
            self.coarse_out
            ), dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[index] + biases[index[2],index[3]]

        # sanity check because numpy indexing confuses me
        for f_i in range(f_c_out):
            for co_i in range(self.coarse_out):
                # create copy of input and output filter
                cf = np.empty_like(data[:,:,0,0])
                cfo = np.empty_like(data[:,:,0,0])
                # set values of input and output
                cf[:] = data[:,:,f_i,co_i]
                cfo[:] = out[:,:,f_i,co_i]
                # subtraction should give biaois
                v = cfo - cf
                for _,val in np.ndenumerate(v):
                    # check each filter result has been added correctly to the bias
                    assert np.allclose(biases[f_i,co_i],val,
                            rtol=1.e-8,atol=1.e-8), "ERROR: the biases don't match!"

        return out
