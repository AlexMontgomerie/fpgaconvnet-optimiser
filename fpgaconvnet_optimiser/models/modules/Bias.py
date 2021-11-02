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
        #TODO add model coefs
        self.rsc_coef["LUT"] =  1 #np.load(
                                #os.path.join(os.path.dirname(__file__),
                                #"../../coefficients/bias_lut.npy"))
        self.rsc_coef["FF"] =   1 #np.load(
                                #os.path.join(os.path.dirname(__file__),
                                #"../../coefficients/bias_ff.npy"))
        self.rsc_coef["BRAM"] = 1 #np.load(
                                #os.path.join(os.path.dirname(__file__),
                                #"../../coefficients/bias_bram.npy"))
        self.rsc_coef["DSP"] =  1 #np.load(
                                #os.path.join(os.path.dirname(__file__),
                                #"../../coefficients/bias_dsp.npy"))



    def utilisation_model(self):#TODO
        return None
    #{
    #        "LUT"   : np.array([self.filters,self.groups,self.data_width,self.cols,self.rows,self.channels]),
    #        "FF"    : np.array([self.filters,self.groups,self.data_width,self.cols,self.rows,self.channels]),
    #        "DSP"   : np.array([self.filters,self.groups,self.data_width,self.cols,self.rows,self.channels]),
    #        "BRAM"  : np.array([self.filters,self.groups,self.data_width,self.cols,self.rows,self.channels]),
    #    }

    def rate_out(self):#TODO
        return None

    def pipeline_depth(self):#TODO
        return None

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

    def functional_model(self,data,biases):
        print(data.shape)
        print(biases.shape)
        f_c_out = int(self.filters/self.coarse_out)
        print(f_c_out)

        # check input dimensionality
        assert data.shape[0] == self.rows                   , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                   , "ERROR: invalid column dimension"
        #assert data.shape[2] == self.channels               , "ERROR: invalid channel dimension"
        #TODO check filter group thing dimension
        assert data.shape[2] == f_c_out                     , "ERROR: invalid filter dimension"
        #assert data.shape[3] == self.coarse_out             , "ERROR: invalid c_out dimension"
        #assert data.shape[3] == self.filters//self.groups   , "ERROR: invalid filter dimension"

        # check bias dimensionality
        assert biases.shape[0] == f_c_out                   , "ERROR: invalid filter dimension"
        #assert biases.shape[1] == self.coarse_out           , "ERROR: invalid c_out dimension"

        #TODO is this required?
        #channels_per_group = self.channels//self.groups
        #filters_per_group  = self.filters//self.groups

        out = np.zeros((
            self.rows,
            self.cols,
            f_c_out,
            #self.coarse_out
            ), dtype=float)
        allocout = np.zeros((
            self.rows,
            self.cols,
            f_c_out,
            #self.coarse_out
            ), dtype=float)

        print("data")
        print(data)
        cf1 = np.empty_like(data[:,:,0])
        cf2 = np.empty_like(data[:,:,0])
        cfo1 = np.empty_like(data[:,:,0])
        cfo2 = np.empty_like(data[:,:,0])

        print("f1")
        cf1[:] = data[:,:,0]
        print(cf1)

        print("f2")
        cf2[:] = data[:,:,1]
        print(cf2)

        print("biases")
        print(biases)

        #TODO how to do this properly
        for index,_ in np.ndenumerate(out):
            out[index] = data[index] + biases[index[2]]

        print("op")
        print(out)
        print("f1")
        cfo1[:] = out[:,:,0]
        v1 = cfo1 - cf1
        print(v1)
        print("f2")
        cfo2[:] = out[:,:,1]
        v2 = cfo2 - cf2
        print(v2)

        # sanity check because numpy indexing confuses me
        for f_i in range(f_c_out):
            # create copy of input and output filter
            cf = np.empty_like(data[:,:,0])
            cfo = np.empty_like(data[:,:,0])
            # set values of input and output
            cf[:] = data[:,:,f_i]
            cfo[:] = out[:,:,f_i]
            # subtraction should give biaois
            v = cfo - cf
            for _,val in np.ndenumerate(v):
                # check each filter result has been added correctly to the bias
                assert np.allclose(biases[f_i], val, rtol=1.e-8,atol=1.e-8), "ERROR: the biases don't match!"

        return out
