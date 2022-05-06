"""
Buffering Module

Stores intermediate compute information such as results from Conv or Pool layers.
During DSE the required size will be calculated to store intermediate results at
branching layers. The position of the buffer layer will then be moved along a
given branch until the buffer size is feasible and the latency of the exit
condition is mitigated/matched. For effective pipelining I think.

Secondary function of the buffer is to "drop" a partial calculation.
Clear a FIFO - takes X number of cycles?
Drop signal will be control signal from the Exit Condition.

Future goal will be to have buffer as an offchip memory link.
In this case, the drop might not be used.
"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os
from dataclasses import dataclass, field
#NOTE using accum resource model for now
from fpgaconvnet_optimiser.tools.resource_model import bram_memory_resource_model

@dataclass
class Buffer(Module):
    ctrledge: int
    drop_mode: bool = True

    def __post_init__(self):
        #NOTE using accum resource model for now
        # load the resource model coefficients
        #self.rsc_coef["LUT"] = np.load(
        #        os.path.join(os.path.dirname(__file__),
        #        "../../coefficients/accum_lut.npy"))
        #self.rsc_coef["FF"] = np.load(
        #        os.path.join(os.path.dirname(__file__),
        #        "../../coefficients/accum_ff.npy"))
        #self.rsc_coef["BRAM"] = np.load(
        #        os.path.join(os.path.dirname(__file__),
        #        "../../coefficients/accum_bram.npy"))
        #self.rsc_coef["DSP"] = np.load(
        #        os.path.join(os.path.dirname(__file__),
        #        "../../coefficients/accum_dsp.npy"))
        #trying buffer coefs
        #print("NEW BUFFER COEFFS")
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/buffer_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/buffer_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/buffer_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/buffer_dsp.npy"))

    def utilisation_model(self):
        return {
            "LUT"   : np.array([self.rows,self.cols,self.channels,self.data_width]),
            "FF"    : np.array([self.rows,self.cols,self.channels,self.data_width]),
            "DSP"   : np.array([self.rows,self.cols,self.channels,self.data_width]),
            "BRAM"  : np.array([self.rows,self.cols,self.channels,self.data_width]),
        }

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["ctrledge"] = self.ctrledge
        info["drop_mode"] = "true" if self.drop_mode == True else "false"
        # return the info
        return info

    def rsc(self,coef=None):
        # use module resource coefficients if none are given
        if coef == None:
            coef = self.rsc_coef
        # get the buffer buffer BRAM estimate
        buf_buffer_bram = bram_memory_resource_model(
                int(self.rows*self.cols*self.channels), self.data_width)
        # get the linear model estimation
        rsc = Module.rsc(self, coef)
        print("linear model:",rsc['BRAM'], "mem rsc model", buf_buffer_bram)
        # add the bram estimation
        rsc["BRAM"] = buf_buffer_bram
        # return the resource usage
        return rsc

    def functional_model(self, data, ctrl_drop):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.channels),dtype=float)

        if self.drop_mode: #non-inverted
            if ctrl_drop == 1.0:
                return
            else:
                return data #pass through
        else: #inverted
            if not ctrl_drop == 1.0:
                return
            else:
                return data #pass through
