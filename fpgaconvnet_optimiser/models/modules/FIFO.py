import numpy as np
import math
import os
import sys

from dataclasses import dataclass, field
from fpgaconvnet_optimiser.models.modules import Module
from fpgaconvnet_optimiser.tools.resource_model import bram_array_resource_model

@dataclass
class FIFO(Module):
    coarse: int
    depth: int

    def __post_init__(self):
        # load the resource model coefficients
        work_dir = os.getcwd()
        os.chdir(sys.path[0])
        self.rsc_coef["LUT"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/fifo_lut.npy"))
        self.rsc_coef["FF"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/fifo_ff.npy"))
        self.rsc_coef["BRAM"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/fifo_bram.npy"))
        self.rsc_coef["DSP"] = np.load(
                os.path.join(os.path.dirname(__file__),
                "../../coefficients/fifo_dsp.npy"))
        os.chdir(work_dir)

    def utilisation_model(self):

        assert self.data_width == 16 or self.data_width == 30
        single_fifo_bram = bram_array_resource_model(self.depth, self.data_width, 'stream')

        model_variable = np.array([
                                    1,
                                    self.data_width,
                                    self.coarse,
                                    self.data_width*self.depth*self.coarse if single_fifo_bram == 0 else 0,
                                    single_fifo_bram*self.coarse,
                                ])

        return {
            "LUT"   : model_variable,
            "FF"    : model_variable,
            "DSP"   : model_variable,
            "BRAM"  : model_variable,
        }

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["coarse"] = self.coarse
        info["depth"] = self.depth
        # return the info
        return info

    def functional_model(self, data):   
        return data