"""
Exit Merge Module

For combining early eixt streams with later exit streams.
The result is passed to the memory write module.
Future work will be adding more inputs OR stacking these modules for more exits.

"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os

class ExitMerge(Module):
    def __init__(
            self,
            rows,
            cols,
            channels,
            #early_exit_edge, #edges record where batch ID comes from
            #late_exit_edge,
            data_width=16
        ):
        # module name
        self.name = "exit_merge"

        # init module
        Module.__init__(self,rows,cols,channels,data_width)

        # init variables
        #self.early_exit_edge = early_exit_edge
        #self.late_exit_edge  = late_exit_edge

        # load resource coefficients
        #TODO resource coefficients file for exit merge module
        #self.rsc_coef = np.load(os.path.join(os.path.dirname(__file__),
        #    "../../coefficients/exit_merge_rsc_coef.npy"))

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
        #how should the FIFOs be laid out?
        return [
            1,
            self.data_width,
            self.data_width*self.channels
        ]

    def pipeline_depth(self):
        #TODO work out if this module can be/needs pipelining
        return 0


    def rsc(self):
        #TODO workout resources
        return {
          "LUT"  : 0, #int(np.dot(self.utilisation_model(), self.rsc_coef[0])),
          "BRAM" : 0,
          "DSP"  : 0,
          "FF"   : 0  #int(np.dot(self.utilisation_model(), self.rsc_coef[3])),
        }

    def functional_model(self, EEdata, LEdata, EE_ID=None, LE_ID=None):
        #Exit merge is not an ONNX or pytorch op
        # check input dimensionality
        assert EEdata.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert EEdata.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert EEdata.shape[2] == self.channels, "ERROR: invalid channel dimension"

        assert LEdata.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert LEdata.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert LEdata.shape[2] == self.channels, "ERROR: invalid channel dimension"

        if EE_ID is not None:
            return np.concatenate(([EE_ID], EEdata))
        elif LE_ID is not None:
            return np.concatenate(([LE_ID], LEdata))
