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
from dataclasses import dataclass, field

@dataclass
class ExitMerge(Module):
    #early_exit_edge, #edges record where batch ID comes from
    #late_exit_edge,
    #batch ID input edge?
    #ID pipeline connection

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
