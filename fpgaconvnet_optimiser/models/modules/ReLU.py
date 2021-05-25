"""
.. figure:: ../../../figures/relu_diagram.png
"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os

class ReLU(Module):
    def __init__(
            self,
            rows,
            cols,
            channels,
            data_width=16
        ):

        # module name
        self.name = "relu"

        # init module
        Module.__init__(self,rows,cols,channels,data_width)

        # load resource coefficients
        #self.rsc_coef = np.load(os.path.join(os.path.dirname(__file__),
        #    "../../coefficients/relu_rsc_coef.npy"))
        rsc_types = ['bram', 'lut', 'dsp', 'ff']
        self.rsc_coef = {}
        for rsc_t in rsc_types:
            filename = "../../coefficients/relu_" + rsc_t + ".npy"
            filersc = np.load(os.path.join(os.path.dirname(__file__), filename))
            self.rsc_coef[rsc_t.upper()] = filersc

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def load_coef(self,rsc_coef_path):
        pass

    def utilisation_model(self):
        return [
            1,
            self.data_width,
            self.data_width*self.rows*self.cols*self.channels
        ]


    def rsc(self, coef=None):
        if coef == None:
            coef = self.rsc_coef
        return {
          "LUT"  : int(np.dot(self.utilisation_model(), coef["LUT"])),
          "BRAM" : 0,
          "DSP"  : 0,
          "FF"   : int(np.dot(self.utilisation_model(), coef["FF"])),
        }

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = max(data[index],0.0)

        return out


