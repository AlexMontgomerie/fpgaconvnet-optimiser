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
            dim,
            batch_size,
            data_width=16
        ):
        
        # module name
        self.name = "relu"
        self.batch_size=batch_size
        self.data_width=data_width
        # init module
        Module.__init__(self,dim,data_width)

        RSC_TYPES=["LUT", "FF", "BRAM", "DSP"]
        self.rsc_coef = {
            "LUT"   : np.array([]),
            "FF"    : np.array([]),
            "DSP"   : np.array([]),
            "BRAM"  : np.array([])
        }
        for rsc in RSC_TYPES:
              self.rsc_coef[rsc]=np.load(os.path.join("/home/wz2320/fpgaconvnet-optimiser/fpgaconvnet_optimiser/coefficients/relu_"+str(rsc).lower()+".npy"))

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
        return {
            "LUT"   : np.array([2 if self.data_width<16 else (self.data_width+4),math.ceil(math.log(self.channels*self.rows*self.cols*self.batch_size,2)) if self.data_width<16 else 2*math.ceil(math.log(self.channels*self.rows*self.cols*self.batch_size,2))]),
            "FF"    : np.array([2 if self.data_width<16 else (self.data_width+4),math.ceil(math.log(self.channels*self.rows*self.cols*self.batch_size,2)) if self.data_width<16 else 2*math.ceil(math.log(self.channels*self.rows*self.cols*self.batch_size,2))]),
            #"FF"    : np.array([1 ,math.ceil(math.log(self.channels*self.rows*self.cols*self.batch_size,2))]),
            "DSP"   : np.array([1]),
            "BRAM"  : np.array([1])
        }


    def rsc(self, coef=None):
        if coef == None:
            coef = self.rsc_coef
        return {
          "LUT"  : int(np.dot(self.utilisation_model()["LUT"], coef["LUT"])),
          "BRAM" : int(np.dot(self.utilisation_model()["BRAM"], coef["BRAM"])),
          "DSP"  : 0,
          "FF"   : int(np.dot(self.utilisation_model()["FF"], coef["FF"])),
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


