from models.modules.Module import Module
import numpy as np
import math

class BatchNorm(Module):
    def __init__(
            self,
            dim,
            data_width=16
        ):
        # init module
        Module.__init__(self,dim,data_width)
    
    
    def load_coef(self,static_coef_path,dynamic_coef_path,rsc_coef_path):
        pass

    def rsc(self):
        return {
          "LUT"  : 0, #int(np.dot(self.utilisation_model(), self.rsc_coef[0])),
          "BRAM" : 0,
          "DSP"  : 0,
          "FF"   : 0 #int(np.dot(self.utilisation_model(), self.rsc_coef[3])),
        }

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self, data, scale, shift):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = scale[index[2]] * ( data[index] + shift[index[2]] )

        return out


