"""
This module performs the max pooling function 
across a kernel-size window of the feature map.

.. figure:: ../../../figures/pool_max_diagram.png
"""
import joblib
from sklearn.neural_network import MLPClassifier
from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os

class Pool(Module):
    def __init__(
            self,
            dim,
            k_size,
            batch_size,
            data_width=16,
            pool_type='max'
        ):
        
        # module name
        self.name = "pool"
        # init module
        Module.__init__(self,dim,data_width)
        
        # init variables
        self.k_size    = k_size
        self.pool_type = pool_type
        self.batch_size= batch_size
        self.data_width= data_width
        RSC_TYPES=["LUT", "FF", "BRAM", "DSP"]
        self.rsc_buildmodel = {
        }
        for rsc in RSC_TYPES:
            self.rsc_buildmodel[rsc]=joblib.load("/home/wz2320/fpgaconvnet-optimiser/fpgaconvnet_optimiser/coefficients/pool_"+str(rsc)+'(randomforest)')         


    def dynamic_model(self, freq, rate, sa_in, sa_out):
        return [
            self.data_width*freq,
            self.data_width*sa_in*freq*rate*self.k_size*self.k_size,
            self.data_width*sa_in*freq*rate,
        ]

    def utilisation_model(self):
#       if self.data_width==32;
#            co1=70
        return {
            "LUT"   : np.array([self.k_size,self.cols,self.rows,self.channels,self.batch_size,self.data_width]),
            "FF"    : np.array([self.k_size,self.cols,self.rows,self.channels,self.batch_size,self.data_width]),
            "DSP"   : np.array([self.k_size,self.cols,self.rows,self.channels,self.batch_size,self.data_width]),
            "BRAM"  : np.array([self.k_size,self.cols,self.rows,self.channels,self.batch_size,self.data_width]),
        }
    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'pool_type'     :  0 if self.pool_type == 'max' else 1,
            'kernel_size'   : self.k_size,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

#    def rsc(self,coef=None):
#        if coef == None:
#            coef = self.rsc_coef
#        return {
#          "LUT"  : int(np.dot(self.utilisation_model()["LUT"], coef["LUT"])),
#          "BRAM" : int(np.dot(self.utilisation_model()["BRAM"], coef["BRAM"])),
#          "DSP"  : 0,
#          "FF"   : int(np.dot(self.utilisation_model()["FF"], coef["FF"])),
#        }

    def rsc(self,buildmodel=None):
        if buildmodel == None:                    
          buildmodel = self.rsc_buildmodel         
        return {
          "LUT"  : int(buildmodel["LUT"].predict(self.utilisation_model()["LUT"].reshape(1, -1))),
          "BRAM" : int(buildmodel["BRAM"].predict(self.utilisation_model()["BRAM"].reshape(1, -1))),
          "DSP"  : int(buildmodel["DSP"].predict(self.utilisation_model()["DSP"].reshape(1, -1))),
          "FF"   : int(buildmodel["FF"].predict(self.utilisation_model()["FF"].reshape(1, -1))),
        }


    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[3] == self.k_size  , "ERROR: invalid column dimension"
        assert data.shape[4] == self.k_size  , "ERROR: invalid column dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels),dtype=float)

        for index,_ in np.ndenumerate(out):
            if self.pool_type == 'max':
                out[index] = np.max(data[index])
            elif self.pool_type == 'avg':
                out[index] = np.mean(data[index])


        return out

