"""
The Glue module is used to combine streams 
used for channel parallelism in the 
Convolution layer together. 

.. figure:: ../../../figures/glue_diagram.png
"""
import joblib
from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os

class Glue(Module):
    def __init__(
            self,
            dim,
            filters,
            coarse_in,
            coarse_out,
            data_width=16,
            acc_width=30
        ):
        
        # module name
        self.name = "glue"
 
        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.filters    = filters
        self.coarse_in  = coarse_in
        self.coarse_out = coarse_out
        self.data_width = data_width
        self.acc_width= acc_width
        RSC_TYPES=["LUT", "FF", "BRAM", "DSP"]
        self.rsc_buildmodel = {
            }
        for rsc in RSC_TYPES:
            self.rsc_buildmodel[rsc]=joblib.load("/home/wz2320/fpgaconvnet-optimiser/fpgaconvnet_optimiser/coefficients/glue_"+str(rsc)+'(randomforest)')        


    def dynamic_model(self, freq, rate, sa_in, sa_out):
        return [
            self.data_width*freq,
            self.data_width*self.sa_in*freq*rate,
            self.data_width*self.sa_in*freq*rate*self.coarse_in*self.coarse_out,
            self.data_width*self.sa_in*freq*rate*self.coarse_out
        ]

    #def utilisation_model(self):
        #return {
            #"LUT"   : np.array([self.coarse_in*self.coarse_out,math.ceil(np.log2(self.filters/self.coarse_out))]),
            #"FF"    : np.array([self.coarse_in*self.coarse_out,math.ceil(np.log2(self.filters/self.coarse_out))]),
            #"DSP"   : np.array([1]),
            #"BRAM"  : np.array([1])
        #}
    def utilisation_model(self):
        return {
            "LUT"   : np.array([self.cols,self.rows,self.channels,self.data_width,self.acc_width,self.filters,self.coarse_in,self.coarse_out]),
            "FF"    : np.array([self.cols,self.rows,self.channels,self.data_width,self.acc_width,self.filters,self.coarse_in,self.coarse_out]),
            "DSP"   : np.array([self.cols,self.rows,self.channels,self.data_width,self.acc_width,self.filters,self.coarse_in,self.coarse_out]),
            "BRAM"  : np.array([self.cols,self.rows,self.channels,self.data_width,self.acc_width,self.filters,self.coarse_in,self.coarse_out]),
        }      

    def channels_in(self):
        return self.filters*self.coarse_in

    def channels_out(self):
        return self.filters

    def get_latency(self):
        return self.rows *self.cols *self.filters / self.coarse_out

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'filters'   : self.filters,
            'coarse_in'     : self.coarse_in,
            'coarse_out'    : self.coarse_out,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    #def rsc(self, coef=None):
        #if coef == None:
            #coef = self.rsc_coef
        #return {
          #"LUT"  : int(np.dot(self.utilisation_model()["LUT"], coef["LUT"])),
          #"BRAM" : int(np.dot(self.utilisation_model()["BRAM"], coef["BRAM"])),
          #"DSP"  : 0,
          #"FF"   : int(np.dot(self.utilisation_model()["FF"], coef["FF"])),
        #}
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

    def functional_model(self,data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == int(self.filters/self.coarse_out) , "ERROR: invalid  dimension"
        assert data.shape[3] == self.coarse_in , "ERROR: invalid  dimension"
        assert data.shape[4] == self.coarse_out , "ERROR: invalid  dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            int(self.filters/self.coarse_out),
            self.coarse_out),dtype=float)

        for index,_ in np.ndenumerate(out):
            for c in range(self.coarse_in):
                out[index] += data[index[0],index[1],index[2],c,index[3]]

        return out

