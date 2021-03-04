"""
The Glue module is used to combine streams 
used for channel parallelism in the 
Convolution layer together. 

.. figure:: ../../../figures/glue_diagram.png
"""

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
            data_width=16
        ):
        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.filters    = filters
        self.coarse_in  = coarse_in
        self.coarse_out = coarse_out

        # load resource coefficients
        self.rsc_coef = np.load(os.path.join(os.path.dirname(__file__),
            "../../coefficients/glue_rsc_coef.npy"))

    def dynamic_model(self, freq, rate, sa_in, sa_out):
        return [
            self.data_width*freq,
            self.data_width*self.sa_in*freq*rate,
            self.data_width*self.sa_in*freq*rate*self.coarse_in*self.coarse_out,
            self.data_width*self.sa_in*freq*rate*self.coarse_out
        ]

    def utilisation_model(self):
        return [
            1,
            self.data_width,
            self.data_width*self.coarse_in*self.coarse_out,
            self.data_width*self.coarse_out
        ]

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

