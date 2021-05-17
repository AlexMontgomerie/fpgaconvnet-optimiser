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
import sys

class Glue(Module):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            filters: int,
            coarse_in: int,
            coarse_out: int,
            data_width=16
        ):
        
        # module name
        self.name = "glue"
 
        # init module
        Module.__init__(self,rows,cols,channels,data_width)

        # init variables
        self.filters    = filters
        self.coarse_in  = coarse_in
        self.coarse_out = coarse_out

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

