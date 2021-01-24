"""
The Fork module provides functionality for 
parallelism within layers. By duplicating the 
streams, it can be used for exploiting 
parallelism across filters in the Convolution 
layers.

.. figure:: ../../../figures/fork_diagram.png
"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math

class Fork(Module):
    def __init__(
            self,
            dim,
            k_size,
            coarse,
            data_width=16
        ):
        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.k_size = k_size
        self.coarse = coarse

        # dynamic power model
        self.dynamic_model = lambda freq, rate, sa_in, sa_out : [
            self.data_width*freq,
            self.data_width*sa_in*freq*rate*self.k_size*self.k_size,
            self.data_width*sa_in*freq*rate*self.k_size*self.k_size*self.coarse
        ]
 
        # utilisation model
        self.utilisation_model = lambda : [
            1,
            self.data_width*self.k_size*self.k_size,
            self.data_width*self.k_size*self.k_size*self.coarse
        ]

    def dynamic_model(self, freq, rate, sa_in, sa_out):
        return [
            self.data_width*freq,
            self.data_width*sa_in*freq*rate*self.k_size*self.k_size,
            self.data_width*sa_in*freq*rate*self.k_size*self.k_size*self.coarse
        ]

    def utilisation_model(self):
        return [
            1,
            self.data_width*self.k_size*self.k_size,
            self.data_width*self.k_size*self.k_size*self.coarse
        ]

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'coarse'    : self.coarse,
            'kernel_size'   : self.k_size,
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
            self.channels,
            self.coarse,
            self.k_size,
            self.k_size),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[
              index[0],
              index[1],
              index[2],
              index[4],
              index[5]]

        return out

