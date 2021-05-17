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
import os
import sys
from typing import Union, List

class Fork(Module):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            k_size: Union[List[int],int],
            coarse: int,
            data_width=16
        ):
        
        # module name
        self.name = "fork"
 
        # init module
        Module.__init__(self,rows,cols,channels,data_width)

        # handle kernel size
        if isinstance(k_size, int):
            k_size = [k_size, k_size]
        elif isinstance(k_size, list):
            assert len(k_size) == 2, "Must specify two kernel dimensions"
        else:
            raise TypeError

        # init variables
        self.k_size = k_size
        self.coarse = coarse

    def utilisation_model(self):
        return [
            1,
            self.data_width*self.k_size[0]*self.k_size[1],
            self.data_width*self.k_size[0]*self.k_size[1]*self.coarse
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

    def rsc(self,coef=None):
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
        assert data.shape[3] == self.k_size[0]  , "ERROR: invalid column dimension"
        assert data.shape[4] == self.k_size[1]  , "ERROR: invalid column dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels,
            self.coarse,
            self.k_size[0],
            self.k_size[1]),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[
              index[0],
              index[1],
              index[2],
              index[4],
              index[5]]

        return out

