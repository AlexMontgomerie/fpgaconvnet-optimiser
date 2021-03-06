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

class Communication(Module):
    def __init__(
            self,
            dim,            
            coarse,
            data_width=16,
            send_nreceive=True,
            pair_id=0,
        ):
        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.send_nreceive = send_nreceive
        self.pair_id = pair_id
        self.coarse = coarse

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
        assert data.shape[3] == self.coarse  , "ERROR: invalid column dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels,
            self.coarse),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[
              index[0],
              index[1],
              index[2],
              index[3]]

        return out

