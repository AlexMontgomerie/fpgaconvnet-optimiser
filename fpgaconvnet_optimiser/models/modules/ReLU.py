"""
.. figure:: ../../../figures/relu_diagram.png
"""

import numpy as np
import math
import os
import sys
from dataclasses import dataclass, field

from fpgaconvnet_optimiser.models.modules import Module

@dataclass
class ReLU(Module):

    def utilisation_model(self):
        return [
            1,
            self.data_width,
            self.data_width*self.rows*self.cols*self.channels
        ]

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


