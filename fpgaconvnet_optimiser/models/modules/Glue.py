"""
The Glue module is used to combine streams
used for channel parallelism in the
Convolution layer together.

.. figure:: ../../../figures/glue_diagram.png
"""

import numpy as np
import math
import os
import sys
from dataclasses import dataclass, field

from fpgaconvnet_optimiser.models.modules import Module

@dataclass
class Glue(Module):
    filters: int
    coarse_in: int
    coarse_out: int

    def utilisation_model(self):
        return [
            1,
            self.data_width,
            self.data_width*self.coarse_in*self.coarse_out,
            self.data_width*self.coarse_out
        ]

    def channels_in(self):
        return self.filters

    def channels_out(self):
        return self.filters

    def get_latency(self):
        return self.rows *self.cols *self.filters / self.coarse_out

    def module_info(self):
        # get the base module fields
        info = Module.module_info(self)
        # add module-specific info fields
        info["filters"] = self.filters
        info["coarse_in"] = self.coarse_in
        info["coarse_out"] = self.coarse_out
        # return the info
        return info

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

