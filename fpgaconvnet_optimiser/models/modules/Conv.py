"""
The convolution module computes the dot product
between the feature map windows and the coefficients
of the convolution module. This module has a tunable
degree of parallelism across the kernel dot product,
affecting the throughput and number of ports of the
on-chip weights storage.

.. figure:: ../../../figures/conv_diagram.png
"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os
import sys
from typing import Union, List

class Conv(Module):
    """
    Conv hardware model class.
    """
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            filters: int,
            fine: int,
            k_size: Union[List[int], int],
            groups: int,
            data_width=16,
            weight_width=8
        ):
        """
        Parameters
        ----------
        rows: int
            row dimension of input featuremap
        cols: int
            column dimension of input featuremap
        channels: int
            channel dimension of input featuremap

        Attributes
        ----------
        filters: int
            output channel dimension of the featuremap.
        fine: int

        rows: int
            row dimension of input featuremap
        cols: int
            column dimension of input featuremap
        channels: int
            channel dimension of input featuremap
        data_width: int
            bitwidth of featuremap pixels (default is 16)
        rsc_coef: list
            list of resource model coefficients. Corresponds
            to `LUT`, `BRAM`, `DSP` and `FF` resources in
            that order.
        """

        # module name
        self.name = "conv"

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
        self.filters = filters
        self.groups  = groups
        self.fine    = fine
        self.k_size  = k_size

        self.weight_width = weight_width

    def dynamic_model(self, freq, rate, sa_in, sa_out):
        return [
            self.data_width*freq,
            self.data_width*sa_in*freq*rate*self.fine/float(self.channels),
            self.data_width*sa_in*freq*rate*self.fine,
            self.data_width*sa_out*freq*rate*self.fine,
            self.data_width*sa_out*freq*rate*self.fine*self.fine/float(self.k_size[0]*self.k_size[1]),
            self.data_width*sa_out*freq*rate*self.fine/float(self.k_size[0]*self.k_size[1]),
        ]

    def utilisation_model(self):
        return [
            1,
            self.data_width*self.k_size[0]*self.k_size[1],
            self.data_width*self.fine,
            self.data_width
        ]

    def channels_out(self):
        return int(self.filters/float(self.groups))

    def rate_in(self):
        return self.fine*self.groups/float(self.k_size[0]*self.k_size[1]*self.filters)

    def rate_out(self):
        return self.fine/float(self.k_size[0]*self.k_size[1])

    def pipeline_depth(self):
        return self.fine

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'filters'   : self.filters,
            'groups'    : self.groups,
            'fine'      : self.fine,
            'kernel_size'   : self.k_size,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def rsc(self,coef=None): # TODO: improve DSP utilisation for different bitwidths
        if coef == None:
            coef = self.rsc_coef
        return {
          "LUT"  : int(np.dot(self.utilisation_model(), coef["LUT"])),
          "BRAM" : 0,
          "DSP"  : self.fine * math.ceil((self.weight_width + self.data_width) / 48), #https://github.com/Xilinx/finn/blob/4fee6ffd8e13f91314ec9086e9ce9b2ea9de15c7/src/finn/custom_op/fpgadataflow/streamingfclayer_batch.py#L368,
          "FF"   : int(np.dot(self.utilisation_model(), coef["FF"])),
        }

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self,data,weights):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[3] == self.k_size[0]  , "ERROR: invalid column dimension"
        assert data.shape[4] == self.k_size[1]  , "ERROR: invalid column dimension"
        # check weight dimensionality
        assert weights.shape[0] == self.channels, "ERROR: invalid channel dimension"
        assert weights.shape[1] == int(self.filters/float(self.groups)) , "ERROR: invalid filter dimension"
        assert weights.shape[2] == self.k_size[0]  , "ERROR: invalid column dimension"
        assert weights.shape[3] == self.k_size[1]  , "ERROR: invalid column dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.channels,
            int(self.filters/self.groups)
        ),dtype=float)

        for index,_ in np.ndenumerate(out):
            for k1 in range(self.k_size[0]):
                for k2 in range(self.k_size[1]):
                    out[index] += data[
                      index[0],index[1],index[2],k1,k2]*weights[
                      index[2],index[3],k1,k2]

        return out

