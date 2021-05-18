"""
The purpose of the accumulation (Accum) module is
to perform the channel-wise accumulation of the
dot product result from the output of the
convolution (Conv) module.  As the data is coming
into the module filter-first, the separate filter
accumulations are buffered until they complete
their accumulation across channels.

.. figure:: ../../../figures/accum_diagram.png
"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os
import sys

class Accum(Module):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            filters: int,
            groups: int,
            data_width=30
        ):

        # module name
        self.name = "accum"

        # init module
        Module.__init__(self,rows,cols,channels,data_width)

        # init variables
        self.filters = filters
        self.groups  = groups

    def utilisation_model(self):
        bram_acc_buffer_size =  (self.filters/self.groups)*self.data_width
        return np.array([
            1,
            #self.data_width,
            self.data_width*self.groups,
            self.data_width*(self.channels/self.groups),
            bram_acc_buffer_size,
            math.ceil( (bram_acc_buffer_size)/18000),
        ])

    def channels_in(self):
        return int((self.channels*self.filters)/(self.groups))

    def channels_out(self):
        return self.filters

    def rate_out(self):
        return (self.groups)/float(self.channels)

    def pipeline_depth(self):
        return (self.channels*self.filters)/(self.groups*self.groups)

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'groups'    : self.groups,
            'channels'  : self.channels_in(),
            'filters'   : self.filters,
            'channels_per_group'  : int(self.channels_in()/self.groups),
            'filters_per_group'   : int(self.filters/self.groups),
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def rsc(self,coef=None):
        if coef == None:
            coef = self.rsc_coef
        # streams
        #bram_input_buffer = math.ceil( ((self.channels*self.filters+1)*self.data_width)/18000)
        #bram_acc_buffer   = math.ceil( ((self.groups*self.filters+1)*self.data_width)/18000)
        acc_buffer =  (self.filters/self.groups)
        acc_fifo_bram = 0
        if acc_buffer*self.data_width >= 512:
            acc_buffer_fifo_data = math.ceil( (acc_buffer*self.data_width)/18000)
            acc_buffer_fifo_addr = math.ceil( math.log(acc_buffer,2) )
            #acc_fifo_bram = max(acc_buffer_fifo_data, acc_buffer_fifo_addr)
            acc_fifo_bram = acc_buffer_fifo_data
        acc_buffer_bram = math.ceil( (acc_buffer*self.data_width)/18000)
        return {
          "LUT"  : int(np.dot(self.utilisation_model(), coef["LUT"])),
          #"BRAM" : int(np.dot(self.utilisation_model(), coef["BRAM"])),
          "BRAM" : acc_fifo_bram,# + acc_fifo_bram,
          "DSP"  : 0,
          "FF"   : int(np.dot(self.utilisation_model(), coef["FF"])),
        }

    def functional_model(self,data):
        # check input dimensionality
        assert data.shape[0] == self.rows                       , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols                       , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels                   , "ERROR: invalid channel dimension"
        assert data.shape[3] == int(self.filters/self.groups)   , "ERROR: invalid filter  dimension"

        channels_per_group = int(self.channels/self.groups)
        filters_per_group  = int(self.filters/self.groups)

        out = np.zeros((
            self.rows,
            self.cols,
            self.filters),dtype=float)

        tmp = np.zeros((
            self.rows,
            self.cols,
            channels_per_group,
            filters_per_group),dtype=float)

        for index,_ in np.ndenumerate(tmp):
            for g in range(self.groups):
                out[index[0],index[1],g*filters_per_group+index[3]] += data[index[0],index[1],g*channels_per_group+index[2],index[3]]

        return out

