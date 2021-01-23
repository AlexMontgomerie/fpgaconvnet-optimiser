"""
.. figure:: ../../../figures/accum_diagram.png
"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math

class Accum(Module):
    def __init__(
            self,
            dim,
            filters,
            groups,
            data_width=32
        ):
        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.filters = filters
        self.groups  = groups

    def utilisation_model(self):
        return [
            1,
            self.data_width,
            self.data_width*self.filters,
            self.data_width*self.channels
        ]

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

    def rsc(self):
        # streams
        #bram_input_buffer = math.ceil( ((self.channels*self.filters+1)*self.data_width)/18000)
        #bram_acc_buffer   = math.ceil( ((self.groups*self.filters+1)*self.data_width)/18000)
        bram_acc_buffer_size =  ((self.filters/self.groups)*(self.channels/self.groups)+1)*self.data_width
        #if bram_acc_buffer_size >= 512:
        bram_acc_buffer = math.ceil( (bram_acc_buffer_size)/18000) 
        return {
          "LUT"  : 0, #int(np.dot(self.utilisation_model(), self.rsc_coef[0])),
          "BRAM" : bram_acc_buffer,
          "DSP"  : 0,
          "FF"   : 0 #int(np.dot(self.utilisation_model(), self.rsc_coef[3])),
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

