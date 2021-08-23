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
from fpgaconvnet_optimiser.tools.hls_helper import stable_array_rsc
import numpy as np
import math
import os
import sys

class Accum(Module):
    def __init__(
            self,
            dim,
            filters,
            groups,
            data_width=30
        ):

        # module name
        self.name = "accum"
        
        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.filters = filters
        self.groups  = groups

        # load resource coefficients
        #work_dir = os.getcwd()
        #os.chdir(sys.path[0])
        #self.rsc_coef = np.load(os.path.join(os.path.dirname(__file__),
        #    "../../coefficients/accum_rsc_coef.npy"))
        #os.chdir(work_dir)

    def utilisation_model(self):
        assert self.data_width == 30

        acc_buffer_size =  int(self.filters/self.groups) if int(self.channels/self.groups) > 1 else 0
        acc_buffer_rsc = stable_array_rsc(self.data_width, acc_buffer_size)
        return np.array([
            1,
            math.log2(self.rows*self.cols*self.groups),
            math.log2(self.channels/self.groups),
            math.log2(self.filters/self.groups),
            self.data_width,
            self.data_width*acc_buffer_size if acc_buffer_rsc['BRAM'] == 0 else 0,
            acc_buffer_rsc['BRAM']
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

        acc_buffer_size =  int(self.filters/self.groups) if int(self.channels/self.groups) > 1 else 0
        acc_buffer_rsc = stable_array_rsc(self.data_width, acc_buffer_size)

        return {
          "LUT"  : int(np.dot(self.utilisation_model(), coef["LUT"])),
          #"BRAM" : int(np.dot(self.utilisation_model(), coef["BRAM"])),
          "BRAM" : acc_buffer_rsc['BRAM'],
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

