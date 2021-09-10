"""
Comparison Module

Compares input to a constant threshold value.

HW TBD.

Attr: cmp_type
values: gt = greater than
        ge = greater than or equal to
        lt = less than
        le = less than or equal to
"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os

class Compare(Module):
    def __init__(
            self,
            rows,
            cols,
            channels,
            threshold,
            cmp_type = 'gt',
            data_width=16
        ):
        # module name
        self.name = "Cmp"
        #TODO additional input for denominator

        # init module
        Module.__init__(self,rows,cols,channels,data_width)

        # init variables
        self.threshold = threshold
        self.cmp_type = cmp_type #TODO currently only supporting gt

        # load resource coefficients
        #TODO resource coefficients file for Cmp module
        #self.rsc_coef = np.load(os.path.join(os.path.dirname(__file__),
        #    "../../coefficients/buffer_rsc_coef.npy"))

    def module_info(self):
        return {
            'type'          : self.__class__.__name__.upper(),
            'rows'          : self.rows_in(),
            'cols'          : self.cols_in(),
            'groups'        : self.groups,
            'channels'      : self.channels_in(),
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def utilisation_model(self):
        #TODO work out what this should be
        #how should the module be laid out?
        return [
            1,
            self.data_width,
            self.data_width*self.channels
        ]

    def pipeline_depth(self):
        #TODO work out if this module can be/needs pipelining
        return 0


    def rsc(self):
        #TODO
        return {
          "LUT"  : 0,
          "BRAM" : 0,
          "DSP"  : 0,
          "FF"   : 0
        }

    def functional_model(self, exp_max_set, exp_sum_set):
        #exp_max is the value from reduce max
        #exp_sum is from the sum of exponentials
        # check input dimensionality
        #assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        #assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        #assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        out = []
        thr_set = np.array(exp_sum_set) * self.threshold
        for (exp_max, thr) in zip(exp_max_set, thr_set):
            if self.cmp_type == 'gt':
                if exp_max > thr:
                    out.append( 1.0)
                else:
                    out.append( 0.0)
            elif self.cmp_type == 'ge':
                if exp_max >= thr:
                    out.append( 1.0)
                else:
                    out.append( 0.0)
            elif self.cmp_type == 'lt':
                if exp_max < thr:
                    out.append( 1.0)
                else:
                    out.append( 0.0)
            elif self.cmp_type == 'le':
                if exp_max <= thr:
                    out.append( 1.0)
                else:
                    out.append( 0.0)
        return np.array(out)
