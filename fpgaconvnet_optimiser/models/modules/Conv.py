from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math

class Conv(Module):
    def __init__(
            self,
            dim,
            filters,
            fine,
            k_size,
            groups,
            data_width=16
        ):
        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.filters = filters
        self.groups  = groups
        self.fine    = fine
        self.k_size  = k_size

    def dynamic_model(self, freq, rate, sa_in, sa_out):
        return [
            self.data_width*freq,
            self.data_width*sa_in*freq*rate*self.fine/float(self.channels),
            self.data_width*sa_in*freq*rate*self.fine,
            self.data_width*sa_out*freq*rate*self.fine,
            self.data_width*sa_out*freq*rate*self.fine*self.fine/float(self.k_size*self.k_size),
            self.data_width*sa_out*freq*rate*self.fine/float(self.k_size*self.k_size),
        ]

    def utilisation_model(self):
        return [
            1,
            self.data_width*self.k_size*self.k_size,
            self.data_width*self.fine,
            self.data_width
        ]

    def channels_out(self):
        return int(self.filters/float(self.groups))

    def rate_in(self):
        return self.fine*self.groups/float(self.k_size*self.k_size*self.filters)

    def rate_out(self):
        return self.fine/float(self.k_size*self.k_size)

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

    def rsc(self): # TODO: improve DSP utilisation for different bitwidths
        return {
          "LUT"  : 0, #int(np.dot(self.utilisation_model(), self.rsc_coef[0])),
          "BRAM" : 0,
          "DSP"  : self.fine,
          "FF"   : 0 #int(np.dot(self.utilisation_model(), self.rsc_coef[3])),
        }

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self,data,weights):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[3] == self.k_size  , "ERROR: invalid column dimension"
        assert data.shape[4] == self.k_size  , "ERROR: invalid column dimension"
        # check weight dimensionality
        assert weights.shape[0] == self.channels, "ERROR: invalid channel dimension"
        assert weights.shape[1] == int(self.filters/float(self.groups)) , "ERROR: invalid filter dimension"
        assert weights.shape[2] == self.k_size  , "ERROR: invalid column dimension"
        assert weights.shape[3] == self.k_size  , "ERROR: invalid column dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.channels,
            int(self.filters/self.groups)
        ),dtype=float)

        for index,_ in np.ndenumerate(out):
            for k1 in range(self.k_size):
                for k2 in range(self.k_size):
                    out[index] += data[
                      index[0],index[1],index[2],k1,k2]*weights[
                      index[2],index[3],k1,k2]

        return out

