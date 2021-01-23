"""
.. figure:: ../../../figures/sliding_window_diagram.png
"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math

class SlidingWindow(Module):
    def __init__(
            self,
            dim,
            k_size,
            stride,
            pad_top,
            pad_right,
            pad_bottom,
            pad_left,
            data_width=16
        ):
        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.k_size = k_size
        self.stride = stride
        self.pad_top    = pad_top
        self.pad_right  = pad_right
        self.pad_bottom = pad_bottom
        self.pad_left   = pad_left

    def dynamic_model(self, freq, rate, sa_in, sa_out):
        return [
            self.data_width*freq,
            self.data_width*self.sa_in*freq*rate,
            self.data_width*self.sa_in*freq*rate*self.k_size*self.k_size,
            self.data_width*self.sa_in*freq*rate*(self.k_size-1),
            self.data_width*self.sa_in*freq*rate*self.k_size*self.k_size*self.rows_out()*self.cols_out()/float(self.rows*self.cols),
        ]

    def utilisation_model(self):
        return [
            1,
            self.data_width,
            self.data_width*self.k_size*self.k_size,
            self.data_width*(self.k_size-1)*self.cols*self.channels,
            self.data_width*self.k_size*self.k_size*self.channels
        ]

    def rows_out(self):
        return int((self.rows_in()-self.k_size+self.pad_top+self.pad_bottom)/self.stride+1)

    def cols_out(self):
        return int((self.cols_in()-self.k_size+self.pad_left+self.pad_right)/self.stride+1)

    def rate_in(self):
        return 1.0 # TODO: maybe need to reduce for padding effect

    def rate_out(self):
        return (self.rows_out()*self.cols_out())/float(self.rows*self.cols)

    def pipeline_depth(self):
        return (self.cols+self.pad_left+self.pad_right)*(self.channels)*(self.k_size-1)+self.channels*self.k_size*(self.k_size-1)

    def wait_depth(self):
        return (self.pad_bottom*self.channels*self.cols+self.pad_left*self.channels+1)

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'stride'    : self.stride,
            'pad_top'       : self.pad_top,
            'pad_right'     : self.pad_right,
            'pad_bottom'    : self.pad_bottom,
            'pad_left'      : self.pad_left,
            'kernel_size'   : self.k_size,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }


    def rsc(self):
        # streams
        #if self.channels > 1: 
        bram_line_buffer = (self.k_size-1)*math.ceil( (((self.cols+self.pad_left+self.pad_right)*self.channels+1)*self.data_width)/18000)
        #if self.channels*self.data_width >= 512:
        bram_frame_buffer = self.k_size*(self.k_size-1)*math.ceil( ((self.channels+1)*self.data_width)/18000)
        return {
          "LUT"  : 0, #int(np.dot(self.utilisation_model(), self.rsc_coef[0])),
          "BRAM" : bram_line_buffer+bram_frame_buffer,
          "DSP"  : 0,
          "FF"   : 0 #int(np.dot(self.utilisation_model(), self.rsc_coef[3])),
        }

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self, data):
        # check input dimensionality
        batch_size = data.shape[0]
        assert data.shape[1] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[2] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[3] == self.channels, "ERROR: invalid channel dimension"

        #pad input
        data_padded = np.ndarray((
            batch_size,
            self.rows + self.pad_bottom + self.pad_top,
            self.cols + self.pad_left   + self.pad_right,
            self.channels),dtype=float)

        for index,_ in np.ndenumerate(data_padded):
            if  (index[1] < self.pad_bottom):
                data_padded[index] = 0
            elif(index[2] < self.pad_left):
                data_padded[index] = 0
            elif(index[1] > self.rows - 1 + self.pad_bottom):
                data_padded[index] = 0
            elif(index[2] > self.cols - 1 + self.pad_left):
                data_padded[index] = 0
            else:
                data_padded[index] = data[
                                        index[0],
                                        index[1]-self.pad_left,
                                        index[2]-self.pad_bottom,
                                        index[3]]

        out = np.ndarray((
            batch_size,
            self.rows_out(),
            self.cols_out(),
            self.channels,
            self.k_size,
            self.k_size),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data_padded[
                          index[0],
                          index[1]*self.stride+index[4],
                          index[2]*self.stride+index[5],
                          index[3]]

        return out

