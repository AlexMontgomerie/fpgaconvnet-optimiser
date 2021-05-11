"""
The Sliding Window module creates sequential windows of the
incoming feature map. This module allows for efficient use 
of the on-chip memory compared to full featuremap caching, 
with only the required number of pixels buffered. This 
stream of feature map windows is used for the convolution 
and pooling functions. 

.. figure:: ../../../figures/sliding_window_diagram.png
"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os
import sys

from fpgaconvnet_optimiser.tools.onnx_helper import _pair

class SlidingWindow(Module):
    """
    Sliding window hardware model class.
    """
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
        """
        Parameters
        ----------
        dim: list
            dimensions of the input featuremap. Should contain
            `channels`, `rows`, `cols` in that order.

        Attributes
        ----------
        k_size: int
            kernel size of the convolution layer.
        stride: int
            both row and column stride of the convolution layer.
        pad_top: int 
            zero padding for the top of the featuremap.
        pad_right: int
            zero padding for the right of the featuremap.
        pad_bottom: int 
            zero padding for the bottom of the featuremap.
        pad_left: int 
            zero padding for the left of the featuremap.
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
        self.name = "sliding_window"

        # init module
        Module.__init__(self,dim,data_width)

        k_size = _pair(k_size)
        stride = _pair(stride)

        # init variables
        self.k_size = k_size
        self.stride = stride
        self.pad_top    = pad_top
        self.pad_right  = pad_right
        self.pad_bottom = pad_bottom
        self.pad_left   = pad_left


        # load resource coefficients
        #work_dir = os.getcwd()
        #os.chdir(sys.path[0])
        #self.rsc_coef = np.load(os.path.join(os.path.dirname((__file__)),
        #    "../../coefficients/sliding_window_rsc_coef.npy"))
        #os.chdir(work_dir)

    def utilisation_model(self):
        return [
            1,
            self.data_width*self.k_size[0]*self.k_size[1],
            self.data_width*(self.k_size[0]-1),
            self.data_width*self.k_size[0]*(self.k_size[1]-1),
            (self.k_size[0]-1)*(((self.cols+self.pad_left+self.pad_right)*self.channels+1)*self.data_width) if ((self.cols+self.pad_left+self.pad_right)*self.channels+1)*self.data_width < 512 else 0,
            (self.k_size[0]-1)*math.ceil( (((self.cols+self.pad_left+self.pad_right)*self.channels+1)*self.data_width)/18000) if ((self.cols+self.pad_left+self.pad_right)*self.channels+1)*self.data_width >= 512 else 0,
            self.k_size[0]*(self.k_size[1]-1)*(self.channels+1)*self.data_width  if self.channels*self.data_width < 512 else 0,
            self.k_size[0]*(self.k_size[1]-1)*math.ceil( ((self.channels+1)*self.data_width)/18000) if self.channels*self.data_width >= 512 else 0,
            self.data_width*self.k_size[0]*self.k_size[1]*self.channels
        ]

    def rows_out(self):
        return int((self.rows_in()-self.k_size[0]+self.pad_top+self.pad_bottom)/self.stride[0]+1)

    def cols_out(self):
        return int((self.cols_in()-self.k_size[1]+self.pad_left+self.pad_right)/self.stride[1]+1)

    def rate_in(self):
        return 1.0 # TODO: maybe need to reduce for padding effect

    def rate_out(self):
        return (self.rows_out()*self.cols_out())/float(self.rows*self.cols)

    def pipeline_depth(self):
        return (self.cols+self.pad_left+self.pad_right)*(self.channels)*(self.k_size[0]-1)+self.channels*self.k_size[0]*(self.k_size[1]-1)

    def wait_depth(self):
        """
        Number of cycles delay before the first pixel is
        consumed by the module from the start signal.

        Returns
        -------
        int
        """
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


    def rsc(self, coef=None):
        """
        the main resources are from the line and frame buffers.
        These use `BRAM` fifos. 

        Returns
        -------
        dict 
            estimated resource usage of the module. Uses the
            resource coefficients for the estimate.
        """
        if coef == None:
            coef = self.rsc_coef
        # stream
        data_width = 18
        # line buffer
        line_size = ((self.cols+self.pad_left+self.pad_right)*self.channels+1)
        bram_line_buffer = 0
        if line_size*self.data_width >= 512:
            bram_line_buffer_data = (self.k_size[0]-1)*math.ceil(line_size*data_width/18000)
            bram_line_buffer_addr = (self.k_size[0]-1)*math.ceil(math.log(line_size,2))
            bram_line_buffer = max(bram_line_buffer_data, bram_line_buffer_addr)
            #bram_line_buffer = bram_line_buffer_data
        # frame buffer
        frame_size = (self.channels+1)
        bram_frame_buffer = 0
        if frame_size*self.data_width >= 512:
            bram_frame_buffer_data = self.k_size[0]*(self.k_size[1]-1)*math.ceil(frame_size*data_width/18000)
            bram_frame_buffer_addr = self.k_size[0]*(self.k_size[1]-1)*math.ceil(math.log(frame_size,2))
            bram_frame_buffer = max(bram_frame_buffer_data, bram_frame_buffer_addr)
            #bram_frame_buffer = bram_frame_buffer_data
        return {
          "LUT"  : int(np.dot(self.utilisation_model(), coef["LUT"])),
          "BRAM" : bram_line_buffer+bram_frame_buffer + self.k_size[0]*self.k_size[1],
          "DSP"  : 0,
          "FF"   : int(np.dot(self.utilisation_model(), coef["FF"])),
        }

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
            self.k_size[0],
            self.k_size[1]),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data_padded[
                          index[0],
                          index[1]*self.stride[0]+index[4],
                          index[2]*self.stride[1]+index[5],
                          index[3]]

        return out

