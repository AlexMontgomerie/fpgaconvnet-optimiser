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

        # init variables
        self.k_size = k_size
        self.stride = stride
        self.pad_top    = pad_top
        self.pad_right  = pad_right
        self.pad_bottom = pad_bottom
        self.pad_left   = pad_left
        self.data_width  =data_width
        RSC_TYPES=["LUT", "FF", "BRAM", "DSP"]
        self.rsc_coef = {
            "LUT"   : np.array([]),
            "FF"    : np.array([]),
            "DSP"   : np.array([]),
            "BRAM"  : np.array([])
        }
        for rsc in RSC_TYPES:
              self.rsc_coef[rsc]=np.load(os.path.join("/home/wz2320/fpgaconvnet-optimiser/fpgaconvnet_optimiser/coefficients/sliding_window_"+str(rsc).lower()+".npy"))

    def utilisation_model(self):
        linebufferfifo_amount=self.k_size-1
        linebufferfifo_depth=(self.cols+self.pad_left+self.pad_right)*(self.channels)
        windowcachefifo_amount=(self.k_size)*(self.k_size-1)
        windowcachefifo_depth=self.channels
        if linebufferfifo_depth>1023:
            linebuffer1=2*(2**math.floor(math.log(linebufferfifo_depth/1024,2)))
        else:
            linebuffer1=1
            
        if linebufferfifo_depth>2047:
            linebuffer2=  5*(2**(math.floor(math.log(linebufferfifo_depth/1024,2))-1))
        elif linebufferfifo_depth>1023:
            linebuffer2= 3 
        else:
            linebuffer2= 2
            
        if linebufferfifo_depth>4095:
            linebuffer3=  5*(2**(math.floor(math.log(linebufferfifo_depth/1024,2))-2))
        elif linebufferfifo_depth>2047:
            linebuffer3= 3 
        elif linebufferfifo_depth>1023:
            linebuffer3= 2 
        else:
            linebuffer3= 1
            
        if linebufferfifo_depth>8191:
            linebuffer4=  5*(2**(math.floor(math.log(linebufferfifo_depth/1024,2))-3))
        elif linebufferfifo_depth>4095:
            linebuffer4= 3 
        elif linebufferfifo_depth>2047:
            linebuffer4= 2 
        else:
            linebuffer4= 1
            
        if linebufferfifo_depth>4095:
            linebuffer5=2*(2**(math.floor(math.log(linebufferfifo_depth/1024,2))-2))
        else:
            linebuffer5=1
            
        if linebufferfifo_depth>1023:
            linebuffer0=4*(2**math.floor(math.log(linebufferfifo_depth/1024,2)))
        else:
            linebuffer0=2            
            
        if self.data_width>27:
            linebuffer=linebuffer0
        elif self.data_width>18:
            linebuffer=linebuffer2        
        elif self.data_width>13:
            linebuffer=linebuffer1
        elif self.data_width>9:
            linebuffer=linebuffer3
        elif self.data_width>4:
            linebuffer=linebuffer4    
        else:
            linebuffer=linebuffer5               
        return {
            "LUT"   : np.array([1,(4+2*self.data_width)*(self.k_size)*(self.k_size),(self.k_size)*(self.k_size-1)*(2*self.data_width+math.floor(math.log(self.channels,2))), (self.k_size-1)*(2*self.data_width+math.floor(math.log(self.channels*self.cols,2))),(self.k_size)*(self.k_size-1),(self.k_size-1)]),
            "FF"    : np.array([1,(4+2*self.data_width)*(self.k_size)*(self.k_size),(self.k_size)*(self.k_size-1)*(2*self.data_width+math.floor(math.log(self.channels,2))), (self.k_size-1)*(2*self.data_width+math.floor(math.log(self.channels*self.cols,2))),(self.k_size)*(self.k_size-1),(self.k_size-1)]),
            "DSP"   : np.array([1]),
            "BRAM"  : np.array([linebufferfifo_amount*linebuffer, windowcachefifo_amount])
            #"BRAM"  : np.array([2**(math.ceil(self.data_width/18)-1)*linebufferfifo_amount*linebuffer1 , windowcachefifo_amount])
            #"BRAM"  : np.array([linebufferfifo_amount*math.ceil(linebufferfifo_depth/1024), windowcachefifo_amount*math.ceil(windowcachefifo_depth/1024)])
                }
        
        #linebufferfifo_amount*math.ceil(linebufferfifo_depth/1024),windowcachefifo_amount*math.ceil(windowcachefifo_depth/1024),


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
            bram_line_buffer_data = (self.k_size-1)*math.ceil(line_size*data_width/18000)
            bram_line_buffer_addr = (self.k_size-1)*math.ceil(math.log(line_size,2))
            bram_line_buffer = max(bram_line_buffer_data, bram_line_buffer_addr)
            #bram_line_buffer = bram_line_buffer_data
        # frame buffer
        frame_size = (self.channels+1)
        bram_frame_buffer = 0
        if frame_size*self.data_width >= 512:
            bram_frame_buffer_data = self.k_size*(self.k_size-1)*math.ceil(frame_size*data_width/18000)
            bram_frame_buffer_addr = self.k_size*(self.k_size-1)*math.ceil(math.log(frame_size,2))
            bram_frame_buffer = max(bram_frame_buffer_data, bram_frame_buffer_addr)
            #bram_frame_buffer = bram_frame_buffer_data
        return {
          "LUT"  : int(np.dot(self.utilisation_model()["LUT"], coef["LUT"])),
          "BRAM" : int(np.dot(self.utilisation_model()["BRAM"], coef["BRAM"])),
          "DSP"  : 0,
          "FF"   : int(np.dot(self.utilisation_model()["FF"], coef["FF"])),
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
            self.k_size,
            self.k_size),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data_padded[
                          index[0],
                          index[1]*self.stride+index[4],
                          index[2]*self.stride+index[5],
                          index[3]]

        return out

