"""
The Fork module provides functionality for 
parallelism within layers. By duplicating the 
streams, it can be used for exploiting 
parallelism across filters in the Convolution 
layers.

.. figure:: ../../../figures/fork_diagram.png
"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os

class Fork(Module):
    def __init__(
            self,
            dim,
            k_size,
            coarse,
            batch_size,
            data_width=16
        ):
        
        # module name
        self.name = "fork"
 
        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.k_size = k_size
        self.coarse = coarse
        self.batch_size=batch_size

        #load resource coefficients
        RSC_TYPES=["LUT", "FF", "BRAM", "DSP"]
        self.rsc_coef = {
            "LUT"   : np.array([]),
            "FF"    : np.array([]),
            "DSP"   : np.array([]),
            "BRAM"  : np.array([])
        }
        for rsc in RSC_TYPES:
              self.rsc_coef[rsc]=np.load(os.path.join(os.path.dirname(__file__),"../../coefficients/fork_"+str(rsc).lower()+".npy"))

    def dynamic_model(self, freq, rate, sa_in, sa_out):
        return [
            self.data_width*freq,
            self.data_width*sa_in*freq*rate*self.k_size*self.k_size,
            self.data_width*sa_in*freq*rate*self.k_size*self.k_size*self.coarse
        ]

    def utilisation_model(self):
        if self.channels*self.rows*self.cols*self.batch_size<2**18+1:
            if self.k_size==1:
                if self.coarse==1:
                    LUT=12
                else:
                    LUT=14   
            if self.k_size==2:
                if self.coarse==1:
                    LUT=14
                else:
                    LUT=13
            else:
                if self.coarse==1:
                    LUT=13
                else:
                    LUT=20      
        elif self.channels*self.rows*self.cols*self.batch_size<2**19+1:
            if self.k_size==1:
                if self.coarse==1:
                    LUT=13
                else:
                    LUT=13   
            if self.k_size==2:
                if self.coarse==1:
                    LUT=13
                else:
                    LUT=16
            else:
                if self.coarse==1:
                    LUT=16
                else:
                    LUT=21      
        elif self.channels*self.rows*self.cols*self.batch_size<2**20+1:
            if self.k_size==1:
                if self.coarse==1:
                    LUT=12
                else:
                    LUT=13   
            if self.k_size==2:
                if self.coarse==1:
                    LUT=13
                else:
                    LUT=14
            else:
                if self.coarse==1:
                    LUT=15
                else:
                    LUT=20 
        elif self.channels*self.rows*self.cols*self.batch_size<2**21+1:
            if self.k_size==1:
                if self.coarse==1:
                    LUT=12
                else:
                    LUT=12   
            if self.k_size==2:
                if self.coarse==1:
                    LUT=14
                else:
                    LUT=14
            else:
                if self.coarse==1:
                    LUT=15
                else:
                    LUT=21       
        elif self.channels*self.rows*self.cols*self.batch_size<2**22+1:
            if self.k_size==1:
                if self.coarse==1:
                    LUT=12
                else:
                    LUT=12   
            if self.k_size==2:
                if self.coarse==1:
                    LUT=13
                else:
                    LUT=17
            else:
                if self.coarse==1:
                    LUT=17
                else:
                    LUT=21  
        elif self.channels*self.rows*self.cols*self.batch_size<2**23+1:
            if self.k_size==1:
                if self.coarse==1:
                    LUT=14
                else:
                    LUT=14   
            if self.k_size==2:
                if self.coarse==1:
                    LUT=14
                else:
                    LUT=17
            else:
                if self.coarse==1:
                    LUT=16
                else:
                    LUT=22  
        else:
            if self.k_size==1:
                if self.coarse==1:
                    LUT=14
                else:
                    LUT=14   
            if self.k_size==2:
                if self.coarse==1:
                    LUT=14
                else:
                    LUT=16
            else:
                if self.coarse==1:
                    LUT=19
                else:
                    LUT=22                                                                                                                                                                                                                                                                
        return {
            "LUT"   : np.array([1,LUT]),
            "FF"    : np.array([1,2+math.ceil(math.log(self.channels*self.rows*self.cols*self.batch_size,2))]),
            "DSP"   : np.array([1]),
            "BRAM"  : np.array([1])
        }

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'coarse'    : self.coarse,
            'kernel_size'   : self.k_size,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def rsc(self,coef=None):
       if coef == None:
            coef = self.rsc_coef
       return {
          "LUT"  : int(np.dot(self.utilisation_model()["LUT"], coef["LUT"])),
          "BRAM" : int(np.dot(self.utilisation_model()["BRAM"], coef["BRAM"])),
          "DSP"  : 0,
          "FF"   : int(np.dot(self.utilisation_model()["FF"], coef["FF"])),
        }

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[3] == self.k_size  , "ERROR: invalid column dimension"
        assert data.shape[4] == self.k_size  , "ERROR: invalid column dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            self.channels,
            self.coarse,
            self.k_size,
            self.k_size),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[
              index[0],
              index[1],
              index[2],
              index[4],
              index[5]]

        return out

