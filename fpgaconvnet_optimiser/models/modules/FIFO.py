from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os
import sys

from fpgaconvnet_optimiser.tools.hls_helper import stream_rsc

class FIFO(Module):
    def __init__(
            self,
            dim,
            coarse,
            depth,
            data_width=16
        ):
        
        # module name
        self.name = "fifo"

        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.coarse = coarse
        self.depth  = depth


        # load resource coefficients
        #work_dir = os.getcwd()
        #os.chdir(sys.path[0])
        #self.rsc_coef = np.load(os.path.join(os.path.dirname(__file__),
        #    "../../coefficients/squeeze_rsc_coef.npy"))
        #os.chdir(work_dir)

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

    def utilisation_model(self):
        assert self.data_width == 16 or self.data_width == 30
        single_fifo_rsc = stream_rsc(self.data_width, self.depth)

        return [
            1,
            self.data_width,
            self.coarse,
            self.data_width*self.depth*self.coarse if single_fifo_rsc['BRAM'] == 0 else 0,
            single_fifo_rsc['BRAM']*self.coarse,
        ]


    def rsc(self,coef=None):
        if coef == None:
            coef = self.rsc_coef

        return {
          "LUT"  : int(np.dot(self.utilisation_model(), coef['LUT'])),
          "BRAM" : stream_rsc(self.data_width, self.depth)['BRAM']*self.coarse,
          "DSP"  : 0,
          "FF"   : int(np.dot(self.utilisation_model(), coef['FF'])),
        }

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self, data):   
        return data