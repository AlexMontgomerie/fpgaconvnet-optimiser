"""
Buffering Module

Stores intermediate compute information such as results from Conv or Pool layers.
During DSE the required size will be calculated to store intermediate results at
branching layers. The position of the buffer layer will then be moved along a
given branch until the buffer size is feasible and the latency of the exit
condition is mitigated/matched. For effective pipelining I think.

Secondary function of the buffer is to "drop" a partial calculation.
Clear a FIFO - takes X number of cycles?
Drop signal will be control signal from the Exit Condition.

Future goal will be to have buffer as an offchip memory link.
In this case, the drop might not be used.
"""

from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os

class Buffer(Module):
    def __init__(
            self,
            rows,
            cols,
            channels,
            ctrledge,
            drop_mode   =True,
            data_width=16
        ):
        # module name
        self.name = "buff"

        # init module
        Module.__init__(self,rows,cols,channels,data_width)

        # init variables
        self.ctrledge = ctrledge
        self.drop_mode = drop_mode
        #self.filters = filters
        #self.groups  = groups

        # load resource coefficients
        #TODO resource coefficients file for buffer module
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
        #how should the FIFOs be laid out?
        return [
            1,
            self.data_width,
            self.data_width*self.channels
        ]

    def pipeline_depth(self):
        #TODO work out if this module can be/needs pipelining
        return 0


    def rsc(self):
        #basic version is just a single FIFO matching input dims
        bram_buffer_size =  self.rows*self.cols*self.channels*self.data_width

        bram_buffer = 0
        if bram_buffer_size >= 512: #taken from Accum.py modules
            bram_buffer = math.ceil( (bram_buffer_size)/18000)
        return {
          "LUT"  : 0, #int(np.dot(self.utilisation_model(), self.rsc_coef[0])),
          "BRAM" : bram_buffer,
          "DSP"  : 0,
          "FF"   : 0 #int(np.dot(self.utilisation_model(), self.rsc_coef[3])),
        }

    def functional_model(self, data, ctrl_drop):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.channels),dtype=float)

        if self.drop_mode: #non-inverted
            if ctrl_drop == 1.0:
                return
            else:
                return data #pass through
        else: #inverted
            if not ctrl_drop == 1.0:
                return
            else:
                return data #pass through
