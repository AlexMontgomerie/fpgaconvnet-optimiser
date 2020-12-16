from models.layers.Layer import Layer

import numpy as np
import pydot

class SplitLayer(Layer):
    def __init__(
            self,
            dim,
            outputs,
            coarse_in   =1,
            coarse_out  =1,
            data_width  =16,
            sa          =0.5,
            sa_out      =0.5
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width)

        # update flags
        self.flags['multi_output'] = True

        # init variables
        self.outputs = outputs

    ## LAYER INFO ##
    def layer_info(self):
        return {
            'type'          : 'SPLIT',
            'buffer_depth'  : self.buffer_depth,
            'rows'          : self.rows,
            'cols'          : self.cols,
            'channels'      : self.channels,
            'outputs'       : self.outputs,
            'coarse'        : self.coarse_in,
            'coarse_in'     : self.coarse_in,
            'coarse_out'    : self.coarse_out,
            'size_in'       : int(self.rows*self.cols*self.channels),
            'size_out'      : int(self.rows_out()*self.cols_out()*self.channels_out()),
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    ## UPDATE MODULES ##
    def update(self):
        pass
