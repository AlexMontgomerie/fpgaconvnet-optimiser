from fpgaconvnet_optimiser.models.modules import BatchNorm
from fpgaconvnet_optimiser.models.layers import Layer

import numpy as np
import math
import tempfile
import pydot

class BatchNormLayer(Layer):
    def __init__(
            self,
            dim,
            coarse_in   =1,
            coarse_out  =1,
            data_width  =12,
            sa          =0.5,
            sa_out      =0.5
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width)
        
        self.data_width = data_width         

        # init variables
        self.scale_layer = None

        # modules
        self.modules = {
            "batch_norm" : BatchNorm(dim)
        }
       
        self.update()

    ## LAYER INFO ##
    def layer_info(self):
        return {
            'type'          : 'BATCH NORM',
            'buffer_depth'  : self.buffer_depth,
            'rows'          : self.rows,
            'cols'          : self.cols,
            'channels'      : self.channels,
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
        # batch norm
        self.modules['batch_norm'].rows     = self.rows_in()
        self.modules['batch_norm'].cols     = self.cols_in()
        self.modules['batch_norm'].channels = int(self.channels/self.coarse_in)
   
    def functional_model(self,data,gamma,beta,batch_size=1):

        
        assert data.shape[0] == self.rows    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR (data): invalid channel dimension"

        assert gamma.shape[0] == self.channels , "ERROR (weights): invalid filter dimension"
        assert beta.shape[0]  == self.channels , "ERROR (weights): invalid filter dimension"

        # instantiate batch norm layer
        batch_norm_layer = torch.nn.BatchNorm2d(self.channels, track_running_stats=False) 

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0) 
        return batch_norm_layer(torch.from_numpy(data))

