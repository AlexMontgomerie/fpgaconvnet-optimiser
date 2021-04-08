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
            data_width  =16,
            sa          =0.5,
            sa_out      =0.5
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width)

        # init variables
        self.scale_layer = None

        # modules
        self.modules = {
            "batch_norm" : BatchNorm(dim, data_width)
        }
        self.update()

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1):
        parameters.batch_size   = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        parameters.coarse_in    = self.coarse_in
        parameters.coarse_out   = self.coarse_out

    ## UPDATE MODULES ##
    def update(self):
        # batch norm
        self.modules['batch_norm'].rows     = self.rows_in()
        self.modules['batch_norm'].cols     = self.cols_in()
        self.modules['batch_norm'].channels = int(self.channels/self.coarse_in)

    def resource(self):

        bn_rsc      = self.modules['batch_norm'].rsc()
        n_filters = float(self.channels) / float(self.coarse_in)                         
        weights_bram_usage = int(math.ceil((2*self.data_width*n_filters)/18000))*self.coarse_in
        
        # Total
        return {
            "LUT"  :  bn_rsc['LUT']*self.coarse_in,
            "FF"   :  bn_rsc['FF']*self.coarse_in,
            "BRAM" :  bn_rsc['BRAM']*self.coarse_in+weights_bram_usage,
            "DSP" :   bn_rsc['DSP']*self.coarse_in
        }
   
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

