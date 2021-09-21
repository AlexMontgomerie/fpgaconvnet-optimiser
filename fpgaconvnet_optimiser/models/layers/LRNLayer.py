from fpgaconvnet_optimiser.models.layers import Layer

class LRNLayer(Layer):
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


