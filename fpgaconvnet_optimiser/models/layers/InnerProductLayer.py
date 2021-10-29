import numpy as np
import math
import pydot
import torch

from fpgaconvnet_optimiser.models.layers.utils import get_factors

from fpgaconvnet_optimiser.tools.resource_model import bram_memory_resource_model

from fpgaconvnet_optimiser.models.modules import SlidingWindow
from fpgaconvnet_optimiser.models.modules import Conv
from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.modules import Accum
from fpgaconvnet_optimiser.models.modules import Glue
from fpgaconvnet_optimiser.models.modules import FIFO
from fpgaconvnet_optimiser.models.layers import Layer

class InnerProductLayer(Layer):
    def __init__(
            self,
            filters: int,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int = 1,
            coarse_out: int = 1,
            input_width: int = 16,
            output_width: int = 16,
            weight_width: int = 16,
            acc_width: int = 16
        ):

        # initialise parent class
        super().__init__(rows, cols, channels, coarse_in,
                coarse_out, data_width=input_width)

        # save the widths
        self.input_width = input_width
        self.output_width = output_width
        self.weight_width = weight_width
        self.acc_width = acc_width

        # update flags
        # self.flags['channel_dependant'] = True
        # self.flags['transformable']     = True

        # save parameters
        self._filters = filters

        # init modules
        self.modules["fork"] = Fork(self.rows_in(), self.cols_in(), self.channels_in(), 1, self.coarse_out)
        self.modules["conv"] = Conv(1,1,self.channels_in()*self.rows_in()*self.cols_in(), self.filters, 1, 1, 1)
        self.modules["accum"] = Accum(1,1,self.channels_in()*self.rows_in()*self.cols_in(), self.filters, 1)
        self.modules["glue"] = Glue(1,1,self.channels_in()*self.rows_in()*self.cols_in(),
                self.filters, self.coarse_in, self.coarse_out)

        self.update()

    @property
    def filters(self) -> int:
        return self._filters

    @filters.setter
    def filters(self, val: int) -> None:
        self._filters = val
        self.update()

    def rows_out(self) -> int:
        return 1

    def cols_out(self) -> int:
        return 1

    def channels_out(self) -> int:
        return self.filters

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.filters      = self.filters
        parameters.input_width  = self.input_width
        parameters.output_width = self.output_width
        parameters.weight_width = self.weight_width
        parameters.acc_width    = self.acc_width

    def update(self): # TODO: update all parameters
        # fork
        self.modules['fork'].rows     = self.rows_in()
        self.modules['fork'].cols     = self.cols_in()
        self.modules['fork'].channels = self.channels_in()//self.coarse_in
        self.modules['fork'].coarse   = self.coarse_out
        self.modules['fork'].data_width = self.input_width
        # conv
        self.modules['conv'].rows     = 1
        self.modules['conv'].cols     = 1
        self.modules['conv'].channels = self.rows_in()*self.cols_in()*self.channels_in()//self.coarse_in
        self.modules['conv'].filters  = self.filters//self.coarse_out
        self.modules['conv'].fine     = 1
        self.modules['conv'].data_width = self.input_width
        self.modules['conv'].weight_width = self.weight_width
        self.modules['conv'].acc_width = self.acc_width
        # accum
        self.modules['accum'].rows     = 1
        self.modules['accum'].cols     = 1
        self.modules['accum'].channels = self.rows_in()*self.cols_in()*self.channels_in()//self.coarse_in
        self.modules['accum'].filters  = self.filters//self.coarse_out
        self.modules['accum'].data_width = self.acc_width
        # glue
        self.modules['glue'].rows = 1
        self.modules['glue'].cols = 1
        self.modules['glue'].filters    = self.filters
        self.modules['glue'].coarse_in  = self.coarse_in
        self.modules['glue'].coarse_out = self.coarse_out
        self.modules['glue'].data_width = self.output_width
        self.modules['glue'].acc_width  = self.acc_width

    def get_weights_reloading_feasible(self):
        return get_factors(int(self.filters/self.coarse_out))

    def get_parameters_size(self):
        weights_size = self.channels * self.filters
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def resource(self):

        # instances
        fork_rsc    = self.modules['fork'].rsc()
        conv_rsc    = self.modules['conv'].rsc()
        accum_rsc   = self.modules['accum'].rsc()
        glue_rsc    = self.modules['glue'].rsc()

        # stream
        fork_out = FIFO(1, 1, 1, self.coarse_in*self.coarse_out, self.buffer_depth)
        fork_out.data_width = self.data_width
        fork_out_rsc = fork_out.rsc()
        conv_out = FIFO(1, 1, 1, self.coarse_in*self.coarse_out, self.buffer_depth)
        conv_out.data_width = self.acc_width
        conv_out_rsc = conv_out.rsc()
        accum_out = FIFO(1, 1, 1, self.coarse_in*self.coarse_out, int(self.modules['accum'].filters / self.modules['accum'].groups + 1))
        accum_out.data_width = self.acc_width
        accum_out_rsc = accum_out.rsc()

        if int(self.channels/self.coarse_in) == 1:
            accum_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            accum_out_rsc = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        #if self.coarse_in == 1:
        #    glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        # Total
        return {
            "LUT"  :  fork_rsc['LUT']*self.coarse_in +
                      conv_rsc['LUT']*self.coarse_in*self.coarse_out +
                      accum_rsc['LUT']*self.coarse_in*self.coarse_out +
                      glue_rsc['LUT'] +
                      fork_out_rsc['LUT'] +
                      conv_out_rsc['LUT'] +
                      accum_out_rsc['LUT'],
            "FF"   :  fork_rsc['FF']*self.coarse_in +
                      conv_rsc['FF']*self.coarse_in*self.coarse_out +
                      accum_rsc['FF']*self.coarse_in*self.coarse_out +
                      glue_rsc['FF'] +
                      fork_out_rsc['FF'] +
                      conv_out_rsc['FF'] +
                      accum_out_rsc['FF'],
            "BRAM" :  fork_rsc['BRAM']*self.coarse_in +
                      conv_rsc['BRAM']*self.coarse_in*self.coarse_out +
                      accum_rsc['BRAM']*self.coarse_in*self.coarse_out +
                      glue_rsc['BRAM'] +
                      fork_out_rsc['BRAM'] +
                      conv_out_rsc['BRAM'] +
                      accum_out_rsc['BRAM'],
            "DSP"  :  fork_rsc['DSP']*self.coarse_in +
                      conv_rsc['DSP']*self.coarse_in*self.coarse_out +
                      accum_rsc['DSP']*self.coarse_in*self.coarse_out +
                      glue_rsc['DSP']
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"fork",str(i)]), label="fork" ))

        for i in range(self.coarse_in):
            for j in range(self.coarse_out):
                cluster.add_node(pydot.Node( "_".join([name,"conv",str(i),str(j)]), label="conv" ))
                cluster.add_edge(pydot.Edge( "_".join([name,"fork",str(i)]) , "_".join([name,"conv",str(i),str(j)]) ))

        for i in range(self.coarse_in):
            for j in range(self.coarse_out):
                cluster.add_node(pydot.Node( "_".join([name,"glue",str(j)]), label="+" ))
                cluster.add_node(pydot.Node( "_".join([name,"accum",str(i),str(j)]), label="accum" ))
                cluster.add_edge(pydot.Edge( "_".join([name,"conv" ,str(i),str(j)]), "_".join([name,"accum",str(i),str(j)]) ))
                cluster.add_edge(pydot.Edge( "_".join([name,"accum",str(i),str(j)]), "_".join([name,"glue",str(j)]) ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"fork",str(i)]) for i in range(self.coarse_in) ]
        nodes_out = [ "_".join([name,"glue",str(i)]) for i in range(self.coarse_out) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self,data,weights,bias,batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters , "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.rows_in()*self.cols_in()*self.channels_in(), "ERROR (weights): invalid channel dimension"


        # instantiate inner product layer
        inner_product_layer = torch.nn.Linear(self.channels_in()*self.rows_in()*self.cols_in(), self.filters, bias=False)

        # update weights
        inner_product_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        inner_product_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # return output featuremap
        data = np.moveaxis(data, -1, 0).flatten()
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return inner_product_layer(torch.from_numpy(data)).detach().numpy()

