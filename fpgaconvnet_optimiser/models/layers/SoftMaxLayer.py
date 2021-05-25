import numpy as np
import math
import pydot
import torch

from fpgaconvnet_optimiser.models.modules import Exponential
from fpgaconvnet_optimiser.models.modules import Accum
from fpgaconvnet_optimiser.models.modules import Div
from fpgaconvnet_optimiser.models.layers import Layer

class SoftMaxLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int,
            coarse_out: int,
            #size_in #number of values to softmax over
            data_width  =16,
        ):
        # initialise parent class
        super().__init__([rows],[cols],[channels],[coarse_in],[coarse_out])

        #update flags TODO
        #update parameters TODO

        #init modules
        self.modules = {
                "exp"   : Exponential(self.rows_in(0), self.cols_in(0), self.channels_in(0)),
                #TODO check filters/groups required for accum module
                "accum" : Accum(self.rows_in(0), self.cols_in(0), self.channels_in(0), 1, 1),
                "div"   : Div(self.rows_in(0), self.cols_in(0), self.channels_in(0))
        }

        self.update()

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1):
        parameters.batch_size   = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in(0)
        parameters.cols_in      = self.cols_in(0)
        parameters.channels_in  = self.channels_in0()
        parameters.rows_out     = self.rows_out(0)
        parameters.cols_out     = self.cols_out(0)
        parameters.channels_out = self.channels_out(0)
        parameters.coarse_in    = self.coarse_in
        parameters.coarse_out   = self.coarse_out

    ## UPDATE MODULES ##
    def update(self):
        # exp TODO check channels are correct
        self.modules['exp'].rows     = self.rows_in(0)
        self.modules['exp'].cols     = self.cols_in(0)
        self.modules['exp'].channels = int(self.channels[0]/self.coarse_in[0])
        # accum
        self.modules['accum'].rows     = self.rows_out(0)
        self.modules['accum'].cols     = self.cols_out(0)
        self.modules['accum'].channels = int(self.channels[0]/self.coarse_in[0])
        # div
        self.modules['div'].rows     = self.rows_in(0)
        self.modules['div'].cols     = self.cols_in(0)
        self.modules['div'].channels = int(self.channels[0]/self.coarse_in[0])

    ### RATES ###
    def rates_graph(self):
        rates_graph = np.zeros( shape=(3,4) , dtype=float )
        # exp
        rates_graph[0,0] = self.modules['exp'].rate_in()
        rates_graph[0,1] = self.modules['exp'].rate_out()
        # accum
        rates_graph[1,1] = self.modules['accum'].rate_in()
        rates_graph[1,2] = self.modules['accum'].rate_out()
        # div
        rates_graph[2,2] = self.modules['div'].rate_in()
        rates_graph[2,3] = self.modules['div'].rate_out()

        return rates_graph

    def resource(self):
        exp_rsc     = self.modules['exp'].rsc()
        accum_rsc   = self.modules['accum'].rsc()
        div_rsc     = self.modules['div'].rsc()

        # Total
        return {
            "LUT"  :  exp_rsc['LUT']*self.coarse_in[0] +
                      accum_rsc['LUT']*self.coarse_in[0] +
                      div_rsc['LUT']*self.coarse_in[0],
            "FF"   :  exp_rsc['FF']*self.coarse_in[0] +
                      accum_rsc['FF']*self.coarse_in[0] +
                      div_rsc['FF']*self.coarse_in[0],
            "BRAM" :  exp_rsc['BRAM']*self.coarse_in[0] +
                      accum_rsc['BRAM']*self.coarse_in[0] +
                      div_rsc['BRAM']*self.coarse_in[0],
            "DSP" :   exp_rsc['DSP']*self.coarse_in[0] +
                      accum_rsc['DSP']*self.coarse_in[0] +
                      div_rsc['DSP']*self.coarse_in[0]
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in[0]):
            cluster.add_node(pydot.Node( "_".join([name,"exp",str(i)]), label="exp" ))

        for i in range(self.coarse_out[0]): #TODO balance this correctly
            cluster.add_node(pydot.Node( "_".join([name,"accum",str(i)]), label="accum" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"exp",str(i)]),
                                         "_".join([name,"accum",str(i)]) ))
            cluster.add_node(pydot.Node( "_".join([name,"div",str(i)]), label="div" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"accum",str(i)]),
                                         "_".join([name,"div",str(i)]) ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"exp",str(i)]) for i in range(self.coarse_in[0]) ]
        nodes_out = [ "_".join([name,"div",str(i)]) for i in range(self.coarse_out[0]) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows_in(0)    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in(0)    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(0), "ERROR (data): invalid channel dimension"

        #instantiate softmax layer
        softmax_layer = torch.nn.Softmax()
        out = softmax_layer(torch.from_numpy(data)).detach().numpy()
        return out
