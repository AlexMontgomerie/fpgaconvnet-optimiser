from fpgaconvnet_optimiser.models.modules import SlidingWindow
from fpgaconvnet_optimiser.models.modules import Conv
from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.modules import Accum
from fpgaconvnet_optimiser.models.modules import Glue
from fpgaconvnet_optimiser.models.layers import Layer

import numpy as np
import math
import pydot
import torch

class InnerProductLayer(Layer):
    def __init__(
            self,
            filters: int,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int,
            coarse_out: int,
            matmul_flag=False
        ):

        # initialise parent class
        super().__init__([rows], [cols], [channels], [coarse_in], [coarse_out])

        self.weight_width = 8

        # update flags
        self.flags['channel_dependant'] = True
        self.flags['transformable']     = True

        # save parameters
        self.filters        = filters
        self.matmul_flag    = matmul_flag

        # init modules
        self.modules = {
            "fork"           : Fork( self.rows_in(0),self.cols_in(0), self.channels_in(0),1,self.coarse_out[0]),
            "conv"           : Conv( 1,1,self.channels_in(0)*self.rows_in(0)*self.cols_in(0),filters,1,1,1),
            "accum"          : Accum(1,1,self.channels_in(0)*self.rows_in(0)*self.cols_in(0),filters,1),
            "glue"           : Glue( 1,1,self.channels_in(0)*self.rows_in(0)*self.cols_in(0),
                filters, self.coarse_in[0], self.coarse_out[0])
        }
        self.update()

    def rows_out(self, port_index):
        return 1

    def cols_out(self, port_index):
        return 1

    def channels_out(self, port_index):
        return self.filters

    def rate_in(self, port_index):
        return abs(self.balance_module_rates(self.rates_graph())[0,0])

    def rate_out(self, port_index):
        return abs(self.balance_module_rates(self.rates_graph())[3,4])

    def update_coarse_in(self, coarse_in):
        self.coarse_in[0]  = coarse_in

    def update_coarse_out(self, coarse_out):
        self.coarse_out[0] = coarse_out

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1):
        parameters.batch_size   = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in(0)
        parameters.cols_in      = self.cols_in(0)
        parameters.channels_in  = self.channels_in0()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        parameters.coarse_in    = self.coarse_in[0]
        parameters.coarse_out   = self.coarse_out[0]
        parameters.filters      = self.filters
        parameters.matmul_flag  = self.matmul_flag

    ## UPDATE MODULES ##
    def update(self): # TODO: update all parameters
        # fork
        self.modules['fork'].rows     = self.rows[0]
        self.modules['fork'].cols     = self.cols[0]
        self.modules['fork'].channels = int(self.channels[0]/self.coarse_in[0])
        self.modules['fork'].coarse   = self.coarse_out[0]
        # conv
        self.modules['conv'].rows     = 1
        self.modules['conv'].cols     = 1
        self.modules['conv'].channels = int(self.rows[0]*self.cols[0]*self.channels[0]/self.coarse_in[0])
        self.modules['conv'].filters  = int(self.filters/(self.coarse_out[0]))
        self.modules['conv'].fine     = 1
        # accum
        self.modules['accum'].rows     = 1
        self.modules['accum'].cols     = 1
        self.modules['accum'].channels = int(self.rows[0]*self.cols[0]*self.channels[0]/self.coarse_in[0])
        self.modules['accum'].filters  = int(self.filters/(self.coarse_out[0]))
        # glue
        self.modules['glue'].rows = 1
        self.modules['glue'].cols = 1
        self.modules['glue'].filters    = self.filters
        self.modules['glue'].coarse_in  = self.coarse_in[0]
        self.modules['glue'].coarse_out = self.coarse_out[0]

    ### RATES ###
    def rates_graph(self):
        rates_graph = np.zeros( shape=(4,5) , dtype=float )
        # fork
        rates_graph[0,0] = self.modules['fork'].rate_in(0)
        rates_graph[0,1] = self.modules['fork'].rate_out(0)
        # conv
        rates_graph[1,1] = self.modules['conv'].rate_in(0)
        rates_graph[1,2] = self.modules['conv'].rate_out(0)
        # accum
        rates_graph[2,2] = self.modules['accum'].rate_in(0)
        rates_graph[2,3] = self.modules['accum'].rate_out(0)
        # glue
        rates_graph[3,3] = self.modules['glue'].rate_in(0)
        rates_graph[3,4] = self.modules['glue'].rate_out(0)

        return rates_graph

    def get_weights_reloading_feasible(self):
        return self.get_factors(int(self.filters/self.coarse_out[0]))

    def get_parameters_size(self):
        weights_size = self.channels[0] * self.filters
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def resource(self):

        fork_rsc    = self.modules['fork'].rsc()
        conv_rsc    = self.modules['conv'].rsc()
        accum_rsc   = self.modules['accum'].rsc()
        if int(self.channels[0]/self.coarse_in[0]) == 1:
            accum_rsc   = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        glue_rsc    = self.modules['glue'].rsc()
        if self.coarse_in == 1:
            glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        # TODO: add to modules instead
        n_filters = float(self.filters*self.channels[0]*self.rows[0]*self.cols[0])/float(self.coarse_in[0]*self.coarse_out[0])
        weights_bram_usage = int(math.ceil(self.weight_width*n_filters/18000))*self.coarse_in[0]*self.coarse_out[0]

        # Total
        return {
            "LUT"  :  fork_rsc['LUT']*self.coarse_in[0] +
                      conv_rsc['LUT']*self.coarse_in[0]*self.coarse_out[0] +
                      accum_rsc['LUT']*self.coarse_in[0]*self.coarse_out[0] +
                      glue_rsc['LUT'],
            "FF"   :  fork_rsc['FF']*self.coarse_in[0] +
                      conv_rsc['FF']*self.coarse_in[0]*self.coarse_out[0] +
                      accum_rsc['FF']*self.coarse_in[0]*self.coarse_out[0] +
                      glue_rsc['FF'],
            "BRAM" :  fork_rsc['BRAM']*self.coarse_in[0] +
                      conv_rsc['BRAM']*self.coarse_in[0]*self.coarse_out[0] +
                      accum_rsc['BRAM']*self.coarse_out[0] +
                      glue_rsc['BRAM'] +
                      weights_bram_usage,
            "DSP"  :  fork_rsc['DSP']*self.coarse_in[0] +
                      conv_rsc['DSP']*self.coarse_in[0]*self.coarse_out[0] +
                      accum_rsc['DSP']*self.coarse_out[0] +
                      glue_rsc['DSP']
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in[0]):
            cluster.add_node(pydot.Node( "_".join([name,"fork",str(i)]), label="fork" ))

        for i in range(self.coarse_in[0]):
            for j in range(self.coarse_out[0]):
                cluster.add_node(pydot.Node( "_".join([name,"conv",str(i),str(j)]), label="conv" ))
                cluster.add_edge(pydot.Edge( "_".join([name,"fork",str(i)]) , "_".join([name,"conv",str(i),str(j)]) ))

        for i in range(self.coarse_in[0]):
            for j in range(self.coarse_out[0]):
                cluster.add_node(pydot.Node( "_".join([name,"glue",str(j)]), label="+" ))
                cluster.add_node(pydot.Node( "_".join([name,"accum",str(i),str(j)]), label="accum" ))
                cluster.add_edge(pydot.Edge( "_".join([name,"conv" ,str(i),str(j)]), "_".join([name,"accum",str(i),str(j)]) ))
                cluster.add_edge(pydot.Edge( "_".join([name,"accum",str(i),str(j)]), "_".join([name,"glue",str(j)]) ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"fork",str(i)]) for i in range(self.coarse_in[0]) ]
        nodes_out = [ "_".join([name,"glue",str(i)]) for i in range(self.coarse_out[0]) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self,data,weights,bias,batch_size=1):

        assert data.shape[0] == self.rows[0]    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols[0]    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels[0], "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters , "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.rows[0]*self.cols[0]*self.channels[0], "ERROR (weights): invalid channel dimension"


        # instantiate convolution layer
        inner_product_layer = torch.nn.Linear(self.channels[0]*self.rows[0]*self.cols[0], self.filters, bias=False)

        # update weights
        inner_product_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        inner_product_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # return output featuremap
        data = np.moveaxis(data, -1, 0).flatten()
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return inner_product_layer(torch.from_numpy(data)).detach().numpy()

