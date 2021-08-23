from fpgaconvnet_optimiser.models.modules import SlidingWindow
from fpgaconvnet_optimiser.models.modules import Conv
from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.modules import Accum
from fpgaconvnet_optimiser.models.modules import Glue
from fpgaconvnet_optimiser.models.modules import FIFO
from fpgaconvnet_optimiser.models.layers import Layer

import numpy as np
import math
import pydot
import torch

class InnerProductLayer(Layer):
    def __init__(
            self,
            dim,
            filters,
            coarse_in   =1,
            coarse_out  =1,
            data_width  =16,
            weight_width =8,
            acc_width   =30,
            sa          =0.5,
            sa_out      =0.5
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width)

        self.weight_width = weight_width
        self.acc_width = acc_width

        # update flags
        self.flags['channel_dependant'] = True
        self.flags['transformable']     = True

        self.filters   = filters

        dim_out = [
            self.filters,
            1,
            1
        ]

        # init modules
        self.modules = {
            "fork"           : Fork( [self.channels,self.rows,self.cols]    ,[1,1],coarse_out,data_width),
            "conv"           : Conv( [self.channels*self.rows*self.cols,1,1],filters,1,[1,1],1,data_width,weight_width,acc_width),
            "accum"          : Accum([self.channels*self.rows*self.cols,1,1],filters,1,acc_width),
            "glue"           : Glue( [self.channels*self.rows*self.cols,1,1],filters,coarse_in,coarse_out,data_width)
        }
        self.update()

        # switching activity
        self.sa     = sa
        self.sa_out = sa_out

    def rows_out(self):
        return 1

    def cols_out(self):
        return 1

    def channels_out(self):
        return self.filters 

    def rate_in(self,index):
        return abs(self.balance_module_rates(self.rates_graph())[0,0])

    def rate_out(self,index):
        return abs(self.balance_module_rates(self.rates_graph())[3,4])

    def update_coarse_in(self, coarse_in):
        self.coarse_in  = coarse_in

    def update_coarse_out(self, coarse_out):
        self.coarse_out = coarse_out

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
        parameters.filters      = self.filters

    ## UPDATE MODULES ##
    def update(self): # TODO: update all parameters
        # fork
        self.modules['fork'].rows     = self.rows
        self.modules['fork'].cols     = self.cols
        self.modules['fork'].channels = int(self.channels/self.coarse_in)
        self.modules['fork'].coarse   = self.coarse_out
        # conv
        self.modules['conv'].rows     = 1
        self.modules['conv'].cols     = 1
        self.modules['conv'].channels = int(self.rows*self.cols*self.channels/self.coarse_in)
        self.modules['conv'].filters  = int(self.filters/(self.coarse_out))
        self.modules['conv'].fine     = 1
        # accum
        self.modules['accum'].rows     = 1
        self.modules['accum'].cols     = 1
        self.modules['accum'].channels = int(self.rows*self.cols*self.channels/self.coarse_in)
        self.modules['accum'].filters  = int(self.filters/(self.coarse_out))
        # glue
        self.modules['glue'].rows = 1
        self.modules['glue'].cols = 1
        self.modules['glue'].filters    = self.filters
        self.modules['glue'].coarse_in  = self.coarse_in 
        self.modules['glue'].coarse_out = self.coarse_out

    ### RATES ###
    def rates_graph(self): 
        rates_graph = np.zeros( shape=(4,5) , dtype=float ) 
        # fork 
        rates_graph[0,0] = self.modules['fork'].rate_in() 
        rates_graph[0,1] = self.modules['fork'].rate_out()
        # conv
        rates_graph[1,1] = self.modules['conv'].rate_in()
        rates_graph[1,2] = self.modules['conv'].rate_out()
        # accum
        rates_graph[2,2] = self.modules['accum'].rate_in()
        rates_graph[2,3] = self.modules['accum'].rate_out()
        # glue
        rates_graph[3,3] = self.modules['glue'].rate_in()
        rates_graph[3,4] = self.modules['glue'].rate_out()

        return rates_graph
 
    def get_weights_reloading_feasible(self):
        return self.get_factors(int(self.filters/self.coarse_out))
 
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
        fork_out = FIFO([1,1,1], self.coarse_in*self.coarse_out, self.buffer_depth, self.data_width)
        fork_out_rsc = fork_out.rsc()
        conv_out = FIFO([1,1,1], self.coarse_in*self.coarse_out, self.buffer_depth, self.acc_width)
        conv_out_rsc = conv_out.rsc()
        accum_out = FIFO([1,1,1], self.coarse_in*self.coarse_out, int(self.modules['accum'].filters / self.modules['accum'].groups + 1), self.acc_width)
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

        assert data.shape[0] == self.rows    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters , "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.rows*self.cols*self.channels, "ERROR (weights): invalid channel dimension"

        
        # instantiate convolution layer
        inner_product_layer = torch.nn.Linear(self.channels*self.rows*self.cols, self.filters, bias=False)
        
        # update weights
        inner_product_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))
        
        # update bias
        inner_product_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))
        
        # return output featuremap
        data = np.moveaxis(data, -1, 0).flatten()
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0) 
        return inner_product_layer(torch.from_numpy(data)).detach().numpy()

