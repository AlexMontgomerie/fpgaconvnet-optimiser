"""
Exit Condition layer

Takes the input and performs Softmax, then takes the maximum value, and
compares it to the exit threshold value specified in the model.
This generates a control signal that will terminate the execution early
or allow the input sample to pass through the graph.

TODO Add other variations of the exit condition.
"""

import torch
import math
import numpy as np
import pydot

#from fpgaconvnet_optimiser.models.modules import SlidingWindow
#from fpgaconvnet_optimiser.models.modules import Pool
from fpgaconvnet_optimiser.models.layers import Layer

class ExitConditionLayer(Layer):
    def __init__(
            self,
            dim,
            ctrledges, #expecting list
            cond_type   = 'top1',
            coarse_in   = 1,
            coarse_out  = 1,
            data_width  = 16
        ):
        Layer.__init__(self, dim, coarse_in, coarse_out, data_width)

        self.ctrledges = ctrledges
        self.cond_type = cond_type

        #update flags

        #init modules
        #TODO
        self.modules = {
        }

        self.update()

    def layer_info(self,parameters,batch_size=1):
        parameters.batch_size   = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in()
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        parameters.coarse       = self.coarse_in
        parameters.coarse_in    = self.coarse_in
        parameters.coarse_out   = self.coarse_out

    def update(self): #TODO
        Layer.update(self)

    def rates_graph(self): #TODO
        rates_graph = np.zeros( shape=(1,2), dtype=float)
        return rates_graph

    def resource(self): #TODO
        mod_rsc = 0#self.modules['mod'].rsc()

        # Total
        return {
            "LUT"  :  mod_rsc['LUT']*self.coarse_in,
            "FF"   :  mod_rsc['FF']*self.coarse_in,
            "BRAM" :  mod_rsc['BRAM']*self.coarse_in,
            "DSP" :   mod_rsc['DSP']*self.coarse_in,
        }

    def visualise(self,name): #TODO replace 'mod' with actual modules used
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"mod",str(i)]), label="mod" ))

        for i in range(self.coarse_out):
            cluster.add_node(pydot.Node( "_".join([name,"mod",str(i)]), label="mod" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"mod",str(i)]) for i in range(self.coarse_in) ]
        nodes_out = [ "_".join([name,"mod",str(i)]) for i in range(self.coarse_out) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self, data, threshold):

        assert data.shape[0] == self.rows    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR (data): invalid channel dimension"

        #instantiate softmax layer
        softmax_layer = torch.nn.Softmax() #TODO move softmax to separate layer
        pk = softmax_layer(torch.from_numpy(data)).detach()
        #get max value
        top1 = torch.max(pk)
        #True = early exit, drop buffered data
        return top1 > threshold
