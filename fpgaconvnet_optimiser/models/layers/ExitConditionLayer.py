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

from fpgaconvnet_optimiser.models.modules import ReduceMax
from fpgaconvnet_optimiser.models.modules import Compare
from fpgaconvnet_optimiser.models.layers import Layer

class ExitConditionLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int,
            coarse_out: int,
            ctrledges: [str], #expecting list
            threshold = 0.5,  #TODO remove default, for the comparison module
            cond_type   = 'top1',
            data_width  = 16
        ):
        super().__init__([rows],[cols],[channels],[coarse_in],[coarse_out])

        self.ctrledges = ctrledges
        self.cond_type = cond_type
        self.threshold = threshold

        #update flags

        #init modules
        #TODO
        self.modules = {
            'redmx'  : ReduceMax(self.rows_in(0), self.cols_in(0), self.channels_in(0)),
            'cmp'   : Compare(self.rows_in(0), self.cols_in(0), self.channels_in(0), threshold)
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
        # exp TODO check channels are correct
        self.modules['redmx'].rows     = self.rows_in(0)
        self.modules['redmx'].cols     = self.cols_in(0)
        self.modules['redmx'].channels = int(self.channels[0]/self.coarse_in[0])
        # div
        self.modules['cmp'].rows     = self.rows_in(0)
        self.modules['cmp'].cols     = self.cols_in(0)
        self.modules['cmp'].channels = int(self.channels[0]/self.coarse_in[0])

    ### RATES ###
    def rates_graph(self): #TODO
        rates_graph = np.zeros( shape=(2,3) , dtype=float )
        # redmx
        rates_graph[0,0] = self.modules['redmx'].rate_in()
        rates_graph[0,1] = self.modules['redmx'].rate_out()
        # cmp
        rates_graph[1,1] = self.modules['cmp'].rate_in()
        rates_graph[1,2] = self.modules['cmp'].rate_out()

        return rates_graph

    def resource(self): #TODO
        redmx_rsc   = self.modules['redmx'].rsc()
        cmp_rsc     = self.modules['cmp'].rsc()

        # Total
        return {
            "LUT"  :  redmx_rsc['LUT']*self.coarse_in[0] +
                      cmp_rsc['LUT']*self.coarse_in[0],
            "FF"   :  redmx_rsc['FF']*self.coarse_in[0] +
                      cmp_rsc['FF']*self.coarse_in[0],
            "BRAM" :  redmx_rsc['BRAM']*self.coarse_in[0] +
                      cmp_rsc['BRAM']*self.coarse_in[0],
            "DSP" :   redmx_rsc['DSP']*self.coarse_in[0] +
                      cmp_rsc['DSP']*self.coarse_in[0]
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in[0]):
            cluster.add_node(pydot.Node( "_".join([name,"redmx",str(i)]), label="redmx" ))

            cluster.add_node(pydot.Node( "_".join([name,"cmp",str(i)]), label="cmp" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"redmx",str(i)]),
                                         "_".join([name,"cmp",str(i)]) ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"redmx",str(i)]) for i in range(self.coarse_in[0]) ]
        nodes_out = [ "_".join([name,"cmp",str(i)]) for i in range(self.coarse_out[0]) ]
        #nodes_out = [ "_".join([name,"cmp",str(i)]) for i in range(len(self.ctrledges)) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self, data, threshold):

        assert data.shape[0] == self.rows    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR (data): invalid channel dimension"

        #pk = softmax_layer(torch.from_numpy(data)).detach()
        #get max value
        top1 = torch.max(torch.from_numpy(data))
        #True = early exit, drop buffered data
        return top1 > threshold
