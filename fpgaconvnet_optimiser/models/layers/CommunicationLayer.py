import torch
import math
import numpy as np
import pydot

from fpgaconvnet_optimiser.models.modules import Communication
from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.modules import Glue
from fpgaconvnet_optimiser.models.layers import Layer

class CommunicationLayer(Layer):
    def __init__(
            self,
            dim,
            groups          =1,
            coarse_in       =1,
            coarse_out      =1,
            data_width      =16,
            send_nreceive   =True,
            pair_id         =0,
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width)

        # update flags
        self.flags['channel_dependant'] = True
        self.flags['transformable']     = True
        # init modules
        self.modules = {
            "comm"  : Communication(dim,coarse_out,send_nreceive),
        }
        self.update()
        #self.load_coef()
    def rate_in(self,index):
        return abs(self.balance_module_rates(self.rates_graph())[0,0])

    def rate_out(self,index):
        return abs(self.balance_module_rates(self.rates_graph())[0,1])

    def get_coarse_in_feasible(self,wr_factor=1):
        return self.get_factors(1)

    def get_coarse_out_feasible(self,wr_factor=1):
        return  self.get_factors(1)
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
        # sliding window
        self.modules['comm'].rows     = self.rows_in()
        self.modules['comm'].cols     = self.cols_in()
        self.modules['comm'].channels = int(self.channels/self.coarse_in)

    ### RATES ### 
    def rates_graph(self):
        rates_graph = np.zeros( shape=(1,2) , dtype=float )
    
        rates_graph[0,0] = self.modules['comm'].rate_in()
        rates_graph[0,1] = self.modules['comm'].rate_out()

        return rates_graph

    def resource(self):

        comm_rsc      = self.modules['comm'].rsc()

        # weight usage
        # Total
        return {
            "LUT"  :  comm_rsc['LUT']*self.coarse_in ,
            "FF"   :  comm_rsc['FF']*self.coarse_in, 
            "BRAM" :  comm_rsc['BRAM']*self.coarse_in,
            "DSP"  :  comm_rsc['DSP']*self.coarse_in}

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"comm",str(i)]), label="comm" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"comm",str(i)]) for i in range(self.coarse_in) ]
        nodes_out = [ "_".join([name,"comm",str(i)]) for i in range(self.coarse_out) ]

        return cluster, nodes_in, nodes_out