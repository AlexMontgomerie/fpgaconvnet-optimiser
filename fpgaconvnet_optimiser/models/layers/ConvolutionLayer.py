import numpy as np
import math
import pydot
import torch

from fpgaconvnet_optimiser.models.modules import SlidingWindow
from fpgaconvnet_optimiser.models.modules import Conv
from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.modules import Accum
from fpgaconvnet_optimiser.models.modules import Glue
from fpgaconvnet_optimiser.models.modules import FIFO
from fpgaconvnet_optimiser.models.layers import Layer

from fpgaconvnet_optimiser.tools.onnx_helper import _pair, _quadruple

class ConvolutionLayer(Layer):
    def __init__(
            self,
            dim,
            filters,
            k_size      =[3,3],
            stride      =[1,1],
            groups      =1,
            pad         =[0,0,0,0],
            coarse_in   =1,
            coarse_out  =1,
            coarse_group = 1,
            fine        =1,
            data_width  =16,
            weight_width =8,
            acc_width   =30,
            sa          =0.5,
            sa_out      =0.5
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width,coarse_group)

        k_size = _pair(k_size)
        stride = _pair(stride)
        pad    = _quadruple(pad)

        # update flags
        self.flags['channel_dependant'] = True
        self.flags['transformable']     = True

        self.weight_width = weight_width
        self.acc_width = acc_width

        # init variables
        self.k_size     = k_size
        self.stride     = stride
        self.groups     = groups
        self.pad_top    = pad[0]
        self.pad_right  = pad[3]
        self.pad_bottom = pad[2]
        self.pad_left   = pad[1]
        assert(self.pad_top == self.pad_bottom and self.pad_left == self.pad_right)
        self.pad        = [self.pad_top, self.pad_left] 
        self.fine       = fine
        self.filters    = filters

        dim_out = [
            self.filters,
            self.rows_out(),
            self.cols_out()
        ]
        # init modules
        self.modules = {
            "sliding_window" : SlidingWindow(dim, k_size, stride, self.pad_top, self.pad_right, self.pad_bottom, self.pad_left, data_width),
            "fork"           : Fork(dim_out,k_size,coarse_out,data_width),
            "conv"           : Conv(dim_out,filters,fine,k_size,groups,data_width,weight_width,acc_width),
            "accum"          : Accum(dim_out,filters,groups,acc_width),
            "glue"           : Glue(dim_out,filters,coarse_in,coarse_out,coarse_group,data_width)
        }
        self.update()
        #self.load_coef()

        # switching activity
        self.sa     = sa
        self.sa_out = sa_out

    def rows_out(self):
        return int(math.floor((self.rows_in()-self.k_size[0]+self.pad_top+self.pad_bottom)/self.stride[0])+1)

    def cols_out(self):
        return int(math.floor((self.cols_in()-self.k_size[1]+self.pad_left+self.pad_right)/self.stride[1])+1)

    def channels_out(self):
        return self.filters

    def rate_in(self,index):
        return abs(self.balance_module_rates(self.rates_graph())[0,0])

    def rate_out(self,index):
        return abs(self.balance_module_rates(self.rates_graph())[4,5])

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
        parameters.fine         = self.fine
        parameters.filters      = self.filters
        parameters.kernel_size.extend(self.k_size)
        parameters.stride.extend(self.stride)
        parameters.coarse_group = self.coarse_group
        parameters.groups       = self.groups
        parameters.pad.extend(self.pad)
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left

    ## UPDATE MODULES ##
    def update(self): 
        # sliding window
        self.modules['sliding_window'].rows     = self.rows_in()
        self.modules['sliding_window'].cols     = self.cols_in()
        self.modules['sliding_window'].channels = int(self.channels/(self.coarse_in*self.coarse_group))
        # fork
        self.modules['fork'].rows     = self.rows_out()
        self.modules['fork'].cols     = self.cols_out()
        self.modules['fork'].channels = int(self.channels/(self.coarse_in*self.coarse_group))
        self.modules['fork'].coarse   = self.coarse_out
        # conv
        self.modules['conv'].rows     = self.rows_out()
        self.modules['conv'].cols     = self.cols_out()
        self.modules['conv'].channels = int(self.channels/(self.coarse_in*self.coarse_group))
        self.modules['conv'].filters  = int(self.filters/(self.coarse_out*self.coarse_group))
        self.modules['conv'].fine     = self.fine
        self.modules['conv'].groups   = int(self.groups/self.coarse_group)
        # accum
        self.modules['accum'].rows     = self.rows_out()
        self.modules['accum'].cols     = self.cols_out()
        self.modules['accum'].channels = int(self.channels/(self.coarse_in*self.coarse_group))
        self.modules['accum'].filters  = int(self.filters/(self.coarse_out*self.coarse_group))
        self.modules['accum'].groups   = int(self.groups/self.coarse_group)
        # glue
        self.modules['glue'].rows       = self.rows_out()
        self.modules['glue'].cols       = self.cols_out()
        self.modules['glue'].filters    = int(self.filters/self.coarse_group) 
        self.modules['glue'].coarse_in  = self.coarse_in
        self.modules['glue'].coarse_out = self.coarse_out
        self.modules['glue'].coarse_group = self.coarse_group

    ### RATES ### 
    def rates_graph(self):
        rates_graph = np.zeros( shape=(5,6) , dtype=float )
        # sliding_window
        if self.k_size[0] == 1 and self.k_size[1] == 1:
            rates_graph[0,0] = 1
            rates_graph[0,1] = 1
        else:
            rates_graph[0,0] = self.modules['sliding_window'].rate_in()
            rates_graph[0,1] = self.modules['sliding_window'].rate_out()
        # fork
        rates_graph[1,1] = self.modules['fork'].rate_in()
        rates_graph[1,2] = self.modules['fork'].rate_out()
        # conv
        rates_graph[2,2] = self.modules['conv'].rate_in()
        rates_graph[2,3] = self.modules['conv'].rate_out()
        # accum
        rates_graph[3,3] = self.modules['accum'].rate_in()
        rates_graph[3,4] = self.modules['accum'].rate_out()
        # glue 
        rates_graph[4,4] = self.modules['glue'].rate_in()
        rates_graph[4,5] = self.modules['glue'].rate_out()

        return rates_graph

    def get_coarse_in_feasible(self,wr_factor=1):
        return self.get_factors(int(self.channels_in()/(self.groups*wr_factor)))

    def get_coarse_out_feasible(self,wr_factor=1):
        return self.get_factors(int(self.channels_out()/(self.groups*wr_factor)))

    def get_coarse_group_feasible(self):
        return self.get_factors(int(self.groups))
        
    def update_coarse_in(self, coarse_in):
        self.coarse_in  = coarse_in

    def update_coarse_out(self, coarse_out):
        self.coarse_out = coarse_out

    def get_fine_feasible(self):
        #return self.get_factors(int(self.k_size*self.k_size))
        if self.k_size[0] != self.k_size[1]:
            assert(self.k_size[0] == 1 or self.k_size[1] == 1)
            return [ 1, max(self.k_size[0],self.k_size[1])]
        else:
            return [ 1, self.k_size[0], self.k_size[0]*self.k_size[1] ]


    def get_weights_reloading_feasible(self):
        return self.get_factors(int(self.filters/(self.groups*self.coarse_out)))

    def get_parameters_size(self):
        weights_size = self.channels * int( self.filters / self.groups ) * self.k_size[0] * self.k_size[1]
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def resource(self):

        # instances
        sw_rsc      = self.modules['sliding_window'].rsc()
        fork_rsc    = self.modules['fork'].rsc()
        conv_rsc    = self.modules['conv'].rsc()
        accum_rsc   = self.modules['accum'].rsc()
        glue_rsc    = self.modules['glue'].rsc()

        # streams
        sw_out = FIFO([1,1,1], self.coarse_in*self.coarse_group*self.k_size[0]*self.k_size[1], self.buffer_depth, self.data_width)
        sw_out_rsc = sw_out.rsc()
        fork_out = FIFO([1,1,1], self.coarse_in*self.coarse_group*self.coarse_out*self.k_size[0]*self.k_size[1], self.buffer_depth, self.data_width)
        fork_out_rsc = fork_out.rsc()
        conv_out = FIFO([1,1,1], self.coarse_in*self.coarse_group*self.coarse_out, self.buffer_depth, self.acc_width)
        conv_out_rsc = conv_out.rsc()
        accum_out = FIFO([1,1,1], self.coarse_in*self.coarse_group*self.coarse_out, int(self.modules['accum'].filters / self.modules['accum'].groups + 1), self.acc_width)
        accum_out_rsc = accum_out.rsc()

        if self.k_size[0] == 1 and self.k_size[1] == 1:
            sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
            sw_out_rsc  = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        #if self.coarse_out == 1:
        #    fork_rsc     = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        #    fork_out_rsc = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        #if int(self.channels/(self.coarse_in*self.group)) == 1:
        #    accum_rsc     = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        #    accum_out_rsc = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        #if self.coarse_in == 1:
        #    glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        #print(sw_rsc['FF'],fork_rsc['FF'],conv_rsc['FF'],accum_rsc['FF'],glue_rsc['FF'])
        #print(sw_out_rsc['FF'],fork_out_rsc['FF'],conv_out_rsc['FF'],accum_out_rsc['FF'])
        # Total
        return {
            "LUT"  :  sw_rsc['LUT']*self.coarse_in*self.coarse_group +
                      fork_rsc['LUT']*self.coarse_in*self.coarse_group +
                      conv_rsc['LUT']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['LUT']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['LUT'] +
                      sw_out_rsc['LUT'] +
                      fork_out_rsc['LUT'] +
                      conv_out_rsc['LUT'] +
                      accum_out_rsc['LUT'],
            "FF"   :  sw_rsc['FF']*self.coarse_in*self.coarse_group +
                      fork_rsc['FF']*self.coarse_in*self.coarse_group +
                      conv_rsc['FF']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['FF']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['FF'] +
                      sw_out_rsc['FF'] +
                      fork_out_rsc['FF'] +
                      conv_out_rsc['FF'] +
                      accum_out_rsc['FF'],
            "BRAM" :  sw_rsc['BRAM']*self.coarse_in*self.coarse_group +
                      fork_rsc['BRAM']*self.coarse_in*self.coarse_group +
                      conv_rsc['BRAM']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['BRAM']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['BRAM'] +
                      sw_out_rsc['BRAM'] +
                      fork_out_rsc['BRAM'] +
                      conv_out_rsc['BRAM'] +
                      accum_out_rsc['BRAM'],
            "DSP" :   sw_rsc['DSP']*self.coarse_in*self.coarse_group +
                      fork_rsc['DSP']*self.coarse_in*self.coarse_group +
                      conv_rsc['DSP']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['DSP']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['DSP']
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)
        nodes_in = []
        nodes_out = []

        for g in range(self.coarse_group):
            for i in range(self.coarse_in):
                cluster.add_node(pydot.Node( "_".join([name,"sw",str(g*self.coarse_in+i)]), label="sw" ))

            for i in range(self.coarse_in):
                cluster.add_node(pydot.Node( "_".join([name,"fork",str(g*self.coarse_in+i)]), label="fork" ))
                cluster.add_edge(pydot.Edge( "_".join([name,"sw",str(g*self.coarse_in+i)]) , "_".join([name,"fork",str(i)]) ))

            for i in range(self.coarse_in):
                for j in range(self.coarse_out):
                    cluster.add_node(pydot.Node( "_".join([name,"conv",str(g*self.coarse_in+i),str(g*self.coarse_out+j)]), label="conv" ))
                    cluster.add_edge(pydot.Edge( "_".join([name,"fork",str(g*self.coarse_in+i)]) , "_".join([name,"conv",str(g*self.coarse_in+i),str(g*self.coarse_out+j)]) ))

            for i in range(self.coarse_in):
                for j in range(self.coarse_out):
                    cluster.add_node(pydot.Node( "_".join([name,"glue",str(g*self.coarse_out+j)]), label="+" ))
                    cluster.add_node(pydot.Node( "_".join([name,"accum",str(g*self.coarse_in+i),str(g*self.coarse_out+j)]), label="accum" ))
                    cluster.add_edge(pydot.Edge( "_".join([name,"conv" ,str(g*self.coarse_in+i),str(g*self.coarse_out+j)]), "_".join([name,"accum",str(g*self.coarse_in+i),str(g*self.coarse_out+j)]) ))
                    cluster.add_edge(pydot.Edge( "_".join([name,"accum",str(g*self.coarse_in+i),str(g*self.coarse_out+j)]), "_".join([name,"glue",str(g*self.coarse_out+j)]) ))

            # get nodes in and out
            for i in range(self.coarse_in):
                nodes_in.append("_".join([name,"sw",str(g*self.coarse_in+i)]))
            
            for j in range(self.coarse_out):
                nodes_out.append("_".join([name,"glue",str(g*self.coarse_out+j)]))

        return cluster, nodes_in, nodes_out

    def functional_model(self,data,weights,bias,batch_size=1):

        assert data.shape[0] == self.rows    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters , "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == int(self.channels/self.groups), "ERROR (weights): invalid channel dimension"
        assert weights.shape[2] == self.k_size[0]  , "ERROR (weights): invalid kernel dimension"
        assert weights.shape[3] == self.k_size[1]  , "ERROR (weights): invalid kernel dimension"

        assert bias.shape[0] == self.filters  , "ERROR (bias): invalid filter dimension"

        # instantiate convolution layer
        convolution_layer = torch.nn.Conv2d(self.channels, self.filters, self.k_size, 
                stride=self.stride, padding=self.pad, groups=self.groups)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))
        
        # update bias
        convolution_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))
        
        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0) 
        return convolution_layer(torch.from_numpy(data)).detach().numpy()

