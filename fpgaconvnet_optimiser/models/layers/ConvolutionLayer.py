import numpy as np
import math
import pydot
import torch
from typing import Union, List

from fpgaconvnet_optimiser.models.modules import SlidingWindow
from fpgaconvnet_optimiser.models.modules import Conv
from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.modules import Accum
from fpgaconvnet_optimiser.models.modules import Glue
from fpgaconvnet_optimiser.models.layers import Layer

class ConvolutionLayer(Layer):
    def __init__(
            self,
            filters: int,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int = 1,
            coarse_out: int = 1,
            coarse_group: int = 1,
            k_size: Union[List[int], int] = 3,
            stride: Union[List[int], int] = 1,
            groups: int = 1,
            pad: Union[List[int], int] = 0,
            fine: int  = 1,
            data_width: int = 16,
            weight_width: int = 8,
            acc_width: int = 30
        ):

        # initialise parent class
        super().__init__([rows],[cols],[channels],[coarse_in],[coarse_out],data_width=data_width)

        # save the widths
        self.weight_width = weight_width
        self.acc_width = acc_width

        # update flags
        self.flags["channel_dependant"] = True
        self.flags["transformable"] = True
        self.flags["has_groups"] = True

        # change coarse to be scalar
        self.coarse_in = coarse_in
        self.coarse_out = coarse_out

        # handle kernel size
        if isinstance(k_size, int):
            k_size = [k_size, k_size]
        elif isinstance(k_size, list):
            assert len(k_size) == 2, "Must specify two kernel dimensions"
        else:
            raise TypeError

        # handle stride
        if isinstance(stride, int):
            stride = [stride, stride]
        elif isinstance(stride, list):
            assert len(stride) == 2, "Must specify two stride dimensions"
        else:
            raise TypeError

        # handle pad
        if isinstance(pad, int):
            pad = [
                    pad - (self.rows[0] - k_size[0] + 2*pad) % stride[0],
                    pad,
                    pad,
                    pad - (self.cols[0] - k_size[1] + 2*pad) % stride[1],
                ]
        elif isinstance(pad, list):
            assert len(pad) == 4, "Must specify four pad dimensions"
        else:
            raise TypeError

        # weight width
        # self.weight_width = weight_width

        # init variables
        self.k_size = k_size
        self.stride = stride
        self.groups = groups
        self.pad = pad
        self.pad_top = pad[0]
        self.pad_right = pad[3]
        self.pad_bottom = pad[2]
        self.pad_left = pad[1]
        self.coarse_group = coarse_group
        self.fine = fine
        self.filters = filters

        # init modules
        self.modules["sliding_window"] = SlidingWindow(self.rows[0],
                                                       self.cols[0],
                                                       self.channels[0],
                                                       k_size,
                                                       stride,
                                                       self.pad_top,
                                                       self.pad_right,
                                                       self.pad_bottom,
                                                       self.pad_left,
                                                       self.data_width)
        self.modules["fork"] = Fork(self.rows_out(), self.cols_out(), self.filters, k_size, self.coarse_out)
        self.modules["conv"] = Conv(self.rows_out(), self.cols_out(), self.filters, filters, fine, k_size, groups)
        self.modules["accum"] = Accum(self.rows_out(), self.cols_out(), self.filters, filters, groups)
        self.modules["glue"] = Glue(self.rows_out(), self.cols_out(), self.filters, filters, self.coarse_in, self.coarse_out)

        self.update()
        #self.load_coef()

    def streams_in(self, port_index=0):
        assert port_index == 0, "convolution layers are only allowed a single port"
        return self.coarse_in

    def streams_out(self, port_index=0):
        assert port_index == 0, "convolution layers are only allowed a single port"
        return self.coarse_out

    def update_coarse_in(self, coarse_in, port_index=0):
        assert port_index == 0, "convolution layers are only allowed a single port"
        self.coarse_in  = coarse_in

    def update_coarse_out(self, coarse_out, port_index=0):
        assert port_index == 0, "convolution layers are only allowed a single port"
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
        parameters.coarse_group = self.coarse_group
        parameters.fine         = self.fine
        parameters.filters      = self.filters
        parameters.kernel_size_x = self.k_size[0]
        parameters.kernel_size_y = self.k_size[1]
        parameters.stride_x = self.stride[0]
        parameters.stride_y = self.stride[1]
        parameters.groups       = self.groups
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left

    ## UPDATE MODULES ##
    def update(self):
        # sliding window
        self.modules['sliding_window'].rows     = self.rows_in()
        self.modules['sliding_window'].cols     = self.cols_in()
        self.modules['sliding_window'].channels = int(self.channels_in()/self.coarse_in*self.coarse_group)
        self.modules['sliding_window'].data_width = self.data_width
        # fork
        self.modules['fork'].rows     = self.rows_out()
        self.modules['fork'].cols     = self.cols_out()
        self.modules['fork'].channels = int(self.channels_in()/self.coarse_in*self.coarse_group)
        self.modules['fork'].coarse   = self.coarse_out
        self.modules['fork'].data_width = self.data_width
        # conv
        self.modules['conv'].rows     = self.rows_out()
        self.modules['conv'].cols     = self.cols_out()
        self.modules['conv'].channels = int(self.channels_in()/self.coarse_in*self.coarse_group)
        self.modules['conv'].filters  = int(self.filters/(self.coarse_out*self.coarse_group))
        self.modules['conv'].fine     = self.fine
        self.modules['conv'].groups   = int(self.groups/self.coarse_group)
        self.modules['conv'].data_width = self.data_width
        self.modules['conv'].weight_width = self.weight_width
        # accum
        self.modules['accum'].rows     = self.rows_out()
        self.modules['accum'].cols     = self.cols_out()
        self.modules['accum'].channels = int(self.channels_in()/(self.coarse_in*self.coarse_group))
        self.modules['accum'].filters  = int(self.filters/(self.coarse_out*self.coarse_group))
        self.modules['accum'].groups   = int(self.groups/self.coarse_group)
        self.modules['accum'].data_width = self.acc_width
        # glue
        self.modules['glue'].rows       = self.rows_out()
        self.modules['glue'].cols       = self.cols_out()
        self.modules['glue'].filters    = int(self.filters/self.coarse_group)
        self.modules['glue'].coarse_in  = self.coarse_in
        self.modules['glue'].coarse_out = self.coarse_out
        self.modules['glue'].data_width = self.acc_width

    ### RATES ###
    def rates_graph(self):
        rates_graph = np.zeros( shape=(5,6) , dtype=float )
        # sliding_window
        if self.k_size == 1:
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

    def get_coarse_in_feasible(self,port_index=0,wr_factor=1):
        assert port_index == 0, "convolution layers are only allowed a single port"
        return self.get_factors(int(self.channels_in()/(self.groups*wr_factor)))

    def get_coarse_out_feasible(self,port_index=0,wr_factor=1):
        assert port_index == 0, "convolution layers are only allowed a single port"
        return self.get_factors(int(self.channels_out()/(self.groups*wr_factor)))

    def get_coarse_group_feasible(self):
        return self.get_factors(int(self.groups))

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
        weights_size = self.channels_in() * int( self.filters / self.groups ) * self.k_size[0] * self.k_size[1]
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def get_operations(self):
        return self.k_size[0]*self.k_size[1]*self.channels_in()*self.filters*self.rows_out()*self.cols_out()

    def resource(self):

        sw_rsc      = self.modules['sliding_window'].rsc()
        fork_rsc    = self.modules['fork'].rsc()
        conv_rsc    = self.modules['conv'].rsc()
        accum_rsc   = self.modules['accum'].rsc()
        glue_rsc    = self.modules['glue'].rsc()

        if self.k_size[0] == 1 and self.k_size[1] == 1:
            sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.coarse_out == 1:
            fork_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if int(self.channels_in()/(self.coarse_in*self.coarse_group)) == 1:
            accum_rsc   = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.coarse_in == 1:
            glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        # weight usage
        n_filters = float(self.filters/self.groups*self.channels_in()*self.k_size[0]*self.k_size[1]) / \
            float(self.fine*self.coarse_in*self.coarse_out*self.coarse_group)
        weights_bram_usage = int(math.ceil((self.weight_width*n_filters)/18000))*self.coarse_in*self.coarse_out*self.coarse_group*self.fine

        # Total
        return {
            "LUT"  :  sw_rsc['LUT']*self.coarse_in*self.coarse_group +
                      fork_rsc['LUT']*self.coarse_in*self.coarse_group +
                      conv_rsc['LUT']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['LUT']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['LUT']*self.coarse_group,
            "FF"   :  sw_rsc['FF']*self.coarse_in*self.coarse_group +
                      fork_rsc['FF']*self.coarse_in*self.coarse_group +
                      conv_rsc['FF']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['FF']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['FF']*self.coarse_group,
            "BRAM" :  sw_rsc['BRAM']*self.coarse_in*self.coarse_group +
                      fork_rsc['BRAM']*self.coarse_in*self.coarse_group +
                      conv_rsc['BRAM']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['BRAM']*self.coarse_out*self.coarse_group +
                      glue_rsc['BRAM']*self.coarse_group +
                      weights_bram_usage,
            "DSP" :   sw_rsc['DSP']*self.coarse_in*self.coarse_group +
                      fork_rsc['DSP']*self.coarse_in*self.coarse_group +
                      conv_rsc['DSP']*self.coarse_in*self.coarse_out*self.coarse_group +
                      accum_rsc['DSP']*self.coarse_in*self.coarse_out*self.coarse_group +
                      glue_rsc['DSP']*self.coarse_group
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

        assert data.shape[0] == self.rows_in(0)    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in(0)    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(0), "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters , "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == int(self.channels_in()/self.groups), "ERROR (weights): invalid channel dimension"
        assert weights.shape[2] == self.k_size[0]  , "ERROR (weights): invalid kernel dimension"
        assert weights.shape[3] == self.k_size[1]  , "ERROR (weights): invalid kernel dimension"

        assert bias.shape[0] == self.filters  , "ERROR (bias): invalid filter dimension"

        # instantiate convolution layer
        convolution_layer = torch.nn.Conv2d(self.channels_in(), self.filters, self.k_size,
                stride=self.stride, padding=self.pad, groups=self.groups)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        convolution_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return convolution_layer(torch.from_numpy(data)).detach().numpy()

