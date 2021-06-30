import torch
import math
import numpy as np
import pydot
from typing import Union, List

from fpgaconvnet_optimiser.models.modules import SlidingWindow
from fpgaconvnet_optimiser.models.modules import Pool
from fpgaconvnet_optimiser.models.layers import Layer

class PoolingLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse: int = 1,
            pool_type   ='max',
            k_size: Union[List[int], int] = 2,
            stride: Union[List[int], int] = 2,
            pad: Union[List[int], int] = 0,
            fine: int = 1,
            data_width: int = 16
        ):

        # initialise parent class
        super().__init__([rows], [cols], [channels], [coarse], [coarse],
                data_width=data_width)

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
                    pad - (self.rows_in() - k_size[0] + 2*pad) % stride[0],
                    pad,
                    pad,
                    pad - (self.cols_in() - k_size[1] + 2*pad) % stride[1],
                ]
        elif isinstance(pad, list):
            assert len(pad) == 4, "Must specify four pad dimensions"
        else:
            raise TypeError

        # update flags
        self.flags['transformable'] = True

        # update parameters
        self.k_size     = k_size
        self.stride     = stride
        self.pad_top    = pad[0]
        self.pad_right  = pad[3]
        self.pad_bottom = pad[2]
        self.pad_left   = pad[1]
        self.fine       = fine
        self.pool_type  = pool_type
        self.coarse = coarse

        #
        if pool_type == 'max':
            self.fine = self.k_size[0] * self.k_size[1]

        # init modules
        self.modules["sliding_window"] = SlidingWindow(rows, cols, channels, self.k_size, self.stride,
                    self.pad_top, self.pad_right, self.pad_bottom, self.pad_left, self.data_width)
        self.modules["pool"] = Pool(rows, cols, channels, k_size)

        # update layer
        self.update()

    def rows_out(self, port_index=0):
        assert port_index == 0, "ERROR: Pooling layers can only have 1 port"
        return int(math.ceil((self.rows_in()-self.k_size[0]+self.pad_top+self.pad_bottom)/self.stride[0])+1)

    def cols_out(self, port_index=0):
        assert port_index == 0, "ERROR: Pooling layers can only have 1 port"
        return int(math.ceil((self.cols_in()-self.k_size[1]+self.pad_left+self.pad_right)/self.stride[1])+1)

    def streams_in(self, port_index=0):
        assert(port_index < self.ports_in)
        return self.coarse

    def streams_out(self, port_index=0):
        assert(port_index < self.ports_out)
        return self.coarse

    def update_coarse_in(self, coarse_in, port_index=0):
        assert(port_index < self.ports_in)
        self.coarse = coarse_in

    def update_coarse_out(self, coarse_out, port_index=0):
        assert(port_index < self.ports_out)
        self.coarse = coarse_out

    ## LAYER INFO ##
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
        parameters.coarse       = self.coarse
        parameters.coarse_in    = self.coarse
        parameters.coarse_out   = self.coarse
        parameters.kernel_size_x = self.k_size[0]
        parameters.kernel_size_y = self.k_size[1]
        parameters.stride_x = self.stride[0]
        parameters.stride_y = self.stride[1]
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left

    ## UPDATE MODULES ##
    def update(self):
        # sliding window
        self.modules['sliding_window'].rows     = self.rows_in()
        self.modules['sliding_window'].cols     = self.cols_in()
        self.modules['sliding_window'].channels = int(self.channels_in()/self.coarse)
        # pool
        self.modules['pool'].rows     = self.rows_out()
        self.modules['pool'].cols     = self.cols_out()
        self.modules['pool'].channels = int(self.channels_in()/self.coarse)

    def get_fine_feasible(self):
        return [1]

    def resource(self):

        sw_rsc      = self.modules['sliding_window'].rsc()
        pool_rsc    = self.modules['pool'].rsc()

        # Total
        return {
            "LUT"  :  sw_rsc['LUT']*self.coarse +
                      pool_rsc['LUT']*self.coarse,
            "FF"   :  sw_rsc['FF']*self.coarse +
                      pool_rsc['FF']*self.coarse,
            "BRAM" :  sw_rsc['BRAM']*self.coarse +
                      pool_rsc['BRAM']*self.coarse,
            "DSP" :   sw_rsc['DSP']*self.coarse +
                      pool_rsc['DSP']*self.coarse
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse):
            cluster.add_node(pydot.Node( "_".join([name,"sw",str(i)]), label="sw" ))

        for i in range(self.coarse):
            cluster.add_node(pydot.Node( "_".join([name,"pool",str(i)]), label="pool" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"sw",str(i)]) , "_".join([name,"pool",str(i)]) ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"sw",str(i)]) for i in range(self.coarse_in[0]) ]
        nodes_out = [ "_".join([name,"pool",str(i)]) for i in range(self.coarse_out[0]) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self,data,batch_size=1):

        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        # instantiate pooling layer
        pooling_layer = torch.nn.MaxPool2d(self.k_size, stride=self.stride, padding=self.pad)

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return pooling_layer(torch.from_numpy(data)).detach().numpy()
