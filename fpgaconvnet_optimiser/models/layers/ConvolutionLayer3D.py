import numpy as np
import math
import pydot
import torch
from typing import Union, List

from fpgaconvnet_optimiser.models.layers.utils import get_factors

from fpgaconvnet_optimiser.tools.resource_model import bram_memory_resource_model

from fpgaconvnet_optimiser.models.modules import SlidingWindow3D
from fpgaconvnet_optimiser.models.modules import Conv3D
from fpgaconvnet_optimiser.models.modules import Fork3D
from fpgaconvnet_optimiser.models.modules import Accum3D
from fpgaconvnet_optimiser.models.modules import Glue3D
from fpgaconvnet_optimiser.models.layers import Layer3D

class ConvolutionLayer3D(Layer3D):

    def format_kernel_size(self, kernel_size):
        if isinstance(kernel_size, int):
            return [kernel_size, kernel_size, kernel_size]
        elif isinstance(kernel_size, list):
            assert len(kernel_size) == 3, "Must specify three kernel dimensions"
            return kernel_size
        else:
            raise TypeError

    def format_stride(self, stride):
        if isinstance(stride, int):
            return [stride, stride, stride]
        elif isinstance(stride, list):
            assert len(stride) == 3, "Must specify three stride dimensions"
            return stride
        else:
            raise TypeError

    def format_pad(self, pad):
        if isinstance(pad, int):
            return [
                    pad - (self.depth_in() - self.kernel_size[0] + 2*pad) % self.stride[0],
                    pad,
                    pad - (self.rows_in() - self.kernel_size[1] + 2*pad) % self.stride[1],
                    pad,
                    pad - (self.cols_in() - self.kernel_size[2] + 2*pad) % self.stride[2],
                    pad,
                ]
        elif isinstance(pad, list):
            assert len(pad) == 6, "Must specify six pad dimensions"
            return pad
        else:
            raise TypeError

    def __init__(
            self,
            filters: int,
            depth: int,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int = 1,
            coarse_out: int = 1,
            coarse_group: int = 1,
            kernel_size: Union[List[int], int] = 3,
            stride: Union[List[int], int] = 1,
            groups: int = 1,
            pad: Union[List[int], int] = 0,
            fine: int  = 1,
            input_width: int = 16,
            output_width: int = 16,
            weight_width: int = 16,
            acc_width: int = 16
        ):

        # initialise parent class
        super().__init__(depth, rows, cols, channels, coarse_in,
                coarse_out, data_width=input_width)

        # save the widths
        self.input_width = input_width
        self.output_width = output_width
        self.weight_width = weight_width
        self.acc_width = acc_width

        # init variables
        self._kernel_size = self.format_kernel_size(kernel_size)
        self._stride = self.format_stride(stride)
        self._pad = self.format_pad(pad)
        self._groups = groups
        self._coarse_group = coarse_group
        self._fine = fine
        self._filters = filters

        self._pad_front = self._pad[0]
        self._pad_back = self._pad[3]
        self._pad_top = self._pad[1]
        self._pad_bottom = self._pad[4]
        self._pad_left = self._pad[2]
        self._pad_right = self._pad[5]

        # init modules
        self.modules["sliding_window"] = SlidingWindow3D(self.depth_in(), self.rows_in(), self.cols_in(), int(self.channels_in()/self.coarse_in),
                self.kernel_size, self.stride, self.pad_front, self.pad_back, self.pad_top, self.pad_bottom, self.pad_left, self.pad_right)
        self.modules["fork"] = Fork3D(self.depth_out(), self.rows_out(), self.cols_out(), int(self.channels_in()/self.coarse_in),
                self.kernel_size, self.coarse_out)
        self.modules["conv"] = Conv3D(self.depth_out(), self.rows_out(), self.cols_out(), int(self.channels_in()/self.coarse_in),
                int(self.filters/self.coarse_out), self.fine, self.kernel_size, self.groups)
        self.modules["accum"] = Accum3D(self.depth_out(), self.rows_out(), self.cols_out(), int(self.channels_in()/self.coarse_in),
                int(self.filters/self.coarse_out), self.groups)
        #TODO: Why are channels here equal to 1?
        self.modules["glue"] = Glue3D(self.depth_out(), self.rows_out(), self.cols_out(), 1, int(self.filters/self.coarse_out),
                self.coarse_in, self.coarse_out)

        self.update()

    @property
    def kernel_size(self) -> List[int]:
        return self._kernel_size

    @property
    def stride(self) -> List[int]:
        return self._stride

    @property
    def pad(self) -> List[int]:
        return self._pad

    @property
    def pad_front(self) -> int:
        return self._pad[0]

    @property
    def pad_back(self) -> int:
        return self._pad[3]

    @property
    def pad_top(self) -> int:
        return self._pad[1]

    @property
    def pad_bottom(self) -> int:
        return self._pad[4]

    @property
    def pad_left(self) -> int:
        return self._pad[2]

    @property
    def pad_right(self) -> int:
        return self._pad[5]

    @property
    def groups(self) -> int:
        return self._groups

    @property
    def coarse_group(self) -> int:
        return self._coarse_group

    @property
    def fine(self) -> int:
        return self._fine

    @property
    def filters(self) -> int:
        return self._filters

    @kernel_size.setter
    def kernel_size(self, val: Union[List[int],int]) -> None:
        self._kernel_size = self.format_kernel_size(val)
        self.update()

    @stride.setter
    def stride(self, val: Union[List[int],int]) -> None:
        self._stride = self.format_stride(val)
        self.update()

    @pad.setter
    def pad(self, val: Union[List[int],int]) -> None:
        self._pad = self.format_pad(val)
        self.pad_front = self._pad[0]
        self.pad_back = self._pad[3]
        self.pad_top = self._pad[1]
        self.pad_bottom = self._pad[4]
        self.pad_left = self._pad[2]
        self.pad_right = self._pad[5]
        self.update()

    @groups.setter
    def groups(self, val: int) -> None:
        self._groups = val
        self.update()

    @fine.setter
    def fine(self, val: int) -> None:
        self._fine = val
        self.update()

    @filters.setter
    def filters(self, val: int) -> None:
        self._filters = val
        self.update()

    @coarse_group.setter
    def coarse_group(self, val: int) -> None:
        assert(val in self.get_coarse_group_feasible())
        self._coarse_group = val
        self.update()

    def depth_out(self) -> int:
        return self.modules["sliding_window"].depth_out()

    def rows_out(self) -> int:
        return self.modules["sliding_window"].rows_out()

    def cols_out(self) -> int:
        return self.modules["sliding_window"].cols_out()

    def channels_out(self) -> int:
        return self.filters

    def layer_info(self,parameters,batch_size=1):
        Layer3D.layer_info(self, parameters, batch_size)
        parameters.filters      = self.filters
        parameters.groups       = self.groups
        parameters.coarse_group = self.coarse_group
        parameters.fine         = self.fine
        parameters.kernel_size.extend([self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]])
        parameters.stride.extend([self.stride[0], self.stride[1], self.stride[2]])
        parameters.pad_front    = self.pad_front
        parameters.pad_back     = self.pad_back
        parameters.pad_top      = self.pad_top
        parameters.pad_right    = self.pad_right
        parameters.pad_bottom   = self.pad_bottom
        parameters.pad_left     = self.pad_left
        parameters.input_width  = self.input_width
        parameters.output_width = self.output_width
        parameters.weight_width = self.weight_width
        parameters.acc_width    = self.acc_width

    def update(self):
        # sliding window
        self.modules['sliding_window'].depth    = self.depth
        self.modules['sliding_window'].rows     = self.rows
        self.modules['sliding_window'].cols     = self.cols
        self.modules['sliding_window'].channels = self.channels//(self.coarse_in*self.coarse_group)
        self.modules['sliding_window'].data_width   = self.input_width
        # fork
        self.modules['fork'].depth    = self.depth_out()
        self.modules['fork'].rows     = self.rows_out()
        self.modules['fork'].cols     = self.cols_out()
        self.modules['fork'].channels = self.channels_in()//(self.coarse_in*self.coarse_group)
        self.modules['fork'].coarse   = self.coarse_out
        self.modules['fork'].data_width     = self.input_width
        # conv
        self.modules['conv'].depth    = self.depth_out()
        self.modules['conv'].rows     = self.rows_out()
        self.modules['conv'].cols     = self.cols_out()
        self.modules['conv'].channels = self.channels//(self.coarse_in*self.coarse_group)
        self.modules['conv'].filters  = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['conv'].fine     = self.fine
        self.modules['conv'].groups   = self.groups//self.coarse_group
        self.modules['conv'].data_width     = self.input_width
        self.modules['conv'].weight_width   = self.weight_width
        self.modules['conv'].acc_width      = self.acc_width
        # accum
        self.modules['accum'].depth    = self.depth_out()
        self.modules['accum'].rows     = self.rows_out()
        self.modules['accum'].cols     = self.cols_out()
        self.modules['accum'].channels = self.channels//(self.coarse_in*self.coarse_group)
        self.modules['accum'].filters  = self.filters//(self.coarse_out*self.coarse_group)
        self.modules['accum'].groups   = self.groups//self.coarse_group
        self.modules['accum'].data_width    = self.acc_width
        # glue
        self.modules['glue'].depth      = self.depth_out()
        self.modules['glue'].rows       = self.rows_out()
        self.modules['glue'].cols       = self.cols_out()
        self.modules['glue'].filters    = self.filters//self.coarse_group
        self.modules['glue'].coarse_in  = self.coarse_in
        self.modules['glue'].coarse_out = self.coarse_out
        self.modules['glue'].data_width = self.output_width
        self.modules['glue'].acc_width  = self.acc_width

    def get_coarse_group_feasible(self):
        return get_factors(self.groups)

    def get_fine_feasible(self):
        if self.kernel_size[0] != self.kernel_size[1] and self.kernel_size[1] == self.kernel_size[2]:
            if self.kernel_size[0] == 1:
                return [1, self.kernel_size[1], self.kernel_size[1]*self.kernel_size[2]]
            elif self.kernel_size[1] == 1:
                return [1, self.kernel_size[0]]
            else:
                return [1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[0]*self.kernel_size[1], self.kernel_size[1]*self.kernel_size[2], self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]]
        elif self.kernel_size[0] == self.kernel_size[1] and self.kernel_size[1] == self.kernel_size[2]:
            if self.kernel_size[0] == 1:
                return [1]
            else:
                return [1, self.kernel_size[0], self.kernel_size[0]*self.kernel_size[1], self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]]
        else:
            return [ 1, self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], self.kernel_size[0]*self.kernel_size[1], self.kernel_size[0]*self.kernel_size[2], self.kernel_size[1]*self.kernel_size[2], self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2] ]

    def get_weights_reloading_feasible(self):
        return get_factors(self.filters//(self.groups*self.coarse_out))

    def get_parameters_size(self):
        weights_size = self.channels_in() * ( self.filters // self.groups ) * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        bias_size = 0
        return {
            "weights"   : weights_size,
            "bias"      : bias_size
        }

    def get_operations(self):
        return self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]*self.channels_in()*self.filters*self.depth_out()*self.rows_out()*self.cols_out()

    def resource(self):

        sw_rsc      = self.modules['sliding_window'].rsc()
        fork_rsc    = self.modules['fork'].rsc()
        conv_rsc    = self.modules['conv'].rsc()
        accum_rsc   = self.modules['accum'].rsc()
        glue_rsc    = self.modules['glue'].rsc()

        if self.kernel_size[0] == 1 and self.kernel_size[1] == 1:
            sw_rsc      = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.coarse_out == 1:
            fork_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if int(self.channels_in()/(self.coarse_in*self.coarse_group)) == 1:
            accum_rsc   = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}
        if self.coarse_in == 1:
            glue_rsc    = {"LUT" : 0,"BRAM" : 0,"DSP" : 0,"FF" : 0}

        # weight usage
        weight_memory_depth = float((self.filters/self.groups)*self.channels_in()*self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]) / \
            float(self.fine*self.coarse_in*self.coarse_out*self.coarse_group)
        weights_bram_usage = bram_memory_resource_model(int(weight_memory_depth),self.weight_width)*self.coarse_in*self.coarse_out*self.coarse_group*self.fine

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

        assert data.shape[0] == self.depth_in()()   , "ERROR (data): invalid depth dimension"
        assert data.shape[1] == self.rows_in()      , "ERROR (data): invalid row dimension"
        assert data.shape[2] == self.cols_in()      , "ERROR (data): invalid column dimension"
        assert data.shape[3] == self.channels_in()  , "ERROR (data): invalid channel dimension"

        assert weights.shape[0] == self.filters                 , "ERROR (weights): invalid filter dimension"
        assert weights.shape[1] == self.channels//self.groups   , "ERROR (weights): invalid channel dimension"
        assert weights.shape[2] == self.kernel_size[0]          , "ERROR (weights): invalid kernel dimension"
        assert weights.shape[3] == self.kernel_size[1]          , "ERROR (weights): invalid kernel dimension"
        assert weights.shape[4] == self.kernel_size[2]          , "ERROR (weights): invalid kernel dimension"

        assert bias.shape[0] == self.filters  , "ERROR (bias): invalid filter dimension"

        # instantiate convolution layer
        convolution_layer = torch.nn.Conv3d(self.channels_in(), self.filters, self.kernel_size,
                stride=self.stride, padding=self.pad[0], groups=self.groups)

        # update weights
        convolution_layer.weight = torch.nn.Parameter(torch.from_numpy(weights))

        # update bias
        convolution_layer.bias = torch.nn.Parameter(torch.from_numpy(bias))

        # return output featuremap
        data = np.moveaxis(data, -1, 0)
        data = np.repeat(data[np.newaxis,...], batch_size, axis=0)
        return convolution_layer(torch.from_numpy(data)).detach().numpy()

