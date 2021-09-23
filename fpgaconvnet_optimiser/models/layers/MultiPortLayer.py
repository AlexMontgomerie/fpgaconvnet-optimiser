"""

"""

import os
import math
import sys
from typing import List
from functools import reduce
import pydot
import collections
from google.protobuf.json_format import MessageToDict
import numpy as np
from dataclasses import dataclass, field

import fpgaconvnet_optimiser.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2

@dataclass
class Layer:
    """
    Base class for all layer models.

    Attributes
    ----------
    buffer_depth: int, default: 0
        depth of incoming fifo buffers for each stream in.
    rows: list int
        row dimension of input featuremap
    cols: list int
        column dimension of input featuremap
    channels: list int
        channel dimension of input featuremap
    ports_in: int
        number of ports into the layer
    ports_out: int
        number of ports out of the layer
    coarse_in: list int
        number of parallel streams per port into the layer.
    coarse_out: list int
        number of parallel streams per port out of the layer.
    data_width: int
        bitwidth of featuremap pixels
    modules: dict
        dictionary of `module` instances that make
        up the layer. These modules are used for the
        resource and performance models of the layer.
    """

    _rows: List[int]
    _cols: List[int]
    _channels: List[int]
    _coarse_in: List[int]
    _coarse_out: List[int]
    _ports_in: int = field(default=1, init=True)
    _ports_out: int = field(default=1, init=True)
    data_width: int = field(default=16, init=True)
    buffer_depth: int = field(default=0, init=False)
    modules: dict = field(default_factory=collections.OrderedDict, init=False)

    # define properties of the layer
    @property
    def rows(self) -> List[int]:
        return self._rows

    @property
    def cols(self) -> List[int]:
        return self._cols

    @property
    def channels(self) -> List[int]:
        return self._channels

    @property
    def coarse_in(self) -> List[int]:
        return self._coarse_in

    @property
    def coarse_out(self) -> List[int]:
        return self._coarse_out

    @property
    def ports_in(self) -> int:
        return self._ports_in

    @property
    def ports_out(self) -> int:
        return self._ports_out

    # create the setter methods for these properties
    @rows.setter
    def rows(self, val: List[int]) -> None:
        assert(len(val) == self.ports_in)
        self._rows = val
        self.update()

    @cols.setter
    def cols(self, val: List[int]) -> None:
        assert(len(val) == self.ports_in)
        self._cols = val
        self.update()

    @channels.setter
    def channels(self, val: List[int]) -> None:
        assert(len(val) == self.ports_in)
        self._channels = val
        self.update()

    @coarse_in.setter
    def coarse_in(self, val: List[int]) -> None:
        assert(len(val) == self.ports_in)
        for i in range(val):
            assert(val[i] in self.coarse_in_feasible(port_index=i))
        self._coarse_in = val
        self.coarse_out = val
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: List[int]) -> None:
        assert(len(val) == self.ports_out)
        for i in range(val):
            assert(val[i] in self.coarse_out_feasible(port_index=i))
        self._coarse_out = val
        self._coarse_in = val
        self.update()

    @ports_in.setter
    def ports_in(self, val: int) -> None:
        self._ports_in = val

    @ports_out.setter
    def ports_out(self, val: int) -> None:
        self._ports_out = val

    def rows_in(self, port_index=0):
        """
        Returns
        -------
        int
            row dimension of the input featuremap
        """
        assert(port_index < self.ports_in)
        return self.rows[port_index]

    def cols_in(self, port_index=0):
        """
        Returns
        -------
        int
            column dimension of the input featuremap
        """
        assert(port_index < self.ports_in)
        return self.cols[port_index]

    def channels_in(self, port_index=0):
        """
        Returns
        -------
        int
            channel dimension of the input featuremap
        """
        assert(port_index < self.ports_in)
        return self.channels[port_index]

    def rows_out(self, port_index=0):
        """
        Returns
        -------
        int
            row dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return list(self.modules.values())[-1].rows_out()

    def cols_out(self, port_index=0):
        """
        Returns
        -------
        int
            column dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return list(self.modules.values())[-1].cols_out()

    def channels_out(self, port_index=0):
        """
        Returns
        -------
        int
            channel dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return list(self.modules.values())[-1].channels_out()*self.coarse_out[port_index]

    def rates_graph(self):

        # create the rates graph
        rates_graph = np.zeros(shape=(len(self.modules.keys()),
                                      len(self.modules.keys())+1) , dtype=float )

        # iterate over modules
        for i, module in enumerate(self.modules.keys()):
            # update rates_graph
            rates_graph[i,i] = self.modules[module].rate_in()
            rates_graph[i,i+1] = self.modules[module].rate_out()

        # return rates_graph
        return rates_graph

    def rate_in(self, port_index=0):
        """
        Parameters
        ----------
        index: int
            index of port into layer

        Returns
        -------
        float
            rate of words into layer. As a fraction of a
            clock cycle.

            default is 1.0
        """
        assert(port_index < self.ports_in)
        return abs(self.balance_module_rates(self.rates_graph())[0,0])

    def rate_out(self, port_index=0):
        """
        Parameters
        ----------
        index: int
            index of port into layer

        Returns
        -------
        float
            rate of words out of the layer. As a fraction
            of a clock cycle.

            default is 1.0
        """
        assert(port_index < self.ports_out)
        return abs(self.balance_module_rates(
            self.rates_graph())[len(self.modules.keys())-1,len(self.modules.keys())])

    def streams_in(self, port_index=0):
        """
        Returns
        -------
        int
            number of parallel streams into the layer.
        """
        assert(port_index < self.ports_in)
        return self.coarse_in[port_index]

    def streams_out(self, port_index=0):
        """
        Returns
        -------
        int
            number of parallel streams out of the layer.
        """
        assert(port_index < self.ports_out)
        return self.coarse_out[port_index]

    def workload_in(self, port_index=0):
        """
        Parameters
        ----------
        index: int
            index of port into layer

        Returns
        -------
        int
            workload into layer from port `index` for a single
            featuremap. This is calculated by
            `rows_in()*cols_in()*channels_in()`.
        """
        assert(port_index < self.ports_in)
        return self.rows_in(port_index) * self.cols_in(port_index) * self.channels_in(port_index)

    def workload_out(self, port_index=0):
        """
        Parameters
        ----------
        index: int
            index of port out of layer

        Returns
        -------
        int
            workload out of layer from port `index` for a
            single featuremap. This is calculated by
            `rows_out()*cols_out()*channels_out()`.
        """
        assert(port_index < self.ports_out)
        return self.rows_out(port_index) * self.cols_out(port_index) * self.channels_out(port_index)

    def size_in(self, port_index=0):
        """
        Returns
        -------
        int
            workload in per stream.
        """
        assert(port_index < self.ports_in)
        return self.rows_in(port_index) * self.cols_in(port_index) * int( self.channels_in(port_index) / self.streams_in(port_index) )

    def size_out(self, port_index=0):
        """
        Returns
        -------
        int
            workload out per stream.
        """
        assert(port_index < self.ports_out)
        return self.rows_out(port_index) * self.cols_out(port_index) * int( self.channels_out(port_index) / self.streams_out(port_index) )

    def width_in(self):
        """
        Returns
        -------
        int
            data width in
        """
        return self.data_width

    def width_out(self):
        """
        Returns
        -------
        int
            data width out
        """
        return self.data_width

    def get_latency_in(self):
        return max([
            abs(self.workload_in(i)/(self.rate_in(i)*self.streams_in(i) )) for
            i in range(self.ports_in) ])

    def get_latency_out(self):
        return max([
            abs(self.workload_out(i)/(self.rate_out(i)*self.streams_out(i)))
            for i in range(self.ports_out) ])

    def get_latency(self):
        return max(self.get_latency_in(), self.get_latency_out())

    def pipeline_depth(self):
        return sum([ self.modules[module].pipeline_depth() for module in self.modules ])

    def wait_depth(self):
        return sum([ self.modules[module].wait_depth() for module in self.modules ])

    def resource(self):
        return {
            "LUT"   : 0,
            "FF"    : 0,
            "BRAM"  : math.ceil(self.buffer_depth*self.data_width/18000)*self.streams_in(),
            "DSP"   : 0
        }

    def get_coarse_in_feasible(self, port_index=0, wr_factor=1):
        assert(port_index < self.ports_in)
        return self.get_factors(int(self.channels_in(port_index)/wr_factor))

    def get_coarse_out_feasible(self, port_index=0, wr_factor=1):
        assert(port_index < self.ports_out)
        return self.get_factors(int(self.channels_out(port_index)/wr_factor))

    def update_coarse_in(self, coarse_in, port_index=0):
        assert(port_index < self.ports_in)
        self.coarse_in[port_index]  = coarse_in
        self.update_coarse_out(coarse_in, port_index=port_index)

    def update_coarse_out(self, coarse_out, port_index=0):
        assert(port_index < self.ports_out)
        self.coarse_out[port_index] = coarse_out
        self.update_coarse_in(coarse_in, port_index=port_index)

    def load_coef(self):
        pass
        # for module in self.modules:
        #     self.modules[module].load_coef(
        #         os.path.join(
        #             os.path.dirname(__file__),
        #             "../../coefficients/{}_rsc_coef.npy".format(module))
        #     )


    def update(self):
        pass

    def layer_info(self, parameters, batch_size=1):
        parameters.batch_size   = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        parameters.coarse_in    = self.streams_in()
        parameters.coarse_out   = self.streams_out()

    def get_operations(self):
        return 0

    def layer_info_dict(self):
        # get parameters
        parameter = fpgaconvnet_pb2.parameter()
        self.layer_info(parameter)
        # convert to dictionary
        return MessageToDict(parameter, preserving_proto_field_name=True)

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in[0]):
            cluster.add_node(pydot.Node( "_".join([name,"edge",str(i)]), label=self.__class__.__name__ ))

        return cluster, "_".join([name,"edge"]), "_".join([name,"edge"])

    def functional_model(self,data,batch_size=1):
        return

    def balance_module_rates(self,rate_graph):

        rate_ratio = [ abs(rate_graph[i,i+1]/rate_graph[i,i]) for i in range(rate_graph.shape[0]) ]

        for i in range(1,rate_graph.shape[0]):
            # start from end
            layer = rate_graph.shape[0]-i

            if abs(rate_graph[layer,layer]) > abs(rate_graph[layer-1,layer]):
                # propogate forward
                for j in range(layer,rate_graph.shape[0]):
                        if(abs(rate_graph[j,j]) <= abs(rate_graph[j-1,j])):
                            break
                        rate_graph[j,j]   = abs(rate_graph[j-1,j])
                        rate_graph[j,j+1] = -rate_graph[j,j]*rate_ratio[j]

            elif abs(rate_graph[layer,layer]) < abs(rate_graph[layer-1,layer]):
                # propogate backward
                for j in range(0,layer):
                        if(abs(rate_graph[layer-j,layer-j]) >= abs(rate_graph[layer-j-1,layer-j])):
                            break
                        rate_graph[layer-j-1,layer-j]   = -abs(rate_graph[layer-j,layer-j])
                        rate_graph[layer-j-1,layer-j-1] = -rate_graph[layer-j-1,layer-j]/rate_ratio[layer-j-1]
        return rate_graph

    def get_factors(self, n):
        return list(set(reduce(list.__add__,
                    ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))
