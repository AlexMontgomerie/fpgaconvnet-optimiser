"""

"""

import os
import math
from typing import List
from functools import reduce
import pydot
from google.protobuf.json_format import MessageToDict
import fpgaconvnet_optimiser.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2

class Layer:
    """
    Base class for all layer models.
    """
    def __init__(
            self,
            rows: List[int],
            cols: List[int],
            channels: List[int],
            coarse_in: List[int],
            coarse_out: List[int],
            ports_in=1,
            ports_out=1,
            data_width=16
        ):
        """
        Parameters
        ----------
        rows: list int
            row dimension of input featuremap
        cols: list int
            column dimension of input featuremap
        channels: list int
            channel dimension of input featuremap

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

        # flags
        self.flags = {
            "multi_input"       : False,
            "multi_output"      : False,
            "channel_dependant" : False,
            "transformable"     : False
        }

        # buffer depth
        self.buffer_depth = 0

        # parameters
        self.rows       = rows
        self.cols       = cols
        self.channels   = channels

        # ports
        self.ports_in   = ports_in
        self.ports_out  = ports_out

        # streams
        self.coarse_in  = coarse_in
        self.coarse_out = coarse_out

        # data width
        self.data_width = data_width

        # init modules
        self.modules = {}

        # power and resource model coefficients
        self.static_coef  = {}
        self.dynamic_coef = {}
        self.rsc_coef     = {}

    def rows_in(self, port_index):
        """
        Returns
        -------
        int
            row dimension of the input featuremap
        """
        assert(port_index < self.ports_in)
        return self.rows[port_index]

    def cols_in(self, port_index):
        """
        Returns
        -------
        int
            column dimension of the input featuremap
        """
        assert(port_index < self.ports_in)
        return self.cols[port_index]

    def channels_in(self, port_index):
        """
        Returns
        -------
        int
            channel dimension of the input featuremap
        """
        assert(port_index < self.ports_in)
        return self.channels[port_index]

    def rows_out(self, port_index):
        """
        Returns
        -------
        int
            row dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return self.rows[port_index]

    def cols_out(self, port_index):
        """
        Returns
        -------
        int
            column dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return self.cols[port_index]

    def channels_out(self, port_index):
        """
        Returns
        -------
        int
            channel dimension of the output featuremap
        """
        assert(port_index < self.ports_out)
        return self.channels[port_index]

    def rate_in(self, port_index):
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
        return 1.0

    def rate_out(self, port_index):
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
        return 1.0

    def streams_in(self, port_index):
        """
        Returns
        -------
        int
            number of parallel streams into the layer.
        """
        assert(port_index < self.ports_in)
        return self.coarse_in[port_index]

    def streams_out(self, port_index):
        """
        Returns
        -------
        int
            number of parallel streams out of the layer.
        """
        assert(port_index < self.ports_out)
        return self.coarse_out[port_index]

    def workload_in(self, port_index):
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

    def workload_out(self, port_index):
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

    def size_in(self, port_index):
        """
        Returns
        -------
        int
            workload in per stream.
        """
        assert(port_index < self.ports_in)
        return self.rows_in(port_index) * self.cols_in(port_index) * int( self.channels_in(port_index) / self.streams_in(port_index) )

    def size_out(self, port_index):
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

    def get_latency(self):
        latency_in  = max([ abs(self.workload_in(i)/(self.rate_in(i)*self.streams_in(i) )) for i in self.ports_in ])
        latency_out = max([ abs(self.workload_out(i)/(self.rate_out(i)*self.streams_out(i))) for i in self.ports_out ])
        return max(latency_in,latency_out)

    def pipeline_depth(self):
        return sum([ self.modules[module].pipeline_depth() for module in self.modules ])

    def wait_depth(self):
        return sum([ self.modules[module].wait_depth() for module in self.modules ])

    def resource(self):
        return {
            "LUT"   : 0,
            "FF"    : 0,
            "BRAM"  : math.ceil(self.buffer_depth/1125)*self.coarse_in[0],
            "DSP"   : 0
        }

    def static_power(self):
        return 0

    def dynamic_power(self, freq, rate):
        return 0

    def power(self,freq,rate):
        return self.static_power() + self.dynamic_power(freq,rate)

    def get_coarse_in_feasible(self, port_index, wr_factor=1):
        return self.get_factors(int(self.channels_in(port_index)/wr_factor))

    def get_coarse_out_feasible(self, port_index, wr_factor=1):
        return self.get_factors(int(self.channels_out(port_index)/wr_factor))

    def update_coarse_in(self, coarse_in):
        self.coarse_in[0]  = coarse_in
        self.coarse_out[0] = coarse_in

    def update_coarse_out(self, coarse_out):
        self.coarse_in[0]  = coarse_out
        self.coarse_out[0] = coarse_out

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

    def layer_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'buffer_depth'  : self.buffer_depth,
            'rows_in'       : self.rows_in(),
            'cols_in'       : self.cols_in(),
            'channels_in'   : self.channels_in(),
            'size_in'       : int(self.workload_in(0)),
            'size_out'      : int(self.workload_out(0)),
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def get_operations(self):
        return 0

    def layer_info_dict(self):
        # get parameters
        parameter = fpgaconvnet_pb2.parameter()
        self.layer_info(parameter)
        # convert to dictionary
        return MessageToDict(parameter, preserving_proto_field_name=True)

    def visualise(self,name): # TODO
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in[0]):
            cluster.add_node(pydot.Node( "_".join([name,"edge",str(i)]), label=self.__class__.__name__ ))

        return cluster, "_".join([name,"edge"]), "_".join([name,"edge"])

    def functional_model(self,data,batch_size=1): # TODO: just leave empty
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

