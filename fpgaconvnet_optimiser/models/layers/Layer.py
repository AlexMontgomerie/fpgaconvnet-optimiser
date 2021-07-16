"""

"""

import pydot
import numpy as np
import fpgaconvnet_optimiser.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2 
from google.protobuf.json_format import MessageToDict
from functools import reduce
import os
import math

class Layer:
    """
    Base class for all layer models.
    """
    def __init__(
            self,
            dim,
            coarse_in,
            coarse_out,
            data_width
        ):
        """
        Parameters
        ----------
        dim: list
            dimensions of the input featuremap. Should contain
            `channels`, `rows`, `cols` in that order.

        Attributes
        ----------
        buffer_depth: int, default: 0
            depth of incoming fifo buffers for each stream in.
        rows: int
            row dimension of input featuremap
        cols: int
            column dimension of input featuremap
        channels: int
            channel dimension of input featuremap
        coarse_in: int
            number of parallel streams into the layer.    
        coarse_out: int
            number of parallel streams out of the layer.
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
        self.rows       = dim[1]
        self.cols       = dim[2]
        self.channels   = dim[0]

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

    def rows_in(self):
        """
        Returns
        -------
        int
            row dimension of the input featuremap
        """
        return self.rows

    def cols_in(self):
        """
        Returns
        -------
        int
            column dimension of the input featuremap
        """
        return self.cols

    def channels_in(self):
        """
        Returns
        -------
        int
            channel dimension of the input featuremap
        """
        return self.channels

    def rows_out(self):
        """
        Returns
        -------
        int
            row dimension of the output featuremap
        """
        return self.rows_in()

    def cols_out(self):
        """
        Returns
        -------
        int
            column dimension of the output featuremap
        """
        return self.cols_in()

    def channels_out(self):
        """
        Returns
        -------
        int
            channel dimension of the output featuremap
        """
        return self.channels_in()

    def rate_in(self,index):
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
        return 1.0

    def rate_out(self,index):
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
        return 1.0

    def streams_in(self):
        """
        Returns
        -------
        int 
            number of parallel streams into the layer.
        """
        return self.coarse_in

    def streams_out(self):
        """
        Returns
        -------
        int 
            number of parallel streams out of the layer.
        """
        return self.coarse_out

    def workload_in(self, index):
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
        return self.rows_in()  * self.cols_in()  * self.channels_in()
        
    def workload_out(self, index):
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
        return self.rows_out() * self.cols_out() * self.channels_out()

    def size_in(self):
        """
        Returns
        -------
        int 
            workload in per stream.
        """
        return self.rows_in()  * self.cols_in()  * int( self.channels_in() / self.coarse_in )
        
    def size_out(self):
        """
        Returns
        -------
        int 
            workload out per stream.
        """
        return self.rows_out() * self.cols_out() * int( self.channels_out() / self.coarse_out )

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
        latency_in  = abs(self.workload_in(0) /(self.rate_in(0) *self.streams_in() ))
        latency_out = abs(self.workload_out(0)/(self.rate_out(0)*self.streams_out()))
        return max(latency_in,latency_out)

    def pipeline_depth(self):
        return sum([ self.modules[module].pipeline_depth() for module in self.modules ])

    def wait_depth(self):
        return sum([ self.modules[module].wait_depth() for module in self.modules ])

    def resource(self):
        return {
            "LUT"   : 0,
            "FF"    : 0,
            "BRAM"  : math.ceil(self.buffer_depth/1125)*self.coarse_in,
            "DSP"   : 0
        }

    def static_power(self):
        return 0

    def dynamic_power(self, freq, rate): 
        return 0

    def power(self,freq,rate):
        return self.static_power() + self.dynamic_power(freq,rate)

    def get_coarse_in_feasible(self,wr_factor=1):
        return self.get_factors(int(self.channels_in()/wr_factor))

    def get_coarse_out_feasible(self,wr_factor=1):
        return self.get_factors(int(self.channels_out()/wr_factor))

    def update_coarse_in(self, coarse_in):
        self.coarse_in  = coarse_in
        self.coarse_out = coarse_in

    def update_coarse_out(self, coarse_out):
        self.coarse_in  = coarse_out
        self.coarse_out = coarse_out

    def load_coef(self):
        for module in self.modules:
            self.modules[module].load_coef(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../coefficients/{}_rsc_coef.npy".format(module))
            )

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

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
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

