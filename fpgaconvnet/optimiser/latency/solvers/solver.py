import copy
import math
import itertools
import random
import numpy as np

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.models.network import Network
from fpgaconvnet.models.layers import ConvolutionLayer, InnerProductLayer, ReLULayer, EltWiseLayer

import fpgaconvnet.optimiser.solvers.solver

from .utils import get_hw_from_dict

class LatencySolver(fpgaconvnet.optimiser.solvers.solver.Solver):

    def __init__(self, net: Network, runtime_parameters=False):

        # initialise base solver
        super().__init__(net, 0)

        # set the dimensionality
        self.dimensionality = 2 # TODO: make parameter

        # dictionary of layers, keyed by their name
        self.building_blocks = {}
        for node in self.net.graph.nodes:
            self.building_blocks[node] = copy.deepcopy(self.net.graph.nodes[node])
            self.building_blocks[node]["exec_nodes"] = [ node ]

        # combine simple layer types
        self.simple_layer_types = [ LAYER_TYPE.ReLU, LAYER_TYPE.EltWise ]
        for layer_type in self.simple_layer_types:
            self.combine(layer_type)

        # flag to say if runtime parameterisable
        self.runtime_parameters = runtime_parameters

    # import shape generation functions
    from .shapes import apply_random_shape
    from .shapes import update_building_block_shape

    # import scheduler functions
    from .scheduler import get_convolution_schedule
    from .scheduler import get_inner_product_schedule
    from .scheduler import get_pooling_schedule
    from .scheduler import get_basic_schedule
    from .scheduler import get_schedule

    def get_layers_of_type(self, layer_type):
        """
        returns a list of the layer keys with the given layer type
        """
        # find all layers of given type
        layers_of_type = []
        for layer in self.building_blocks:
            # layers of the same type
            if self.building_blocks[layer]["type"] == layer_type:
                layers_of_type.append(layer)

        # return layers
        return layers_of_type

    def combine(self, layer_type, discriminate=[], num_layers=-1):

        # get the layers of the given type
        layers_of_type = self.get_layers_of_type(layer_type)

        # further discriminate the layers to combine TODO
        layers_to_combine = layers_of_type

        # escape if there are no layers to combine
        if len(layers_to_combine) == 0:
            return

        # create a new layer name by combining
        new_layer_name = "_".join(layers_to_combine)

        # parameters to create new hardware node
        parameters = None

        # get the superset layer for the given layer type:
        match layer_type:
            case LAYER_TYPE.Convolution:

                # get all the parameter keys
                max_param_keys = [ "groups",  "kernel_rows", "kernel_cols",
                        "stride_rows", "stride_cols", "pad_top", "pad_bottom",
                        "pad_left", "pad_right" ]
                min_param_keys = [ "filters", "rows", "cols", "channels", "fine",
                        "coarse_in", "coarse_out", "coarse_group" ]

                # add 3D specific parameters
                if self.dimensionality == 3:
                    max_param_keys.extend(["kernel_depth",
                        "stride_depth", "pad_front", "pad_back"])
                    min_param_keys.append("depth")

                # get the parameters
                parameters = { key: self.get_max_attr_of_hw_nodes(
                    layers_to_combine, key) for key in max_param_keys }
                parameters.update({ key: self.get_min_attr_of_hw_nodes(
                    layers_to_combine, key) for key in min_param_keys })

            case LAYER_TYPE.InnerProduct:

                # get all the parameter keys
                min_param_keys = [ "filters", "rows", "cols", "channels",
                        "coarse_in", "coarse_out", "coarse_group" ]

                # add 3D specific parameters
                if self.dimensionality == 3:
                    min_param_keys.append("depth")

                # get the parameters
                parameters = { key: self.get_min_attr_of_hw_nodes(
                    layers_to_combine, key) for key in min_param_keys }

            case LAYER_TYPE.Pooling:

                # get all the parameter keys
                max_param_keys = [ "kernel_rows", "kernel_cols",
                        "stride_rows", "stride_cols", "pad_top", "pad_bottom",
                        "pad_left", "pad_right" ]
                min_param_keys = [ "rows", "cols", "channels", "coarse" ]

                # add 3D specific parameters
                if self.dimensionality == 3:
                    max_param_keys.extend(["kernel_depth",
                        "stride_depth", "pad_front", "pad_back"])
                    min_param_keys.append("depth")

                # get the parameters
                parameters = { key: self.get_max_attr_of_hw_nodes(
                    layers_to_combine, key) for key in max_param_keys }
                parameters.update({ key: self.get_min_attr_of_hw_nodes(
                    layers_to_combine, key) for key in min_param_keys })

            case LAYER_TYPE.ReLU:

                min_param_keys = [ "rows", "cols", "channels", "coarse" ]

                # add 3D specific parameters
                if self.dimensionality == 3:
                    min_param_keys.append("depth")

                # get the parameters
                parameters = { key: self.get_min_attr_of_hw_nodes(
                    layers_to_combine, key) for key in min_param_keys }

            case LAYER_TYPE.EltWise:

                min_param_keys = [ "rows", "cols", "channels", "coarse" ]
                max_param_keys = [ "ports_in" ]

                # add 3D specific parameters
                if self.dimensionality == 3:
                    min_param_keys.append("depth")

                # get the parameters
                parameters = { key: self.get_min_attr_of_hw_nodes_multi(
                    layers_to_combine, key) for key in min_param_keys }

                # TODO: decide on op type and broadcast
                parameters["op_type"] = "mul"
                parameters["broadcast"] = True

            case _:
                raise NotImplementedError(layer_type)

        # get all the execution nodes from the layers to combine
        exec_nodes = list(itertools.chain(*[
                self.building_blocks[hw_node]["exec_nodes"] \
                        for hw_node in layers_to_combine ]))

        # create a new layer from these parameters
        self.building_blocks[new_layer_name] = {
            "type": layer_type,
            "hw": get_hw_from_dict(layer_type,
                parameters, self.dimensionality),
            "exec_nodes": exec_nodes,
        }

        # remove the combined layers
        if len(layers_to_combine) > 1:
            for layer in layers_to_combine:
                del self.building_blocks[layer]

    def get_max_attr_of_hw_nodes(self, hw_nodes, attr):
        return max([ getattr(self.building_blocks[hw_node]["hw"], attr) \
                for hw_node in hw_nodes ])

    def get_min_attr_of_hw_nodes(self, hw_nodes, attr):
        return min([ getattr(self.building_blocks[hw_node]["hw"], attr) \
                for hw_node in hw_nodes ])

    def get_max_attr_of_hw_nodes_multi(self, hw_nodes, attr):
        return max([ getattr(self.building_blocks[hw_node]["hw"], attr)[0] \
                for hw_node in hw_nodes ])

    def get_min_attr_of_hw_nodes_multi(self, hw_nodes, attr):
        return min([ getattr(self.building_blocks[hw_node]["hw"], attr)[0] \
                for hw_node in hw_nodes ])


    def seperate(self, node):
        """
        method to seperate out hardware nodes in `self.building_blocks`
        """
        # iterate over exec_nodes
        for exec_node in self.building_blocks[node]["exec_nodes"]:
            # add hardware of exec_node to the latency nodes
            self.building_blocks[exec_node] = copy.deepcopy(self.net.graph.nodes[exec_node])
            self.building_blocks[exec_node]["exec_nodes"] = [ exec_node ]
            # keep performance parameters the same (coarse, fine, ...)
            # TODO

        # delete the original node from the latency nodes
        del self.building_blocks[node]

    def get_resources(self):
        """
        returns the sum of the resources of all nodes in the building_blocks
        """
        return {
            "LUT": sum([ node["hw"].resource()["LUT"] \
                    for _, node in self.building_blocks.items() ]),
            "FF": sum([ node["hw"].resource()["FF"] \
                    for _, node in self.building_blocks.items() ]),
            "DSP": sum([ node["hw"].resource()["DSP"] \
                    for _, node in self.building_blocks.items() ]),
            "BRAM": sum([ node["hw"].resource()["BRAM"] \
                    for _, node in self.building_blocks.items() ]),
        }

    def validate(self):
        """
        check that all `building_blocks` have valid parameters
        """
        # iterate over laytency nodes
        for hw_node in self.building_blocks:
            # switch-case on layer type
            match self.building_blocks[hw_node]["type"]:
                case LAYER_TYPE.Convolution:
                    # iterate over the execution nodes
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"]:
                        # assertions to check parameters are correct
                        assert self.net.graph.nodes[exec_node]["hw"].kernel_size[0] <= \
                                self.building_blocks[hw_node]["hw"].kernel_size[0]
                        assert self.net.graph.nodes[exec_node]["hw"].kernel_size[1] <= \
                                self.building_blocks[hw_node]["hw"].kernel_size[1]

    def get_building_block(self, exec_node):
        """
        find the corresponding hardware node for the node to be executed
        """
        for hw_node in self.building_blocks:
            if exec_node in self.building_blocks[hw_node]["exec_nodes"]:
                return hw_node
        raise StopIteration(f"could not find hardware for execution node {exec_node}")

    def evaluate_latency(self):
        """
        evaluate the latency for the execution of the graph. Maps the
        nodes of the `self.net.graph` to those of the `self.building_blocks`.
        The latency is the sum of the execution of all these elements.
        """
        # total execution latency
        total_latency = 0

        # get the schedule
        schedule = self.get_schedule()

        # iterate over nodes in the execution graph
        for exec_node in self.net.graph:

            # find the hardware node
            hw_node = self.get_building_block(exec_node)

            # get the latency of the node for all scheduled executions
            if self.runtime_parameters:
                total_latency += sum([ get_hw_from_dict(
                    self.building_blocks[hw_node]["type"],
                    param, self.dimensionality).latency() \
                        for param in schedule[exec_node] ])
            else:
                total_latency += len(schedule[exec_node]) * \
                    self.building_blocks[hw_node]["hw"].latency()

        # return the overall latency
        return total_latency


