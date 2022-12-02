import copy
import math
import itertools
import random
import numpy as np

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.models.network import Network
from fpgaconvnet.models.layers import ConvolutionLayer, InnerProductLayer, ReLULayer, EltWiseLayer

import fpgaconvnet.optimiser.solvers.solver

from .utils import *

class Solver(fpgaconvnet.optimiser.solvers.solver.Solver):

    def __init__(self, net: Network, runtime_parameters=False):

        # initialise base solver
        super().__init__(net, 0)

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

        # get the hardware for all the layers to combine
        layers_to_combine_hw = [ self.building_blocks[layer]["hw"] \
                for layer in layers_to_combine ]

        # create a new layer name by combining
        new_layer_name = "_".join(layers_to_combine)

        # get the superset layer for the given layer type:
        match layer_type:
            case LAYER_TYPE.ReLU:
                pass
            case LAYER_TYPE.Convolution:

                # get all the parameter keys
                max_param_keys = [ "groups",  "kernel_rows", "kernel_cols",
                        "stride_rows", "stride_cols", "pad_top", "pad_bottom",
                        "pad_left", "pad_right" ]
                min_param_keys = [ "filters", "rows", "cols", "channels", "fine",
                        "coarse_in", "coarse_out", "coarse_group" ]

                # get the parameters
                parameters = { key: self.get_max_attr_of_hw_nodes(
                    layers_to_combine, key) for key in max_param_keys }
                parameters.update({ key: self.get_min_attr_of_hw_nodes(
                    layers_to_combine, key) for key in min_param_keys })

                # get all the execution nodes from the layers to combine
                exec_nodes = list(itertools.chain(*[
                        self.building_blocks[layer]["exec_nodes"] \
                                for layer in layers_to_combine ]))

                # create a new layer from these parameters
                self.building_blocks[new_layer_name] = {
                    "type": LAYER_TYPE.Convolution,
                    "hw": get_convolution_from_dict(parameters),
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
                    self.building_blocks[hw_node]["type"], param).latency() \
                        for param in schedule[exec_node] ])
            else:
                total_latency += len(schedule[exec_node])*self.building_blocks[hw_node]["hw"].latency()

        # return the overall latency
        return total_latency

    def get_schedule(self):
        """
        returns a (unoptimised) schedule for the execution of the hardware for
        `self.net.graph`. Need to choose the configuration of the input shapes,
        the choice in coarse factors, and the fine factor. We will want to use
        as much of the hardware as possible for each run
        """
        # create a schedule for each node of the execution graph
        schedule = {}

        # iterate over nodes in the execution graph
        for exec_node in self.net.graph.nodes:
            # add a blank list to the schedule for this node
            schedule[exec_node] = []
            # find the hardware node
            hw_node = self.get_building_block(exec_node)
            # get the parameters for the exec node
            base_param = self.net.graph.nodes[exec_node]["hw"].layer_info_dict()
            # handle different hardware types
            match self.net.graph.nodes[exec_node]["type"]:
                case LAYER_TYPE.Convolution:
                    # choose the largest factor for fine that's below the hardware's fine
                    fine = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].fine,
                            self.net.graph.nodes[exec_node]["hw"].get_fine_feasible()))[-1]
                    # do the same for the coarse factors TODO: improve the channel, coarse trade-off
                    coarse_in = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in,
                            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))[-1]
                    coarse_out = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_out,
                            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))[-1]
                    coarse_group = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_group,
                            self.net.graph.nodes[exec_node]["hw"].get_coarse_group_feasible()))[-1]
                    # get the repetition of each dimension
                    row_repetition = math.ceil(
                        self.net.graph.nodes[exec_node]["hw"].rows_out() / \
                                self.building_blocks[hw_node]["hw"].rows_out())
                    col_repetition = math.ceil(
                        self.net.graph.nodes[exec_node]["hw"].cols_out() / \
                                self.building_blocks[hw_node]["hw"].cols_out())
                    # channel_repetition = math.ceil(
                    #     self.net.graph.nodes[exec_node]["hw"].channels_in() / \
                    #             self.building_blocks[hw_node]["hw"].channels_in())
                    # filter_repetition = math.ceil(
                    #     self.net.graph.nodes[exec_node]["hw"].filters / \
                    #             self.building_blocks[hw_node]["hw"].filters)
                    # TODO: at the moment, assume filters and channels always fit
                    # iterate across each dimension to be repeated
                    for h in range(row_repetition):
                        for w in range(col_repetition):
                            # for c in range(channel_repetition): # TODO
                            #     for f in range(filter_repetition): #TODO
                            # get the greatest spatial dimensions for each execution
                            rows_out = min(self.building_blocks[hw_node]["hw"].rows_out(),
                                    base_param["rows_out"]-h*self.building_blocks[hw_node]["hw"].rows_out())
                            cols_out = min(self.building_blocks[hw_node]["hw"].cols_out(),
                                    base_param["cols_out"]-w*self.building_blocks[hw_node]["hw"].cols_out())
                            # convert the output dimensions to input dimensions
                            rows_in = (rows_out*base_param["stride"][0]) + base_param["kernel_size"][0]-base_param["pad_bottom"]-base_param["pad_top"]-1
                            cols_in = (cols_out*base_param["stride"][1]) + base_param["kernel_size"][1]-base_param["pad_left"]-base_param["pad_right"]-1
                            # add the parameters to the schedule
                            param = copy.deepcopy(base_param)
                            param["rows_in"] = rows_in
                            param["cols_in"] = cols_in
                            param["fine"] = fine
                            param["coarse_in"] = coarse_in
                            param["coarse_out"] = coarse_out
                            param["coarse_group"] = coarse_group
                            # append to the schedule
                            schedule[exec_node].append(param)
                case _:
                    # in the default case, assume it's just run once with the exec_node's parameters
                    schedule[exec_node].append(base_param)

            # change rows_in, cols_in, depth_in, etc... to rows, cols, depth, ...
            for i in range(len(schedule[exec_node])):
                schedule[exec_node][i]["rows"] = schedule[exec_node][i]["rows_in"]
                schedule[exec_node][i]["cols"] = schedule[exec_node][i]["cols_in"]
                schedule[exec_node][i]["channels"] = schedule[exec_node][i]["channels_in"]
                if "depth_in" in schedule[exec_node][i]:
                    schedule[exec_node][i]["depth"] = schedule[exec_node][i]["depth_in"]

        # return the schedule
        return schedule


    def validate_schedule(self):
        pass

    def apply_random_shape(self, hw_node, rand_shape_range = [10, 10, 10] ,
            use_previous_shape: bool = False) -> np.array:
        """
        get a random shape for executing the featuremap.
        """

        # get the previous input shape
        prev_input_shape = self.building_blocks[hw_node]["hw"].shape_in()
        prev_output_shape = self.building_blocks[hw_node]["hw"].shape_out()

        # get the max shape for the input and output
        max_input_shape = [ max([ self.net.graph.nodes[exec_node]["hw"].shape_in()[i] \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]) for \
                    i in range(len(prev_input_shape)) ]
        max_output_shape = [ max([ self.net.graph.nodes[exec_node]["hw"].shape_out()[i] \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]) for \
                    i in range(len(prev_output_shape)) ]

        if use_previous_shape:
            # get a random shape based on the previous (within a range)
            next_input_shape = [ random.randint(
                    max(1, prev_input_shape[i]-rand_shape_range[i]),
                    min(prev_input_shape[i]+rand_shape_range[i], max_input_shape[i])) for \
                            i in range(len(prev_input_shape)) ]
            next_output_shape = [ random.randint(
                    max(1, prev_output_shape[i]-rand_shape_range[i]),
                    min(prev_output_shape[i]+rand_shape_range[i], max_output_shape[i])) for \
                            i in range(len(prev_output_shape)) ]
        else:
            # get a random shape
            next_input_shape = [ random.randint(1, max_dim) for max_dim in max_input_shape ]
            next_output_shape = [ random.randint(1, max_dim) for max_dim in max_output_shape ]

        # update the next shape for specific hardware types
        self.update_building_block_shape(hw_node,
                next_input_shape, max_input_shape,
                next_output_shape, max_output_shape)

    def update_building_block_shape(self, hw_node, next_input_shape,
            max_input_shape, next_output_shape, max_output_shape):

        # update the next shape for specific hardware types
        match self.building_blocks[hw_node]["type"]:
            case LAYER_TYPE.Convolution:
                # get the max kernel size
                max_kernel_size = [
                    max([ self.net.graph.nodes[exec_node]["hw"].kernel_size[0] \
                        for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]),
                    max([ self.net.graph.nodes[exec_node]["hw"].kernel_size[1] \
                        for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]),
                ]
                # make sure rows are greater than the kernel size
                # TODO: get the actual min shape
                self.building_blocks[hw_node]["hw"].rows = max(max_kernel_size[0]+1, next_input_shape[0])
                self.building_blocks[hw_node]["hw"].cols = max(max_kernel_size[1]+1, next_input_shape[1])
                # fix channels to be max TODO: do we want to have runtime channels?
                self.building_blocks[hw_node]["hw"].channels = max_input_shape[2]
                # set a random filter dimension
                max_filters = max([ self.net.graph.nodes[exec_node]["hw"].filters \
                        for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]),
                # self.building_blocks[node]["hw"].filters = random.randint(1, max_filters) TODO: support properly
                self.building_blocks[hw_node]["hw"].filters = max_output_shape[2]
            case LAYER_TYPE.Pooling:
                # get the max kernel size
                max_kernel_size = [
                    max([ self.net.graph.nodes[exec_node]["hw"].kernel_size[0] \
                        for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]),
                    max([ self.net.graph.nodes[exec_node]["hw"].kernel_size[1] \
                        for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]),
                ]
                # make sure rows are greater than the kernel size
                self.building_blocks[hw_node]["hw"].rows = max(max_kernel_size[0]+1, next_input_shape[0])
                self.building_blocks[hw_node]["hw"].cols = max(max_kernel_size[1]+1, next_input_shape[1])
            # TODO: handle the other layer types
            case _:
                self.building_blocks[hw_node]["hw"].rows = max_input_shape[0]
                self.building_blocks[hw_node]["hw"].cols = max_input_shape[1]
                self.building_blocks[hw_node]["hw"].channels = max_input_shape[2]

        # update the hw node
        self.building_blocks[hw_node]["hw"].update()


