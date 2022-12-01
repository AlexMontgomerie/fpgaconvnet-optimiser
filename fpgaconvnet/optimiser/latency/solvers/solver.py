import copy
import math
import itertools
import random
import numpy as np

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.models.network import Network
from fpgaconvnet.models.layers import ConvolutionLayer, InnerProductLayer, ReLULayer, EltWiseLayer

import fpgaconvnet.optimiser.solvers.solver

class Solver(fpgaconvnet.optimiser.solvers.solver.Solver):

    def __init__(self, net: Network):

        # initialise base solver
        super().__init__(net, 0)

        # dictionary of layers, keyed by their name
        self.latency_nodes = {}
        for node in self.net.graph.nodes:
            self.latency_nodes[node] = copy.deepcopy(self.net.graph.nodes[node])
            self.latency_nodes[node]["exec_nodes"] = [ node ]

        # combine simple layer types
        self.simple_layer_types = [ LAYER_TYPE.ReLU, LAYER_TYPE.EltWise ]
        for layer_type in self.simple_layer_types:
            self.combine(layer_type)

    def get_layers_of_type(self, layer_type):
        """
        returns a list of the layer keys with the given layer type
        """
        # find all layers of given type
        layers_of_type = []
        for layer in self.latency_nodes:
            # layers of the same type
            if self.latency_nodes[layer]["type"] == layer_type:
                layers_of_type.append(layer)

        # return layers
        return layers_of_type

    def combine(self, layer_type, discriminate=[], num_layers=-1):

        # get the layers of the given type
        layers_of_type = self.get_layers_of_type(layer_type)

        # further discriminate the layers to combine TODO
        layers_to_combine = layers_of_type

        # get the hardware for all the layers to combine
        layers_to_combine_hw = [ self.latency_nodes[layer]["hw"] \
                for layer in layers_to_combine ]

        # create a new layer name by combining
        new_layer_name = "_".join(layers_to_combine)

        # get the superset layer for the given layer type:
        match layer_type:
            case LAYER_TYPE.ReLU:
                pass
            case LAYER_TYPE.Convolution:

                # get the max of the parameters
                filters = max([ node.filters for node in layers_to_combine_hw ])
                rows = max([ node.rows for node in layers_to_combine_hw ])
                cols = max([ node.cols for node in layers_to_combine_hw ])
                channels = max([ node.channels for node in layers_to_combine_hw ])
                fine = min([ node.fine for node in layers_to_combine_hw ])
                groups = max([ node.groups for node in layers_to_combine_hw ])
                coarse_in = min([ node.coarse_in for node in layers_to_combine_hw ])
                coarse_out = min([ node.coarse_out for node in layers_to_combine_hw ])
                coarse_group = min([ node.coarse_group for node in layers_to_combine_hw ])
                kernel_size = [
                        max([ node.kernel_size[0] for node in layers_to_combine_hw ]),
                        max([ node.kernel_size[1] for node in layers_to_combine_hw ]),
                ]
                stride = [
                        min([ node.stride[0] for node in layers_to_combine_hw ]),
                        min([ node.stride[1] for node in layers_to_combine_hw ]),
                ]
                pad = [
                        max([ node.pad[0] for node in layers_to_combine_hw ]),
                        max([ node.pad[1] for node in layers_to_combine_hw ]),
                        max([ node.pad[2] for node in layers_to_combine_hw ]),
                        max([ node.pad[3] for node in layers_to_combine_hw ]),
                ]

                # get all the execution nodes from the layers to combine
                exec_nodes = list(itertools.chain(*[
                        self.latency_nodes[layer]["exec_nodes"] \
                                for layer in layers_to_combine ]))

                # create a new layer from these parameters
                self.latency_nodes[new_layer_name] = {
                    "type": LAYER_TYPE.Convolution,
                    "hw": ConvolutionLayer(
                            filters, rows, cols, channels, coarse_in,
                            coarse_out, coarse_group, kernel_size,
                            stride, groups, pad, fine),
                    "exec_nodes": exec_nodes,
                }

        # remove the combined layers
        if len(layers_to_combine) > 1:
            for layer in layers_to_combine:
                del self.latency_nodes[layer]

    def seperate(self, node):

        # iterate over exec_nodes
        for exec_node in self.latency_nodes[node]["exec_nodes"]:
            # add hardware of exec_node to the latency nodes
            self.latency_nodes[exec_node] = copy.deepcopy(self.net.graph.nodes[exec_node])
            self.latency_nodes[exec_node]["exec_nodes"] = [ exec_node ]
            # keep performance parameters the same (coarse, fine, ...)
            # TODO

        # delete the original node from the latency nodes
        del self.latency_nodes[node]

    def get_resources(self):
        """
        returns the sum of the resources of all nodes in the latency_nodes
        """
        return {
            "LUT": sum([ node["hw"].resource()["LUT"] \
                    for _, node in self.latency_nodes.items() ]),
            "FF": sum([ node["hw"].resource()["FF"] \
                    for _, node in self.latency_nodes.items() ]),
            "DSP": sum([ node["hw"].resource()["DSP"] \
                    for _, node in self.latency_nodes.items() ]),
            "BRAM": sum([ node["hw"].resource()["BRAM"] \
                    for _, node in self.latency_nodes.items() ]),
        }

    def validate(self):
        """
        check that all `latency_nodes` have valid parameters
        """
        # iterate over laytency nodes
        for node in self.latency_nodes:
            # switch-case on layer type
            match self.latency_nodes[node]["type"]:
                case LAYER_TYPE.Convolution:
                    # iterate over the execution nodes
                    for exec_node in self.latency_nodes[node]["exec_nodes"]:
                        # assertions to check parameters are correct
                        assert self.net.graph.nodes[exec_node]["hw"].kernel_size[0] <= \
                                self.latency_nodes[node]["hw"].kernel_size[0]
                        assert self.net.graph.nodes[exec_node]["hw"].kernel_size[1] <= \
                                self.latency_nodes[node]["hw"].kernel_size[1]

    def get_latency_node(self, exec_node):
        """
        find the corresponding hardware node for the node to be executed
        """
        for node in self.latency_nodes:
            if exec_node in self.latency_nodes[node]["exec_nodes"]:
                return node
        raise StopIteration(f"could not find hardware for execution node {exec_node}")

    def evaluate_latency(self):
        """
        evaluate the latency for the execution of the graph. Maps the
        nodes of the `self.net.graph` to those of the `self.latency_nodes`.
        The latency is the sum of the execution of all these elements.
        """
        # total execution latency
        total_latency = 0
        # get the schedule
        schedule = self.get_schedule()
        # iterate over nodes in the execution graph
        for node in self.net.graph:
            # find the hardware node
            hw_node = self.get_latency_node(node)
            # handle different hardware types
            match self.net.graph.nodes[node]["type"]:
                case LAYER_TYPE.Convolution:
                    # get repetition
                    repetition = 1
                    repetition *= 1 # TODO row repetition
                    repetition *= 1 # TODO col repetition
                    repetition *= 1 # TODO channel repetition
                    repetition *= 1 # TODO filter repetition
                    # get the latency for the node
                    # TODO add run time parameters to the latency estimate
                    node_latency = self.latency_nodes[hw_node]["hw"].latency()
                case LAYER_TYPE.Pooling:
                    # get repetition
                    repetition = 1
                    repetition *= 1 # TODO row repetition
                    repetition *= 1 # TODO col repetition
                    repetition *= 1 # TODO channel repetition
                    repetition *= 1 # TODO filter repetition
                    # get the latency for the node
                    # TODO add run time parameters to the latency estimate
                    node_latency = self.latency_nodes[hw_node]["hw"].latency()
                case LAYER_TYPE.Squeeze:
                    # get repetition
                    repetition = 1
                    repetition *= 1 # TODO row repetition
                    repetition *= 1 # TODO col repetition
                    repetition *= 1 # TODO channel repetition
                    repetition *= 1 # TODO filter repetition
                    # get the latency for the node
                    # TODO add run time parameters to the latency estimate
                    node_latency = self.latency_nodes[hw_node]["hw"].latency()
                case LAYER_TYPE.InnerProduct:
                    # get repetition
                    repetition = 1
                    repetition *= 1 # TODO row repetition
                    repetition *= 1 # TODO col repetition
                    repetition *= 1 # TODO channel repetition
                    repetition *= 1 # TODO filter repetition
                    # get the latency for the node
                    # TODO add run time parameters to the latency estimate
                    node_latency = self.latency_nodes[hw_node]["hw"].latency()
                case LAYER_TYPE.ReLU:
                    # get repetition
                    repetition = 1
                    repetition *= 1 # TODO row repetition
                    repetition *= 1 # TODO col repetition
                    repetition *= 1 # TODO channel repetition
                    repetition *= 1 # TODO filter repetition
                    # get the latency for the node
                    # TODO add run time parameters to the latency estimate
                    node_latency = self.latency_nodes[hw_node]["hw"].latency()
                case _:
                    raise NotImplementedError(f"layer type not implemented")

            # add the node latency to the total latency
            total_latency += repetition*node_latency
        # return the overall latency
        return total_latency

    def apply_random_shape(self, node, rand_shape_range = [10, 10, 10] ,
            use_previous_shape: bool = False) -> np.array:
        """
        get a random shape for executing the featuremap.
        """
        # get the max shape for the input
        max_input_shape = {
            "rows" : max([ self.graph.nodes[exec_node]["hw"].rows_in \
                    for exec_node in self.latency_nodes[node]["exec_nodes"] ]),
            "cols" : max([ self.graph.nodes[exec_node]["hw"].cols_in \
                    for exec_node in self.latency_nodes[node]["exec_nodes"] ]),
            "channels" : max([ self.graph.nodes[exec_node]["hw"].channels_in \
                    for exec_node in self.latency_nodes[node]["exec_nodes"] ]),
        }

        if use_previous_shape:
            # get the previous input shape
            prev_input_shape = {
                "rows" : self.latency_nodes[node]["hw"].rows_in,
                "cols" : self.latency_nodes[node]["hw"].cols_in,
                "channels" : self.latency_nodes[node]["hw"].channels_in,
            }
            # get a random shape based on the previous (within a range)
            next_input_shape = {
                "rows" : random.randint(
                    max(1, prev_input_shape["rows"]-rand_shape_range[0]),
                    min(prev_input_shape["rows"]+rand_shape_range[0], max_input_shape["rows"])),
                "cols" : random.randint(
                    max(1, prev_input_shape["cols"]-rand_shape_range[0]),
                    min(prev_input_shape["cols"]+rand_shape_range[0], max_input_shape["cols"])),
                "channels" : random.randint(
                    max(1, prev_input_shape["channels"]-rand_shape_range[0]),
                    min(prev_input_shape["channels"]+rand_shape_range[0], max_input_shape["channels"])),
            }
        else:
            # get a random shape based on the previous (within a range)
            next_input_shape = {
                "rows" : random.randint(1, max_input_shape["rows"]),
                "cols" : random.randint(1, max_input_shape["cols"]),
                "channels" : random.randint(1, max_input_shape["channels"]),
            }

        # update the next shape for specific hardware types
        match self.net.graph.nodes[node]["type"]:
            case LAYER_TYPE.Convolution:
                # get the max kernel size
                max_kernel_size = [
                    max([ self.graph.nodes[exec_node]["hw"].kernel_size[0] \
                        for exec_node in self.latency_nodes[node]["exec_nodes"] ]),
                    max([ self.graph.nodes[exec_node]["hw"].kernel_size[1] \
                        for exec_node in self.latency_nodes[node]["exec_nodes"] ]),
                ]
                # make sure rows are greater than the kernel size
                self.latency_nodes[node]["hw"].rows = max(max_kernel_size[0]+1, next_input_shape["rows"])
                self.latency_nodes[node]["hw"].cols = max(max_kernel_size[1]+1, next_input_shape["cols"])
                # fix channels to be max TODO: do we want to have runtime channels?
                self.latency_nodes[node]["hw"].channels = max_input_shape["channels"]
                # set a random filter dimension
                max_filters = max([ self.graph.nodes[exec_node]["hw"].filters \
                        for exec_node in self.latency_nodes[node]["exec_nodes"] ]),
                # self.latency_nodes[node]["hw"].filters = random.randint(1, max_filters) TODO: support properly
                self.latency_nodes[node]["hw"].filters = max_filters
            case LAYER_TYPE.Pooling:
                # get the max kernel size
                max_kernel_size = [
                    max([ self.graph.nodes[exec_node]["hw"].kernel_size[0] \
                        for exec_node in self.latency_nodes[node]["exec_nodes"] ]),
                    max([ self.graph.nodes[exec_node]["hw"].kernel_size[1] \
                        for exec_node in self.latency_nodes[node]["exec_nodes"] ]),
                ]
                # make sure rows are greater than the kernel size
                self.latency_nodes[node]["hw"].rows = max(max_kernel_size[0]+1, next_input_shape["rows"])
                self.latency_nodes[node]["hw"].cols = max(max_kernel_size[1]+1, next_input_shape["cols"])
            # TODO: handle the other layer types
            case _:
                self.latency_nodes[node]["hw"].rows = max_input_shape["rows"]
                self.latency_nodes[node]["hw"].cols = max_input_shape["cols"]
                self.latency_nodes[node]["hw"].channels = max_input_shape["channels"]



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
            hw_node = self.get_latency_node(exec_node)
            # get the parameters for the exec node
            base_param = self.net.graph.nodes[exec_node]["hw"].layer_info_dict()
            # handle different hardware types
            match self.net.graph.nodes[exec_node]["type"]:
                case LAYER_TYPE.Convolution:
                    # choose the largest factor for fine that's below the hardware's fine
                    fine = list(filter(lambda f: f <= self.latency_nodes[hw_node]["hw"].fine,
                            self.net.graph.nodes[exec_node]["hw"].get_fine_feasible()))[-1]
                    # do the same for the coarse factors TODO: improve the channel, coarse trade-off
                    coarse_in = list(filter(lambda f: f <= self.latency_nodes[hw_node]["hw"].coarse_in,
                            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))[-1]
                    coarse_out = list(filter(lambda f: f <= self.latency_nodes[hw_node]["hw"].coarse_out,
                            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))[-1]
                    coarse_group = list(filter(lambda f: f <= self.latency_nodes[hw_node]["hw"].coarse_group,
                            self.net.graph.nodes[exec_node]["hw"].get_coarse_group_feasible()))[-1]
                    # get the repetition of each dimension
                    row_repetition = math.ceil(
                        self.net.graph.nodes[exec_node]["hw"].rows_out() / \
                                self.latency_nodes[hw_node]["hw"].rows_out())
                    col_repetition = math.ceil(
                        self.net.graph.nodes[exec_node]["hw"].cols_out() / \
                                self.latency_nodes[hw_node]["hw"].cols_out())
                    # channel_repetition = math.ceil(
                    #     self.net.graph.nodes[exec_node]["hw"].channels_in() / \
                    #             self.latency_nodes[hw_node]["hw"].channels_in())
                    # filter_repetition = math.ceil(
                    #     self.net.graph.nodes[exec_node]["hw"].filters / \
                    #             self.latency_nodes[hw_node]["hw"].filters)
                    # TODO: at the moment, assume filters and channels always fit
                    # iterate across each dimension to be repeated
                    for h in range(row_repetition):
                        for w in range(col_repetition):
                            # for c in range(channel_repetition): # TODO
                            #     for f in range(filter_repetition): #TODO
                            # get the greatest spatial dimensions for each execution
                            rows_out = min(self.latency_nodes[hw_node]["hw"].rows_out(),
                                    base_param["rows_out"]-h*self.latency_nodes[hw_node]["hw"].rows_out())
                            cols_out = min(self.latency_nodes[hw_node]["hw"].cols_out(),
                                    base_param["cols_out"]-w*self.latency_nodes[hw_node]["hw"].cols_out())
                            # convert the output dimensions to input dimensions
                            rows_in = (rows_out*base_param["stride"][0]) + base_param["kernel_size"][0]-base_param["pad_bottom"]-base_param["pad_top"]-1
                            cols_in = (cols_out*base_param["stride"][1]) + base_param["kernel_size"][1]-base_param["pad_left"]-base_param["pad_right"]-1
                            # add the parameters to the schedule
                            param = copy.deepcopy(base_param)
                            param["fine"] = fine
                            param["coarse_in"] = coarse_in
                            param["coarse_out"] = coarse_out
                            param["coarse_group"] = coarse_group
                            # append to the schedule
                            schedule[exec_node].append(param)
                case _:
                    # in the default case, assume it's just run once with the exec_node's parameters
                    schedule[exec_node].append(base_param)

        # return the schedule
        return schedule


    def validate_schedule(self):
        pass

    # def apply_random_transform(self, node):

    #     # transforms to use
    #     transforms = [ "input_shape", "coarse" ]

    #     # add extra transforms for Convolution and Inner Product layer
    #     if self.latency_nodes[node]["type"] == LAYER_TYPE.Convolution:
    #         transforms += [ "fine" ]

    #     # choose a random transform
    #     transform = random.choice(transforms)

    #     # apply the transforms
    #     match transform:
    #         case "input_shape":

    #             # get a random shape for the node
    #             input_shape = self.get_random_arbitrary_shape(node)

    #             # apply the random shape
    #             self.latency_nodes[node]["hw"].rows = input_shape["rows"]
    #             self.latency_nodes[node]["hw"].cols = input_shape["cols"]
    #             self.latency_nodes[node]["hw"].channels = input_shape["channels"]

    #             # for convolution layers change the filter dimension # TODO: need to support the current coarse factor (or reset it)
    #             if self.latency_nodes[node]["type"] == LAYER_TYPE.Convolution:
    #                 # self.latency_nodes[node]["hw"].filters = input_shape["filters"]
    #                 pass

    #         case "coarse":

    #             # coarse in, out and group

    #         case "fine":
    #             pass
    #         case _:
    #             raise NotImplementedError(f"transform {transform} nit implemented")


    #     # update the latency node
    #     self.latency_nodes[node].update()
