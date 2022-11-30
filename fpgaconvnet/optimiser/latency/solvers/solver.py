import copy
import itertools

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.models.network import Network
from fpgaconvnet.models.layers import *

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
                fine = max([ node.fine for node in layers_to_combine_hw ])
                groups = max([ node.groups for node in layers_to_combine_hw ])
                coarse_in = max([ node.coarse_in for node in layers_to_combine_hw ])
                coarse_out = max([ node.coarse_out for node in layers_to_combine_hw ])
                coarse_group = max([ node.coarse_group for node in layers_to_combine_hw ])
                kernel_size = [
                        max([ node.kernel_size[0] for node in layers_to_combine_hw ]),
                        max([ node.kernel_size[1] for node in layers_to_combine_hw ]),
                ]
                stride = [
                        max([ node.stride[0] for node in layers_to_combine_hw ]),
                        max([ node.stride[1] for node in layers_to_combine_hw ]),
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

        """
        LAYER_TYPE.Convolution : fpgaconvnet_pb2.layer.layer_type.CONVOLUTION,
        LAYER_TYPE.InnerProduct : fpgaconvnet_pb2.layer.layer_type.INNER_PRODUCT,
        LAYER_TYPE.Pooling : fpgaconvnet_pb2.layer.layer_type.POOLING,
        LAYER_TYPE.ReLU : fpgaconvnet_pb2.layer.layer_type.RELU,
        LAYER_TYPE.Squeeze : fpgaconvnet_pb2.layer.layer_type.SQUEEZE,
        LAYER_TYPE.Concat : fpgaconvnet_pb2.layer.layer_type.CONCAT,
        LAYER_TYPE.BatchNorm : fpgaconvnet_pb2.layer.layer_type.BATCH_NORM,
        LAYER_TYPE.Split : fpgaconvnet_pb2.layer.layer_type.SPLIT,
        LAYER_TYPE.AveragePooling : fpgaconvnet_pb2.layer.layer_type.AVERAGE_POOLING,
        LAYER_TYPE.EltWise: fpgaconvnet_pb2.layer.layer_type.ELTWISE,
        LAYER_TYPE.NOP : fpgaconvnet_pb2.layer.layer_type.SQUEEZE,
        """

    def seperate(self, layer_type, discriminate=[], num_layers=-1):
        pass

    def apply_random_transform(self, latency_node):
        pass

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
