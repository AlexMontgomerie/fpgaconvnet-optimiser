import copy
import math
import itertools
import random
import secrets
from dataclasses import dataclass, field

import numpy as np

from fpgaconvnet.tools.layer_enum import  LAYER_TYPE
from fpgaconvnet.models.network import Network

from fpgaconvnet.optimiser.latency.solvers.utils import get_hw_from_dict
import fpgaconvnet.optimiser.solvers.solver

@dataclass
class LatencySolver(fpgaconvnet.optimiser.solvers.solver.Solver):
    runtime_parameters: bool = False

    def __post_init__(self):

        # get the model's dimensionality from the Network
        self.dimensionality = self.net.dimensionality

        # dictionary of layers, keyed by their name
        self.building_blocks = {}
        for node in self.net.graph.nodes:
            self.building_blocks[node] = copy.deepcopy(self.net.graph.nodes[node])
            self.building_blocks[node]["exec_nodes"] = [ node ]

        # combine simple layer types
        self.simple_layer_types = [ LAYER_TYPE.ReLU, LAYER_TYPE.EltWise,
                LAYER_TYPE.Sigmoid, LAYER_TYPE.SiLU, LAYER_TYPE.GlobalPooling ]
        for layer_type in self.simple_layer_types:
            self.combine(layer_type)

    # import shape generation transform functions
    from fpgaconvnet.optimiser.latency.transforms.shapes import apply_random_shape
    from fpgaconvnet.optimiser.latency.transforms.shapes import update_building_block_shape

    # import combine transform functions
    from fpgaconvnet.optimiser.latency.transforms.combine import get_max_attr_of_hw_nodes
    from fpgaconvnet.optimiser.latency.transforms.combine import get_min_attr_of_hw_nodes
    from fpgaconvnet.optimiser.latency.transforms.combine import get_max_attr_of_hw_nodes_multi
    from fpgaconvnet.optimiser.latency.transforms.combine import get_min_attr_of_hw_nodes_multi
    from fpgaconvnet.optimiser.latency.transforms.combine import combine

    # import seperate transform functions
    from fpgaconvnet.optimiser.latency.transforms.seperate import seperate

    # import scheduler functions
    from fpgaconvnet.optimiser.latency.solvers.scheduler import get_convolution_schedule
    from fpgaconvnet.optimiser.latency.solvers.scheduler import get_inner_product_schedule
    from fpgaconvnet.optimiser.latency.solvers.scheduler import get_pooling_schedule
    from fpgaconvnet.optimiser.latency.solvers.scheduler import get_basic_schedule
    from fpgaconvnet.optimiser.latency.solvers.scheduler import get_schedule

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

    def check_building_blocks(self):
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
                        if self.dimensionality == 3:
                            assert self.net.graph.nodes[exec_node]["hw"].kernel_depth <= \
                                    self.building_blocks[hw_node]["hw"].kernel_depth
                        # check channels in and out are greater than all exec nodes
                        # TODO: handle properly in scheduler, and remove here
                        assert self.net.graph.nodes[exec_node]["hw"].channels_in() <= \
                                self.building_blocks[hw_node]["hw"].channels_in()
                        assert self.net.graph.nodes[exec_node]["hw"].channels_out() <= \
                                self.building_blocks[hw_node]["hw"].channels_out()
                case LAYER_TYPE.InnerProduct:
                    # iterate over the execution nodes
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"]:
                        # check channels in and out are greater than all exec nodes
                        # TODO: handle properly in scheduler, and remove here
                        assert self.net.graph.nodes[exec_node]["hw"].channels_in() <= \
                                self.building_blocks[hw_node]["hw"].channels_in()
                        assert self.net.graph.nodes[exec_node]["hw"].channels_out() <= \
                                self.building_blocks[hw_node]["hw"].channels_out()

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
                # add node execution
                total_latency += sum([ get_hw_from_dict(
                    self.building_blocks[hw_node]["type"],
                    param, self.dimensionality).latency() \
                        for param in schedule[exec_node] ])
                # add extra penalty for reconfiguration # TODO: need to tune with real data
                total_latency += 1000 * len(schedule[exec_node])
            else:
                total_latency += len(schedule[exec_node]) * \
                    self.building_blocks[hw_node]["hw"].latency()

        # return the overall latency
        return total_latency

    def get_cost(self):
        return self.evaluate_latency()

    def check_resources(self):
        # get the resources
        resources = self.get_resources()
        # check against board constraints
        assert resources['FF']   <= self.net.platform.get_ff(), "ERROR: FF usage exceeded"
        assert resources['LUT']  <= self.platform.get_lut()   , "ERROR: LUT usage exceeded"
        assert resources['DSP']  <= self.platform.get_dsp()   , "ERROR: DSP usage exceeded"
        assert resources['BRAM'] <= self.platform.get_bram()  , "ERROR: BRAM usage exceeded"

