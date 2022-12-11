import copy
import math
import itertools
import random
import secrets
from dataclasses import dataclass, field
from collections import Counter, namedtuple

import numpy as np

from fpgaconvnet.tools.layer_enum import  LAYER_TYPE
from fpgaconvnet.models.network import Network

from fpgaconvnet.optimiser.latency.solvers.utils import get_hw_from_dict, get_runtime_latency
import fpgaconvnet.optimiser.solvers.solver

@dataclass
class LatencySolver(fpgaconvnet.optimiser.solvers.solver.Solver):
    runtime_parameters: bool = True
    transforms: list = field(default_factory=lambda:[
        'shape', 'coarse', 'fine', 'combine', 'seperate'])

    def __post_init__(self):

        # get the model's dimensionality from the Network
        self.dimensionality = self.net.dimensionality

        # dictionary of layers, keyed by their name
        self.building_blocks = {}
        for node in self.net.graph.nodes:
            self.building_blocks[node] = copy.deepcopy(self.net.graph.nodes[node])
            self.building_blocks[node]["exec_nodes"] = [ node ]

        # combine simple layer types
        self.simple_layer_types = [ LAYER_TYPE.ReLU, LAYER_TYPE.EltWise, LAYER_TYPE.Pooling,
                LAYER_TYPE.Sigmoid, LAYER_TYPE.SiLU, LAYER_TYPE.GlobalPooling ]
        for layer_type in self.simple_layer_types:
            self.combine(layer_type)

    # import shape generation transform functions
    from fpgaconvnet.optimiser.latency.transforms.shapes import apply_random_shape
    from fpgaconvnet.optimiser.latency.transforms.shapes import apply_inherited_shape
    from fpgaconvnet.optimiser.latency.transforms.shapes import apply_min_shape
    from fpgaconvnet.optimiser.latency.transforms.shapes import update_building_block_shape

    # import combine transform functions
    from fpgaconvnet.optimiser.latency.transforms.combine import get_max_attr_of_hw_nodes
    from fpgaconvnet.optimiser.latency.transforms.combine import get_min_attr_of_hw_nodes
    from fpgaconvnet.optimiser.latency.transforms.combine import get_max_attr_of_hw_nodes_multi
    from fpgaconvnet.optimiser.latency.transforms.combine import get_min_attr_of_hw_nodes_multi
    from fpgaconvnet.optimiser.latency.transforms.combine import combine

    # import seperate transform functions
    from fpgaconvnet.optimiser.latency.transforms.seperate import seperate

    # import fine transform functions
    from fpgaconvnet.optimiser.latency.transforms.fine import apply_random_fine_node

    # import coarse transform functions
    from fpgaconvnet.optimiser.latency.transforms.coarse import apply_random_coarse_node
    from fpgaconvnet.optimiser.latency.transforms.coarse import fix_coarse_node

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

    def solver_status(self, temp, cost=None):
        """
        prints out the current status of the solver.
        """
        # objective
        objectives = [ 'latency', 'throughput', 'power']
        objective  = objectives[self.objective]
        # cost
        if cost is None:
            cost = self.get_cost()
        # Resources
        resources = self.get_resources()
        BRAM = resources['BRAM']
        DSP  = resources['DSP']
        LUT  = resources['LUT']
        FF   = resources['FF']
        print("TEMP:\t {temperature:.4f}, COST:\t {cost:.4f} ({objective}), RESOURCE:\t {DSP}\t{BRAM}\t{FF}\t{LUT}\t(DSP|BRAM|FF|LUT)".format(
            temperature=temp, cost=cost,objective=objective,DSP=int(DSP),BRAM=int(BRAM),FF=int(FF),LUT=int(LUT)))

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

    def get_resources_util(self):
        # get resources
        resources = self.get_resources()
        return {
            "LUT": (resources["LUT"]/self.net.platform.get_lut())*100.0,
            "FF": (resources["FF"]/self.net.platform.get_ff())*100.0,
            "DSP": (resources["DSP"]/self.net.platform.get_dsp())*100.0,
            "BRAM": (resources["BRAM"]/self.net.platform.get_bram())*100.0,
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
                        assert self.net.graph.nodes[exec_node]["hw"].kernel_rows <= \
                                self.building_blocks[hw_node]["hw"].kernel_rows
                        assert self.net.graph.nodes[exec_node]["hw"].kernel_cols <= \
                                self.building_blocks[hw_node]["hw"].kernel_cols
                        if self.dimensionality == 3:
                            assert self.net.graph.nodes[exec_node]["hw"].kernel_depth <= \
                                    self.building_blocks[hw_node]["hw"].kernel_depth
                        # check channels in and out are greater than all exec nodes
                        # TODO: handle properly in scheduler, and remove here
                        # assert self.net.graph.nodes[exec_node]["hw"].channels_in() <= \
                        #         self.building_blocks[hw_node]["hw"].channels_in()
                        # assert self.net.graph.nodes[exec_node]["hw"].channels_out() <= \
                        #         self.building_blocks[hw_node]["hw"].channels_out()
                case LAYER_TYPE.InnerProduct:
                    # iterate over the execution nodes
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"]:
                        pass
                        # check channels in and out are greater than all exec nodes
                        # TODO: handle properly in scheduler, and remove here
                        # assert self.net.graph.nodes[exec_node]["hw"].channels_in() <= \
                        #         self.building_blocks[hw_node]["hw"].channels_in()
                        # assert self.net.graph.nodes[exec_node]["hw"].channels_out() <= \
                        #         self.building_blocks[hw_node]["hw"].channels_out()

    def get_building_block(self, exec_node):
        """
        find the corresponding hardware node for the node to be executed
        """
        for hw_node in self.building_blocks:
            if exec_node in self.building_blocks[hw_node]["exec_nodes"]:
                return hw_node
        raise StopIteration(f"could not find hardware for execution node {exec_node}")

    def evaluate_latency_exec_node(self, schedule, exec_node):

        # find the hardware node
        hw_node = self.get_building_block(exec_node)

        # get the latency of the node for all scheduled executions
        if self.runtime_parameters:

            # initialise latency at zero
            latency = 0

            # remove data types from parameters (HACK)
            for param in schedule[exec_node]:
                param.pop("data_t", None)
                param.pop("input_t", None)
                param.pop("output_t", None)
                param.pop("acc_t", None)
                param.pop("weight_t", None)
                param.pop("kernel_size", None)
                param.pop("stride", None)
                param.pop("pad", None)

            # turn the parameters into something hashable
            param_type = namedtuple('param', schedule[exec_node][0])
            param_tuple = [ param_type(**param) for param in schedule[exec_node] ]

            # get a reduced count of the parameters
            param_cntr = Counter(param_tuple)

            # get the latency for each repeated parameter execution
            for param, repetition in param_cntr.items():
                latency += repetition*get_runtime_latency(
                    self.building_blocks[hw_node]["type"],
                    self.building_blocks[hw_node]["hw"],
                    param._asdict(), self.dimensionality)

            # # add extra penalty for reconfiguration # TODO: need to tune with real data
            # latency += 1000 * len(schedule[exec_node])
        else:
            latency = len(schedule[exec_node]) * \
                self.building_blocks[hw_node]["hw"].latency()

        # return the latency (in clock cycles)
        return latency

    def evaluate_latency(self):
        """
        evaluate the latency for the execution of the graph. Maps the
        nodes of the `self.net.graph` to those of the `self.building_blocks`.
        The latency is the sum of the execution of all these elements.
        """
        # total execution latency
        total_latency = 0

        # get the schedule
        schedule, _ = self.get_schedule()

        # iterate over nodes in the execution graph
        for exec_node in self.net.graph:

            # evaluate the latency of the node for the schedule
            total_latency += self.evaluate_latency_exec_node(schedule, exec_node)

        # latency in ms
        total_latency = total_latency / (self.net.platform.board_freq*1e3)

        # return the overall latency
        return total_latency

    def get_cost(self):
        return self.evaluate_latency()

    def check_resources(self):
        # get the resources
        resources = self.get_resources()
        # check against board constraints
        assert resources['FF']   <= self.net.rsc_allocation*self.net.platform.get_ff()  , "ERROR: FF usage exceeded"
        assert resources['LUT']  <= self.net.rsc_allocation*self.net.platform.get_lut() , "ERROR: LUT usage exceeded"
        assert resources['DSP']  <= self.net.rsc_allocation*self.net.platform.get_dsp() , "ERROR: DSP usage exceeded"
        assert resources['BRAM'] <= self.net.rsc_allocation*self.net.platform.get_bram(), "ERROR: BRAM usage exceeded"

    def apply_transform(self, transform, hw_node, exec_node):

        # switch case across transforms
        match transform:
            case "fine":
                self.apply_random_fine_node(hw_node)
            case "coarse":
                self.apply_random_coarse_node(hw_node)
            case "combine":
                # get the hardware type of exec node
                layer_type = self.net.graph.nodes[exec_node]["type"]
                # combine layers of that type
                self.combine(layer_type)
            case "seperate":
                self.seperate(hw_node)
            case "shape":
                # self.apply_random_shape(hw_node)
                self.apply_inherited_shape(hw_node)
                self.fix_coarse_node(hw_node)

    def report(self):
        """
        generate a report of the time taken to execute each node of the
        execution graph, and how many repetitions occured.
        """

        # get the schedule
        schedule, iteration_space = self.get_schedule()

        # empty dictionary
        report = { "general": {}, "per_layer": {} }

        # iterate over execution nodes
        for exec_node in self.net.graph.nodes():
            # get the latency of the node (in ms)
            latency = self.evaluate_latency_exec_node(schedule, exec_node)
            latency = latency / (self.net.platform.board_freq*1e3)
            # create the report for the node
            report["per_layer"][exec_node] = {
                "type" : str(self.net.graph.nodes[exec_node]["type"]),
                "hw_node" : self.get_building_block(exec_node),
                "latency" : latency,
                "repetitions" : len(schedule[exec_node]),
                "iteration_space" : iteration_space[exec_node],
            }

        # add general information
        total_latency_per_bb = { hw_node: 0 for hw_node in self.building_blocks }
        for val in report["per_layer"].values():
            total_latency_per_bb[val["hw_node"]] += val["latency"]
        report["general"]["building_block_latency"] = total_latency_per_bb

        # add total number of building blocks
        report["general"]["total_building_blocks"] = len(self.building_blocks)

        # return the report
        return report

    def per_layer_table(self):
        """
        generate a report of the time taken to execute each node of the
        execution graph, and how many repetitions occured.
        """

        # get the schedule
        schedule, iteration_space = self.get_schedule()

        table = {
            "exec_node": [],
            "hw_node": [],
            "type": [],
            "latency": [],
            "repetitions": [],
            "iteration_space": [],
        }

        # iterate over execution nodes
        for exec_node in self.net.graph.nodes():
            # get the latency of the node (in ms)
            latency = self.evaluate_latency_exec_node(schedule, exec_node)
            latency = latency / (self.net.platform.board_freq*1e3)
            # update the table
            table["exec_node"].append(exec_node)
            table["hw_node"].append(hw_node)
            table["type"].append(str(self.net.graph.nodes[exec_node]["type"]))
            table["latency"].append(latency)
            table["repetitions"].append(len(schedule[exec_node]))
            table["iteration_space"].append(iteration_space[exec_node])

        # return the report
        return table


    def config(self):
        return { node: hw["hw"].layer_info_dict() for node, hw in self.building_blocks.items() }
