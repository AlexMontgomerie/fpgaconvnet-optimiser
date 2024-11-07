import sys
import numpy as np
import json
import copy
import random
import math
from dataclasses import dataclass, field
from tqdm import tqdm

from fpgaconvnet.optimiser.solvers import Solver
import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

import fpgaconvnet.optimiser.transforms as transforms
from fpgaconvnet.optimiser.transforms.helper import get_all_layers
import itertools

from fpgaconvnet.models.partition import Partition

LATENCY   =0
THROUGHPUT=1

@dataclass
class Increment(Solver):

    def __post_init__(self):
        assert False, "this solver is not yet fully implemented" # TODO

    def increment_partition(self, partition: Partition):

        # values for next level of parallelism
        parallelism_next = {}

        # performance changes
        delta_obj = {}
        delta_obj_per_dsp = {}

        # get the current performance
        curr_obj = partition.get_cycle()

        # get the current dsp usage
        curr_dsp = self.get_partition_resource(partition)["DSP"]

        def calc_delta_obj_per_dsp(diff_obj, diff_dsp):
            # print(diff_obj, diff_dsp)
            if diff_obj > 0 and diff_dsp > 0:
                return float(diff_obj) / float(diff_dsp)
            elif diff_obj > 0 and diff_dsp == 0:
                return diff_obj * 10000
            else:
                return 0

        def eval_next_parallelism(layer, dim):

            # get the current parallelism
            dim_curr = getattr(partition.graph.nodes[layer]['hw'], dim)

            # find the next degree of parallelism
            parallelism_next[layer][dim] = min([d for d in \
                getattr(partition.graph.nodes[layer]['hw'],
                    f"get_{dim}_feasible")() \
                    if d > dim_curr], default=dim_curr)

            # increment the parallelism across this dimension
            setattr(partition.graph.nodes[layer]['hw'],
                    dim, parallelism_next[layer][dim])

            # update the node to reflect performance and resources
            partition.update()
            partition.remove_squeeze()

            # get the difference in performance and resources
            obj_diff = curr_obj - partition.get_cycle()
            dsp_diff = (self.get_partition_resource(partition)["DSP"] - curr_dsp)

            # print(layer, dim, parallelism_next[layer][dim], obj_diff, dsp_diff)

            # update the performance changes
            delta_obj[layer][dim] = obj_diff
            delta_obj_per_dsp[layer][dim] = calc_delta_obj_per_dsp(obj_diff, dsp_diff)

            # reset the parallelism
            setattr(partition.graph.nodes[layer]['hw'], dim, dim_curr)

            # update the node to reflect performance and resources
            partition.graph.nodes[layer]['hw'].update()

        # remove squeeze layers
        partition.remove_squeeze()

        # iterate over nodes in the partition
        # for layer in tqdm(partition.graph.nodes, desc=""):
        for layer in partition.graph.nodes:

            # add the layer to the parallelism and perforamnce changes
            parallelism_next[layer] = {}
            delta_obj[layer] = {}
            delta_obj_per_dsp[layer] = {}

            # different cases for the types of layers
            if partition.graph.nodes[layer]['type'] == LAYER_TYPE.Convolution:

                # evaluate coarse in, coarse out, coarse group, and fine
                eval_next_parallelism(layer, "coarse_in")
                eval_next_parallelism(layer, "coarse_out")
                eval_next_parallelism(layer, "coarse_group")
                eval_next_parallelism(layer, "fine")

            elif partition.graph.nodes[layer]['type'] == LAYER_TYPE.InnerProduct:

                # evaluate coarse in and coarse out
                eval_next_parallelism(layer, "coarse_in")
                eval_next_parallelism(layer, "coarse_out")

            elif partition.graph.nodes[layer]['type'] == LAYER_TYPE.Squeeze:

                # do not change
                pass

            else:

                # evaluate coarse
                eval_next_parallelism(layer, "coarse_in")

        # flatten the objective per dsp delta into a list
        delta_obj_per_dsp_flat = []
        for layer in delta_obj_per_dsp:
            for parallelism_type, delta in delta_obj_per_dsp[layer].items():
                delta_obj_per_dsp_flat.append((layer, parallelism_type, delta))

        # sort the delta per objective
        delta_obj_per_dsp_flat.sort(key=lambda x: x[2], reverse=True)

        # print(delta_obj_per_dsp.keys())
        print(delta_obj_per_dsp_flat[0])

        # update the node with the most efficient parallelism choice
        chosen_layer = delta_obj_per_dsp_flat[0][0]
        chosen_parallelism_type = delta_obj_per_dsp_flat[0][1]
        chosen_factor = parallelism_next[chosen_layer][chosen_parallelism_type]
        setattr(partition.graph.nodes[chosen_layer]['hw'],
                chosen_parallelism_type, chosen_factor)

        # update the partition
        partition.update()

    def run_solver(self):
        for i in range(100):
            self.increment_partition(self.net.partitions[0])
            print(self.net.partitions[0].get_cycle())