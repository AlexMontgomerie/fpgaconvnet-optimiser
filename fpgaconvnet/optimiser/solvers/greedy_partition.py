import sys
import numpy as np
import json
import copy
import random
import math
from dataclasses import dataclass, field

from fpgaconvnet.optimiser.solvers import Solver
import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

import fpgaconvnet.optimiser.transforms.weights_reloading as weights_reloading
import fpgaconvnet.optimiser.transforms.partition as partition
import fpgaconvnet.optimiser.transforms.coarse as coarse
import fpgaconvnet.optimiser.transforms.fine as fine

LATENCY   =0
THROUGHPUT=1

START_LOOP=1

@dataclass
class GreedyPartition(Solver):
    coarse_in_first: list = field(default_factory=list)

    def merge_memory_bound_partitions(self):
        print("resolving memory bound partitions")
        reject_list = []

        while True:

            # cache the network
            net= copy.deepcopy(self.net)

            self.net.update_partitions()
            cost = self.get_cost()

            for i in range(len(self.net.partitions)):
                self.net.partitions[i].need_optimise = False

            input_memory_bound = []
            output_memory_bound = []

            for partition_index, partition in enumerate(self.net.partitions):
                for i in range(len(self.net.partitions)):
                    self.net.partitions[i].remove_squeeze()
                horizontal_merges = partition.get_all_horizontal_merges(self.net, partition_index)
                self.net.update_partitions()

                if horizontal_merges[1]:
                    if horizontal_merges[1] not in reject_list:
                        if partition.is_input_memory_bound() and self.net.partitions[horizontal_merges[1][0]].wr_factor == 1 \
                                or partition.get_latency(self.net.platform.board_freq) < self.net.platform.reconf_time:
                            input_memory_bound.append(partition_index)

                if horizontal_merges[0]:
                    if horizontal_merges[0] not in reject_list:
                        if partition.is_output_memory_bound() and self.net.partitions[horizontal_merges[0][0]].wr_factor == 1 \
                                or partition.get_latency(self.net.platform.board_freq) < self.net.platform.reconf_time:
                            output_memory_bound.append(partition_index)

            memory_bound = input_memory_bound + output_memory_bound
            if len(memory_bound) == 0:
                self.net = net
                break

            # remove all auxiliary layers
            for i in range(len(self.net.partitions)):
                self.net.partitions[i].remove_squeeze()

            ## Choose slowest partition
            partition_latencys = [ self.net.partitions[partition_index].get_latency(
                self.net.platform.board_freq) for partition_index in memory_bound ]
            partition_index = memory_bound[partition_latencys.index(max(partition_latencys))]

            horizontal_merges = self.get_all_horizontal_merges(partition_index)

            if horizontal_merges[0] and partition_index in output_memory_bound:
                self.net.partitions[horizontal_merges[0][0]].reset()
                self.net.partitions[horizontal_merges[0][1]].reset()
                weights_reloading.apply_max_weights_reloading(self.net, horizontal_merges[0][0])
                partition.merge_horizontal(self.net, *horizontal_merges[0])
                current_merge = horizontal_merges[0]

            elif horizontal_merges[1] and partition_index in input_memory_bound:
                self.net.partitions[horizontal_merges[1][0]].reset()
                self.net.partitions[horizontal_merges[1][1]].reset()
                weights_reloading.apply_max_weights_reloading(self.net, horizontal_merges[1][0])
                partition.merge_horizontal(self.net, *horizontal_merges[1])
                current_merge = horizontal_merges[1]
            print(current_merge)
            self.net.update_partitions()
            status = self.run_solver()

            if not status or self.get_cost() >= cost:
                self.net= net
                reject_list.append(current_merge)
                print("reject")
            else:
                for i, merge in enumerate(reject_list):
                    if merge[0] >= current_merge[1]:
                        reject_list[i] = (merge[0]-1,merge[1]-1)
                print("accept")

    def empirical_solver(self, partition_index, phase_name):
        optimiser_phase = getattr(self.net.partitions[partition_index], phase_name)

        net = copy.deepcopy(self.net)
        while optimiser_phase():
            self.net.update_partitions()

            try:
                self.check_resources()
                self.check_constraints()

                net = copy.deepcopy(self.net)
            except AssertionError as error:
                if phase_name not in ["apply_more_coarse_favour_coarse_in", "apply_more_coarse_favour_coarse_out"]:
                    break

        self.net = net
        self.net.update_partitions()
        #print(partition_index,"ultilised DSP:", self.partitions[partition_index].get_resource_usage()['DSP'])

    def get_max_dsp_combination(self, partition_index):
        partition = copy.deepcopy(self.net.partitions[partition_index])
        weights_reloading.remove_weights_reloading_transform(self.net, partition_index)

        partition_dsp_product = []

        for layer in graphs.ordered_node_list(partition.graph):
            layer_dsp_product = []
            node_hw = copy.deepcopy(partition.graph.nodes[layer]['hw'])
            if partition.graph.nodes[layer]['type'] == LAYER_TYPE.Convolution:
                coarse_in_feasible = node_hw.get_coarse_in_feasible()
                coarse_out_feasible = node_hw.get_coarse_out_feasible()
                coarse_group_feasible = node_hw.get_coarse_group_feasible()
                fine_feasible = node_hw.get_fine_feasible()

                for coarse_group in coarse_group_feasible:
                    for coarse_in in coarse_in_feasible:
                        for coarse_out in coarse_out_feasible:
                            for fine in fine_feasible:
                                node_hw.coarse_group = coarse_group
                                node_hw.coarse_in = coarse_in
                                node_hw.coarse_out = coarse_out
                                node_hw.fine = fine
                                node_hw.update()
                                layer_dsp_product.append(node_hw.resource()['DSP'])

            elif partition.graph.nodes[layer]['type'] == LAYER_TYPE.InnerProduct:
                coarse_in_feasible = node_hw.get_coarse_in_feasible()
                coarse_out_feasible = node_hw.get_coarse_out_feasible()

                for coarse_in in coarse_in_feasible:
                    for coarse_out in coarse_out_feasible:
                        node_hw.coarse_in = coarse_in
                        node_hw.coarse_out = coarse_out
                        node_hw.update()
                        layer_dsp_product.append(node_hw.resource()['DSP'])

            if len(layer_dsp_product) > 0:
                layer_dsp_product = list(set(layer_dsp_product))
                partition_dsp_product.append(layer_dsp_product)

        if len(partition_dsp_product) < 5:
            partition_dsp_combination = [0]
            for layer_dsp_product in partition_dsp_product:
                existing_dsp_combination = partition_dsp_combination
                partition_dsp_combination = []
                for dsp_combination in existing_dsp_combination:
                    for dsp in layer_dsp_product:
                        partition_dsp_combination.append(dsp_combination+dsp)
                partition_dsp_combination = list(set(partition_dsp_combination))
                partition_dsp_combination = sorted(partition_dsp_combination)

            all_dsp_combination = list(filter(lambda x: x < (self.rsc_allocation*self.net.platform.get_dsp()), partition_dsp_combination))

            max_dsp_combination = max(all_dsp_combination)
        else:
            print("Might lead to program hanging. Abort getting max dsp combination")
            max_dsp_combination = int(self.rsc_allocation*self.net.platform.get_dsp())

        return max_dsp_combination

    def get_all_wr_feasible(self, partition_index):
        partition = copy.deepcopy(self.net.partitions[partition_index])

        partition.remove_weights_reloading_transform()
        partition.graph.nodes[partition.wr_layer]['hw'].coarse_out = 1
        all_wr_feasible = partition.graph.nodes[partition.wr_layer]['hw'].get_weights_reloading_feasible()

        return all_wr_feasible

    def run_solver(self, log=True):
        # update all partitions
        self.net.update_partitions()

        # Setup
        cost = self.get_cost()

        start = False

        # # use uram to store weight #TODO
        # count =0
        # for partition in reversed(self.net.partitions):
        #     for layer in reversed(graphs.ordered_node_list(partition.graph)):
        #         if partition.graph.nodes[layer]['type'] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
        #             partition_resource_usage = partition.get_resource_usage()
        #             if "URAM" in self.net.platform.resources:
        #                 if partition_resource_usage['BRAM'] / self.platform['constraints']['BRAM'] > partition_resource_usage['URAM'] / self.platform['constraints']['URAM']:
        #                     partition.graph.nodes[layer]["hw"].modules['conv'].uram_weight = True
        #                     count +=1
        #             else:
        #                 break
        try:
            self.check_resources()
            self.check_constraints()
            start = True
        except AssertionError as error:
            print("ERROR: Exceeds resource usage (trying to find valid starting point)")
            bad_partitions = self.get_resources_bad_partitions()

        # Attempt to find a good starting point
        if not start:
            transforms_config = self.transforms_config
            self.transforms_config = self.fix_starting_point_config
            self.get_transforms()

            for i in range(START_LOOP):
                transform = random.choice(self.transforms)
                partition_index = list(bad_partitions.keys())[-1]
                self.apply_transform(transform,partition_index)
                self.net.update_partitions()

                try:
                    self.check_resources()
                    self.check_constraints()
                    break
                except AssertionError as error:
                    bad_partitions = self.get_resources_bad_partitions()

            self.transforms_config = transforms_config
            self.get_transforms()
        try:
            self.check_resources()
            self.check_constraints()
        except AssertionError as error:
            print("ERROR: Exceeds resource usage")
            return False

        assert "partition" not in self.transforms

        for partition_index in range(len(self.net.partitions)):
            if not self.net.partitions[partition_index].need_optimise:
                continue

            max_dsp = self.get_max_dsp_combination(partition_index)

            for phase in ["apply_more_fine", "apply_less_weight_reloading"]:
                self.empirical_optimiser(partition_index, phase)

            all_wr_feasible = self.get_all_wr_feasible(partition_index) # TODO
            sorted_wr_feasible = np.sort(all_wr_feasible)
            sorted_wr_feasible = list(filter(lambda x: x>=self.net.partitions[partition_index].wr_factor, sorted_wr_feasible))

            for wr_factor in sorted_wr_feasible:
                net = copy.deepcopy(self.net)
                # get the current cost
                cost = self.get_cost([partition_index])

                weights_reloading.remove_weights_reloading_transform(self.net, partition_index)
                self.net.partitions[partition_index].wr_factor = int(wr_factor)
                weights_reloading.apply_weights_reloading_transform(self.net, partition_index)
                self.net.update_partitions()

                if partition_index in self.coarse_in_first:
                    coarse_phases = ["apply_more_coarse_favour_coarse_in",
                                     "apply_more_coarse_fix_coarse_in"]
                else:
                    coarse_phases = ["apply_more_coarse_favour_coarse_out",
                                     "apply_more_coarse_fix_coarse_out"]

                for phase in coarse_phases:
                    self.empirical_optimiser(partition_index,phase)
                    if self.net.partitions[partition_index].get_resource_usage()['DSP'] == max_dsp:
                        break


                if self.get_cost([partition_index]) >= cost:
                    self.net = net

                if self.net.partitions[partition_index].get_resource_usage()['DSP'] == max_dsp:
                    break

                if self.objective != 1:
                    break

            print(partition_index,"single partition cost:",self.get_cost([partition_index]))
            print("ultilised DSP:", self.partitions[partition_index].get_resource_usage()['DSP'],
                  "max DSP:", max_dsp)
            self.solver_status()

        return True
