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

import fpgaconvnet.optimiser.transforms as transforms
from fpgaconvnet.optimiser.transforms.helper import get_all_layers
import itertools

LATENCY   =0
THROUGHPUT=1

START_LOOP=1

@dataclass
class GreedyPartition(Solver):
    coarse_in_first: list = field(default_factory=list)
    merge_ongoing: bool = False

    def reset_partition(self, partition_index):
        partition = self.net.partitions[partition_index]
        partition.remove_squeeze()
        transforms.remove_weights_reloading_transform(partition)
        partition.need_optimise = True

        for node in partition.graph.nodes():
            partition.graph.nodes[node]["hw"].coarse_in = 1
            partition.graph.nodes[node]["hw"].coarse_out = 1
            if partition.graph.nodes[node]["type"] == LAYER_TYPE.Convolution:
                partition.graph.nodes[node]["hw"].coarse_group = 1
                partition.graph.nodes[node]["hw"].fine = 1

    def merge_memory_bound_partitions(self):
        print("resolving memory bound partitions")
        reject_list = []
        self.merge_ongoing = True

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
                horizontal_merges = transforms.get_all_horizontal_merges(self.net, partition_index)
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

            horizontal_merges = transforms.get_all_horizontal_merges(self.net, partition_index)

            if horizontal_merges[0] and partition_index in output_memory_bound:
                self.reset_partition(horizontal_merges[0][0])
                self.reset_partition(horizontal_merges[0][1])
                transforms.apply_max_weights_reloading(self.net.partitions[horizontal_merges[0][0]])
                transforms.merge_horizontal(self.net, *horizontal_merges[0])
                current_merge = horizontal_merges[0]

            elif horizontal_merges[1] and partition_index in input_memory_bound:
                self.reset_partition(horizontal_merges[1][0])
                self.reset_partition(horizontal_merges[1][1])
                transforms.apply_max_weights_reloading(self.net.partitions[horizontal_merges[1][0]])
                transforms.merge_horizontal(self.net, *horizontal_merges[1])
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

    def balance_coarse(self, partition_index):
        net = copy.deepcopy(self.net)
        try:
            self.check_resources()
            self.check_constraints()
        except AssertionError as error:
            partition = self.net.partitions[partition_index] 
            feasible_layers = get_all_layers(partition.graph, LAYER_TYPE.Convolution)
            if len(feasible_layers) == 1:
                node = feasible_layers[0]
                current_coarse_in = partition.graph.nodes[node]['hw'].coarse_in
                current_coarse_out = partition.graph.nodes[node]['hw'].coarse_out
                coarse_in_feasible = partition.graph.nodes[node]['hw'].get_coarse_in_feasible()
                coarse_out_feasible = partition.graph.nodes[node]['hw'].get_coarse_out_feasible()   

                all_coarse_combination = list(itertools.product(coarse_in_feasible, coarse_out_feasible))
                all_coarse_combination = list(filter(lambda x: x[0] * x[1] == current_coarse_in*current_coarse_out, all_coarse_combination))
                all_coarse_combination.remove((current_coarse_in, current_coarse_out))

                prev_latency = partition.graph.nodes[node]['hw'].latency()
                prev_rsc = partition.get_resource_usage()

                for comb in all_coarse_combination:
                    partition = self.net.partitions[partition_index]
                    partition.graph.nodes[node]['hw'].coarse_in = comb[0]
                    partition.graph.nodes[node]['hw'].coarse_out = comb[1]
                    partition.update()
                    current_latency = partition.graph.nodes[node]['hw'].latency()
                    current_rsc = partition.get_resource_usage()

                    if current_rsc["LUT"] >= prev_rsc["LUT"] or current_latency > prev_latency:
                        self.net = net
                    else:
                        break

    def adjust_squeeze(self, partition_index):
        net = copy.deepcopy(self.net)
        try:
            self.check_resources()
            self.check_constraints()
        except AssertionError as error:
            partition = self.net.partitions[partition_index]
            prev_rsc = partition.get_resource_usage()
            prev_cycle = partition.get_cycle()
            partition.reduce_squeeze_fanout()
            partition.update()
            current_cycle = partition.get_cycle()
            current_rsc = partition.get_resource_usage()
            if current_rsc["LUT"] >= prev_rsc["LUT"] or current_cycle > prev_cycle:
                self.net = net         
    
    def empirical_solver(self, partition_index, optimiser_phase, fast_mode = True):
        net = copy.deepcopy(self.net)
        reject_list = []
        while True:
            skip_second_slowest_node = True#self.merge_ongoing and optimiser_phase in [transforms.apply_more_coarse_favour_coarse_in, transforms.apply_more_coarse_favour_coarse_out]
            # skipping avoids some cases of local minima
            status, node = optimiser_phase(self.net.partitions[partition_index], reject_list, skip_second_slowest_node)
            if not status:
                break
            self.net.update_partitions()
            self.allocate_uram()
            self.adjust_squeeze(partition_index)
            self.balance_coarse(partition_index)

            try:
                self.check_resources()
                self.check_constraints()

                net = copy.deepcopy(self.net)
            except AssertionError as error:
                if fast_mode: # break to save optimisation time
                    break
                else:
                    reject_list.append(node)
                    self.net = copy.deepcopy(net)

        self.net = net
        self.net.update_partitions()
        #print(partition_index,"ultilised DSP:", self.partitions[partition_index].get_resource_usage()['DSP'])

    def get_max_dsp_combination(self, partition):
        partition = copy.deepcopy(partition)
        transforms.remove_weights_reloading_transform(partition)

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

            all_dsp_combination = list(filter(lambda x: x < (self.net.rsc_allocation*self.net.platform.get_dsp()), partition_dsp_combination))

            max_dsp_combination = max(all_dsp_combination)
        else:
            print("Might lead to program hanging. Abort getting max dsp combination")
            max_dsp_combination = int(self.net.rsc_allocation*self.net.platform.get_dsp())

        return max_dsp_combination

    def get_all_wr_feasible(self, partition):
        partition = copy.deepcopy(partition)

        transforms.remove_weights_reloading_transform(partition)
        partition.graph.nodes[partition.wr_layer]['hw'].coarse_out = 1
        all_wr_feasible = partition.graph.nodes[partition.wr_layer]['hw'].get_weights_reloading_feasible()

        return all_wr_feasible

    def allocate_uram(self):
        if self.net.platform.get_uram() == 0:
            return
        # reset all uram flags
        for partition in self.net.partitions:
            for layer in graphs.ordered_node_list(partition.graph):
                if partition.graph.nodes[layer]['type'] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
                    partition.graph.nodes[layer]["hw"].use_uram = False

        # balance between bram and uram
        for partition in reversed(self.net.partitions):
            for layer in reversed(graphs.ordered_node_list(partition.graph)):
                if partition.graph.nodes[layer]['type'] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
                    partition_resource_usage = partition.get_resource_usage()
                    if partition_resource_usage['BRAM'] >= self.net.platform.get_bram() * self.net.rsc_allocation:
                        partition.graph.nodes[layer]["hw"].use_uram = True
                    else:
                        break

    def run_solver(self, log=True):
        # update all partitions
        self.net.update_partitions()
        self.allocate_uram()

        # Setup
        cost = self.get_cost()

        start = False

        try:
            self.check_resources()
            self.check_constraints()
            start = True
        except AssertionError as error:
            print("ERROR: Exceeds resource usage")
            return False

        assert "partition" not in self.transforms

        for partition_index in range(len(self.net.partitions)):
            # don't use enumerate, copy.deepcopy creates the new partition object 
            if not self.net.partitions[partition_index].need_optimise:
                continue

            max_dsp = self.get_max_dsp_combination(self.net.partitions[partition_index])

            for phase in [transforms.apply_more_fine, transforms.apply_less_weight_reloading]:
                self.empirical_solver(partition_index, phase)

            if "weights_reloading" in self.transforms and self.net.partitions[partition_index].wr_layer:
                all_wr_feasible = self.get_all_wr_feasible(self.net.partitions[partition_index]) # TODO
                sorted_wr_feasible = np.sort(all_wr_feasible)
                sorted_wr_feasible = list(filter(lambda x: x>=self.net.partitions[partition_index].wr_factor, sorted_wr_feasible))
            else:
                sorted_wr_feasible = [1]

            for wr_factor in sorted_wr_feasible:
                net = copy.deepcopy(self.net)
                # get the current cost
                cost = self.get_cost([partition_index])

                transforms.remove_weights_reloading_transform(self.net.partitions[partition_index])
                self.net.partitions[partition_index].wr_factor = int(wr_factor)
                transforms.apply_weights_reloading_transform(self.net.partitions[partition_index])
                self.net.update_partitions()

                if partition_index in self.coarse_in_first:
                    coarse_phases = [transforms.apply_more_coarse_favour_coarse_in,
                                     transforms.apply_more_coarse_fix_coarse_in]
                else:
                    coarse_phases = [transforms.apply_more_coarse_favour_coarse_out,
                                     transforms.apply_more_coarse_fix_coarse_out]

                for phase in coarse_phases:
                    self.empirical_solver(partition_index,phase)
                    if self.net.partitions[partition_index].get_resource_usage()['DSP'] == max_dsp:
                        break


                if self.get_cost([partition_index]) >= cost:
                    self.net = net

                if self.net.partitions[partition_index].get_resource_usage()['DSP'] == max_dsp:
                    break

                if self.objective != 1:
                    break

            print(partition_index,"single partition cost:",self.get_cost([partition_index]))
            print("ultilised DSP:", self.net.partitions[partition_index].get_resource_usage()['DSP'],
                  "max DSP:", max_dsp)
            self.solver_status()

        return True
