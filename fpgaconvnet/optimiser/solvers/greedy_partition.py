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
    enable_weights_streaming: bool = True
    targets: dict = field(default_factory=lambda: {
    'latency'    :  0.0, 'throughput' : float("inf")})

    def check_targets_met(self):
        # stop the optimiser if targets are already met
        if self.objective == LATENCY:
            return self.net.get_latency(self.platform.board_freq, self.inter_delay()) <= self.targets['latency']
        elif self.objective == THROUGHPUT:
            return self.net.get_throughput(self.platform.board_freq, self.inter_delay()) >= self.targets['throughput']

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
            #if self.check_targets_met(): # slow to run for networks with many nodes
            #    return

            # cache the network
            net= copy.deepcopy(self.net)

            self.update_partitions()
            cost = self.get_cost()

            for i in range(len(self.net.partitions)):
                self.net.partitions[i].need_optimise = False

            input_memory_bound = []
            output_memory_bound = []

            for partition_index, partition in enumerate(self.net.partitions):
                for i in range(len(self.net.partitions)):
                    self.net.partitions[i].remove_squeeze()
                horizontal_merges = transforms.get_all_horizontal_merges(self.net, partition_index)
                self.update_partitions()

                if horizontal_merges[1]:
                    if horizontal_merges[1] not in reject_list:
                        if self.multi_fpga \
                            or partition.is_input_memory_bound() and self.net.partitions[horizontal_merges[1][0]].wr_factor == 1 \
                            or partition.get_latency(self.platform.board_freq) < self.platform.reconf_time:
                            input_memory_bound.append(partition_index)

                if horizontal_merges[0]:
                    if horizontal_merges[0] not in reject_list:
                        if self.multi_fpga \
                            or partition.is_output_memory_bound() and self.net.partitions[horizontal_merges[0][0]].wr_factor == 1 \
                            or partition.get_latency(self.platform.board_freq) < self.platform.reconf_time:
                                
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
                self.platform.board_freq) for partition_index in memory_bound ]
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
            self.update_partitions()
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

    def adjust_coarse(self, partition_index, force_run=True, constrain_resource=True, constrain_latency=True):
        # enumerate all possible coarse combinations
        # Warning: This function can be very slow

        if not force_run:
            partition = self.net.partitions[partition_index]
            partition_resource_usage = partition.get_resource_usage()
            if partition_resource_usage["LUT"] <= self.rsc_allocation * self.platform.get_lut():
                return False

        node_latencys = np.array([ partition.graph.nodes[layer]['hw'].latency() \
            for layer in graphs.ordered_node_list(partition.graph) ])
        node_index = list(reversed(np.argsort(node_latencys)))[0]
        node = graphs.ordered_node_list(partition.graph)[node_index]

        if partition.graph.nodes[node]['type'] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
            current_coarse_in = partition.graph.nodes[node]['hw'].coarse_in
            current_coarse_out = partition.graph.nodes[node]['hw'].coarse_out
            coarse_in_feasible = partition.graph.nodes[node]['hw'].get_coarse_in_feasible()
            coarse_out_feasible = partition.graph.nodes[node]['hw'].get_coarse_out_feasible()   

            all_coarse_combination = list(itertools.product(coarse_in_feasible, coarse_out_feasible))
            all_coarse_combination = list(filter(lambda x: x[0] * x[1] == current_coarse_in*current_coarse_out, all_coarse_combination))
            all_coarse_combination.remove((current_coarse_in, current_coarse_out))

            current_latency = partition.graph.nodes[node]['hw'].latency()
            current_rsc = partition.get_resource_usage()

            for comb in all_coarse_combination:
                net = copy.deepcopy(self.net)
                partition = self.net.partitions[partition_index]
                partition.graph.nodes[node]['hw'].coarse_in = comb[0]
                partition.graph.nodes[node]['hw'].coarse_out = comb[1]
                partition.update()
                new_latency = partition.graph.nodes[node]['hw'].latency()
                new_rsc = partition.get_resource_usage()

                if constrain_resource and new_rsc["LUT"] >= current_rsc["LUT"] \
                    or constrain_latency and new_latency > current_latency:
                    self.net = net
                else:
                    return True
        return False

    def adjust_squeeze(self, partition_index, run_pass=None, constrain_resource=True, constrain_latency=True):
        # eliminate squeeze layers when possible to save resources
        # Warning: This function can hurt the performance

        if run_pass is None:
            try:
                self.check_resources()
                #self.check_constraints() # slow to run for networks with many nodes
                run_pass = False
            except AssertionError as error:
                run_pass = True

            if run_pass:
                net = copy.deepcopy(self.net)
                partition = self.net.partitions[partition_index]
                current_rsc = partition.get_resource_usage()
                current_latency = partition.get_cycle()
                partition.reduce_squeeze_fanout()
                partition.update()
                self.allocate_memory(partition_index)
                new_latency = partition.get_cycle()
                new_rsc = partition.get_resource_usage()
                if constrain_resource and new_rsc["LUT"] >= current_rsc["LUT"] \
                    or constrain_latency and new_latency > current_latency:
                    self.net = net
                else:
                    return True

    def balance_coarse(self, partition_index, force_run=True, constrain_resource=True, constrain_latency=True):
        # balance coarse_in and coarse_out to avoid large squeeze layers which affect frquency
        # this function is similar to adjust_squeeze but it is less aggressive
        # Warning: This function can be very slow

        if not force_run:
            partition = self.net.partitions[partition_index]
            partition_resource_usage = partition.get_resource_usage()
            if partition_resource_usage["LUT"] <= self.rsc_allocation * self.platform.get_lut():
                return False

        net = copy.deepcopy(self.net)
        partition = self.net.partitions[partition_index]
        feasible_layers = get_all_layers(partition.graph, LAYER_TYPE.Convolution)
        feasible_layers += get_all_layers(partition.graph, LAYER_TYPE.InnerProduct)
        if len(feasible_layers) == 0:
            return False
        node_coarse_diff = [ abs(partition.graph.nodes[layer]['hw'].coarse_in - partition.graph.nodes[layer]['hw'].coarse_out)  
            for layer in feasible_layers ]
        node_index = list(reversed(np.argsort(node_coarse_diff)))[0]
        node = feasible_layers[node_index]
        current_coarse_in = partition.graph.nodes[node]['hw'].coarse_in
        current_coarse_out = partition.graph.nodes[node]['hw'].coarse_out
        current_latency = partition.get_cycle()
        coarse_in_feasible = partition.graph.nodes[node]['hw'].get_coarse_in_feasible()
        coarse_out_feasible = partition.graph.nodes[node]['hw'].get_coarse_out_feasible() 
        if current_coarse_in == current_coarse_out:
            return False
        elif current_coarse_in > current_coarse_out:
            coarse_in_feasible = list(filter(lambda x: x < current_coarse_in, coarse_in_feasible))
            coarse_out_feasible = list(filter(lambda x: x > current_coarse_out, coarse_out_feasible))
            if len(coarse_in_feasible) == 0 or len(coarse_out_feasible) == 0:
                return False
            new_coarse_in = list(sorted(coarse_in_feasible))[-1]
            new_coarse_out = list(sorted(coarse_out_feasible))[0]
            if new_coarse_in < new_coarse_out:
                return False # reject cross-over
        else:
            coarse_in_feasible = list(filter(lambda x: x > current_coarse_in, coarse_in_feasible))
            coarse_out_feasible = list(filter(lambda x: x < current_coarse_out, coarse_out_feasible))
            if len(coarse_in_feasible) == 0 or len(coarse_out_feasible) == 0:
                return False
            new_coarse_in = list(sorted(coarse_in_feasible))[0]
            new_coarse_out = list(sorted(coarse_out_feasible))[-1]
            if new_coarse_in > new_coarse_out:
                return False # reject cross-over
        partition.graph.nodes[node]['hw'].coarse_in = new_coarse_in
        partition.graph.nodes[node]['hw'].coarse_out = new_coarse_out
        partition.update()
        self.allocate_memory(partition_index)
        new_latency = partition.get_cycle()
        try:
            self.check_resources()
            resource_valid = True
        except AssertionError as error:
            resource_valid = False
        if constrain_resource and (not resource_valid) \
            or constrain_latency and new_latency > current_latency:
            self.net = net
            return False
        else:
            return True
            
    def empirical_solver(self, partition_index, optimiser_phase, fast_mode = True):
        net = copy.deepcopy(self.net)
        reject_list = []
        changed = False
        while True:
            if self.multi_fpga and partition_index > 0:
                if self.net.partitions[partition_index].get_interval() \
                    <= self.net.get_interval(list(range(partition_index))):
                    return
            #if self.check_targets_met(): # slow to run for networks with many nodes
            #    return

            skip_second_slowest_node = (self.net.batch_size > 1)
            status, node = optimiser_phase(self.net.partitions[partition_index], reject_list, skip_second_slowest_node)
            if not status:
                break
            self.update_partitions()
            pass_status = self.adjust_coarse(partition_index, force_run=False)
            pass_status |= self.balance_coarse(partition_index, force_run=False)
            if not pass_status:
                pass_status |= self.allocate_memory(partition_index)
        
            if self.enable_weights_streaming:
                # due to off-chip bandwidth limitation, the fastest node may be slowed down
                current_latency = net.partitions[partition_index].get_interval() * net.partitions[partition_index].slow_down_factor
                new_latency = self.net.partitions[partition_index].get_interval() * self.net.partitions[partition_index].slow_down_factor
                if new_latency > current_latency:
                    self.net = copy.deepcopy(net)
                    break

            try:
                self.check_resources()
                #self.check_constraints() # slow to run for networks with many nodes

                net = copy.deepcopy(self.net)
                changed = True
            except AssertionError as error:
                if fast_mode: # break to save optimisation time
                    break
                else:
                    reject_list.append(node)
                    self.net = copy.deepcopy(net)

        self.net = net
        self.update_partitions()
        return changed

    def get_all_wr_feasible(self, partition):
        partition = copy.deepcopy(partition)

        transforms.remove_weights_reloading_transform(partition)
        partition.graph.nodes[partition.wr_layer]['hw'].coarse_out = 1
        all_wr_feasible = partition.graph.nodes[partition.wr_layer]['hw'].get_weights_reloading_feasible()

        return all_wr_feasible

    def resolve_offchip_bandwidth_constraint(self, partition_index):
        partition = self.net.partitions[partition_index]

        bw_in = partition.get_bandwidth_in(self.platform.board_freq)
        bw_out = partition.get_bandwidth_out(self.platform.board_freq)
        bw_data_total = sum(bw_in) + sum(bw_out)
        bw_max = self.rsc_allocation*self.platform.get_mem_bw()
        if bw_data_total > bw_max:
            return False

        partition.slow_down_factor = 1.0
        bw_weight = sum(partition.get_bandwidth_weight(self.platform.board_freq))
        partition.slow_down_factor = bw_weight / (bw_max - bw_data_total) + 1e-2 # avoid precision error
        bw_weight = sum(partition.get_bandwidth_weight(self.platform.board_freq))
        assert bw_weight <= bw_max - bw_data_total
        return True

    def allocate_memory(self, partition_index):
        types = [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]
        partition = self.net.partitions[partition_index]
        layers = []
        # reset all flags
        for layer in graphs.ordered_node_list(partition.graph):
            if partition.graph.nodes[layer]['type'] in types:
                partition.graph.nodes[layer]["hw"].use_uram = False
                partition.graph.nodes[layer]["hw"].stream_weights = 0
                layers.append(layer)
        if len(layers) == 0:
            return False

        # balance between bram and uram
        if self.platform.get_uram() == 0:
            return False
        partition_resource_usage = partition.get_resource_usage()
        sorted_layers = sorted(layers, key=lambda layer: partition.graph.nodes[layer]["hw"].weights_ram_usage , reverse=True)
        for layer in sorted_layers:
            node = partition.graph.nodes[layer]["hw"]
            if node.weights_ram_usage == 0:
                continue
            curr_bram_util = partition_resource_usage['BRAM'] / self.platform.get_bram()
            curr_uram_util = partition_resource_usage['URAM'] / self.platform.get_uram()
            node.use_uram = True
            partition_resource_usage = partition.get_resource_usage()
            new_bram_util = partition_resource_usage['BRAM'] / self.platform.get_bram()
            new_uram_util = partition_resource_usage['URAM'] / self.platform.get_uram()
            if new_bram_util < new_uram_util and curr_bram_util > curr_uram_util:
                node.use_uram = False
                break

        if not self.enable_weights_streaming:
            return False
        partition_resource_usage = partition.get_resource_usage()
        curr_bram_util = partition_resource_usage['BRAM'] / self.platform.get_bram()
        curr_uram_util = partition_resource_usage['URAM'] / self.platform.get_uram()
        ram_utilization = self.ram_usage #self.rsc_allocation
        while curr_bram_util > ram_utilization or curr_uram_util > ram_utilization:
            partition_copy = copy.deepcopy(partition)
            def _validate(layer):
                node = partition_copy.graph.nodes[layer]["hw"]
                return node.weights_ram_usage > 0 #node.stream_unit() * node.stream_step(0.1)
            def _compare(layer):
                node = copy.deepcopy(partition.graph.nodes[layer]["hw"])
                node.stream_weights += min(node.weights_ram_usage, node.stream_unit() * node.stream_step(0.1))   
                return node.stream_cycles()
            filtered_layers = list(filter(_validate, layers))
            if len(filtered_layers) == 0: #nothing left to move, double buffer?
                break
            if curr_bram_util <= ram_utilization:
                filtered_layers = list(filter(lambda layer: partition.graph.nodes[layer]["hw"].use_uram, filtered_layers))
            elif curr_uram_util <= ram_utilization:
                filtered_layers = list(filter(lambda layer: not partition.graph.nodes[layer]["hw"].use_uram, filtered_layers))
        
            sorted_layers = sorted(filtered_layers, key=_compare, reverse=True)
            if len(sorted_layers) == 0:
                return False # is there anything left to move?
            layer = sorted_layers[0]
            node = partition.graph.nodes[layer]["hw"] 
            assert node.weights_ram_usage + node.stream_weights == math.ceil(node.weight_array_depth/node.weight_array_unit_depth) * node.weight_array_num * math.ceil(node.weight_array_width/node.weight_array_num/node.weight_array_unit_width)
            node.stream_weights += min(node.weights_ram_usage, node.stream_unit() * node.stream_step(0.1))    
            partition_resource_usage = partition.get_resource_usage()
            curr_bram_util = partition_resource_usage['BRAM'] / self.platform.get_bram()
            curr_uram_util = partition_resource_usage['URAM'] / self.platform.get_uram()

            try:
                self.check_memory_bandwidth()
            except AssertionError as error: 
                pass_status = self.resolve_offchip_bandwidth_constraint(partition_index)
                if not pass_status:
                    self.net.partitions[partition_index] = partition_copy
                    break
        return True
 
    def run_solver(self, log=True):
        # update all partitions
        self.update_partitions()
        for partition_index in range(len(self.net.partitions)):
            self.allocate_memory(partition_index)
            
        # Setup
        cost = self.get_cost()

        start = False

        try:
            self.check_resources()
            self.check_memory_bandwidth()
            #self.check_constraints() # slow to run for networks with many nodes
            start = True
        except AssertionError as error:
            print("ERROR: Exceeds resource usage")
            return False

        assert "partition" not in self.transforms

        for partition_index in range(len(self.net.partitions)):
            # don't use enumerate, copy.deepcopy creates the new partition object 
            if not self.net.partitions[partition_index].need_optimise:
                continue

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
                self.update_partitions()

                if partition_index in self.coarse_in_first:
                    coarse_phases = [transforms.apply_more_coarse_favour_coarse_in,
                                     transforms.apply_more_coarse_fix_coarse_in,
                                     transforms.apply_more_coarse_fix_coarse_out,
                                     transforms.apply_more_coarse_favour_coarse_in]
                else:
                    coarse_phases = [transforms.apply_more_coarse_favour_coarse_out,
                                     transforms.apply_more_coarse_fix_coarse_out,
                                     transforms.apply_more_coarse_fix_coarse_in,
                                     transforms.apply_more_coarse_favour_coarse_in]

                while True:
                    changed = False
                    for phase in coarse_phases:
                        changed = changed or self.empirical_solver(partition_index,phase)
                    if not changed:
                        break

                if self.get_cost([partition_index]) >= cost:
                    self.net = net

                if self.objective != 1:
                    break
            self.balance_coarse(partition_index)
            print(partition_index,"single partition cost:",self.get_cost([partition_index]))
            self.solver_status()
            print("slowdown:", self.net.partitions[0].slow_down_factor)

        return True
