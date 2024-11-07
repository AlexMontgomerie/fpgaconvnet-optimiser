import copy
import itertools
import pickle
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import fpgaconvnet.tools.graphs as graphs
import numpy as np
from fpgaconvnet.models.layers.utils import stream_buffer
from fpgaconvnet.tools.layer_enum import LAYER_TYPE
from tabulate import tabulate

import fpgaconvnet.optimiser.transforms as transforms
import wandb
from fpgaconvnet.optimiser.solvers import Solver

LATENCY   =0
THROUGHPUT=1
START_LOOP=1

@dataclass
class GreedyPartition(Solver):
    coarse_in_first: list = field(default_factory=list)
    merge_ongoing: bool = False
    targets: dict = field(default_factory=lambda: {
    'latency'    :  0.0, 'throughput' : float("inf")})

    def __post_init__(self):
        if self.wandb_enabled:
            self.pre_merge_log_tbl = wandb.Table(columns=["part", f"part_{'latency' if self.objective == LATENCY else 'throughput'}", "part_optim_time", "slowdown", f"net_{'latency' if self.objective == LATENCY else 'throughput'}", "URAM %", "BRAM %", "DSP %", "LUT %", "FF %", "BW %", "BW"])
            self.final_log_tbl = wandb.Table(columns=["part", f"part_{'latency' if self.objective == LATENCY else 'throughput'}", "part_optim_time", "slowdown", f"net_{'latency' if self.objective == LATENCY else 'throughput'}", "URAM %", "BRAM %", "DSP %", "LUT %", "FF %", "BW %", "BW"])

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
            net_partitions = pickle.loads(pickle.dumps(self.net.partitions))

            self.update_partitions()
            cost = self.get_cost()

            for i in range(len(self.net.partitions)):
                self.net.partitions[i].need_optimise = False

            input_memory_bound = []
            output_memory_bound = []

            for partition_index, partition in enumerate(self.net.partitions):
                # remove all auxiliary layers. This might not needed at the 1st iteration of the loop but its mandatory for the rest of the iterations to avoid errors in get_all_horizontal_merges call
                for i in range(len(self.net.partitions)):
                    self.net.partitions[i].remove_squeeze()
                horizontal_merges = transforms.get_all_horizontal_merges(self.net, partition_index)
                self.update_partitions()

                if horizontal_merges[1]:
                    if horizontal_merges[1] not in reject_list:
                        #if self.multi_fpga \
                        #or partition.is_input_memory_bound() and self.net.partitions[horizontal_merges[1][0]].wr_factor == 1 \
                        #or partition.get_latency(self.platform.board_freq) < self.platform.reconf_time:
                        input_memory_bound.append(partition_index)

                if horizontal_merges[0]:
                    if horizontal_merges[0] not in reject_list:
                        #if self.multi_fpga \
                        #    or partition.is_output_memory_bound() and self.net.partitions[horizontal_merges[0][0]].wr_factor == 1 \
                        #    or partition.get_latency(self.platform.board_freq) < self.platform.reconf_time:
                        output_memory_bound.append(partition_index)

            memory_bound = input_memory_bound + output_memory_bound
            if len(memory_bound) == 0:
                self.net.partitions = net_partitions
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
                current_merge = horizontal_merges[0]
            elif horizontal_merges[1] and partition_index in input_memory_bound:
                current_merge = horizontal_merges[1]

            data = [["Attemting Partition Merging:", f"(Part {current_merge[0] + 1}, Part {current_merge[1] + 1})", "", "Total Partitions:", len(self.net.partitions)]]
            data_table = tabulate(data, headers="firstrow", tablefmt="youtrack")
            print(data_table)

            self.reset_partition(current_merge[0])
            self.reset_partition(current_merge[1])
            transforms.apply_max_weights_reloading(self.net.partitions[current_merge[0]])
            transforms.merge_horizontal(self.net, *current_merge)
            self.update_partitions()
            self.allocate_memory(current_merge[0])
            status = self.run_solver(log_final=True)

            if not status or self.get_cost() >= cost:
                self.net.partitions = net_partitions
                reject_list.append(current_merge)
                data = [["Outcome:", "Merge Rejected"]]
            else:
                for i, merge in enumerate(reject_list):
                    if merge[0] >= current_merge[1]:
                        reject_list[i] = (merge[0]-1,merge[1]-1)
                data = [["Outcome:", "Merge Accepted"]]
            data_table = tabulate(data, headers="firstrow", tablefmt="youtrack")
            print(data_table)

        if self.wandb_enabled:
            wandb.log({"final_solver": self.final_log_tbl})

    def validate_partition_resource(self, partition, partition_index):
        try:
            self.check_partition_resources(partition, partition_index)
            valid = True
        except AssertionError as error:
            valid = False
        return valid

    def optimise_coarse(self, partition_index):
        def _get_rsc_bottleneck(partition_resource_usage, platform):
            dsp_util = partition_resource_usage['DSP'] / platform.get_dsp()
            lut_util = partition_resource_usage['LUT'] / platform.get_lut()
            ff_util = partition_resource_usage['FF'] / platform.get_ff()
            if dsp_util > lut_util and dsp_util > ff_util:
                return 'DSP'
            elif lut_util > ff_util:
                return 'LUT'
            else:
                return 'FF'

        partition = self.net.partitions[partition_index]
        cycles = partition.get_cycle()
        partition_resource_usage = self.get_partition_resource(partition, bram_to_lut=False)
        bottleneck = _get_rsc_bottleneck(partition_resource_usage, self.platform)

        conv_layers = []
        other_layers = []
        for layer in graphs.ordered_node_list(partition.graph):
            if partition.graph.nodes[layer]['type'] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
                conv_layers.append(layer)
            elif partition.graph.nodes[layer]['type'] != LAYER_TYPE.Squeeze:
                other_layers.append(layer)

        # for conv layers, enumerate combinations with same product
        for layer in conv_layers:
            node = partition.graph.nodes[layer]['hw']
            current_coarse_in = node.coarse_in
            current_coarse_out = node.coarse_out
            coarse_in_feasible = node.get_coarse_in_feasible()
            coarse_out_feasible = node.get_coarse_out_feasible()
            candidates = list(itertools.product(coarse_in_feasible, coarse_out_feasible))
            candidates = list(filter(lambda x: x[0] * x[1] == current_coarse_in*current_coarse_out, candidates))
            filtered_candidates = []
            for comb in candidates:
                partition_copy = pickle.loads(pickle.dumps(partition))
                node_copy = partition_copy.graph.nodes[layer]['hw']
                node_copy.coarse_in = comb[0]
                node_copy.coarse_out = comb[1]
                partition_copy.update()
                cycles_copy = partition_copy.get_cycle()
                partition_copy_resource_usage = self.get_partition_resource(partition_copy, bram_to_lut=False)
                if cycles_copy > cycles \
                    or partition_copy_resource_usage[bottleneck] >= partition_resource_usage[bottleneck] \
                    or not self.validate_partition_resource(partition_copy, partition_index):
                    continue
                filtered_candidates.append((comb[0], comb[1], partition_copy_resource_usage[bottleneck]))
            if len(filtered_candidates) == 0:
                continue
            sorted_candidates = sorted(filtered_candidates, key=lambda x: x[2])
            node.coarse_in = sorted_candidates[0][0]
            node.coarse_out = sorted_candidates[0][1]
            partition.update()
            partition_resource_usage = self.get_partition_resource(partition, bram_to_lut=False)
            bottleneck = _get_rsc_bottleneck(partition_resource_usage, self.platform)

        self.allocate_memory(partition_index)
        if bottleneck == 'DSP':
            return

        # for other layers, enumerate all combinations
        for layer in other_layers:
            node = partition.graph.nodes[layer]['hw']
            current_coarse_in = node.coarse_in
            if isinstance(current_coarse_in, Iterable):
                current_coarse_in = current_coarse_in[0]
            coarse_in_feasible = node.get_coarse_in_feasible()
            coarse_in_feasible = list(filter(lambda x: x > current_coarse_in, coarse_in_feasible))
            candidates = list((x, x) for x in coarse_in_feasible)
            filtered_candidates = []
            for comb in candidates:
                partition_copy = pickle.loads(pickle.dumps(partition))
                node_copy = partition_copy.graph.nodes[layer]['hw']
                node_copy.coarse_in = comb[0]
                node_copy.coarse_out = comb[1]
                partition_copy.update()
                cycles_copy = partition_copy.get_cycle()
                partition_copy_resource_usage = self.get_partition_resource(partition_copy, bram_to_lut=False)
                if cycles_copy > cycles \
                    or partition_copy_resource_usage[bottleneck] >= partition_resource_usage[bottleneck] \
                    or not self.validate_partition_resource(partition_copy, partition_index):
                    continue
                filtered_candidates.append((comb[0], comb[1], partition_copy_resource_usage[bottleneck]))
            if len(filtered_candidates) == 0:
                continue
            sorted_candidates = sorted(filtered_candidates, key=lambda x: x[2])
            node.coarse_in = sorted_candidates[0][0]
            node.coarse_out = sorted_candidates[0][1]
            partition.update()
            partition_resource_usage = self.get_partition_resource(partition, bram_to_lut=False)
            bottleneck = _get_rsc_bottleneck(partition_resource_usage, self.platform)

        self.allocate_memory(partition_index)

    def empirical_solver(self, partition_index, optimiser_phase, fast_mode = True):
        net_partitions = pickle.loads(pickle.dumps(self.net.partitions))
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
            if optimiser_phase == transforms.apply_more_fine:
                status, node = optimiser_phase(self.net.partitions[partition_index], reject_list, skip_second_slowest_node, self.sparse_fine_threshold)
            else:
                status, node = optimiser_phase(self.net.partitions[partition_index], reject_list, skip_second_slowest_node)
            if not status:
                break
            self.update_partitions()

            self.allocate_memory(partition_index)
            if self.off_chip_streaming:
                # due to off-chip bandwidth limitation, the fastest node may be slowed down
                current_interval = net_partitions[partition_index].get_interval() * net_partitions[partition_index].slow_down_factor
                new_interval = self.net.partitions[partition_index].get_interval() * self.net.partitions[partition_index].slow_down_factor
                if new_interval > current_interval:
                    self.net.partitions = pickle.loads(pickle.dumps(net_partitions))
                    break

            try:
                self.check_resources()
                #self.check_constraints() # slow to run for networks with many nodes

                net_partitions = pickle.loads(pickle.dumps(self.net.partitions))
                changed = True
            except AssertionError as error:
                if fast_mode: # break to save optimisation time
                    break
                else:
                    reject_list.append(node)
                    self.net.partitions = pickle.loads(pickle.dumps(net_partitions))

        self.net.partitions = net_partitions
        self.update_partitions()
        return changed

    def get_all_wr_feasible(self, partition):
        partition = pickle.loads(pickle.dumps(partition))

        transforms.remove_weights_reloading_transform(partition)
        partition.graph.nodes[partition.wr_layer]['hw'].coarse_out = 1
        all_wr_feasible = partition.graph.nodes[partition.wr_layer]['hw'].get_weights_reloading_feasible()

        return all_wr_feasible

    def resolve_offchip_bandwidth_constraint(self, partition_index, bandwidth_max):
        partition = self.net.partitions[partition_index]
        partition.slow_down_factor = 1.0
        bandwidth_total = partition.get_total_bandwidth(self.platform.board_freq)
        partition.slow_down_factor = bandwidth_total / bandwidth_max + 1e-2 # avoid precision error
        bandwidth_total = partition.get_total_bandwidth(self.platform.board_freq)
        assert bandwidth_total <= bandwidth_max

        return True

    def allocate_memory(self, partition_index):
        types = [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]
        partition = self.net.partitions[partition_index]
        conv_layers = []
        # reset all flags
        for layer in graphs.ordered_node_list(partition.graph):
            partition.graph.nodes[layer]["hw"].stream_inputs = \
                [False] * len(partition.graph.nodes[layer]["hw"].stream_inputs)
            partition.graph.nodes[layer]["hw"].stream_outputs = \
                [False] * len(partition.graph.nodes[layer]["hw"].stream_outputs)
            if partition.graph.nodes[layer]['type'] in types:
                partition.graph.nodes[layer]["hw"].use_uram = False
                partition.graph.nodes[layer]["hw"].stream_weights = 0
                conv_layers.append(layer)
        # update squeeze layers after the reset
        self.net.update_partitions()
        if len(conv_layers) == 0:
            return False

        # balance between bram and uram
        partition_resource_usage = self.get_partition_resource(partition)
        if self.platform.get_uram() > 0 and self.balance_bram_uram:
            sorted_conv_layers = sorted(conv_layers, key=lambda layer: partition.graph.nodes[layer]["hw"].weights_ram_usage , reverse=True)
            for layer in sorted_conv_layers:
                node = partition.graph.nodes[layer]["hw"]
                if node.weights_ram_usage == 0:
                    continue
                curr_bram_util = partition_resource_usage['BRAM'] / self.platform.get_bram()
                curr_uram_util = partition_resource_usage['URAM'] / self.platform.get_uram()
                node.use_uram = True
                partition_resource_usage = self.get_partition_resource(partition)
                new_bram_util = partition_resource_usage['BRAM'] / self.platform.get_bram()
                new_uram_util = partition_resource_usage['URAM'] / self.platform.get_uram()
                if new_bram_util <= new_uram_util and curr_bram_util >= curr_uram_util:
                    node.use_uram = abs(new_bram_util-new_uram_util) < abs(curr_bram_util-curr_uram_util)
                    break
            partition_resource_usage = self.get_partition_resource(partition)
            curr_bram_util = partition_resource_usage['BRAM'] / self.platform.get_bram()
            curr_uram_util = partition_resource_usage['URAM'] / self.platform.get_uram()
        else:
            curr_bram_util = partition_resource_usage['BRAM'] / self.platform.get_bram()
            curr_uram_util = 0

        if not self.off_chip_streaming:
            return False

        candidates = [(layer, "weights", 0) for layer in conv_layers]
        candidates += [(layer, "inputs", 0) for layer in conv_layers]
        for layer in graphs.ordered_node_list(partition.graph):
            if partition.graph.nodes[layer]['type'] in [LAYER_TYPE.Concat, LAYER_TYPE.EltWise]:
                for i in range(partition.graph.nodes[layer]["hw"].ports_in):
                    candidates.append((layer, "inputs", i))

        ram_utilization = self.ram_usage
        while curr_bram_util > ram_utilization or curr_uram_util > ram_utilization:
            bottleneck = "URAM" if curr_bram_util <= ram_utilization else "BRAM"
            partition_copy = pickle.loads(pickle.dumps(partition))

            def _validate(entry):
                layer, data_type, index = entry
                node = partition.graph.nodes[layer]["hw"]
                if data_type == "weights":
                    if bottleneck == "URAM":
                        return node.weights_ram_usage > 0 and node.use_uram
                    else:
                        return node.weights_ram_usage > 0 and not node.use_uram
                else:
                    if bottleneck == "URAM":
                        return False
                    else:
                        return node.inputs_ram_usage[index] > 0
            filtered_candidates = list(filter(_validate, candidates))
            if len(filtered_candidates) == 0: # is there anything left to move?
                return False

            def _step(partition, layer, data_type, index, weight_step_size=0.1):
                node = partition.graph.nodes[layer]["hw"]
                if data_type == "weights":
                    node.stream_weights += min(node.weights_ram_usage, node.stream_unit() * node.stream_step(weight_step_size))
                else:
                    partition.remove_squeeze()
                    node.stream_inputs[index] = True
                    if layer not in graphs.get_input_nodes(partition.graph, allow_multiport=True):
                        prev_layer = graphs.get_prev_nodes(partition.graph,layer)[index]
                        prev_node = partition.graph.nodes[prev_layer]["hw"]
                        for j, l in enumerate(graphs.get_next_nodes(partition.graph,prev_layer)):
                            if l == layer:
                                break
                        prev_node.stream_outputs[j] = True
                    partition.add_squeeze()
            def _compare(entry):
                layer, data_type, index = entry
                partition_copy = pickle.loads(pickle.dumps(partition))
                node_copy = partition_copy.graph.nodes[layer]["hw"]
                _step(partition_copy, layer, data_type, index)
                node = partition.graph.nodes[layer]["hw"]
                delta_on_chip_mem = node.resource()[bottleneck] - node_copy.resource()[bottleneck]
                if data_type == "weights":
                    delta_on_chip_mem += node_copy.stream_buffer()
                delta_off_chip_bw = partition_copy.get_total_bandwidth(self.platform.board_freq) - partition.get_total_bandwidth(self.platform.board_freq)
                assert delta_on_chip_mem >= 0
                assert delta_off_chip_bw > 0
                return delta_on_chip_mem / delta_off_chip_bw

            sorted_candidates = sorted(filtered_candidates, key=_compare, reverse=True)
            layer, data_type, index = sorted_candidates[0]
            node = partition.graph.nodes[layer]["hw"]
            _step(partition, layer, data_type, index)
            partition_resource_usage = self.get_partition_resource(partition)
            curr_bram_util = partition_resource_usage['BRAM'] / self.platform.get_bram()
            curr_uram_util = partition_resource_usage['URAM'] / self.platform.get_uram() if self.platform.get_uram() > 0 else 0

            bandwidth_total = partition.get_total_bandwidth(self.platform.board_freq)
            bandwidth_max = self.rsc_allocation*self.platform.get_mem_bw()
            if bandwidth_total > bandwidth_max:
                pass_status = self.resolve_offchip_bandwidth_constraint(partition_index, bandwidth_max)
                if not pass_status:
                    self.net.partitions[partition_index] = partition_copy
                    break
        return True

    def remove_conv_squeeze(self, partition_index):
        types = [LAYER_TYPE.Convolution]
        partition = self.net.partitions[partition_index]
        layers = []
        for layer in graphs.ordered_node_list(partition.graph):
            if partition.graph.nodes[layer]['type'] in types:
                layers.append(layer)
        partition_resource_usage = self.get_partition_resource(partition)
        curr_lut_util = partition_resource_usage['LUT'] / self.platform.get_lut()
        curr_dsp_util = partition_resource_usage['DSP'] / self.platform.get_dsp()
        if len(layers) > 0 and curr_lut_util > self.rsc_allocation and curr_dsp_util < self.rsc_allocation:
            transforms.fine.apply_complete_fine(partition)
            return True
        else:
            return False

    def run_solver(self, log=True, log_final=False) -> bool:
        # update all partitions
        self.update_partitions()
        for partition_index in range(len(self.net.partitions)):
            self.remove_conv_squeeze(partition_index)
            self.allocate_memory(partition_index)
        # Setup
        cost = self.get_cost()

        start = False

        try:
            self.check_resources()
            #self.check_constraints() # slow to run for networks with many nodes
            start = True
        except AssertionError as error:
            print(f"ERROR: Exceeds resource usage:\n{error}")
            return False

        assert "partition" not in self.transforms

        for partition_index in range(len(self.net.partitions)):
            part_start_time = time.perf_counter()
            # don't use enumerate, copy.deepcopy creates the new partition object
            if not self.net.partitions[partition_index].need_optimise:
                continue

            if self.net.partitions[partition_index].is_sparse():
                sparse_fine_threshold_list = [1.2, 1.01]
            else:
                sparse_fine_threshold_list = [1]

            for sparse_fine_threshold in sparse_fine_threshold_list:
                self.sparse_fine_threshold = sparse_fine_threshold
                for phase in [transforms.apply_more_fine, transforms.apply_less_weight_reloading]:
                    self.empirical_solver(partition_index, phase)
                if "weights_reloading" in self.transforms and self.net.partitions[partition_index].wr_layer:
                    all_wr_feasible = self.get_all_wr_feasible(self.net.partitions[partition_index]) # TODO
                    sorted_wr_feasible = np.sort(all_wr_feasible)
                    sorted_wr_feasible = list(filter(lambda x: x>=self.net.partitions[partition_index].wr_factor, sorted_wr_feasible))
                else:
                    sorted_wr_feasible = [1]
                for wr_factor in sorted_wr_feasible:
                    net_partitions = pickle.loads(pickle.dumps(self.net.partitions))
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
                        if changed:
                            self.optimise_coarse(partition_index)
                        else:
                            break
                    self.allocate_memory(partition_index)
                    new_cost = self.get_cost([partition_index])
                    if new_cost >= cost:
                        self.net.partitions = net_partitions
                    if self.objective != 1:
                        break

            part_opt_time = time.perf_counter() - part_start_time
            self.total_opt_time += part_opt_time
            part_cost = self.get_cost([partition_index]) if self.objective == LATENCY else -self.get_cost([partition_index])
            data = [[f"single partition (Part {partition_index + 1}) cost ({'latency' if self.objective == LATENCY else 'throughput'}):",
                     f"{part_cost:.4f}",
                     "",
                     "slowdown:",
                     self.net.partitions[partition_index].slow_down_factor,
                     "partition optimisation time:",
                     f"{part_opt_time:.2f} sec",
                     "total optimisation time:",
                     f"{self.total_opt_time:.2f} sec",]]
            data_table = tabulate(data, tablefmt="double_outline")
            print(data_table)
            if self.wandb_enabled:
                self.wandb_log()
                if log_final:
                    self.final_log_tbl.add_data(partition_index+1, part_cost, part_opt_time, self.net.partitions[partition_index].slow_down_factor, -1, -1, -1, -1, -1, -1, -1, -1)
                    self.solver_status(wandb_tbl=self.final_log_tbl)
                else:
                    self.pre_merge_log_tbl.add_data(partition_index+1, part_cost, part_opt_time, self.net.partitions[partition_index].slow_down_factor, -1, -1, -1, -1, -1, -1, -1, -1)
                    self.solver_status(wandb_tbl=self.pre_merge_log_tbl)
            else:
                self.solver_status()

        if self.wandb_enabled:
            if not log_final:
                wandb.log({"pre_merge_solver": self.pre_merge_log_tbl})
        return True
