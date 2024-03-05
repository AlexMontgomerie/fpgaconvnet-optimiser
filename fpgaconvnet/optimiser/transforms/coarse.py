"""
Input and output channel dimension parallelism of Layers. For a convolution node, this is how filters are run in parallel

.. note::
    The `coarse_in` and `coarse_out` variables are limited to factors of `channels_in()` and `channels_out()`

.. note::
    For all nodes except for `fpgaconvnet_optimser.models.nodes.ConvolutionLayer`, `fpgaconvnet_optimser.models.nodes.InnerProductLayer` and `fpgaconvnet_optimser.models.nodes.SqueezeLayer` must have identical `coarse_in` and `coarse_out`

"""

import random
import numpy as np
from collections.abc import Iterable
import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

transformable_nodes = [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]

def apply_random_coarse_node(partition, node):
    # choose coarse in or coarse out
    coarse_type = random.choice(['coarse_in','coarse_group','coarse_out'])
    # apply coarse folding
    ## coarse in
    if coarse_type == 'coarse_in':
        # choose random coarse in factor
        coarse_in = random.choice(partition.graph.nodes[node]['hw'].get_coarse_in_feasible())
        # update coarse folding for both node info and actual nodes
        partition.graph.nodes[node]['hw'].coarse_in = coarse_in
        # check if transformable node
        if not partition.graph.nodes[node]['type'] in transformable_nodes:
            # if not, update both node info
            partition.graph.nodes[node]['hw'].coarse_out = coarse_in
    ## coarse group
    if coarse_type == 'coarse_group' and partition.graph.nodes[node]['type'] == LAYER_TYPE.Convolution:
        # choose random coarse in factor
        coarse_group = random.choice(partition.graph.nodes[node]['hw'].get_coarse_group_feasible())
        # update coarse folding for both node info and actual nodes
        partition.graph.nodes[node]['hw'].coarse_group = coarse_group
    ## coarse out
    if coarse_type == 'coarse_out':
        # choose random coarse out factor
        coarse_out = random.choice(partition.graph.nodes[node]['hw'].get_coarse_out_feasible())
        # update coarse folding for both node info and actual nodes
        partition.graph.nodes[node]['hw'].coarse_out = coarse_out
        # check if transformable node
        if not partition.graph.nodes[node]['type'] in transformable_nodes:
            # if not, update both node info
            partition.graph.nodes[node]['hw'].coarse_in = coarse_out

def apply_max_coarse(partition):
    # iterate over nodes
    for node in partition.graph.nodes():
        # apply max coarse to each node
        partition.apply_max_coarse_node(partition_index, node)

def apply_max_coarse_node(partition, node):
    # choose max coarse in and out
    coarse_in  = partition.graph.nodes[node]['hw'].get_coarse_in_feasible()[-1]
    coarse_out = partition.graph.nodes[node]['hw'].get_coarse_out_feasible()[-1]
    # update both coarse in and out
    partition.graph.nodes[node]['hw'].coarse_in  = coarse_in
    partition.graph.nodes[node]['hw'].coarse_out = coarse_out
    # update coarse group
    if partition.graph.nodes[node]['type'] == LAYER_TYPE.Convolution:
        coarse_group = partition.graph.nodes[node]['hw'].get_coarse_group_feasible()[-1]
        partition.graph.nodes[node]['hw'].coarse_group = coarse_group

def fix_coarse(partition):
    # iterate over nodes
    for node in partition.graph.nodes():
        # check if coarse in is greater than max feasible coarse in
        coarse_in = partition.graph.nodes[node]['hw'].coarse_in
        # todo: fix multi ports
        if isinstance(coarse_in, Iterable):
            coarse_in = coarse_in[0]
        coarse_in_feasible = partition.graph.nodes[node]['hw'].get_coarse_in_feasible()
        if coarse_in not in coarse_in_feasible:
            partition.graph.nodes[node]['hw'].coarse_in = max(filter(lambda x: x <= coarse_in, coarse_in_feasible))
        
        # check if coarse out is greater than max feasible coarse out
        coarse_out = partition.graph.nodes[node]['hw'].coarse_out
        # todo: fix multi ports
        if isinstance(coarse_out, Iterable):
            coarse_out = coarse_out[0]
        coarse_out_feasible = partition.graph.nodes[node]['hw'].get_coarse_out_feasible()
        if coarse_out not in coarse_out_feasible:
            partition.graph.nodes[node]['hw'].coarse_out = max(filter(lambda x: x <= coarse_out, coarse_out_feasible))

        # check if coarse group is greater than max feasible coarse group
        if partition.graph.nodes[node]['type'] == LAYER_TYPE.Convolution:
            coarse_group = partition.graph.nodes[node]['hw'].coarse_group
            coarse_group_feasible = partition.graph.nodes[node]['hw'].get_coarse_group_feasible()
            if coarse_group not in coarse_group_feasible:
                partition.graph.nodes[node]['hw'].coarse_group = max(filter(lambda x: x <= coarse_group, coarse_group_feasible))

def apply_more_coarse(partition, reject_list, skip_second_slowest_node, coarse_in_first, fix_coarse):
    partition.remove_squeeze()

    node_latencys = np.array([ partition.graph.nodes[layer]['hw'].latency() \
    for layer in graphs.ordered_node_list(partition.graph) ])

    for node_index in reversed(np.argsort(node_latencys, kind='mergesort')):
        layer = graphs.ordered_node_list(partition.graph)[node_index]

        if layer in reject_list:
            continue

        current_coarse_in = partition.graph.nodes[layer]['hw'].coarse_in
        # todo: fix multi ports
        if isinstance(current_coarse_in, Iterable):
            current_coarse_in = current_coarse_in[0]
        current_coarse_out = partition.graph.nodes[layer]['hw'].coarse_out
        # todo: fix multi ports
        if isinstance(current_coarse_out, Iterable):
                current_coarse_out = current_coarse_out[0]
        if partition.graph.nodes[layer]['type'] == LAYER_TYPE.Convolution:
            current_coarse_group = partition.graph.nodes[layer]['hw'].coarse_group
        else:
            current_coarse_group = 1

        current_coarse_product = current_coarse_in * current_coarse_out * current_coarse_group

        coarse_in_feasible = partition.graph.nodes[layer]['hw'].get_coarse_in_feasible()
        coarse_out_feasible = partition.graph.nodes[layer]['hw'].get_coarse_out_feasible()
        if partition.graph.nodes[layer]['type'] == LAYER_TYPE.Convolution and partition.graph.nodes[layer]["hw"].groups != 1:
            coarse_group_feasible = partition.graph.nodes[layer]['hw'].get_coarse_group_feasible()
        else:
            coarse_group_feasible = [1]


        all_coarse_combination = []
        if partition.graph.nodes[layer]['type'] in transformable_nodes:
            for coarse_group in coarse_group_feasible:
                for coarse_in in coarse_in_feasible:
                    for coarse_out in coarse_out_feasible:
                        if fix_coarse:
                            if coarse_in_first:
                                if coarse_group*coarse_in*coarse_out > current_coarse_product \
                                and coarse_group >= current_coarse_group \
                                and coarse_in >= current_coarse_in \
                                and coarse_out == current_coarse_out:
                                    all_coarse_combination.append((coarse_group,coarse_in,coarse_out,coarse_group*coarse_in*coarse_out))
                            else:
                                if coarse_group*coarse_in*coarse_out > current_coarse_product \
                                and coarse_group >= current_coarse_group \
                                and coarse_in == current_coarse_in \
                                and coarse_out >= current_coarse_out:
                                    all_coarse_combination.append((coarse_group,coarse_in,coarse_out,coarse_group*coarse_in*coarse_out))
                        else:
                            if coarse_group*coarse_in*coarse_out > current_coarse_product:
                                all_coarse_combination.append((coarse_group,coarse_in,coarse_out,coarse_group*coarse_in*coarse_out))

        else:
            for coarse_group in coarse_group_feasible:
                for coarse_in in coarse_in_feasible:
                    coarse_out = coarse_in
                    if coarse_group*coarse_in*coarse_out > current_coarse_product \
                    and coarse_group >= current_coarse_group \
                    and coarse_in >= current_coarse_in:
                        all_coarse_combination.append((coarse_group,coarse_in,coarse_out,coarse_group*coarse_in*coarse_out))


        if len(all_coarse_combination) > 0:
            all_coarse_combination = sorted(all_coarse_combination, key=lambda x: (x[3]))
            next_coarse_product = all_coarse_combination[0][3]
            all_coarse_combination = list(filter(lambda x: x[3]==next_coarse_product, all_coarse_combination))
            if coarse_in_first:
                all_coarse_combination = sorted(all_coarse_combination, key=lambda x: (x[2],x[1],x[0]))
            else:
                all_coarse_combination = sorted(all_coarse_combination, key=lambda x: (x[1],x[2],x[0]))

            for selected_coarse_combination in all_coarse_combination:
                if partition.graph.nodes[layer]['type'] == LAYER_TYPE.Convolution:
                    partition.graph.nodes[layer]['hw'].coarse_group = int(selected_coarse_combination[0])
                partition.graph.nodes[layer]['hw'].coarse_in = int(selected_coarse_combination[1])
                partition.graph.nodes[layer]['hw'].coarse_out = int(selected_coarse_combination[2])
                partition.graph.nodes[layer]['hw'].update()
                if partition.graph.nodes[layer]['hw'].latency() < node_latencys[node_index]:
                    return True, layer
                else:
                    partition.graph.nodes[layer]['hw'].coarse_in = current_coarse_in
                    partition.graph.nodes[layer]['hw'].coarse_out = current_coarse_out
                    if partition.graph.nodes[layer]['type'] == LAYER_TYPE.Convolution:
                        partition.graph.nodes[layer]['hw'].coarse_group = current_coarse_group
                    partition.graph.nodes[layer]['hw'].update()
        if skip_second_slowest_node:
            break

    return False, None

def apply_more_coarse_favour_coarse_in(partition, reject_list=[], skip_second_slowest_node=False):
    return apply_more_coarse(partition, reject_list, skip_second_slowest_node, True, False)

def apply_more_coarse_favour_coarse_out(partition, reject_list=[], skip_second_slowest_node=False):
    return apply_more_coarse(partition, reject_list, skip_second_slowest_node, False, False)

def apply_more_coarse_fix_coarse_out(partition, reject_list=[], skip_second_slowest_node=False):
    return apply_more_coarse(partition, reject_list, skip_second_slowest_node, True, True)

def apply_more_coarse_fix_coarse_in(partition, reject_list=[], skip_second_slowest_node=False):
    return apply_more_coarse(partition, reject_list, skip_second_slowest_node,False, True)