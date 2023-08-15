"""
Reduces on-chip memory usage by creating partial featuremaps across partition iterations
"""

import copy
import random
import numpy as np

import fpgaconvnet.tools.graphs as graphs

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

import fpgaconvnet.optimiser.transforms.helper as helper
from fpgaconvnet.optimiser.transforms.coarse import fix_coarse

transformable_layers = [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]

def apply_random_weights_reloading(partition):
    # get the weights reloading layer in partition
    partition.wr_layer = partition.get_wr_layer()
    if partition.wr_layer:
        # remove weights reloading transform
        remove_weights_reloading_transform(partition)
        # choose random weights reloading
        wr_factor = random.choice(partition.graph.nodes[partition.wr_layer]['hw'].get_weights_reloading_feasible())
        # update partition weights reloading factor
        partition.wr_factor = wr_factor
    else:
        # update modules weights reloading factor to 1
        partition.wr_factor = 1
    # apply weights reloading transform
    apply_weights_reloading_transform(partition)

def apply_max_weights_reloading(partition):
    # get the weights reloading layer in partition
    partition.wr_layer = partition.get_wr_layer()
    if partition.wr_layer:
        # remove weights reloading transform
        remove_weights_reloading_transform(partition)
        # choose random weights reloading
        wr_factor = max(partition.graph.nodes[partition.wr_layer]['hw'].get_weights_reloading_feasible())
        # update modules weights reloading factor
        partition.wr_factor = wr_factor
    else:
        # update modules weights reloading factor to 1
        partition.wr_factor = 1
    # apply weights reloading transform
    apply_weights_reloading_transform(partition)

def remove_weights_reloading_transform(partition):
    # if there is a wr layer
    if partition.wr_layer:
        # update number of filters in wr layer
        filters = partition.graph.nodes[partition.wr_layer]['hw'].filters
        partition.graph.nodes[partition.wr_layer]['hw'].filters = filters*partition.wr_factor
        # iterate until the end to update the rest of the channels
        layers_after = graphs.get_next_nodes_all(partition.graph, partition.wr_layer)
        for layer in layers_after:
            ## get channels and reduce by wr factor
            channels = partition.graph.nodes[layer]['hw'].channels_in()
            if channels == filters:
                partition.graph.nodes[layer]['hw'].channels = channels*partition.wr_factor
            # otherwise, there could be a flatten layer
    # set wr_factor to 1
    partition.wr_factor = 1
    # fix the coarse factors
    fix_coarse(partition)

def apply_weights_reloading_transform(partition):
    # if there is a wr layer
    if partition.wr_layer:
        # update number of filters in wr layer
        filters = partition.graph.nodes[partition.wr_layer]['hw'].filters
        partition.graph.nodes[partition.wr_layer]['hw'].filters = filters//partition.wr_factor
        # iterate until the end to update the rest of the channels
        layers_after = graphs.get_next_nodes_all(partition.graph, partition.wr_layer)
        for layer in layers_after:
            ## get channels and reduce by wr factor
            channels = partition.graph.nodes[layer]['hw'].channels_in()
            if channels == filters:
                partition.graph.nodes[layer]['hw'].channels = filters//partition.wr_factor
    # fix the coarse factors
    fix_coarse(partition)

def apply_less_weight_reloading(partition, reject_list=[], skip_second_slowest_node=False):
    # get the weights reloading layer in partition
    partition.wr_layer = partition.get_wr_layer()
    wr_factor = partition.wr_factor
    if partition.wr_layer and wr_factor > 1:
        # remove weights reloading transform
        remove_weights_reloading_transform(partition)
        sorted_wr_feasible = np.sort(partition.graph.nodes[partition.wr_layer]['hw'].get_weights_reloading_feasible())[::-1]
        wr_factor_index = sorted_wr_feasible.tolist().index(wr_factor) + 1
        partition.wr_factor = int(sorted_wr_feasible[wr_factor_index])
        # apply the weights reloading transform
        apply_weights_reloading_transform(partition)
        return True, partition.wr_layer
    return False, None
