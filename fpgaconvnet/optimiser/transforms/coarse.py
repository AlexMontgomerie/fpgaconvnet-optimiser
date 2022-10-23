"""
Input and output channel dimension parallelism of Layers. For a convolution node, this is how filters are run in parallel

.. note::
    The `coarse_in` and `coarse_out` variables are limited to factors of `channels_in()` and `channels_out()`

.. note::
    For all nodes except for `fpgaconvnet_optimser.models.nodes.ConvolutionLayer`, `fpgaconvnet_optimser.models.nodes.InnerProductLayer` and `fpgaconvnet_optimser.models.nodes.SqueezeLayer` must have identical `coarse_in` and `coarse_out`

"""

import random

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

transformable_nodes = [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]

def apply_random_coarse_node(partition, node):
    # choose coarse in or coarse out
    coarse_type = random.choice(['coarse_in','coarse_out'])
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

def fix_coarse(partition):
    # iterate over nodes
    for node in partition.graph.nodes():
        # check if coarse in is greater than max feasible coarse in
        coarse_in = partition.graph.nodes[node]['hw'].coarse_in
        coarse_in_max = partition.graph.nodes[node]['hw'].get_coarse_in_feasible()[-1]
        if coarse_in > coarse_in_max:
            partition.graph.nodes[node]['hw'].coarse_in = coarse_in_max
        # check if coarse out is greater than max feasible coarse out
        coarse_out = partition.graph.nodes[node]['hw'].coarse_out
        coarse_out_max = partition.graph.nodes[node]['hw'].get_coarse_out_feasible()[-1]
        if coarse_out > coarse_out_max:
            partition.graph.nodes[node]['hw'].coarse_out = coarse_out_max

