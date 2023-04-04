"""
Defines the parallelism for the kernel x kernel dot product of the `fpgaconvnet_optimiser.models.modules.Conv` module.

.. note::
    The `fine` parameter is limited to `[1,kernel_size,kernel_size*kernel_size]`
"""

import random
import numpy as np
from fpgaconvnet.optimiser.transforms.helper import get_all_layers
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def apply_random_fine_node(partition, node):

    # feasible nodes
    feasible_nodes = get_all_layers(
            partition.graph, LAYER_TYPE.Convolution)

    # check node can have fine transform applied
    if node in feasible_nodes:
        # choose random fine
        fine = random.choice(partition.graph.nodes[node]['hw'].get_fine_feasible())
        # update modules fine grain folding factor
        partition.graph.nodes[node]['hw'].fine = fine

def apply_complete_fine(partition):
    # iterate over nodes node info
    for node in partition.graph.nodes():
        # choose to apply to convolution node only
        if partition.graph.nodes[node]['type'] == LAYER_TYPE.Convolution:
            # choose max fine for convolution node
            fine = partition.graph.nodes[node]['hw'].get_fine_feasible()[-1]
            partition.graph.nodes[node]['hw'].fine = fine

def apply_more_fine(partition, reject_list=[], skip_second_slowest_node=False):
    # feasible layers
    feasible_layers = get_all_layers(partition.graph, LAYER_TYPE.Convolution)
    feasible_layers = [ layer for layer in feasible_layers if layer not in reject_list ]

    if len(feasible_layers) > 0:
        node_latencys = np.array([ partition.graph.nodes[layer]['hw'].latency() \
            for layer in feasible_layers])

        for node_index in reversed(np.argsort(node_latencys)):
            layer = feasible_layers[node_index]
            current_fine = partition.graph.nodes[layer]['hw'].fine
            fine_feasible = partition.graph.nodes[layer]['hw'].get_fine_feasible()
            if current_fine < fine_feasible[-1]:
                fine_index = fine_feasible.index(current_fine) + 1
                partition.graph.nodes[layer]['hw'].fine = fine_feasible[fine_index]
                partition.graph.nodes[layer]['hw'].update()
                if partition.graph.nodes[layer]['hw'].latency() < node_latencys[node_index]:
                    return True, layer
                else:
                    partition.graph.nodes[layer]['hw'].fine = current_fine
                    partition.graph.nodes[layer]['hw'].update()
            if skip_second_slowest_node:
                break

    return False, None
