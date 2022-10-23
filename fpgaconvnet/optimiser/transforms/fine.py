"""
Defines the parallelism for the kernel x kernel dot product of the `fpgaconvnet_optimiser.models.modules.Conv` module.

.. note::
    The `fine` parameter is limited to `[1,kernel_size,kernel_size*kernel_size]`
"""

import random
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

