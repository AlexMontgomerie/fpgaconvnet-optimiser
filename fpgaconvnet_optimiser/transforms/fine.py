"""
Defines the parallelism for the kernel x kernel dot product of the `fpgaconvnet_optimiser.models.modules.Conv` module. 

.. note::
    The `fine` parameter is limited to `[1,kernel_size,kernel_size*kernel_size]` 
"""

import random
import fpgaconvnet_optimiser.transforms.helper
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE
import fpgaconvnet_optimiser.tools.graphs as graphs
import numpy as np

def apply_random_fine_layer(self, layer):

    # feasible layers
    feasible_layers = fpgaconvnet_optimiser.transforms.helper.get_all_layers(self.graph, LAYER_TYPE.Convolution)

    # check layer can have fine transform applied
    if layer in feasible_layers:
        # choose random fine
        fine = random.choice(self.graph.nodes[layer]['hw'].get_fine_feasible())
        # update modules fine grain folding factor
        self.graph.nodes[layer]['hw'].fine = fine

def apply_complete_fine(self):
    # iterate over layers node info
    for layer in self.graph.nodes():
        # choose to apply to convolution layer only
        if self.graph.nodes[layer]['type'] == LAYER_TYPE.Convolution:
            # choose max fine for convolution layer
            fine = self.graph.nodes[layer]['hw'].get_fine_feasible()[-1]
            self.graph.nodes[layer]['hw'].fine = fine


def apply_more_fine(self):

    # feasible layers
    feasible_layers = fpgaconvnet_optimiser.transforms.helper.get_all_layers(self.graph, LAYER_TYPE.Convolution)

    if len(feasible_layers) > 0:
        node_latencys = np.array([ self.graph.nodes[layer]['hw'].get_latency() \
            for layer in feasible_layers])

        node_index = np.argsort(node_latencys)[-1]
        layer = feasible_layers[node_index]
        if self.graph.nodes[layer]['hw'].fine < self.graph.nodes[layer]['hw'].get_fine_feasible()[-1]:
            fine_index = self.graph.nodes[layer]['hw'].get_fine_feasible().index(self.graph.nodes[layer]['hw'].fine) + 1
            self.graph.nodes[layer]['hw'].fine = self.graph.nodes[layer]['hw'].get_fine_feasible()[fine_index]
            return True
    
    return False