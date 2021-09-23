"""
Defines the parallelism for the kernel x kernel dot product of the `fpgaconvnet_optimiser.models.modules.Conv` module.

.. note::
    The `fine` parameter is limited to `[1,kernel_size,kernel_size*kernel_size]`
"""

import random
import fpgaconvnet_optimiser.transforms.helper
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

def apply_random_fine_layer(self, layer):

    # feasible layers
    feasible_layers = fpgaconvnet_optimiser.transforms.helper.get_all_layers(self.graph, LAYER_TYPE.Convolution)

    # check layer can have fine transform applied
    if layer in feasible_layers:
        # choose random fine
        self.graph.nodes[layer]['hw'].fine = random.choice(self.graph.nodes[layer]['hw'].get_fine_feasible())

def apply_complete_fine(self):
    # iterate over layers node info
    for node in self.graph.nodes():
        # choose to apply to convolution layer only
        if self.graph.nodes[node]['type'] == LAYER_TYPE.Convolution:
            # choose max fine for convolution layer
            self.graph.nodes[node]['hw'].fine = self.graph.nodes[node]['hw'].get_fine_feasible()[-1]

