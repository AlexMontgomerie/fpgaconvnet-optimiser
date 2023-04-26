"""
Defines whether to use skipping_windows in the `fpgaconvnet_optimiser.models.modules.Conv` module.

"""

from fpgaconvnet.optimiser.transforms.helper import get_all_layers
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def apply_complete_skipping_windows(partition):
    # iterate over nodes node info
    for node in partition.graph.nodes():
        # choose to apply to convolution node only
        if partition.graph.nodes[node]['type'] == LAYER_TYPE.Convolution:
            # choose max fine for convolution node
            partition.graph.nodes[node]['hw'].skipping_windows = True