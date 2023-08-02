"""
Defines whether to use skipping_windows in the `fpgaconvnet_optimiser.models.modules.Conv` module.

"""

from fpgaconvnet.optimiser.transforms.helper import get_all_layers
from fpgaconvnet.tools.layer_enum import LAYER_TYPE
import numpy as np

def apply_complete_skipping_windows(partition):
    # iterate over nodes node info
    for node in partition.graph.nodes():
        # choose to apply to convolution node only
        if partition.graph.nodes[node]['type'] == LAYER_TYPE.Convolution:
            # choose max fine for convolution node

            hw = partition.graph.nodes[node]['hw']

            hw.skipping_windows = True
            if len(hw.sparsity) > 0:
            # reject if pointwise or low sparsity
                weights = np.arange(hw.sparsity.shape[1])
                avg_sparsity = np.sum(weights * hw.sparsity, axis = 1)/(hw.sparsity.shape[1] - 1)
                if hw.kernel_rows == 1 and hw.kernel_cols == 1:
                        hw.skipping_windows = False
                        hw.window_sparsity = []
                        hw.sparsity = []
                elif np.mean(avg_sparsity) < 0.1:
                        print("Sparsity too less")
                        hw.skipping_windows = False
                        hw.window_sparsity = []
                        hw.sparsity = []

            else:
                hw.skipping_windows = False
                hw.window_sparsity = []