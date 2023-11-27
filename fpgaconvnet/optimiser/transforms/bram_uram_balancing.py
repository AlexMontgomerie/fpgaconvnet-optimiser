"""
Balances the usage of BRAM and URAM for the weights of convolutional layers.
"""
import numpy as np
import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

transformable_layers = [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]

def apply_random_bram_uram_balancing(self, partition):
    conv_layers = []
    # reset all flags
    for layer in graphs.ordered_node_list(partition.graph):
        partition.graph.nodes[layer]["hw"].stream_inputs = \
            [False] * len(partition.graph.nodes[layer]["hw"].stream_inputs)
        partition.graph.nodes[layer]["hw"].stream_outputs = \
            [False] * len(partition.graph.nodes[layer]["hw"].stream_outputs)
        if partition.graph.nodes[layer]['type'] in transformable_layers:
            partition.graph.nodes[layer]["hw"].use_uram = False
            partition.graph.nodes[layer]["hw"].stream_weights = 0
            conv_layers.append(layer)

    # update squeeze layers after the reset
    self.net.update_partitions()
    if len(conv_layers) == 0:
        return False

    # balance between bram and uram
    bram_memory_bits = self.platform.get_bram() * 18000
    uram_memory_bits = self.platform.get_uram() * 288000
    total_memory_bits = bram_memory_bits + uram_memory_bits

    for layer in conv_layers:
        memory_type = np.random.choice(['BRAM', 'URAM'], p=[bram_memory_bits/total_memory_bits, uram_memory_bits/total_memory_bits])
        node = partition.graph.nodes[layer]["hw"]
        node.use_uram = memory_type == 'URAM'