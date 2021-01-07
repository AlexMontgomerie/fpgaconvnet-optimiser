import random

from tools.layer_enum import LAYER_TYPE

def apply_random_fine_layer(self, partition_index, layer):

    # feasible layers
    feasible_layers = self.conv_layers + self.pool_layers
    
    if layer in feasible_layers:
        # choose random fine
        fine = random.choice(self.partitions[partition_index].graph.nodes[layer]['hw'].get_fine_feasible())
        # update modules fine grain folding factor
        self.partitions[partition_index].graph.nodes[layer]['hw'].fine = fine

def apply_complete_fine_partition(self, partition_index):
    # iterate over layers node info
    for layer in self.partitions[partition_index].graph.nodes():
        # choose to apply to convolution layer only
        if self.partitions[partition_index].graph.nodes[layer]['type'] == LAYER_TYPE.Convolution:
            # choose max fine for convolution layer
            fine = self.partitions[partition_index].graph.nodes[layer]['hw'].get_fine_feasible()[-1]
            self.partitions[partition_index].graph.nodes[layer]['hw'].fine = fine

def apply_complete_fine(self):
    for partition_index in range(len(self.partitions)):
        self.apply_complete_fine_partition(partition_index)
