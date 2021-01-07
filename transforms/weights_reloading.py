import copy
import random

import tools.graphs as graphs
import transforms.helper as helper

from tools.layer_enum import LAYER_TYPE

transformable_layers = [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]

def get_wr_layer(self, partition_index):
    # iterative function to find weights reloading layer
    def _wr_layer(layer):
        if self.partitions[partition_index].graph.nodes[layer]['type'] == LAYER_TYPE.Concat:
            return None
        if self.partitions[partition_index].graph.nodes[layer]['type'] in transformable_layers:
            return layer
        if self.partitions[partition_index].graph.in_degree(layer) == 0:
            return None
        prev_node = graphs.get_prev_nodes(self.partitions[partition_index].graph,layer)[0]
        return _wr_layer( prev_node )
    # start from the end
    output_node = graphs.get_output_nodes(self.partitions[partition_index].graph)[0]
    if ( self.graph.in_degree(output_node) == 0 ) and ( self.graph.nodes[output_node]['type'] in transformable_layers ):
        return output_node
    else:
        return _wr_layer( output_node )

def get_weights_reloading_factors(self, partition_index):
    # weights reloading factors per layer
    wr_factors = {}
    # get wr layer and factor
    wr_layer  = self.partitions[partition_index].wr_layer
    wr_factor = self.partitions[partition_index].wr_factor
    if wr_layer:
        # get nodes before
        layers_before = graphs.get_prev_nodes_all(self.partitions[partition_index].graph, wr_layer)
        for layer in layers_before:
            wr_factors[layer] = [1,1]
        # get wr for wr layer
        wr_factors[wr_layer] = [1,wr_factor]
        # get nodes after
        layers_after = graphs.get_next_nodes_all(self.partitions[partition_index].graph, wr_layer)
        for layer in layers_after:
            wr_factors[layer] = [wr_factor,wr_factor]
    else:
        # iterate over layers
        for layer in self.partitions[partition_index].graph.nodes():
            # assign wr of 1 to both in and out
            wr_factors[layer] = [1,1]
    # return wr factors
    return wr_factors

def apply_random_weights_reloading(self, partition_index):
    # get the weights reloading layer in partition
    wr_layer = self.get_wr_layer(partition_index)
    if wr_layer:
        # choose random weights reloading
        wr_factor = random.choice(self.graph.nodes[wr_layer]['hw'].get_weights_reloading_feasible())
        # update partition weights reloading factor
        self.partitions[partition_index].wr_layer  = wr_layer 
        self.partitions[partition_index].wr_factor = wr_factor
    else:
        # update modules weights reloading factor to 1
        self.partitions[partition_index].wr_layer  = None
        self.partitions[partition_index].wr_factor = 1

def apply_max_weights_reloading(self, partition_index):
    # get the weights reloading layer in partition
    wr_layer = self.get_wr_layer(partition_index)
    if wr_layer:
        # choose random weights reloading
        wr_factor = max(self.graph.nodes[wr_layer]['hw'].get_weights_reloading_feasible())
        # update modules weights reloading factor
        self.partitions[partition_index].wr_layer  = wr_layer 
        self.partitions[partition_index].wr_factor = wr_factor
    else:
        # update modules weights reloading factor to 1
        self.partitions[partition_index].wr_layer  = None
        self.partitions[partition_index].wr_factor = 1

def fix_weights_reloading(self, partition_index):
    # get wr layer and factor
    wr_factor = self.partitions[partition_index].wr_factor
    wr_layer  = self.get_wr_layer(partition_index)
    self.partitions[partition_index].wr_layer = wr_layer
    # skip if wr layer is none
    if self.get_wr_layer(partition_index) == None:
        self.partitions[partition_index].wr_layer  = None
        self.partitions[partition_index].wr_factor = 1
        return 
    # check the wr_layer is correct
    if not wr_layer == self.get_wr_layer(partition_index):
        # reset weights reloading
        self.apply_max_weights_reloading(partition_index)
        return
    # check that wr factor is correct
    if not wr_factor in self.graph.nodes[wr_layer]['hw'].get_weights_reloading_feasible():
        # reset weights reloading
        self.apply_max_weights_reloading(partition_index)
        return

def apply_weights_reloading_transform(self, partition_index):
    # get wr layer and factor
    wr_layer  = self.partitions[partition_index].wr_layer
    wr_factor = self.partitions[partition_index].wr_factor
    if wr_layer:
        # update number of filters in wr layer
        filters = self.graph.nodes[wr_layer]['hw'].filters
        self.partitions[partition_index].graph.nodes[wr_layer]['hw'].filters = int(filters/wr_factor)
        # iterate until the end to update the rest of the channels
        layers_after = graphs.get_next_nodes_all(self.partitions[partition_index].graph, wr_layer)
        for layer in layers_after:
            ## get channels and reduce by wr factor
            channels = self.graph.nodes[layer]['hw'].channels
            self.partitions[partition_index].graph.nodes[layer]['hw'].channels = int(channels/wr_factor)
