"""
Reduces on-chip memory usage by creating partial featuremaps across partition iterations
"""

import copy
import random

import fpgaconvnet_optimiser.tools.graphs as graphs
import fpgaconvnet_optimiser.transforms.helper as helper

from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

import numpy as np

transformable_layers = [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]

def get_wr_layer(self):
    # iterative function to find weights reloading layer
    def _wr_layer(layer):
        if self.graph.nodes[layer]['type'] == LAYER_TYPE.Concat:
            return None
        if self.graph.nodes[layer]['type'] in transformable_layers:
            return layer
        if self.graph.in_degree(layer) == 0:
            return None
        prev_node = graphs.get_prev_nodes(self.graph,layer)[0]
        return _wr_layer( prev_node )
    # start from the end
    output_node = graphs.get_output_nodes(self.graph)[0]
    if ( self.graph.in_degree(output_node) == 0 ) and ( self.graph.nodes[output_node]['type'] in transformable_layers ):
        return output_node
    else:
        return _wr_layer( output_node )

def get_weights_reloading_factors(self):
    # weights reloading factors per layer
    wr_factors = {}
    if wr_layer:
        # get nodes before
        layers_before = graphs.get_prev_nodes_all(self.graph, self.wr_layer)
        for layer in layers_before:
            wr_factors[layer] = [1,1]
        # get wr for wr layer
        wr_factors[self.wr_layer] = [1,self.wr_factor]
        # get nodes after
        layers_after = graphs.get_next_nodes_all(self.graph, self.wr_layer)
        for layer in layers_after:
            wr_factors[layer] = [self.wr_factor,self.wr_factor]
    else:
        # iterate over layers
        for layer in self.graph.nodes():
            # assign wr of 1 to both in and out
            wr_factors[layer] = [1,1]
    # return wr factors
    return wr_factors

def apply_random_weights_reloading(self):
    # get the weights reloading layer in partition
    self.wr_layer = self.get_wr_layer()
    # remove weights reloading transform
    self.remove_weights_reloading_transform()
    if self.wr_layer:
        # choose random weights reloading
        wr_factor = random.choice(self.graph.nodes[self.wr_layer]['hw'].get_weights_reloading_feasible())
        # update partition weights reloading factor
        self.wr_factor = wr_factor
    else:
        # update modules weights reloading factor to 1
        self.wr_factor = 1
    # apply weights reloading transform
    self.apply_weights_reloading_transform()

def apply_max_weights_reloading(self):
    # get the weights reloading layer in partition
    self.wr_layer = self.get_wr_layer()
    # remove weights reloading transform
    self.remove_weights_reloading_transform()
    if self.wr_layer:
        # choose random weights reloading
        wr_factor = max(self.graph.nodes[self.wr_layer]['hw'].get_weights_reloading_feasible())
        # update modules weights reloading factor
        self.wr_factor = wr_factor
    else:
        # update modules weights reloading factor to 1
        self.wr_factor = 1
    # apply weights reloading transform
    self.apply_weights_reloading_transform()

def remove_weights_reloading_transform(self):
    # if there is a wr layer
    if self.wr_layer:
        # update number of filters in wr layer
        filters = self.graph.nodes[self.wr_layer]['hw'].filters
        self.graph.nodes[self.wr_layer]['hw'].filters = filters*self.wr_factor
        # iterate until the end to update the rest of the channels
        layers_after = graphs.get_next_nodes_all(self.graph, self.wr_layer)
        for layer in layers_after:
            ## get channels and reduce by wr factor
            channels = self.graph.nodes[layer]['hw'].channels_in()
            self.graph.nodes[layer]['hw'].channels = channels*self.wr_factor
    # set wr_factor to 1
    self.wr_factor = 1

def apply_weights_reloading_transform(self):
    # if there is a wr layer
    if self.wr_layer:
        # update number of filters in wr layer
        filters = self.graph.nodes[self.wr_layer]['hw'].filters
        self.graph.nodes[self.wr_layer]['hw'].filters = filters//self.wr_factor
        # iterate until the end to update the rest of the channels
        layers_after = graphs.get_next_nodes_all(self.graph, self.wr_layer)
        for layer in layers_after:
            ## get channels and reduce by wr factor
            channels = self.graph.nodes[layer]['hw'].channels_in()
            self.graph.nodes[layer]['hw'].channels = channels//self.wr_factor

    self.fix_coarse()     

def apply_less_weight_reloading(self):
    # get the weights reloading layer in partition
    self.wr_layer = self.get_wr_layer()
    wr_factor = self.wr_factor

    if self.wr_layer and wr_factor > 1:
        # remove weights reloading transform
        self.remove_weights_reloading_transform()
        sorted_wr_feasible = np.sort(self.graph.nodes[self.wr_layer]['hw'].get_weights_reloading_feasible())[::-1]
        wr_factor_index = sorted_wr_feasible.tolist().index(wr_factor) + 1
        self.wr_factor = int(sorted_wr_feasible[wr_factor_index])
        # apply weights reloading transform
        self.apply_weights_reloading_transform()
        return True

    return False
