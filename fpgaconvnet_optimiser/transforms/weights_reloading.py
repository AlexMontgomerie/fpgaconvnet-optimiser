"""
Reduces on-chip memory usage by creating partial featuremaps across partition iterations
"""

import copy
import random

import fpgaconvnet_optimiser.tools.graphs as graphs
import fpgaconvnet_optimiser.transforms.helper as helper

from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE, from_onnx_op_type

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
            #print("rm wr layer after layer:", layer)
            #print(channels)
            #print(self.graph.nodes[layer]['type'])
            if self.graph.nodes[layer]['type'] == LAYER_TYPE.If:
                #FIXME assigns a list to work with multiport layers
                self.graph.nodes[layer]['hw'].channels = [channels*self.wr_factor,
                        channels*self.wr_factor]
            else:
                self.graph.nodes[layer]['hw'].channels = channels*self.wr_factor
    # set wr_factor to 1
    self.wr_factor = 1

def apply_weights_reloading_transform(self):
    # if there is a wr layer
    if self.wr_layer:
        # update number of filters in wr layer
        filters = self.graph.nodes[self.wr_layer]['hw'].filters
        self.graph.nodes[self.wr_layer]['hw'].filters = filters//self.wr_factor
        # make sure the coarse out factor is not larger than the filters
        self.graph.nodes[self.wr_layer]['hw'].coarse_out = min(
            max(self.graph.nodes[self.wr_layer]['hw'].get_coarse_out_feasible()),
            self.graph.nodes[self.wr_layer]['hw'].coarse_out
        )
        # iterate until the end to update the rest of the channels
        layers_after = graphs.get_next_nodes_all(self.graph, self.wr_layer)
        #print("LAYERS AFTER:", layers_after)
        for layer in layers_after:
            ## get channels and reduce by wr factor
            channels = self.graph.nodes[layer]['hw'].channels_in()
            #print("LAYER",layer)
            #print("TYPE",self.graph.nodes[layer]['type'])
            #print("CO",self.graph.nodes[layer]['hw'].coarse_out)
            if self.graph.nodes[layer]['type'] == LAYER_TYPE.Split:
                #FIXME assigns a list to work with multiport layers
                #print("found split to apply wr to")
                self.graph.nodes[layer]['hw'].channels = [channels//self.wr_factor]

                # make sure the coarse out factor is not larger than the filters
                #print(max(self.graph.nodes[layer]['hw'].get_coarse_in_feasible()))
                #print(self.graph.nodes[layer]['hw'].coarse_in[0])
                #print(max(self.graph.nodes[layer]['hw'].get_coarse_out_feasible()))
                co = self.graph.nodes[layer]['hw'].coarse_out[0]

                self.graph.nodes[layer]['hw'].coarse_in = [min(
                    max(self.graph.nodes[layer]['hw'].get_coarse_in_feasible()),
                    self.graph.nodes[layer]['hw'].coarse_in[0]
                )]
                # make sure the coarse out factor is not larger than the filters
                self.graph.nodes[layer]['hw'].coarse_out = [min(
                    max(self.graph.nodes[layer]['hw'].get_coarse_out_feasible()),
                    co
                )]

            else:
                self.graph.nodes[layer]['hw'].channels = channels//self.wr_factor
                # make sure the coarse out factor is not larger than the filters
                self.graph.nodes[layer]['hw'].coarse_in = min(
                    max(self.graph.nodes[layer]['hw'].get_coarse_in_feasible()),
                    self.graph.nodes[layer]['hw'].coarse_in
                )
                # make sure the coarse out factor is not larger than the filters
                self.graph.nodes[layer]['hw'].coarse_out = min(
                    max(self.graph.nodes[layer]['hw'].get_coarse_out_feasible()),
                    self.graph.nodes[layer]['hw'].coarse_out
                )
