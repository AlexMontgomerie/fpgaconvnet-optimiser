"""
Input and output channel dimension parallelism of Layers. For a convolution layer, this is how filters are run in parallel

.. note::
    The `coarse_in` and `coarse_out` variables are limited to factors of `channels_in()` and `channels_out()`

.. note::
    For all layers except for `fpgaconvnet_optimser.models.layers.ConvolutionLayer`, `fpgaconvnet_optimser.models.layers.InnerProductLayer` and `fpgaconvnet_optimser.models.layers.SqueezeLayer` must have identical `coarse_in` and `coarse_out`

"""

import random

import fpgaconvnet_optimiser.tools.graphs as graphs
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

transformable_layers = [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]

def apply_random_coarse_layer(self, layer):
    # choose coarse in or coarse out
    coarse_type = random.choice(['coarse_in','coarse_out'])
    # apply coarse folding
    ## coarse in
    if coarse_type == 'coarse_in':
        # choose random coarse in factor
        coarse_in = random.choice(self.graph.nodes[layer]['hw'].get_coarse_in_feasible())
        # if input layer, make sure streams aren't too large 
        if layer in graphs.get_input_nodes(self.graph):
            coarse_in = min(coarse_in, self.max_streams_in) 
        # update coarse folding for both node info and actual layers 
        self.graph.nodes[layer]['hw'].coarse_in = coarse_in
        # check if transformable layer
        if not self.graph.nodes[layer]['type'] in transformable_layers:
            # if not, update both layer info
            self.graph.nodes[layer]['hw'].coarse_out = coarse_in 
    ## coarse out
    if coarse_type == 'coarse_out':
        # choose random coarse out factor
        coarse_out = random.choice(self.graph.nodes[layer]['hw'].get_coarse_out_feasible())
        # if output layer, make sure streams aren't too large 
        if layer in graphs.get_output_nodes(self.graph):
            coarse_out = min(coarse_out, self.max_streams_out) 
        # update coarse folding for both node info and actual layers 
        self.graph.nodes[layer]['hw'].coarse_out = coarse_out
        # check if transformable layer
        if not self.graph.nodes[layer]['type'] in transformable_layers:
            # if not, update both layer info
            self.graph.nodes[layer]['hw'].coarse_in = coarse_out 

def fix_coarse(self):
    # iterate over layers
    for node in self.graph.nodes():
        # check if coarse in is greater than max feasible coarse in
        coarse_in = self.graph.nodes[node]['hw'].coarse_in
        coarse_in_max = self.graph.nodes[node]['hw'].get_coarse_in_feasible()[-1]
        if coarse_in > coarse_in_max:
            self.graph.nodes[node]['hw'].coarse_in = coarse_in_max
        # check if coarse out is greater than max feasible coarse out
        coarse_out = self.graph.nodes[node]['hw'].coarse_out
        coarse_out_max = self.graph.nodes[node]['hw'].get_coarse_out_feasible()[-1]
        if coarse_out > coarse_out_max:
            self.graph.nodes[node]['hw'].coarse_out = coarse_out_max
            
