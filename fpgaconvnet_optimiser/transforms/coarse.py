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
from fpgaconvnet_optimiser.transforms.helper import get_factors

transformable_layers = [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]

def apply_random_coarse_layer(self, layer):
    """
    Applies a random coarse in or coarse out factor to the given layer.

    .. note::
        The input and output layer of the partition are constrained to have `coarse_in` 
        and `coarse_out` factors within `ports_in*(port_width/data_width)` and 
        `ports_out*(port_width/data_width)` respectively.

    Parameters
    ----------
    layer: str
        name of layer to update coarse factor

    """
    # choose coarse in or coarse out
    if self.graph.nodes[layer]['hw'].groups == 1:
        coarse_type = random.choice(['coarse_in','coarse_out'])
    else:
        #coarse_type = random.choice(['coarse_in','coarse_out', 'coarse_group'])
        coarse_type = random.choice(['coarse_group'])
    # apply coarse folding
    ## coarse in
    if coarse_type == 'coarse_in':
        # get all feasible coarse in
        coarse_in_feasible = self.graph.nodes[layer]['hw'].get_coarse_in_feasible()

        if self.graph.in_degree(layer) != 0:
            prev_node = graphs.get_prev_nodes(self.graph,layer)[0]
            if self.graph.nodes[prev_node]['hw'].groups != 1:
                coarse_in_feasible = [ x for x in coarse_in_feasible if (x in get_factors(self.graph.nodes[prev_node]['hw'].groups))]

        # if input layer, make sure streams aren't too large 
        #if layer in graphs.get_input_nodes(self.graph):
        #    coarse_in_feasible = [ x for x in coarse_in_feasible if ((x * self.graph.nodes[layer]['hw'].coarse_group  <= self.max_streams_in) or x == 1)]
        # choose random coarse in factor
        coarse_in = random.choice(coarse_in_feasible)
        # update coarse folding for both node info and actual layers 
        self.graph.nodes[layer]['hw'].update_coarse_in(coarse_in)
    ## coarse out
    if coarse_type == 'coarse_out':
        # get all feasible coarse out 
        coarse_out_feasible = self.graph.nodes[layer]['hw'].get_coarse_out_feasible()

        if self.graph.out_degree(layer) != 0:
            next_node = graphs.get_next_nodes(self.graph,layer)[0]
            if self.graph.nodes[next_node]['hw'].groups != 1:
                coarse_out_feasible = [ x for x in coarse_out_feasible if (x in get_factors(self.graph.nodes[next_node]['hw'].groups))]

        # if output layer, make sure streams aren't too large 
        #if layer in graphs.get_output_nodes(self.graph):
        #    coarse_out_feasible = [ x for x in coarse_out_feasible if ((x * self.graph.nodes[layer]['hw'].coarse_group <= self.max_streams_out) or x == 1)]
        # choose random coarse out factor
        coarse_out = random.choice(coarse_out_feasible)
        # update coarse folding for both node info and actual layers 
        self.graph.nodes[layer]['hw'].update_coarse_out(coarse_out)
    ## coarse group
    if coarse_type == 'coarse_group':
        # get all feasible coarse group 
        coarse_group_feasible = self.graph.nodes[layer]['hw'].get_coarse_group_feasible()
        # if input layer, make sure streams aren't too large 
        #if layer in graphs.get_input_nodes(self.graph):
        #    coarse_group_feasible = [ x for x in coarse_group_feasible if ((x * self.graph.nodes[layer]['hw'].coarse_in  <= self.max_streams_in) or x == 1)]
        # if output layer, make sure streams aren't too large 
        #if layer in graphs.get_output_nodes(self.graph):
        #    coarse_group_feasible = [ x for x in coarse_group_feasible if ((x * self.graph.nodes[layer]['hw'].coarse_out <= self.max_streams_out) or x == 1)]
        # choose random coarse out factor
        coarse_group = random.choice(coarse_group_feasible)
        # update coarse folding for both node info and actual layers 
        self.graph.nodes[layer]['hw'].update_coarse_group(coarse_group)

def fix_coarse(self):
    # iterate over layers
    for node in self.graph.nodes():
        # check if coarse in is greater than max feasible coarse in
        coarse_in = self.graph.nodes[node]['hw'].coarse_in
        coarse_in_max = max(self.graph.nodes[node]['hw'].get_coarse_in_feasible())
        self.graph.nodes[node]['hw'].update_coarse_in(min(coarse_in,coarse_in_max))
        # check if coarse out is greater than max feasible coarse out
        coarse_out = self.graph.nodes[node]['hw'].coarse_out
        coarse_out_max = max(self.graph.nodes[node]['hw'].get_coarse_out_feasible())
        self.graph.nodes[node]['hw'].update_coarse_out(min(coarse_out,coarse_out_max))            
        # check if coarse group is greater than max feasible coarse out
        coarse_group = self.graph.nodes[node]['hw'].coarse_group
        coarse_group_max = max(self.graph.nodes[node]['hw'].get_coarse_group_feasible())
        self.graph.nodes[node]['hw'].update_coarse_group(min(coarse_group,coarse_group_max))
            