"""
Input and output channel dimension parallelism of Layers. For a convolution layer, this is how filters are run in parallel

.. note::
    The `coarse_in` and `coarse_out` variables are limited to factors of `channels_in()` and `channels_out()`

.. note::
    For all layers except for `fpgaconvnet_optimser.models.layers.ConvolutionLayer`, `fpgaconvnet_optimser.models.layers.InnerProductLayer` and `fpgaconvnet_optimser.models.layers.SqueezeLayer` must have identical `coarse_in` and `coarse_out`

"""

import logging
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
    # get possible coarse folding types
    coarse_types = ["coarse_in", "coarse_out"]
    if self.graph.nodes[layer]['type'] == LAYER_TYPE.Convolution:
        coarse_types.append("coarse_group")
    # choose coarse in or coarse out
    coarse_type = random.choice(coarse_types)
    # apply coarse folding
    ## coarse in
    if coarse_type == 'coarse_in':
        # get all feasible coarse in
        coarse_in_feasible = self.graph.nodes[layer]['hw'].get_coarse_in_feasible()
        # if input layer, make sure streams aren't too large
        if layer in graphs.get_input_nodes(self.graph):
            coarse_in_feasible = [ x for x in coarse_in_feasible if x <= self.max_streams_in ]
        # update coarse folding for layer
        coarse_in_factor = random.choice(coarse_in_feasible)
        self.graph.nodes[layer]['hw'].coarse_in = coarse_in_factor
        # log the applied transform
        logging.info(f"applying coarse in factor of {coarse_in_factor} to {layer}")
    ## coarse out
    elif coarse_type == 'coarse_out':
        # get all feasible coarse out
        coarse_out_feasible = self.graph.nodes[layer]['hw'].get_coarse_out_feasible()
        # if output layer, make sure streams aren't too large
        if layer in graphs.get_output_nodes(self.graph):
            coarse_out_feasible = [ x for x in coarse_out_feasible if x <= self.max_streams_out ]
        # choose random coarse out factor
        coarse_out_factor = random.choice(coarse_out_feasible)
        self.graph.nodes[layer]['hw'].coarse_out = coarse_out_factor
        # log the applied transform
        logging.info(f"applying coarse out factor of {coarse_out_factor} to {layer}")
    ## coarse group
    elif coarse_type == 'coarse_group':
        # get all feasible coarse group
        coarse_group_feasible = self.graph.nodes[layer]['hw'].get_coarse_group_feasible()
        # choose random coarse group factor
        coarse_group_factor = random.choice(coarse_group_feasible)
        self.graph.nodes[layer]['hw'].coarse_group = coarse_group_factor
        # log the applied transform
        logging.info(f"applying coarse group factor of {coarse_group_factor} to {layer}")

def fix_coarse(self):
    # iterate over layers
    for node in self.graph.nodes():
        # check if coarse in is greater than max feasible coarse in
        coarse_in = self.graph.nodes[node]['hw'].streams_in
        coarse_in_max = self.graph.nodes[node]['hw'].get_coarse_in_feasible()[-1]
        self.graph.nodes[node]['hw'].coarse_in = min(coarse_in,coarse_in_max)
        # check if coarse out is greater than max feasible coarse out
        coarse_out = self.graph.nodes[node]['hw'].streams_out
        coarse_out_max = self.graph.nodes[node]['hw'].get_coarse_out_feasible()[-1]
        self.graph.nodes[node]['hw'].coarse_out = min(coarse_out,coarse_out_max)
        if self.graph.nodes[node]['hw'].flags["has_groups"]:
            coarse_group = self.graph.nodes[node]['hw'].coarse_group
            coarse_group_max = self.graph.nodes[node]['hw'].get_coarse_group_feasible()[-1]
            self.graph.nodes[node]['hw'].update_coarse_group(min(coarse_group,coarse_group_max))

