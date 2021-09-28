import json
import copy
import fpgaconvnet_optimiser.tools.graphs as graphs
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

def update(self):

    ## remove auxiliary layers
    self.remove_squeeze()

    ## update streams in and out
    input_node  = graphs.get_input_nodes(self.graph)[0]
    output_node = graphs.get_output_nodes(self.graph)[0]

    self.streams_in = min(self.max_streams_in, self.graph.nodes[input_node]["hw"].streams_in())
    self.streams_out = min(self.max_streams_out, self.graph.nodes[output_node]["hw"].streams_out())

    ## add auxiliary layers
    self.add_squeeze()

    ## update streams in and out
    self.input_nodes = graphs.get_input_nodes(self.graph)
    self.output_nodes = graphs.get_output_nodes(self.graph)

    ## update sizes
    self.size_in  = self.graph.nodes[self.input_nodes[0]]['hw'].size_in()
    self.size_out = self.graph.nodes[self.input_nodes[0]]['hw'].size_out()
    if self.wr_layer != None:
        self.size_wr = self.graph.nodes[self.wr_layer]['hw'].get_parameters_size()['weights']
    else:
        self.size_wr = 0

