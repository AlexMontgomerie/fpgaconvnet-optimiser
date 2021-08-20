import json
import copy
import fpgaconvnet_optimiser.tools.graphs as graphs
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

def update_modules(self):
    for layer in self.graph.nodes():
        self.graph.nodes[layer]['hw'].update()

def update_coefficients(self):
    for node in self.graph.nodes():
        self.graph.nodes[node]['hw'].load_coef()
        
def update_bitwidth(self, data_width):
    for node in self.graph.nodes():
        if self.graph.nodes[node]['type'] == LAYER_TYPE.Convolution:
            self.graph.nodes[node]['hw'].data_width = data_width
            self.graph.nodes[node]['hw'].update()
            
    

