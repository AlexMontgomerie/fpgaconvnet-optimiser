import json
import copy
import tools.graphs as graphs

def update_modules(self):
    for layer in self.graph.nodes():
        self.graph.nodes[layer]['hw'].update()

def update_coefficients(self):
    for node in self.graph.nodes():
        self.graph.nodes[node]['hw'].load_coef()

