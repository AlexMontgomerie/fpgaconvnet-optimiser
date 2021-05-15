import json
import copy
import fpgaconvnet_optimiser.tools.graphs as graphs

def update_modules(self):
    for layer in self.graph.nodes():
        self.graph.nodes[layer]['hw'].update()

def update_coefficients(self):
    for node in self.graph.nodes():
        self.graph.nodes[node]['hw'].load_coef()

def update_partition_index(self):
    for partition, id in enumerate(self.partitions):
        partition.set_id(id)