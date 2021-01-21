from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

def get_all_layers(graph, layer_type):
    layers= []
    for node in graph.nodes():
        if graph.nodes[node]['type'] == layer_type:
            layers.append(node)
    return layers

def get_factors(n):
    return list(set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

