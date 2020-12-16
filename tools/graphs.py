import numpy as np
import scipy
import copy
from numpy.linalg import matrix_rank
import networkx as nx
from networkx.algorithms.dag import ancestors
from networkx.algorithms.dag import descendants

def print_graph(graph):
    for node, edges in graph.adjacency():
        edges = list(edges)
        print(f"{node}\t: {edges}")

def get_input_nodes(graph):
    return [ edge for edge, deg in graph.in_degree() if not deg ]

def get_output_nodes(graph):
    return [ edge for edge, deg in graph.out_degree() if not deg ]

def get_next_nodes(graph, node):
    return list(graph.successors(node))

def get_prev_nodes(graph, node):
    return list(graph.predecessors(node))

def get_next_nodes_all(graph, node):
    return list(descendants(graph, node))

def get_prev_nodes_all(graph, node):
    return list(ancestors(graph, node))

def ordered_node_list(graph): # TODO: make work for parallel networks
    return list( nx.topological_sort(graph) )

def split_graph_horizontal(graph,edge):
    prev_nodes = get_prev_nodes_all(graph,edge[1])
    next_nodes = get_next_nodes_all(graph,edge[0])
    prev_graph = graph.subgraph(prev_nodes).copy()
    next_graph = graph.subgraph(next_nodes).copy()
    return prev_graph, next_graph

def split_graph_vertical(graph,nodes):
    input_node = get_input_nodes(graph)[0]
    # find left side graph
    left_nodes = []
    for node in nodes[0]:
        left_nodes.extend( get_next_nodes_all(graph,node) )
    left_nodes.extend(input_node) 
    left_graph = graph.subgraph(left_nodes).copy()
    # find right side graph
    right_nodes = []
    for node in nodes[1]:
        right_nodes.extend( get_next_nodes_all(graph,node) )
    right_nodes.extend(input_node) 
    right_graph = graph.subgraph(right_nodes).copy()
    return left_graph, right_graph

def merge_graphs_horizontal(graph_prev, graph_next):
    graph = nx.compose(graph_prev,graph_next)
    prev_output_node = get_output_nodes(graph_prev)[0]
    next_input_node  = get_input_nodes(graph_next)[0]
    graph.add_edge(prev_output_node, next_input_node)
    return graph

def merge_graphs_vertical(graph_prev, graph_next):
    pass

def to_json(graph_in):
    graph_out = nx.DiGraph()
    for node in ordered_node_list(graph_in):
        graph_out.add_node(node.replace("/","_"))
        for edge in get_next_nodes(graph_in, node):
            graph_out.add_edge(node.replace("/","_"),edge.replace("/","_"))
    return nx.jit_data(graph_out)

def from_json(data):
    return nx.jit_graph(data,create_using=nx.DiGraph())

def view_graph(graph,filepath):
    _, name = os.path.split(filepath)
    g = pydot.Dot(graph_type='digraph')
    g.set_node_defaults(shape='record')
    for node in graph:
        #if graph.nodes[node]['type'] == 'Input':
        #    node_type = 'INPUT'
        #    rows      = graph.nodes[node]['rows']
        #    cols      = graph.nodes[node]['cols']
        #    channels  = graph.nodes[node]['channels']
        if graph.nodes[node]['type'] == LAYER_TYPE.Concat:
            layer_info = graph.nodes[node]['hw'].layer_info()
            node_type  = layer_info['type']            
            rows       = layer_info['rows']            
            cols       = layer_info['cols']            
            channels   = str(layer_info['channels'])
        else:
            layer_info = graph.nodes[node]['hw'].layer_info()
            node_type  = layer_info['type']            
            rows       = layer_info['rows']            
            cols       = layer_info['cols']            
            channels   = layer_info['channels']            
        g.add_node(pydot.Node(node,
            label="{{ {node}|type: {type} \n dim: [{rows}, {cols}, {channels}]  }}".format(
            node=node,
            type=node_type,
            rows=rows,
            cols=cols,
            channels=channels)))
        for edge in graph[node]:
            #g.add_edge(pydot.Edge(node,edge,splines="ortho"))
            g.add_edge(pydot.Edge(node,edge,splines="line"))
    g.write_png('outputs/images/'+name+'.png')


###########################################


"""
def check_graphs_equivalent(graph_a,graph_b):
    # check all nodes there
    node_list_a = get_node_list(graph_a) 
    node_list_b = get_node_list(graph_b) 
    if not set(node_list_a) == set(node_list_b):
        return False
    # check all edges there
    edge_list_a = get_edge_list(graph_a) 
    edge_list_b = get_edge_list(graph_b) 
    if not set(edge_list_a) == set(edge_list_b):
        return False
    return True

def split_graph_horizontal(graph,edge):
    graph_a = {}
    graph_b = {}
    nodes_before = get_nodes_before(graph,edge[0])
    nodes_after  = get_nodes_after(graph,edge[0])
    # add all nodes before to graph a
    for node in nodes_before:
        graph_a[node] = graph[node]
    # add final node of graph a
    graph_a[edge[0]] = []
    # add all nodes after to graph a
    for node in nodes_after:
        graph_b[node] = graph[node]
    # return both graphs
    return graph_a, graph_b

def split_graph_vertical(graph,nodes):
    input_node  = get_first_node(graph)
    output_node = get_last_node(graph)
    graph_a = { input_node : nodes[0] }
    graph_b = { input_node : nodes[1] }
    # iterative function
    def _iterate_graph(graph_ref,graph_new,node):
        graph_new[node] = graph_ref[node]
        if graph_ref[node]:
            return _iterate_graph(graph_ref,graph_new,graph_ref[node][0])
    # iterate for each graph
    for branch in graph_a[input_node]:
        _iterate_graph(graph,graph_a,branch)
    for branch in graph_b[input_node]:
        _iterate_graph(graph,graph_b,branch)
    del graph_a[output_node]
    del graph_b[output_node]
    graph_a[output_node] = []
    graph_b[output_node] = []
    return graph_a, graph_b

def merge_graphs_horizontal(graph_a,graph_b):
    # get the connecting edge
    node_from = get_last_node(graph_a)
    node_to   = get_first_node(graph_b)
    # return the combined graph with the updated edge
    graph_a[node_from] = [node_to]
    graph_a.update(graph_b)
    return graph_a   

def merge_graphs_vertical(graph_a,graph_b):
    # update input edge 
    input_node = get_first_node(graph_a)
    graph_b[input_node].extend(graph_a[input_node])
    # update graphs
    graph_a.update(graph_b)
    return graph_a

def add_node(graph,start_node,new_node,end_node):
    graph[start_node].append(new_node)
    graph[start_node].remove(end_node)
    graph[new_node] = [end_node]

def remove_node(graph,start_node,old_node,end_node):
    graph[start_node].append(end_node)
    graph[start_node].remove(old_node)
    del graph[old_node]

"""

