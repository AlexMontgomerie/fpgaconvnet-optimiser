import numpy as np
import copy

import tools.graphs as graphs
import tools.matrix as matrix

from models.layers.SqueezeLayer import SqueezeLayer

from tools.layer_enum import LAYER_TYPE

def add_squeeze(self, partition_index):
    # find mismatching streams  
    #graphs.print_graph(self.partitions[partition_index]['graph'])
    streams_matrix = matrix.get_streams_matrix(
        self.partitions[partition_index]['graph']
    )
    edge_list = matrix.get_edge_list_matrix(
        self.partitions[partition_index]['graph']
    )
    err = np.sum(streams_matrix,axis=1)
    # iterate over stream difference
    for edge in range(err.shape[0]):
        # mismatch
        if err[edge] != 0:
            # add node to graph
            start_node = edge_list[edge][0]
            end_node   = edge_list[edge][1]
            new_node   = "_".join([start_node,"squeeze",end_node])
            # add node to node info
            self.partitions[partition_index]['graph'].add_node(new_node,type=LAYER_TYPE.Squeeze,
                hw=SqueezeLayer(
                    [
                        self.partitions[partition_index]['graph'].nodes[start_node]['hw'].channels_out(),
                        self.partitions[partition_index]['graph'].nodes[start_node]['hw'].rows_out(),
                        self.partitions[partition_index]['graph'].nodes[start_node]['hw'].cols_out()
                    ], 
                    self.partitions[partition_index]['graph'].nodes[start_node]['hw'].coarse_out,
                    self.partitions[partition_index]['graph'].nodes[end_node]['hw'].coarse_in
                )
            )
            # add node to graph
            self.partitions[partition_index]['graph'].add_edge(start_node,new_node)
            self.partitions[partition_index]['graph'].add_edge(new_node,end_node)
            self.partitions[partition_index]['graph'].remove_edge(start_node,end_node)

    # check difference in input streams 
    input_node  = graphs.get_input_nodes(self.partitions[partition_index]['graph'])[0]
    if self.partitions[partition_index]['streams_in'] != self.partitions[partition_index]['graph'].nodes[input_node]['hw'].coarse_in:
        # add node to graph
        new_node  = "_".join([input_node,"squeeze"])
        # add node to node info
        self.partitions[partition_index]['graph'].add_node(new_node, type=LAYER_TYPE.Squeeze,
            hw=SqueezeLayer(
                [
                    self.partitions[partition_index]['graph'].nodes[input_node]['hw'].channels_in(),
                    self.partitions[partition_index]['graph'].nodes[input_node]['hw'].rows_in(),
                    self.partitions[partition_index]['graph'].nodes[input_node]['hw'].cols_in()
                ], 
                self.partitions[partition_index]['streams_in'],
                self.partitions[partition_index]['graph'].nodes[input_node]['hw'].coarse_in
            )
        )
        # add edge to graph
        self.partitions[partition_index]['graph'].add_edge(new_node,input_node)
    # check difference in output streams 
    output_node = graphs.get_output_nodes(self.partitions[partition_index]['graph'])[0]
    if self.partitions[partition_index]['streams_out'] != self.partitions[partition_index]['graph'].nodes[output_node]['hw'].coarse_out:
        # add node to graph
        new_node  = "_".join(["squeeze",output_node])
        # add node to node info
        self.partitions[partition_index]['graph'].add_node(new_node,type=LAYER_TYPE.Squeeze,
            hw=SqueezeLayer(
                [
                    self.partitions[partition_index]['graph'].nodes[output_node]['hw'].channels_out(),
                    self.partitions[partition_index]['graph'].nodes[output_node]['hw'].rows_out(),
                    self.partitions[partition_index]['graph'].nodes[output_node]['hw'].cols_out()
                ], 
                self.partitions[partition_index]['graph'].nodes[output_node]['hw'].coarse_out,
                self.partitions[partition_index]['streams_out']
            )
        )
        self.partitions[partition_index]['graph'].add_edge(output_node,new_node)

def remove_squeeze(self,partition_index):
    # remove input squeeze module
    input_node = graphs.get_input_nodes(self.partitions[partition_index]['graph'])[0]
    if self.partitions[partition_index]['graph'].nodes[input_node]['type'] == LAYER_TYPE.Squeeze:
        self.partitions[partition_index]['graph'].remove_node(input_node)
    # remove input squeeze module
    output_node = graphs.get_output_nodes(self.partitions[partition_index]['graph'])[0]
    if self.partitions[partition_index]['graph'].nodes[output_node]['type'] == LAYER_TYPE.Squeeze:
        self.partitions[partition_index]['graph'].remove_node(output_node)
    # remove intermediate squeeze modules
    remove_nodes = []
    for node in self.partitions[partition_index]['graph'].nodes():
        if self.partitions[partition_index]['graph'].nodes[node]['type'] == LAYER_TYPE.Squeeze:
            # add squeeze nodes to list
            remove_nodes.append(node)
            # place edge back
            prev_node = graphs.get_prev_nodes(self.partitions[partition_index]['graph'],node)[0]
            next_node = graphs.get_next_nodes(self.partitions[partition_index]['graph'],node)[0]
            self.partitions[partition_index]['graph'].add_edge(prev_node,next_node)
    # remove squeeze nodes
    #graphs.print_graph(self.partitions[partition_index]['graph'])
    #print(remove_nodes)
    self.partitions[partition_index]['graph'].remove_nodes_from(remove_nodes)

def fix_split(self,partition_index):
    # graphs to be reused
    graph     = self.partitions[partition_index]['graph']
    graph_inv = graphs.get_graph_inv(graph)
    # iterate over layers in partition
    for layer in self.partitions[partition_index]['graph'].nodes:
        # check if split layer
        if self.partitions[partition_index]['graph'].nodes[layer]['type'] == LAYER_TYPE.Split:
            # fix the branches in split layer
            if set(graph[layer]) != set(self.graph[layer]):
                # update the number of outputs of the split layer 
                self.partitions[partition_index]['graph'].nodes[layer]['hw'].outputs = len(graph[layer])

def fix_concat(self,partition_index):
    # graphs to be reused
    graph     = self.partitions[partition_index]['graph']
    graph_inv = graphs.get_graph_inv(graph)
    # iterate over layers in partition
    for layer in self.partitions[partition_index]['graph'].nodes:
        # check if concat layer
        if self.partitions[partition_index]['graph'].nodes[layer]['type'] == LAYER_TYPE.Concat:
            # fix the branches in concat layer
            if set(graph_inv[layer]) != set(graphs.get_graph_inv(self.graph)[layer]):
                # update the number of outputs of the concat layer 
                self.partitions[partition_index]['graph'].nodes[layer]['hw'].n_input = len(graph_inv[layer])
                channels = []
                # correct the channels for each input
                for prev_layer in graph_inv[layer]:
                    index = graphs.get_graph_inv(self.graph)[layer].index(prev_layer)
                    channels.append(self.partitions[partition_index]['graph'].nodes[layer]['hw'].channels[index])
                # update channels
                self.partitions[partition_index]['graph'].nodes[layer]['hw'].channels = copy.copy(channels)

def remove_redundant_split(self,partition_index):
    # graphs to be reused
    graph     = self.partitions[partition_index]['graph']
    graph_inv = graphs.get_graph_inv(graph)
    # iterate over layers in partition
    for layer in self.partitions[partition_index]['graph'].nodes:
        # check if split layer
        if self.partitions[partition_index]['graph'].nodes[layer]['type'] == LAYER_TYPE.Split:
            # first check if it only has one stream out
            if len(graph[layer]) == 1:
                # remove this split layer from layer info
                del self.partitions[partition_index]['graph'].nodes[layer]
                # remove this split layer from the graph
                ## check if split layer is the input 
                if layer == self.partitions[partition_index]['input_nodes'][0]:
                    ## remove from graph
                    del self.partitions[partition_index]['graph'][layer]
                else:
                    ## remove from graph
                    start_node = graph_inv[layer][0]
                    end_node   = graph[layer][0]
                    graphs.remove_node(self.partitions[partition_index]['graph'],start_node,layer,end_node)

def remove_redundant_concat(self,partition_index):
    # graphs to be reused
    graph     = self.partitions[partition_index]['graph']
    graph_inv = graphs.get_graph_inv(graph)
    # iterate over layers in partition
    for layer in self.partitions[partition_index]['graph'].nodes:
        # check if concat layer
        if self.partitions[partition_index]['graph'].nodes[layer]['type'] == LAYER_TYPE.Concat:
            # first check if it only has one stream in
            if len(graph_inv[layer]) == 1:
                # remove this concat layer from layer info
                del self.partitions[partition_index]['graph'].nodes[layer]
                # remove this concat layer from the graph
                ## check if concat layer is the output 
                if layer == self.partitions[partition_index]['output_nodes'][0]:
                    ## remove from graph
                    del self.partitions[partition_index]['graph'][layer]
                    prev_layer = graph_inv[layer][0]
                    self.partitions[partition_index]['graph'][prev_layer] = []
                else:
                    ## remove from graph
                    start_node = graph_inv[layer][0]
                    end_node   = graph[layer][0]
                    graphs.remove_node(self.partitions[partition_index]['graph'],start_node,layer,end_node)

def add_buffer(self,partition_index): #TODO
    pass

def remove_buffer(self,partition_index): #TODO
    pass

