from build.lib.fpgaconvnet_optimiser.tools.graphs import print_graph
import numpy as np
import copy

import fpgaconvnet_optimiser.tools.graphs as graphs
import fpgaconvnet_optimiser.tools.matrix as matrix

from fpgaconvnet_optimiser.models.layers import CommunicationLayer, SqueezeLayer

from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

def add_squeeze(self):
    # find mismatching streams  
    streams_matrix = matrix.get_streams_matrix(self.graph)
    edge_list = matrix.get_edge_list_matrix(self.graph)
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
            self.graph.add_node(new_node,type=LAYER_TYPE.Squeeze,
                hw=SqueezeLayer(
                    [
                        self.graph.nodes[start_node]['hw'].channels_out(),
                        self.graph.nodes[start_node]['hw'].rows_out(),
                        self.graph.nodes[start_node]['hw'].cols_out()
                    ], 
                    self.graph.nodes[start_node]['hw'].coarse_out,
                    self.graph.nodes[end_node]['hw'].coarse_in
                )
            )
            # add node to graph
            self.graph.add_edge(start_node,new_node)
            self.graph.add_edge(new_node,end_node)
            self.graph.remove_edge(start_node,end_node)

    # check difference in input streams 
    input_node  = graphs.get_input_nodes(self.graph)[0]
    if self.streams_in != self.graph.nodes[input_node]['hw'].coarse_in:
        # add node to graph
        new_node  = "_".join([input_node,"squeeze"])
        # add node to node info
        self.graph.add_node(new_node, type=LAYER_TYPE.Squeeze,
            hw=SqueezeLayer(
                [
                    self.graph.nodes[input_node]['hw'].channels_in(),
                    self.graph.nodes[input_node]['hw'].rows_in(),
                    self.graph.nodes[input_node]['hw'].cols_in()
                ], 
                self.streams_in,
                self.graph.nodes[input_node]['hw'].coarse_in
            )
        )
        # add edge to graph
        self.graph.add_edge(new_node,input_node)
    # check difference in output streams 
    output_node = graphs.get_output_nodes(self.graph)[0]
    if self.streams_out != self.graph.nodes[output_node]['hw'].coarse_out:
        # add node to graph
        new_node  = "_".join(["squeeze",output_node])
        # add node to node info
        self.graph.add_node(new_node,type=LAYER_TYPE.Squeeze,
            hw=SqueezeLayer(
                [
                    self.graph.nodes[output_node]['hw'].channels_out(),
                    self.graph.nodes[output_node]['hw'].rows_out(),
                    self.graph.nodes[output_node]['hw'].cols_out()
                ], 
                self.graph.nodes[output_node]['hw'].coarse_out,
                self.streams_out
            )
        )
        self.graph.add_edge(output_node,new_node)

def remove_squeeze(self):
    # remove input squeeze module
    input_node = graphs.get_input_nodes(self.graph)[0]
    if self.graph.nodes[input_node]['type'] == LAYER_TYPE.Squeeze:
        self.graph.remove_node(input_node)
    # remove input squeeze module
    output_node = graphs.get_output_nodes(self.graph)[0]
    if self.graph.nodes[output_node]['type'] == LAYER_TYPE.Squeeze:
        self.graph.remove_node(output_node)
    # remove intermediate squeeze modules
    remove_nodes = []
    for node in self.graph.nodes():
        if self.graph.nodes[node]['type'] == LAYER_TYPE.Squeeze:
            # add squeeze nodes to list
            remove_nodes.append(node)
            # place edge back
            prev_node = graphs.get_prev_nodes(self.graph,node)[0]
            next_node = graphs.get_next_nodes(self.graph,node)[0]
            self.graph.add_edge(prev_node,next_node)
    # remove squeeze nodes
    self.graph.remove_nodes_from(remove_nodes)

def add_communication(self,platform):
    new_comm_link=0
    if platform['connections_in']!=[0]:
        # check input node
        input_node = graphs.get_input_nodes(self.graph)[0]
        if self.graph.nodes[input_node]['type'] != LAYER_TYPE.Communication:
            # add node to graph
            new_node  = "_".join(["comm_in",str(platform['connections_in'][0]),input_node])
            self.graph.add_node(new_node,type=LAYER_TYPE.Communication,
                hw=CommunicationLayer(
                    dim=[
                        self.graph.nodes[input_node]['hw'].channels_out(),
                        self.graph.nodes[input_node]['hw'].rows_out(),
                        self.graph.nodes[input_node]['hw'].cols_out()
                    ], 
                    coarse_in=self.graph.nodes[input_node]['hw'].coarse_out,
                    coarse_out=self.streams_out,
                    send_nreceive=True,
                    pair_id=new_comm_link
                )
            )
            # add node to graph
            self.graph.add_edge(new_node,input_node)
            
        
    if platform['connections_out']!=[0]:
        # Check if output node is communication layer 
        output_node = graphs.get_output_nodes(self.graph)[0]
        if self.graph.nodes[output_node]['type'] != LAYER_TYPE.Communication:
            # add node to graph
            new_node  = "_".join(["comm_out",str(platform['connections_out'][0]),output_node])
            self.graph.add_node(new_node,type=LAYER_TYPE.Communication,
                hw=CommunicationLayer(
                    dim=[
                        self.graph.nodes[output_node]['hw'].channels_out(),
                        self.graph.nodes[output_node]['hw'].rows_out(),
                        self.graph.nodes[output_node]['hw'].cols_out()
                    ], 
                    coarse_in=self.graph.nodes[output_node]['hw'].coarse_out,
                    coarse_out=self.streams_out,
                    send_nreceive=True,
                    pair_id=new_comm_link
                )
            )
            self.graph.add_edge(output_node,new_node)
   #print("ID:{id:002d} ".format(id=self.get_id()))


def remove_communication(self):
    # remove input squeeze module
    try:
        input_node = graphs.get_input_nodes(self.graph)[0]
        if self.graph.nodes[input_node]['type'] == LAYER_TYPE.Communication:
            self.graph.remove_node(input_node)
    except:
        print("Unable to remove input communication")
    try:
        if(graphs.get_output_nodes(self.graph)):
            output_node = graphs.get_output_nodes(self.graph)[0]
            if self.graph.nodes[output_node]['type'] == LAYER_TYPE.Communication:
                self.graph.remove_node(output_node)
    except:
        print("Partition:{}".format(self.get_id()))
        print_graph(self.graph)
        print("Unable to remove output communication")
