import fpgaconvnet_optimiser.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from google.protobuf.text_format import MessageToString
from google.protobuf.json_format import MessageToJson
import numpy as np
import os
import json
import networkx as nx
import copy

import fpgaconvnet_optimiser.tools.graphs as graphs
import fpgaconvnet_optimiser.tools.onnx_helper as onnx_helper
import fpgaconvnet_optimiser.tools.layer_enum
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

def get_model_input_node(self, partition_index):
    input_node = self.partitions[partition_index].input_nodes[0]
    while True:
        try:
            onnx_node = onnx_helper.get_model_node(self.model, input_node, self.submodels)
            return onnx_node.input[0]
        except NameError:
            # input is not included in onnx data
            # TODO: change this so that you get PREVIOUS node in ONNX graph
            print("Finding next available ONNX op")
            input_node = graphs.get_next_nodes(self.partitions[partition_index].graph,input_node)
            if len(input_node) < 1:
                return "NA"
            input_node = input_node[0]


def get_model_output_node(self, partition_index):
    output_node = self.partitions[partition_index].output_nodes[0]
    while True:
        try:
            onnx_node = onnx_helper.get_model_node(self.model, output_node, self.submodels)
            return onnx_node.output[0]
        except NameError:
            # output is not included in onnx data
            # TODO: change this so that you get PREVIOUS node in ONNX graph
            print("Finding previous available ONNX op")
            output_node = graphs.get_prev_nodes(self.partitions[partition_index].graph,output_node)
            if len(output_node) < 1:
                return "NA"
            output_node =output_node[0]

def gen_layer_name(self, partition_index, layer_name): # layer in protobuf form
    layer_type = self.partitions[partition_index].graph.nodes[layer_name]['type']
    #FIXME use protobuf correctly to get all caps version
    layer_type_str = str(layer_type)[11:].upper() # remove 'LAYER_TYPE.'
    if layer_name.isnumeric(): # preprend with type to avoid macro issue
        return f'{layer_type_str}{layer_name}'
    else:
        return layer_name

def add_stream(stream, s_name, control=False, split=False):
    stream.name  = s_name
    stream.ctrl = control
    stream.split = split
    return stream

def add_stream_in(self, p_i, node, layer, s_name="in", control=False, coarse=None, split=False):
    new_stream  = layer.streams_in.add()
    add_stream(new_stream, s_name, control,split)
    if coarse is None:
        new_stream.coarse = self.partitions[p_i].graph.nodes[node]['hw'].coarse_in[0]
    else:
        new_stream.coarse = coarse

def add_stream_out(self, p_i, node, layer, s_name="out", control=False, coarse=None, split=False):
    new_stream  = layer.streams_out.add()
    add_stream(new_stream, s_name, control,split)
    if coarse is None:
        new_stream.coarse = self.partitions[p_i].graph.nodes[node]['hw'].coarse_out[0]
    else:
        new_stream.coarse = coarse

def save_all_partitions(self,filepath, separate_partitions=False): # TODO: update
    # create protocol buffer
    partitions = fpgaconvnet_pb2.partitions()
    # iterate over partions
    for i in range(len(self.partitions)):
        # create partition
        partition = partitions.partition.add()
        # add partition info
        partition.id = i
        partition.ports = 1 # TODO
        partition.input_node  = self.get_model_input_node(i) #self.partitions[i]['input_nodes'][0]
        partition.output_node = self.get_model_output_node(i) #self.partitions[i]['output_nodes'][0]
        partition.batch_size  = self.partitions[i].batch_size
        partition.weights_reloading_factor = self.partitions[i].wr_factor
        partition.weights_reloading_layer  = str(self.partitions[i].wr_layer)
        # add all layers (in order)
        for node in graphs.ordered_node_list(self.partitions[i].graph):
            # create layer
            layer = partition.layers.add()
            layer_name = self.gen_layer_name(i, node.replace("/","_"))
            layer.name = node.replace("/","_") #layer_name
            layer.type = fpgaconvnet_optimiser.tools.layer_enum.to_proto_layer_type(self.partitions[i].graph.nodes[node]['type'])
            # add stream(s) in
            prev_nodes = graphs.get_prev_nodes(self.partitions[i].graph, node)
            node_in_list = []
            if not prev_nodes:
                node_in_list.append(layer_name)
                self.add_stream_in(i, node, layer)
            else:
                for prev_node in prev_nodes:
                    prev_layer_name = self.gen_layer_name(i, prev_node.replace("/","_"))
                    node_in_list.append(prev_layer_name)
                    if self.partitions[i].graph.nodes[prev_node]['type'] in [LAYER_TYPE.Split]:
                        self.add_stream_in(i, node, layer,prev_layer_name+"_stream",split=True)
                    else:
                        self.add_stream_in(i,node,layer, "_".join([prev_layer_name, layer_name]))
            layer.node_in.extend(node_in_list)
            # add stream(s) out
            next_nodes = graphs.get_next_nodes(self.partitions[i].graph, node)
            node_out_list = []
            #NOTE add any new control output layers
            if not next_nodes \
                and self.partitions[i].graph.nodes[node]['type'] not in [LAYER_TYPE.Greater]:
                node_out_list.append(layer_name)
                self.add_stream_out(i, node, layer)
            else:
                for next_node in next_nodes:
                    next_layer_name = self.gen_layer_name(i, next_node.replace("/","_"))
                    node_out_list.append(next_layer_name)
                    if self.partitions[i].graph.nodes[node]['type'] in [LAYER_TYPE.Split]:
                        self.add_stream_out(i,node,layer,layer_name+"_stream",split=True)
                    else:
                        self.add_stream_out(i,node,layer,"_".join([layer_name, next_layer_name]))


            layer.node_out.extend(node_out_list)
            # add parameters
            self.partitions[i].graph.nodes[node]['hw'].layer_info(layer.parameters, batch_size=self.partitions[i].batch_size)
            # add weights key
            if self.partitions[i].graph.nodes[node]['type'] in [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]:
                layer.weights_path = self.partitions[i].graph.nodes[node]['inputs']['weights']
                layer.bias_path    = self.partitions[i].graph.nodes[node]['inputs']['bias']

            if self.partitions[i].graph.nodes[node]['type'] in [ LAYER_TYPE.Greater]:
                #add control signals
                #print(self.partitions[i].graph.nodes[node]['hw'].ctrledges)
                for ce in self.partitions[i].graph.nodes[node]['hw'].ctrledges:
                    c_name = layer_name+"-ctrlout"
                    if ce not in self.partitions[i].graph.nodes:
                        #NOTE in latency profiling mode with broken connections
                        print("WARNING: Greater in latency profiling mode with broken connections")
                        c_name = "out" #FIXME introduce profiling controls on the backend

                out_size = self.partitions[i].graph.nodes[node]['hw'].ctrl_out_size + 1 #NOTE +1 for ID pipeline
                self.add_stream_out(i, node, layer, c_name, control=True, coarse=out_size)

            if self.partitions[i].graph.nodes[node]['type'] in [LAYER_TYPE.Buffer]:
                #print("I AM A BUFFER LAYER")
                #print(self.partitions[i].graph.nodes[node]['hw'].ctrledge)
                ce = self.partitions[i].graph.nodes[node]['hw'].ctrledge
                if ce not in self.partitions[i].graph.nodes:
                    #NOTE in latency profiling mode with broken connections
                    print("WARNING: Buffer in latency profiling mode with broken connections")
                    c_name = "in" #FIXME introduce profiling controls on the backend
                else:
                    c_name= self.gen_layer_name(i, ce) + "-ctrlout"
                self.add_stream_in(i, node, layer, c_name, control=True, coarse=1)

            if self.partitions[i].graph.nodes[node]['type'] in [LAYER_TYPE.Split]:
                #NOTE to get the profiling working
                if len(layer.streams_out) == 1:
                    layer.parameters.ports_out = 1

        # save in JSON format
        if separate_partitions:
            with open(os.path.join(filepath,f"{self.name}-stage-{i}.json"),"w") as f:
                f.write(MessageToJson(partitions,preserving_proto_field_name=True))
            # reset partitions
            partitions = fpgaconvnet_pb2.partitions()

    # # save protobuf message
    # with open(os.path.join(filepath,f"{self.name}.prototxt"),"w") as f:
    #     f.write(MessageToString(partitions))

    # save in JSON format
    if not separate_partitions:
        with open(os.path.join(filepath,f"{self.name}.json"),"w") as f:
            f.write(MessageToJson(partitions,preserving_proto_field_name=True))
            #json.dump(MessageToJson(partitions),f)

def save_partition_subgraphs(self, filepath, partition_index):
    # fragment partition into individual partitions
    main_partition = self.partitions[partition_index]

    # NOTE: Assumptions
    # one GREATER (comparison) layer
    # one IF (exit) layer
    # first node in the graph is the start
    # there will be layers between backbone and exit
    # assuming simple paths will always align with exits

    # TODO might be more reliable to do this split when parsing the subgraphs

    #find comparison layer, IF layer, and starting layer
    for node in graphs.ordered_node_list(main_partition.graph):
        if not graphs.get_prev_nodes(main_partition.graph, node):
            first_layer = node
            print("top node:", node)

        if main_partition.graph.nodes[node]['type'] in [LAYER_TYPE.Greater]:
            comparison_layer = node
            print("comparison node:", node)

        if main_partition.graph.nodes[node]['type'] in [LAYER_TYPE.If]:
            exit_layer = node
            print("exit node:", node)

    backbone_subgraph = nx.shortest_path(main_partition.graph, first_layer, comparison_layer)

    subgraph_nodes=[backbone_subgraph]
    # get remaining paths to exit layer, minus exit layer and nodes before buffer
    for ee_subgraph in nx.all_simple_paths(main_partition.graph, first_layer, exit_layer):
        ee_subgraph = ee_subgraph[:-1]  # remove exit layer
        node = ee_subgraph[0] # remove modules up to first buffer
        while main_partition.graph.nodes[node]['type'] not in [LAYER_TYPE.Buffer]:
            ee_subgraph.pop(0)
            node = ee_subgraph[0]
        subgraph_nodes.append(ee_subgraph)

    # remove weights reloading transform
    self.partitions[partition_index].remove_weights_reloading_transform()
    # turn node lists into graph objects
    subgraphs = [main_partition.graph.subgraph(sgn).copy() for sgn in subgraph_nodes]

    for _ in range(len(subgraphs) - 1): # assuming this will be > 1
        self.partitions.insert(partition_index,copy.deepcopy(self.partitions[partition_index]))

    # fix wr layer as in transforms.split_horizontal
    wr_layer = self.partitions[partition_index].wr_layer
    for i in range(len(subgraphs)):
        pi = partition_index + i
        self.partitions[pi].graph   = subgraphs[i]
        if wr_layer not in self.partitions[pi].graph.nodes:
            self.partitions[pi].wr_layer  = self.partitions[pi].get_wr_layer()
            self.partitions[pi].wr_factor = 1
        self.partitions[pi].apply_weights_reloading_transform()

    self.update_partitions()
    self.save_all_partitions(filepath, separate_partitions=True)
