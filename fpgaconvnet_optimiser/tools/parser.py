from graphviz import Digraph
import pydot
import os
import random
import copy
import onnx
import onnx.utils
import onnx.numpy_helper
import networkx as nx

import fpgaconvnet_optimiser.tools.graphs as graphs
#import fpgaconvnet_optimiser.tools.onnx_helper as onnx_helper
import onnx_helper #TODO MAKE SURE TO CHANGE BACK

from fpgaconvnet_optimiser.models.layers import BatchNormLayer
from fpgaconvnet_optimiser.models.layers import ConvolutionLayer
from fpgaconvnet_optimiser.models.layers import InnerProductLayer
from fpgaconvnet_optimiser.models.layers import PoolingLayer
from fpgaconvnet_optimiser.models.layers import ReLULayer
from fpgaconvnet_optimiser.models.layers import LRNLayer
from fpgaconvnet_optimiser.models.layers import SoftMaxLayer
#EE layers
from fpgaconvnet_optimiser.models.layers import BufferLayer
#from fpgaconvnet_optimiser.models.layers import SplitLayer
from fpgaconvnet_optimiser.models.layers import ExitConditionLayer
from fpgaconvnet_optimiser.models.layers import ExitSelectLayer

#from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE
from layer_enum import LAYER_TYPE #TODO MAKE SURE TO CHANGE BACK

def _layer_type(op_type):
    layer_types = {
        "Conv"      : LAYER_TYPE.Convolution,
        "Gemm"      : LAYER_TYPE.InnerProduct,
        "Relu"      : LAYER_TYPE.ReLU,
        "MaxPool"   : LAYER_TYPE.Pooling,
        "LRN"       : LAYER_TYPE.LRN,
        "Reshape"   : LAYER_TYPE.Transpose,
        "Softmax"   : LAYER_TYPE.Softmax,
        "Dropout"   : LAYER_TYPE.Dropout,
        "Flatten"   : LAYER_TYPE.Flatten,
        "BatchNormalization" : LAYER_TYPE.BatchNorm,
        "GlobalAveragePool"  : LAYER_TYPE.Pooling,
        "AveragePool"        : LAYER_TYPE.Pooling,
        "Add"       : LAYER_TYPE.Eltwise,
        "Cast"      : LAYER_TYPE.Cast,
        "Clip"      : LAYER_TYPE.Clip,
        "Shape"     : LAYER_TYPE.Shape,
        "Squeeze"   : LAYER_TYPE.Squeeze,
        #placeholder layers for branching + exit decision
        "If"        : LAYER_TYPE.If,
        "ReduceMax" : LAYER_TYPE.ReduceMax,
        "Greater"   : LAYER_TYPE.Greater,
        "Identity"  : LAYER_TYPE.Identity,
        #hw layer to help split dataflow
        "Split"     : LAYER_TYPE.Split,
        #flexble buffer point for intermediate results
        "Buffer"    : LAYER_TYPE.Buffer,
    }
    return layer_types.get(op_type, lambda: TypeError)

def remove_node(graph, node): # TODO: move to tools.graphs
    prev_nodes = graphs.get_prev_nodes(graph,node)
    next_nodes = graphs.get_next_nodes(graph,node)
    graph.remove_node(node)
    for prev_node in prev_nodes:
        for next_node in next_nodes:
            graph.add_edge(prev_node,next_node)

def filter_node_types(graph, layer_type):
    remove_nodes = []
    for node in graph.nodes():
        if graph.nodes[node]['type'] == layer_type:
            remove_nodes.append(node)
    for node in remove_nodes:
        remove_node(graph,node)

def build_graph(model):
    # graph structure
    graph = nx.DiGraph()
    submodels = [] #links to the subgraphs in If ops
    ctrledges = [] #the name/ID of the if nodes [[ifnode,then,else, cond]]
    edges = [] #dataflow edges
    exitedges = [] #dataflow from exits to parent If op
    #TODO this would be the point to add in branch execution rates

    # add all nodes from network
    for node in model.graph.node:
        # get name of node
        name = onnx_helper._name(node)

        print(name, ":", node.op_type)

        # add node to graph
        graph.add_node( name, type=_layer_type(node.op_type), hw=None, inputs={} )
        if _layer_type(node.op_type) in [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]:
            graph.nodes[name]['inputs'] = { "weights": "", "bias": "" }

        #add subgraphs to the network
        if _layer_type(node.op_type) == LAYER_TYPE.If:
            print("IFNODE", name)
            ifnode = [name, None, None, None]
            #access the subgraphs
            for subgraph in node.attribute:
                print("attr: ", subgraph.name)

                submodels.append(subgraph) #record link to submodels
                subnode_head = onnx_helper._name(subgraph.g.node[0])
                if subgraph.name == "then_branch":
                    ifnode[1] = subnode_head
                elif subgraph.name == "else_branch":
                    ifnode[2] = subnode_head
                else:
                    print("Incorrect branch names")
                    raise NameError

                last_name = None
                for subnode in subgraph.g.node:
                    subname = onnx_helper._name(subnode)
                    print(subname, ":", subnode.op_type)
                    # add sub graph node to graph
                    graph.add_node(subname, type=_layer_type(subnode.op_type), hw=None, inputs={} )
                    last_name=subname
                exitedges.append((last_name, name)) #dataflow from last node in branch to If op
            ctrledges.append(ifnode)

    # add all edges from network
    for name in graph.nodes():
        # get node from model
        node = onnx_helper.get_model_node(model, name, submodels=submodels)
        # add edges into node
        for input_node in node.input:
            # add initializers
            if onnx_helper.get_model_initializer(model, input_node) is not None:
                # get input details
                input_details = onnx_helper.get_model_input(model, input_node)
                # convolution inputs
                if graph.nodes[name]["type"] == LAYER_TYPE.Convolution:
                    if len(input_details.type.tensor_type.shape.dim) == 4:
                        graph.nodes[name]['inputs']['weights'] = input_node
                    if len(input_details.type.tensor_type.shape.dim) == 1:
                        graph.nodes[name]['inputs']['bias'] = input_node
                # inner product inputs
                if graph.nodes[name]["type"] == LAYER_TYPE.InnerProduct:
                    if len(input_details.type.tensor_type.shape.dim) == 2:
                        graph.nodes[name]['inputs']['weights'] = input_node
                    if len(input_details.type.tensor_type.shape.dim) == 1:
                        graph.nodes[name]['inputs']['bias'] = input_node
                continue
            input_node = onnx_helper._format_name(input_node)
            if input_node != name:
                edges.append((input_node, name))
        # add eges out of node
        for output_node in node.output:
            output_node = onnx_helper._format_name(output_node)
            if output_node in graph.nodes():
                if output_node != name:
                    edges.append((name,output_node))
    # add edges to graph
    #print("EDGES\n",edges)
    for edge in edges:
        graph.add_edge(*edge)
    # return graph
    return submodels, graph, ctrledges, exitedges

def add_split_nodes(graph, ctrledges):
    #adding the split nodes for branching
    splitnodes = []
    for node in graph.nodes:
        successors = graphs.get_next_nodes(graph, node)
        #print(successors)
        if len(successors) > 1: #general split node placement
            splitnodes.append((node, successors))
    print("splitnodes:", splitnodes)

    save_nodes = []
    for i,(node,successors) in enumerate(splitnodes):
        splitname = "split_" + str(i)
        graph.add_node(splitname, type=LAYER_TYPE.Split, hw=None, inputs={} )
        for succ in successors:
            graph.remove_edge(node, succ)
            graph.add_edge(splitname, succ)
        graph.add_edge(node, splitname)
        save_nodes.append(splitname)
    return save_nodes

def add_buffer_nodes(graph, ctrledges):
    #adding buffer nodes to store/buffer compute at branch points
    def _add_buffer_nodes(graph, ctrledges, node, instance, save_nodes):
        predec = graphs.get_prev_nodes(graph, node)
        if len(predec) > 1:
            raise Exception("Multiple predecessors not supported")
        buffername = "buffer_" + str(instance)
        graph.add_node(buffername, type=LAYER_TYPE.Buffer, hw=None, inputs={} )
        #insert buffer layer
        graph.remove_edge(predec[0], node)
        graph.add_edge(predec[0], buffername)
        graph.add_edge(buffername, node)
        #update ctrledge to point to buffer
        save_nodes.append(buffername)
        return buffername

    save_nodes = []
    i=0
    for branch_start in ctrledges:
        #then branch = [1], else branch = [2]
        buffname = _add_buffer_nodes(graph, ctrledges, branch_start[1], i, save_nodes)
        branch_start[1] = buffname
        buffname = _add_buffer_nodes(graph, ctrledges, branch_start[2], i+1, save_nodes)
        branch_start[2] = buffname
        i+=2
    return save_nodes

def update_crtledges(graph, ctrledges):
    #ASSUMPTION: that target layer will be immediate predecessor
    for node in ctrledges: #will perform at each If op
        #ctrledges[i][0] is the If node
        predec = graphs.get_prev_nodes(graph, node[0])
        if len(predec) > 1:
            raise Exception("Multiple predecessors not supported")
        if graph.nodes[predec[0]]["type"] not in [LAYER_TYPE.Greater]:
            raise Exception("Other layer types not supported")
        graph.remove_edge(predec[0], node[0]) #remove dataflow edge
        node[3] = node[0] #set If op to have ctrl edge
        node[0] = predec[0] #Replace with predecessor

def find_ctrl_origin(graph, ctrledges, node):
    for ctrl in ctrledges:
        if node == ctrl[1]:
            #then branch link so EE
            return ctrl[0], True
        elif node == ctrl[2]:
            #else branch link so not EE
            return ctrl[0], False
        elif node == ctrl[3]: #for linking exit selct layer
            #bit of a hack
            return ctrl[0], None
    raise Exception("Node has no control input")

def add_hardware(model, submodels, graph, ctrledges, other_nodes):
    # iterate over nodes in graph
    all_nodes = [*model.graph.node]
    for submodel in submodels:
        for subnode in submodel.g.node:
            all_nodes.append(subnode)
    for node in all_nodes:
        # get node name
        name = onnx_helper._name(node)
        # check if node in graph
        if not name in graph.nodes():
            continue
        # Convolution layer
        if graph.nodes[name]['type'] == LAYER_TYPE.Convolution:
            # get number of filters
            weights_input = graph.nodes[name]["inputs"]["weights"]
            weights_dim = onnx_helper.get_model_input(model,weights_input)
            filters = int(weights_dim.type.tensor_type.shape.dim[0].dim_value)
            # get node attributes
            attr = onnx_helper._format_attr(node.attribute)
            # default attributes
            attr.setdefault("group", 1)
            attr.setdefault("strides", [1,1])
            attr.setdefault("pads", [0,0,0,0])
            attr.setdefault("dilations", [1,1])
            # create convolution layer hardware
            graph.nodes[name]['hw'] = ConvolutionLayer(
                filters,
                0, # initialise rows to 0
                0, # initialise cols to 0
                0, # initialise channels to 0
                1, # initialise coarse in to 0
                1, # initialise coarse out to 0
                k_size =attr["kernel_shape"][0],
                stride =attr["strides"][0],
                pad    =attr["pads"][0],
                groups =attr["group"]
            )
            continue
        # FC Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.InnerProduct:
            # get number of filters
            weights_input = graph.nodes[name]["inputs"]["weights"]
            weights_dim = onnx_helper.get_model_input(model,weights_input)
            filters = int(weights_dim.type.tensor_type.shape.dim[0].dim_value)
            # create inner product layer hardware
            graph.nodes[name]['hw'] = InnerProductLayer(
                filters,
                0, # initialise rows to 0
                0, # initialise cols to 0
                0, # initialise channels to 0
                1, # initialise coarse in to 0
                1, # initialise coarse out to 0
            )
            continue
        # Pooling layer
        if graph.nodes[name]['type'] == LAYER_TYPE.Pooling:
            # get node attributes
            attr = onnx_helper._format_attr(node.attribute)
            # default attributes
            attr.setdefault("strides", [1,1])
            attr.setdefault("pads", [0,0,0,0])
            attr.setdefault("dilations", [1,1])
            # create pooling layer hardware
            graph.nodes[name]['hw'] = PoolingLayer(
                0, # initialise rows to 0
                0, # initialise cols to 0
                0, # initialise channels to 0
                1, # initialise coarse in to 0
                1, # initialise coarse out to 0
                pool_type = 'max', # TODO: change so that it does AVG also
                k_size =attr["kernel_shape"][0],
                stride =attr["strides"][0],
                pad    =attr["pads"][0]
            )
            continue
        # ReLU Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.ReLU:
            # create relu layer hardware
            graph.nodes[name]['hw'] = ReLULayer(
                0, # initialise rows to 0
                0, # initialise cols to 0
                0, # initialise channels to 0
                1, # initialise coarse in to 0
                1, # initialise coarse out to 0
            )
            continue
        # BatchNorm Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.BatchNorm:
            graph.nodes[name]['hw'] = BatchNormLayer(
                0, # initialise rows to 0
                0, # initialise cols to 0
                0, # initialise channels to 0
                1, # initialise coarse in to 0
                1, # initialise coarse out to 0
            )
            continue
        #top1 exit criterion layer
        if graph.nodes[name]['type'] == LAYER_TYPE.Greater:
            #need to have some idea of the hw layer for EC
            #add the control edges to the buffers/drop points
            for ctrl in ctrledges:
                if name == ctrl[0]:
                    ctrlout = ctrl[1:]
            if len(ctrlout) == 0:
                raise NameError("Control edges not found")
            print("CTRL OUT", ctrlout)
            graph.nodes[name]['hw'] = ExitConditionLayer([0,0,0], ctrlout)
            continue
        #early exit layer
        if graph.nodes[name]['type'] == LAYER_TYPE.If:
            #with two exits it makes sense to pull from the if
            #will need to generalise assumptions for >2 exits
            #graph - two dataflow inputs, pick either or on hw level
            ctrl_origin, _ = find_ctrl_origin(graph, ctrledges, name)
            graph.nodes[name]['hw'] = ExitSelectLayer([0,0,0], ctrl_origin)
            continue
        print(name, node.op_type)
        raise NameError

    #add hardware for the non-ONNX nodes
    for name in other_nodes:
        #split layer
        if graph.nodes[name]['type'] == LAYER_TYPE.Split:
            #has input and minimum two outputs
            #connect up the inputs and outputs - might be done thru graph
            continue
        #buffer layer
        if graph.nodes[name]['type'] == LAYER_TYPE.Buffer:
            #buffer point to be moved up and down links-depending on exit laten
            #maybe do a size calc here?
            #might need to specify link from EC to here?
            ctrl_origin, EE_flag = find_ctrl_origin(graph, ctrledges, name)
            #ASSUMPTION: then branch corresponds to EE
            graph.nodes[name]['hw'] = BufferLayer([0,0,0], ctrl_origin, drop_mode=EE_flag)
            continue
        print(name, graph.nodes[name]['type'])
        raise NameError


def add_dimensions(model, graph):
    # add input dimensions
    input_channels  = int(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
    input_rows      = int(model.graph.input[0].type.tensor_type.shape.dim[2].dim_value)
    input_cols      = int(model.graph.input[0].type.tensor_type.shape.dim[3].dim_value)
    # update input node hardware
    input_node = graphs.get_input_nodes(graph)[0]
    graph.nodes[input_node]['hw'].channels[0]  = input_channels
    graph.nodes[input_node]['hw'].rows[0]      = input_rows
    graph.nodes[input_node]['hw'].cols[0]      = input_cols
    # iterate over layers in model
    nodes = list(graph.nodes())
    nodes.remove(input_node)
    print("node list\n", graph.nodes)
    for node in nodes:
        print("add_dimensions() node:", node)
        # find previous node
        prev_nodes = graphs.get_prev_nodes(graph, node)
        for prev_node in prev_nodes: # TODO: support parallel networks
            print("add_dimensions() prev_node:", node)
            # get previous node output dimensions
            dim = onnx_helper._out_dim(model, prev_node)
            # update input dimensions
            graph.nodes[node]['hw'].channels[0] = dim[0]
            graph.nodes[node]['hw'].rows[0]     = dim[1]
            graph.nodes[node]['hw'].cols[0]     = dim[2]

def parse_net(filepath,view=True):
    print("GOT HERE")
    # load onnx model
    model = onnx_helper.load(filepath)

    # get graph
    submodels, graph, ctrledges, exitedges = build_graph(model)
    print("parse_net(): CTRL EDGES\n", ctrledges)

    # remove input node
    remove_nodes = []
    for node in graph.nodes:
        if "type" not in graph.nodes[node]:
            remove_nodes.append(node)
    #print("parse_net(): REMOVING", remove_nodes)
    for node in remove_nodes:
        graph.remove_node(node)

    # remove unnecessary nodes
    filter_node_types(graph, LAYER_TYPE.Dropout)
    filter_node_types(graph, LAYER_TYPE.Transpose)
    filter_node_types(graph, LAYER_TYPE.Flatten)
    filter_node_types(graph, LAYER_TYPE.Clip)
    filter_node_types(graph, LAYER_TYPE.Cast)
    filter_node_types(graph, LAYER_TYPE.Squeeze)
    filter_node_types(graph, LAYER_TYPE.Shape)
    #TODO softmax needed for exit condition, remove filter when ONNX input updated
    filter_node_types(graph, LAYER_TYPE.Softmax)
    filter_node_types(graph, LAYER_TYPE.LRN)

    #remove ReduceMax since it's implied as part of the EC
    filter_node_types(graph, LAYER_TYPE.ReduceMax)

    #add in split and buffer/offchip store layer nodes
    other_nodes = add_split_nodes(graph, ctrledges)
    #print("updated edges-split", graph.edges)
    other_nodes += add_buffer_nodes(graph, ctrledges)
    #print("updated edges-buffer\n", graph.edges)
    print("Other nodes",other_nodes)

    #shift control edge start from If to Greater (the layer standing in as EC)
    #append the ctrl edge from the Greater to If and remove the data edge
    update_crtledges(graph, ctrledges)
    print("updated ctrledges", ctrledges)

    #determine Early Exit points (Identity operations, edge to exit)
    for eedge in exitedges:
        graph.add_edge(*eedge)
    #remove pass through node
    filter_node_types(graph, LAYER_TYPE.Identity)
    print("updated edges-exits,identity\n", graph.edges)
    #TODO separate softmax layer from other layers in model

    # add hardware to graph
    add_hardware(model, submodels, graph, ctrledges, other_nodes)

    # add layer dimensions
    add_dimensions(model, graph)

    # update all layers
    for node in graph.nodes:
        graph.nodes[node]['hw'].update()

    return model, graph
