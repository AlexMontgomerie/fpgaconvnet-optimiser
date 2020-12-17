from graphviz import Digraph
import pydot
import os
import random
import copy 
import onnx
import onnx.utils
import onnx.numpy_helper
import networkx as nx

import tools.graphs as graphs
import tools.onnx_helper as onnx_helper

from models.layers.BatchNormLayer       import BatchNormLayer
from models.layers.ConvolutionLayer     import ConvolutionLayer
from models.layers.InnerProductLayer    import InnerProductLayer
from models.layers.PoolingLayer         import PoolingLayer
from models.layers.ReLULayer            import ReLULayer

from tools.layer_enum import LAYER_TYPE

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
    # add all nodes from network
    for node in model.graph.node:
        # get name of node
        name = onnx_helper._name(node)
        # add node to graph
        graph.add_node( name, type=_layer_type(node.op_type), hw=None, inputs={} )
        if _layer_type(node.op_type) in [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]:
            graph.nodes[name]['inputs'] = { "weights": "", "bias": "" } 
    # add all edges from network
    edges = []
    for name in graph.nodes():
        # get node from model
        node = onnx_helper.get_model_node(model, name)
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
    for edge in edges:
        graph.add_edge(*edge)
    # return graph
    return graph

def add_hardware(model, graph):
    # iterate over nodes in graph
    for node in model.graph.node:
        # get node name
        name = onnx_helper._name(node)
        # check if node in graph
        if not name in graph.nodes():
            continue
        # Concat Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.Concat:
            n_input = len(layer.bottom)
            graph.nodes[name]['hw'] = ConcatLayer([[0]*n_input,0,0],
                n_input
            )
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
            graph.nodes[name]['hw'] = ConvolutionLayer([0,0,0],
                #layer.convolution_param.num_output,
                filters,
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
            graph.nodes[name]['hw'] = InnerProductLayer([0,0,0],
                filters
            )
            continue
        # LRN Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.LRN:
            graph.nodes[name]['hw'] = LRNLayer([0,0,0])
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
            graph.nodes[name]['hw'] = PoolingLayer([0,0,0],
                pool_type = 'max', # TODO: change so that it does AVG also
                k_size =attr["kernel_shape"][0],
                stride =attr["strides"][0],
                pad    =attr["pads"][0]
            )
            continue
        # ReLU Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.ReLU:
            # create relu layer hardware
            graph.nodes[name]['hw'] = ReLULayer([0,0,0])
            continue
        # Sigmoid Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.Sigmoid:
            graph.nodes[name]['hw'] = SigmoidLayer([0,0,0])
            continue
        # SoftMax Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.Softmax:
            graph.nodes[name]['hw'] = SoftMaxLayer([0,0,0])
            continue
        # Eltwise Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.Eltwise:
            #n_input = len(layer.bottom)
            n_input = 2
            graph.nodes[name]['hw'] = EltwiseLayer([0,0,0],n_input)
            continue
        # BatchNorm Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.BatchNorm:
            graph.nodes[name]['hw'] = BatchNormLayer([0,0,0])
            continue
        # Scale Layer
        if graph.nodes[name]['type'] == LAYER_TYPE.Scale:
            graph.nodes[name]['hw'] = ScaleLayer([0,0,0])
            continue
        #raise NameError
        print(name,graph.nodes[name]['type'])

def add_dimensions(model, graph):
    # add input dimensions
    input_channels  = int(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value)
    input_rows      = int(model.graph.input[0].type.tensor_type.shape.dim[2].dim_value)
    input_cols      = int(model.graph.input[0].type.tensor_type.shape.dim[3].dim_value)
    # update input node hardware
    input_node = graphs.get_input_nodes(graph)[0]
    graph.nodes[input_node]['hw'].channels  = input_channels
    graph.nodes[input_node]['hw'].rows      = input_rows
    graph.nodes[input_node]['hw'].cols      = input_cols
    # iterate over layers in model
    nodes = list(graph.nodes())
    nodes.remove(input_node)
    for node in nodes:
        # find previous node
        prev_nodes = graphs.get_prev_nodes(graph, node)
        for prev_node in prev_nodes: # TODO: support parallel networks
            # get previous node output dimensions
            dim = onnx_helper._out_dim(model, prev_node) 
            # update input dimensions
            graph.nodes[node]['hw'].channels = dim[0]
            graph.nodes[node]['hw'].rows     = dim[1]
            graph.nodes[node]['hw'].cols     = dim[2]

def parse_net(filepath,view=True):

    # load onnx model
    model = onnx_helper.load(filepath)
    
    # get graph
    graph = build_graph(model)
    
    # remove input node
    remove_nodes = []
    for node in graph.nodes:
        if "type" not in graph.nodes[node]:
            print(node)
            remove_nodes.append(node)
    for node in remove_nodes:
        graph.remove_node(node)

    #graphs.print_graph(graph)
    print(graph.nodes)
    
    # remove unnecessary nodes
    filter_node_types(graph, LAYER_TYPE.Dropout)
    filter_node_types(graph, LAYER_TYPE.Transpose)
    filter_node_types(graph, LAYER_TYPE.Flatten)
    filter_node_types(graph, LAYER_TYPE.Clip)
    filter_node_types(graph, LAYER_TYPE.Cast)
    filter_node_types(graph, LAYER_TYPE.Squeeze)
    filter_node_types(graph, LAYER_TYPE.Shape)

    # add hardware to graph
    add_hardware(model, graph)

    # add layer dimensions
    add_dimensions(model, graph)

    return model, graph

if __name__=="__main__":
    pass
    #net = parse_net("data/models/resnet.prototxt")
    # lenet:
    #net = parse_net("data/models/lenet.onnx")
    # alexnet: 
    net = parse_net("data/models/alexnet.onnx")
    # vgg: 
    #net = parse_net("data/models/vgg16.onnx")
    # vgg batchnorm:
    #net = parse_net("data/models/vgg16bn.onnx")
    # mobilenetv2: 
    #net = parse_net("data/models/mobilenetv1.onnx")
    #net = parse_net("data/models/alexnet.prototxt")
    #net = parse_net("data/models/googlenet_short.prototxt")
    #net = parse_net("data/models/inception_test.prototxt")
    #net = parse_net("data/models/googlenet.prototxt")
    #net = parse_net("data/models/mobilenet.prototxt")
    #net = parse_net("data/models/vgg16.prototxt")
    #net = parse_net("data/models/multipath.prototxt")
    #net = parse_net("data/models/nin.prototxt")
    #net = parse_net("data/models/lenet.prototxt")

    #net = parse_net("data/models/resnet50.prototxt")
    #net = parse_net("data/models/resnet101.prototxt")
    #net = parse_net("data/models/vgg16.prototxt")
    #net = parse_net("data/models/inceptionv4.prototxt")
    #net = parse_net("data/models/yolov3.prototxt")
    #net = parse_net("data/models/unet.prototxt")
    #print(net)
