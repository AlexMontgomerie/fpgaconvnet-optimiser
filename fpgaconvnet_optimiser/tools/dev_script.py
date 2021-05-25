from graphviz import Digraph
import pydot
import os
import random
import copy
import onnx
#import onnx.utils
#import onnx.numpy_helper
import networkx as nx
#import onnxoptimizer as optimizer
import argparse

#import fpgaconvnet_optimiser.tools.graphs as graphs
import fpgaconvnet_optimiser.tools.onnx_helper as onnx_helper
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

#from fpgaconvnet_optimiser.models.layers import BatchNormLayer
#from fpgaconvnet_optimiser.models.layers import ConvolutionLayer
#from fpgaconvnet_optimiser.models.layers import InnerProductLayer
#from fpgaconvnet_optimiser.models.layers import PoolingLayer
#from fpgaconvnet_optimiser.models.layers import ReLULayer
#from fpgaconvnet_optimiser.models.layers import LRNLayer
#from fpgaconvnet_optimiser.models.layers import SoftMaxLayer

import fpgaconvnet_optimiser.tools.parser as parser
#import parser

#importing vis relevant files
from fpgaconvnet_optimiser.models.network import Network
from fpgaconvnet_optimiser.models.partition import Partition

def parser_expr():
    print("Parser experiments")

    #attempt to parse the graph and see what errors
    #exits BEFORE softmax
    #filepath = "/home/benubu/phd/fpgaconvnet-optimiser/examples/models/speedy-brn-top1ee-bsf.onnx"
    #exits AFTER softmax
    filepath = "/home/benubu/phd/fpgaconvnet-optimiser/examples/models/speedy-brn-top1ee-bsf-trnInc-sftmx.onnx"
    #filepath = "/home/benubu/phd/fpgaconvnet-optimiser/examples/models/pt_fulltest.onnx"
    model, submodels, graph, ctrledges = \
        parser.parse_net(filepath, view=False) #check what view does

    print(graph.nodes)
    print(graph.edges)

    for node in graph.nodes:
        #print(graph.nodes[node]['hw'])
        if graph.nodes[node]['type'] == LAYER_TYPE.Greater:
            #seeing if I can find the threshold...
            vi = onnx_helper.get_model_value_info(model, node, submodels)
            print(vi)

        if graph.nodes[node]['type'] == LAYER_TYPE.Convolution:
            print("name:", node)
            #print("w:", graph.nodes[node]['inputs']['weights'])
            #print("b:", graph.nodes[node]['inputs']['bias'])
            print(graph.nodes[node]['inputs'])

    for node in model.graph.node:
        #looking for constant feeding into greater
        op = parser._layer_type(node.op_type)
        if op != LAYER_TYPE.Greater:
            continue
        for input_node in node.input:
            input_details = onnx_helper.get_model_input(model,input_node)
            print("INPUT DEETS:", input_details) #outputs type info - only if input
            print("RAW NODE:", input_node) #just outputs the name/number

def vis_expr():
    print("Visualiser experiments")
    #exits BEFORE softmax
    #filepath = "/home/benubu/phd/fpgaconvnet-optimiser/examples/models/speedy-brn-top1ee-bsf.onnx"
    #exits AFTER softmax
    filepath = "/home/benubu/phd/fpgaconvnet-optimiser/examples/models/speedy-brn-top1ee-bsf-trnInc-sftmx.onnx"
    #filepath = "/home/benubu/phd/fpgaconvnet-optimiser/examples/models/pt_fulltest.onnx"

    #taking filepath as model_path
    name = 'branchynet' #taking name as branchynet
    #leave the rest of the networks as default

    test_net = Network(name, filepath) #rest as defaults

    test_outpath = "/home/benubu/phd/test_out-new-mods-sftmx.png"
    test_net.visualise(test_outpath)


def main():
    parser = argparse.ArgumentParser(description="script for running experiments")
    parser.add_argument('--expr',choices=['parser','vis'],
                        help='for testing parser or vis')
    args = parser.parse_args()

    if args.expr == 'parser':
        parser_expr()
    elif args.expr == 'vis':
        vis_expr()
    else:
        raise NameError("Experiment doesn\'t exist")

if __name__ == "__main__":
    main()
