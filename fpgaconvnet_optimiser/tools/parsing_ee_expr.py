from graphviz import Digraph
import pydot
import os
import random
import copy
import onnx
import onnx.utils
import onnx.numpy_helper
import networkx as nx


import onnxoptimizer as optimizer

#import fpgaconvnet_optimiser.tools.graphs as graphs
#import fpgaconvnet_optimiser.tools.onnx_helper as onnx_helper

#from fpgaconvnet_optimiser.models.layers import BatchNormLayer
#from fpgaconvnet_optimiser.models.layers import ConvolutionLayer
#from fpgaconvnet_optimiser.models.layers import InnerProductLayer
#from fpgaconvnet_optimiser.models.layers import PoolingLayer
#from fpgaconvnet_optimiser.models.layers import ReLULayer
#from fpgaconvnet_optimiser.models.layers import LRNLayer
#from fpgaconvnet_optimiser.models.layers import SoftMaxLayer

#from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

#import the current parser functions to see what they can do
import parser

def main():
    print("Parser experiments")

    #attempt to parse the graph and see what errors
    filepath = "/home/benubu/phd/fpgaconvnet-optimiser/examples/models/speedy-brn-top1ee-bsf.onnx"
    #filepath = "/home/benubu/phd/fpgaconvnet-optimiser/examples/models/pt_fulltest.onnx"
    model, graph, ctrledges = parser.parse_net(filepath, view=False) #check what view does

    print(graph.nodes)
    print(graph.edges)
    print(ctrledges)

    for node in graph.nodes:
        print(graph.nodes[node]['hw'])

if __name__ == "__main__":
    main()
