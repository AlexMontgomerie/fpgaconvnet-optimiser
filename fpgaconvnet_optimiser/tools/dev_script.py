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
from datetime import datetime as dt

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

def parser_expr(filepath):
    #attempt to parse the graph and see what errors
    print("Parser experiments")

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

def vis_expr(filepath):
    print("Visualiser experiments")

    #taking filepath as model_path
    name = 'branchynet' #taking name as branchynet
    #leave the rest of the networks as default

    test_net = Network(name, filepath) #rest as defaults

    timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")
    test_outpath = "/home/localadmin/phd/opt-vis-outputs/test_out-sftmxCmp"
    test_outpath += "-" + timestamp + ".png"
    test_net.visualise(test_outpath)

def output_network(filepath, is_branchy, save_name=None):
    #save the json files
    print("outputing experiments for backend")

    # create a new network
    if is_branchy:
        net = Network("branchynet", filepath)
    else:
        net = Network(save_name, filepath)

    # load from json format
    #net.load_network(".json") #for loading previous network config
    net.batch_size = 1 #256
    net.update_platform("/home/localadmin/phd/fpgaconvnet-optimiser/examples/platforms/zedboard.json")
    # update partitions
    net.update_partitions()
    # create report
    #net.create_report("report.json") # for resrc usage

    #timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")
    print("Saving Network")
    net.save_all_partitions("tmp") # NOTE saves as one partition
    #net.get_schedule_csv("scheduler.csv") #for scheduler for running on board

    if is_branchy:
        # split into partitions for timing
        # NOTE partitions are incomplete but are correct otherwise
        print("Saving branchynet FOR PROFILING LATENCY")
        net.save_partition_subgraphs("tmp", partition_index=0)

def main():
    parser = argparse.ArgumentParser(description="script for running experiments")
    parser.add_argument('--expr',choices=['parser','vis', 'out', 'out_brn'],
                        help='for testing parser, vis or outputing network json')
    parser.add_argument('--save_name', type=str, help='save name for json file')
    args = parser.parse_args()

    #exits BEFORE softmax
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/speedy-brn-top1ee-bsf.onnx"
    #exits AFTER softmax
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/speedy-brn-top1ee-bsf-trnInc-sftmx.onnx"
    #Removed softmax layer before exit results - only used for exit condition
    filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/speedy-brn-top1ee-bsf-lessOps-trained.onnx"
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/pt_fulltest.onnx"

    #switched pool to floor mode, adjusted FC layers to match resulting sizes
    filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/ceil_false.onnx"

    #switched pool to floor mode, adjusted FC layers to match resulting sizes, adjusted conv layer padding to fit
    filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/io_match.onnx"

    #changed conv to no bias, normalised data set
    filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/io_match_trained_norm.onnx"
    # raised threshold, removed bias from conv and FC layers
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/io_match_trained_norm_thr_high.onnx"

    # just the first exit, trained, normed, no bias, for branchynet
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/brn_first_exit.onnx"

    # just the second exit, trained, normed, no bias, for branchynet
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/brn_second_exit.onnx"

    # just a trained, fc layer, normed, no bias, for mnist
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/fc_layer.onnx"

    # just a trained, fc layer, normed, WITH bias, for mnist
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/fc_layer_bias.onnx"

    #lenet example filepath
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/lenet.onnx"

    # pool and relu layer
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/pool_relu_layers.onnx"

    # conv pool relu fc
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/conv_pool_relu_fc.onnx"

    # conv pool fc
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/conv_pool_fc.onnx"

    # pool relu fc
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/pool_relu_fc.onnx"

    # conv 1 channel out, fc
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/conv_fc.onnx"

    # conv 5 channel out, pool, relu
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/conv5c.onnx"

    # conv 5 channel out, fc
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/conv5c_fc.onnx"

    # conv 5 channel out, fc with bias (gemm)
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/conv5c_fc-bias.onnx"

    #FC and conv have no bias, normalised data set, threshold at .9
    filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/io-match_trained_norm_no-bias.onnx"

    if args.expr == 'parser':
        parser_expr(filepath)
    elif args.expr == 'vis':
        vis_expr(filepath)
    elif args.expr == 'out':
        output_network(filepath, False, args.save_name)
    elif args.expr == 'out_brn':
        output_network(filepath, True)
    else:
        raise NameError("Experiment doesn\'t exist")

if __name__ == "__main__":
    main()
