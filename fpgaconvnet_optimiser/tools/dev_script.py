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

from fpgaconvnet_optimiser.models.network import Network
from fpgaconvnet_optimiser.models.partition import Partition
from fpgaconvnet_optimiser.optimiser.simulated_annealing import SimulatedAnnealing

#for optimiser
import yaml
#for graphing
import numpy as np
import matplotlib.pyplot as plt
import json

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

def output_network(args,filepath, is_branchy):
    #save the json files
    print("outputing experiments for backend")

    # create a new network
    if is_branchy and args.save_name is None:
        net = Network("branchynet", filepath)
    else:
        net = Network(args.save_name, filepath)

    # load from json format
    #net.load_network(".json") #for loading previous network config
    net.batch_size = 1 #256
    net.update_platform("/home/localadmin/phd/fpgaconvnet-optimiser/examples/platforms/zc706.json")
    # update partitions
    net.update_partitions()
    # create report
    #net.create_report("report.json") # for resrc usage

    #timestamp = dt.now().strftime("%Y-%m-%d_%H%M%S")
    print("Saving Network")
    net.save_all_partitions("tmp") # NOTE saves as one partition
    #net.get_schedule_csv("scheduler.csv") #for scheduler for running on board
    print("#################### Finished saving full network #######################")

    if is_branchy:
        # split into partitions for timing
        # NOTE partitions are incomplete but are correct otherwise
        print("Saving branchynet FOR PROFILING LATENCY")
        net.save_partition_subgraphs("tmp", partition_index=0)


###########################################################
####################### optimiser expr ####################
###########################################################

def optim_expr(args, filepath, is_branchy, opt_path):
    #opt_path is path of optimiser example config .yml file
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print("gen op path")

    #add optimiser config
    with open(opt_path,"r") as f:
        optimiser_config = yaml.load(f, Loader=yaml.Loader)
        print("loading opt conf")

    print("Starting optimisation")
    net = SimulatedAnnealing(
            args.save_name,
            filepath,#args.model_path,
            T=float(optimiser_config["annealing"]["T"]),
            T_min=float(optimiser_config["annealing"]["T_min"]),
            k=float(optimiser_config["annealing"]["k"]),
            cool=float(optimiser_config["annealing"]["cool"]),
            iterations=int(optimiser_config["annealing"]["iterations"]),
            transforms_config=optimiser_config["transforms"],
            checkpoint=bool(optimiser_config["general"]["checkpoints"]),
            checkpoint_path=os.path.join(args.output_path,"checkpoint"),
            rsc_allocation=float(optimiser_config["general"]["resource_allocation"])
            )
    net.DEBUG=True #NOTE this is required, object doesnt have DEBUG unless declared
    net.objective  = 1 #NOTE throughput objective (default is latency)
    print("generated simulated annealing object")

    #updating params
    net.batch_size = 1 #256 #since batch size is 1 for testing - latency obj co-optim
    net.update_platform("/home/localadmin/phd/fpgaconvnet-optimiser/examples/platforms/zc706.json")
    # update partitions
    net.update_partitions()

    # complete fine transform for conv layers is more resource efficient
    if bool(optimiser_config["transforms"]["fine"]["start_complete"]):
        print("applying fine max transform")
        for partition_index in range(len(net.partitions)):
            net.partitions[partition_index].apply_complete_fine()

    #print("Saving Network - no partition")
    #net.save_all_partitions(args.output_path) # NOTE saves as one partition
    #net.get_schedule_csv("scheduler.csv") #for scheduler for running on board
    #print("#################### Finished saving full network #######################")

    #print("Pre Number of partitions:",len(net.partitions))
    #saving un-optimised, unsplit network
    old_name = net.name
    net.name = old_name+"-noOpt-noESplit"
    net.save_all_partitions(args.output_path)
    print("Saved no opt, no exit split")
    # network function to create ee partitions
    net.name = old_name+"-noOpt"
    net.exit_split(partition_index=0)
    print("Exit split complete")
    net.save_all_partitions(args.output_path)
    print("Saved no opt")
    #print("Post Number of partitions:",len(net.partitions))
    net.name = old_name

    auto_flag=True #carry out lots of runs at different rsc if true
    if not auto_flag: #one run on partitions at optimiser_example specified rsc usage
        net.run_optimiser()
        net.update_partitions()

        #create folder to store results - percentage/iteration
        post_optim_path = os.path.join(args.output_path,
                "post_optim-rsc{}p".format(int(net.rsc_allocation*100)))
        if not os.path.exists(post_optim_path):
            os.makedirs(post_optim_path)
        # save all partitions
        net.save_all_partitions(post_optim_path)
        print("Partitions saved")
        # visualise network
        #net.visualise(os.path.join(post_optim_path,"topology.png"))
        # create report
        net.create_report(os.path.join(post_optim_path,
            "report_{}.json".format(net.name)))

    ### FOR LOOP FOR REPEATED OPTIM ###
    #NOTE expose these to the expr top level
    rsc_limits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    full_sa_runs = 5

    if auto_flag:
        for rsc in rsc_limits:
            for sa_i in range(full_sa_runs):
                #deep copy the network
                nets = [copy.deepcopy(net), copy.deepcopy(net)]

                #remove other partition
                nets[0].partitions.pop(0)
                nets[1].partitions.pop(1)

                #change the network name
                if len(nets[0].partitions[0].graph.nodes) > len(nets[1].partitions[0].graph.nodes):
                    nets[0].name = nets[0].name+"-ee1-rsc{}p-iter{}".format(int(rsc*100),sa_i)
                    nets[1].name = nets[1].name+"-eef-rsc{}p-iter{}".format(int(rsc*100),sa_i)
                else:
                    nets[0].name = nets[0].name+"-eef-rsc{}p-iter{}".format(int(rsc*100),sa_i)
                    nets[1].name = nets[1].name+"-ee1-rsc{}p-iter{}".format(int(rsc*100),sa_i)

                for split in nets:
                    split.rsc_allocation = rsc
                    print("\nRunning split: {}".format(split.name))
                    pass_flag = split.run_optimiser() #true = pass

                    if pass_flag:
                        # update all partitions
                        split.update_partitions()

                        #create folder to store results - percentage/iteration
                        post_optim_path = os.path.join(args.output_path,
                                "post_optim-rsc{}p".format(int(rsc*100)))
                        if not os.path.exists(post_optim_path):
                            os.makedirs(post_optim_path)

                        # save all partitions
                        split.save_all_partitions(post_optim_path)
                        print("Partitions saved")

                        # visualise network
                        #split.visualise(os.path.join(post_optim_path,"topology.png"))

                        # create report
                        split.create_report(os.path.join(post_optim_path,
                            "report_{}.json".format(split.name)))

                        # create scheduler
                        #split.get_schedule_csv(os.path.join(args.output_path,"scheduler.csv"))

def data_split(args):
    rsc_names = ["LUT","FF","BRAM","DSP"]
    #generate csv and some graph data
    print("save name",args.save_name)
    print("op path",args.output_path)
    print("ip path",args.input_path)

    print("cwd",os.getcwd())
    print("dir list",os.listdir())

    os.chdir(args.input_path)
    print("cwd 2",os.getcwd())
    dirs_list = os.listdir()
    print("dir list 2",dirs_list)

    ee1_data = {"throughput":[],"resource_max":[]}
    eef_data = {"throughput":[],"resource_max":[]}

    #go through each dir and check if file or not
    for dirs in dirs_list:
        if not os.path.isfile(dirs):
            print("in dir",dirs)
            reports = os.listdir(dirs)
            #print("reports", reports)

            rsc_p = int(dirs.split("-")[1][3:-1])
            print("rsc_p",rsc_p)

            for repf in reports:
                if 'report' in repf :

                    #print("found report")
                    open_repf = open(os.path.join(dirs,repf),"r")
                    repf_data = json.loads(open_repf.read())

                    #pull out throughput info, rsc usage, maybe platform?
                    #"network"
                    #    "performance"
                    #    "throughput"
                    #"max_resource_usage"
                    #    "LUT"
                    #    "FF"
                    #    "BRAM"
                    #    "DSP"
                    platform_dict = repf_data["platform"]["constraints"]
                    throughput = float(repf_data["network"]["performance"]["throughput"])
                    rsc_dict = repf_data["network"]["max_resource_usage"]
                    #print("RSC names")
                    actual_rsc = [float(rsc_dict[rn])/float(platform_dict[rn])
                            for rn in rsc_names]
                    actual_rsc_max = max(actual_rsc)

                    #print(repf, throughput, actual_rsc_max)

                    if 'ee1' in repf:
                        ee1_data["throughput"].append(throughput)
                        ee1_data["resource_max"].append(actual_rsc_max)
                    elif 'eef' in repf:
                        eef_data["throughput"].append(throughput)
                        eef_data["resource_max"].append(actual_rsc_max)
                    else:
                        raise IndexError("not an exit in range")

    #checking size of data
    print("ee1 len:{}".format(len(ee1_data["resource_max"])))
    print("eef len:{}".format(len(eef_data["resource_max"])))
    #generate graphs of max resource vs throughput
    fig, ax = plt.subplots()
    ax.scatter(ee1_data["resource_max"], ee1_data["throughput"], c="blue", label='EE1')
    ax.scatter(eef_data["resource_max"], eef_data["throughput"], c="black", label='EEF')
    ax.set(xlabel='Resource Max (%)', ylabel='Throughput (sample/s)',
            title='Exit resource throughput plot')
    ax.legend(loc='best')
    ax.grid()
    #save plot
    if args.save_name is not None:
        fig.savefig("dual_plot_test-{}.png".format(args.save_name))
    else:
        fig.savefig("dual_plot_test.png")
    print("Saved Graphs")

def main():
    parser = argparse.ArgumentParser(description="script for running experiments")
    parser.add_argument('--expr',
            choices=['parser','vis', 'out', 'out_brn', 'opt_brn', 'data_split'],
            help='for testing parser, vis or outputing network json')

    parser.add_argument('--save_name', type=str, help='save name for json file')

    parser.add_argument('-o','--output_path', metavar='PATH', required=True,
            help='Path to output directory')

    parser.add_argument('-i', '--input_path', metavar='PATH',
            help='folder location for report JSONs')

    #parser.add_argument('--objective', choices=['throughput','latency'], required=True,
    #            help='Optimiser objective')
    #parser.add_argument('--optimiser', choices=['simulated_annealing', 'improve', 'greedy_partition'],
    #            default='improve', help='Optimiser strategy')
    #parser.add_argument('--optimiser_config_path', metavar='PATH', required=True,
    #        help='Configuration file (.yml) for optimiser')


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

    #brn se - less layers to simplify debug, fc has bias, 2 conv, 3 fc only
    #filepath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/models/brn_se_SMOL.onnx"


    #optimiser path - taken from opt example
    optpath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/optimiser_example.yml"

    if args.expr == 'parser':
        parser_expr(filepath)
    elif args.expr == 'vis':
        vis_expr(filepath)
    elif args.expr == 'out':
        output_network(filepath, False, args.save_name)
    elif args.expr == 'out_brn':
        if args.save_name is not None:
            output_network(filepath, True, args.save_name)
        else:
            output_network(filepath, True)
    elif args.expr == 'opt_brn':
        optim_expr(args, filepath, True, optpath)
    elif args.expr == 'data_split':
        data_split(args)
    else:
        raise NameError("Experiment doesn\'t exist")

if __name__ == "__main__":
    main()
