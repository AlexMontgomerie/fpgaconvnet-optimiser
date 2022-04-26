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

def optim_expr(args,filepath,is_branchy,opt_path,plat_path):
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
    net.update_platform(plat_path)
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

#pull out throughput info, rsc usage
#"network"
#    "performance"
#    "throughput"
#"max_resource_usage"
#    "LUT"
#    "FF"
#    "BRAM"
#    "DSP"
def extract_rpt_data(data_dict, input_path, report_str=""):
    #resources to search through
    rsc_names = ["LUT","FF","BRAM","DSP"]
    dirs_list = os.listdir(input_path)
    platform_dict={}
    for dirs in dirs_list:
        current_path = os.path.join(input_path,dirs)
        if (not os.path.isfile(current_path)) and\
            ("post_optim" in dirs):
            reports = os.listdir(current_path)
            rsc_p = int(dirs.split("-")[1][3:-1])
            print("In directory:",dirs,"\nResource percentage:",rsc_p)

            for repf in reports:
                # check for report and report contains specified string in the title
                if 'report' in repf and report_str in repf:
                    #open report file
                    open_repf = open(os.path.join(current_path,repf),"r")
                    #load report into json (dict)
                    repf_json = json.loads(open_repf.read())
                    #get dict of available resources for platform used
                    platform_dict = repf_json["platform"]["constraints"]
                    #get overall throughput
                    throughput = float(repf_json["network"]["performance"]["throughput"])
                    #get overall resource usage
                    rsc_dict = repf_json["network"]["max_resource_usage"]
                    #get the percentage of available resources for all the resource types
                    actual_rsc = []
                    for rn in rsc_names:
                        res_percent = float(rsc_dict[rn])/float(platform_dict[rn])
                        #data_dict[rn].append(res_percent)
                        data_dict[rn].append(rsc_dict[rn]) #store actual resource value
                        actual_rsc.append([res_percent,rn])
                    #get maximum of each of the resource types (limiting resource possibly)
                    idx_max = max(range(len(actual_rsc)), key=actual_rsc.__getitem__)
                    #print(repf,"::",actual_rsc,"::",actual_rsc[idx_max])
                    #append lists with report data
                    data_dict["throughput"].append(throughput)
                    data_dict["resource_max"].append(actual_rsc[idx_max][0])
                    data_dict["limiting_resource"].append(actual_rsc[idx_max][1])
                    data_dict["report_name"].append(repf)
    #need the platform stats
    return platform_dict

def combine_network_sections(args, ee1_data, eef_data,
        platform_dict, baseline_data, eef_exit_fraction=0.5):
    rsc_names = ["LUT","FF","BRAM","DSP"]

    combined_dict ={"report_name":[],"throughput":[],
                    "ee1_throughput":[], "eef_throughput":[],
                    "resource_max":[], "limiting_resource":[],
                    "LUT":[], "FF":[], "BRAM":[], "DSP":[]}

    ee1_len = len(ee1_data["report_name"])
    eef_len = len(eef_data["report_name"])
    for ee1_idx in range(ee1_len):
        for eef_idx in range(eef_len):
            ee1_thr = ee1_data["throughput"][ee1_idx]
            eef_thr = float(eef_data["throughput"][eef_idx])/eef_exit_fraction
            #pair up each
            combined_dict["report_name"].append(
                    (ee1_data["report_name"][ee1_idx],eef_data["report_name"][eef_idx]))
            #raw throughputs
            combined_dict["ee1_throughput"].append(ee1_thr)
            combined_dict["eef_throughput"].append(eef_data["throughput"][eef_idx])
            #minimum of the exit throughputs (limiting thr)
            combined_dict["throughput"].append(min(ee1_thr, eef_thr))
            #for getting the limiting rsc
            actual_rsc = []
            for rn in rsc_names:
                rsc_sum = ee1_data[rn][ee1_idx]+eef_data[rn][eef_idx]
                res_percent = rsc_sum/float(platform_dict[rn])
                combined_dict[rn].append(res_percent)
                #data_dict[rn].append(rsc_dict[rn]) #store actual resource value
                actual_rsc.append([res_percent,rn])
            idx_max = max(range(len(actual_rsc)), key=actual_rsc.__getitem__)
            combined_dict["resource_max"].append(actual_rsc[idx_max][0])
            combined_dict["limiting_resource"].append(actual_rsc[idx_max][1])

    return combined_dict

def pareto_front(data_dict, resource_string):
    #generate list of indices for pareto front
    #get list of throughputs and resource(s)

    point_len = len(data_dict["throughput"])
    #construct numpy array of throughput and resource max
    np_data = np.empty((point_len,2))
    for i,(thr,rsc) in enumerate(
            zip(data_dict["throughput"],data_dict[resource_string])):
        np_data[i][0] = 1/float(thr)
        np_data[i][1] = rsc
    #print("PARETO\n",np_data)

    is_efficient = np.arange(np_data.shape[0])
    point_total = np_data.shape[0]
    next_point_index = 0
    while next_point_index<len(np_data):
        nd_point_mask = np.any(np_data<np_data[next_point_index], axis=1)
        nd_point_mask[next_point_index] = True
        is_efficient = is_efficient[nd_point_mask]
        np_data = np_data[nd_point_mask]
        next_point_index = np.sum(nd_point_mask[:next_point_index])+1
    #return mask
        #is_efficient_mask = np.zeros(point_total, dtype=bool)
        #is_efficient_mask[is_efficient] = True
    #print("efficieny mask\n", is_efficient_mask)
    #print("is eff\n",is_efficient)
    return is_efficient


#function to reduce copy pasting code for graph gen
def _gen_graph(args,ee_flag,baseline_flag,rsc_str,
        ee1_data,eef_data,baseline_data):
    #generate graphs of chosen resource vs throughput
    #plot baseline or otherwise as chosen
    fig, ax = plt.subplots()

    subtitle_str = ""
    if ee_flag:
        print("ee1 len:{}".format(len(ee1_data[rsc_str])))
        print("eef len:{}".format(len(eef_data[rsc_str])))
        ax.scatter(ee1_data[rsc_str], ee1_data["throughput"], c="blue", label='EE1')
        ax.scatter(eef_data[rsc_str], eef_data["throughput"], c="black", label='EEF')
        subtitle_str+= "-EE-"

    if baseline_flag:
        print("base len:{}".format(len(baseline_data[rsc_str])))
        ax.scatter(baseline_data[rsc_str], baseline_data["throughput"], c="red", label='BASE')
        subtitle_str+="-BASE-"

    #fix the title of the plot
    ax.set(xlabel='Fraction of {}s'.format(rsc_str), ylabel='Throughput (sample/s)',
            title='Exit resource throughput plot {}\n({})'.format(subtitle_str,args.save_name))
    ax.legend(loc='best')
    ax.grid()
    plt.xlim(0.05,0.95)
    #save plot
    if args.save_name is not None:
        fig.savefig(os.path.join(args.output_path,"plot_{}_{}.png".format(args.save_name,rsc_str)))
    else:
        fig.savefig(os.path.join(args.output_path,"plot_{}.png".format(rsc_str)))
    print("Saved {} Graph".format(rsc_str))

def gen_graph(args):
    '''
    function for graphing optimiser results for EE and baseline

    if -bi only then only do baseline results
        red colour
        graph labelled baseline

    if -i then just do EE results
        EE1 is blue
        EEF is black
        graph labelled early exit

    if -i ... -bi ... then plot both

    make folder paths part of the saving process somehow, report?
        network name
        -i path
        number of points for EE1 and EEF
        -bi path
        number of points for baseline
    '''
    rsc_names = ["LUT","FF","BRAM","DSP"]
    #print("save name",args.save_name)
    #print("op path",args.output_path)
    if not os.path.exists(args.output_path):
        print("Creating output path:",args.output_path)
        os.makedirs(args.output_path)
    #print("ip path",args.input_path)
    #print("baseline ip path",args.baseline_input_path)
    #print("current dir",os.getcwd())

    if args.input_path is None:
        #doing baseline
        assert(args.baseline_input_path is not None)
        baseline_flag=True
        ee_flag=False
    elif args.baseline_input_path is None:
        #do EE only
        assert(args.input_path is not None)
        baseline_flag=False
        ee_flag=True
    else:
        #neither are none so do joint graph
        print("Joint graph")
        baseline_flag=True
        ee_flag=True

    #init data dicts
    ee1_data = {"report_name":[], "throughput":[],
                "resource_max":[], "limiting_resource":[],
                "LUT":[], "FF":[], "BRAM":[], "DSP":[]}
    eef_data = copy.deepcopy(ee1_data)
    baseline_data = copy.deepcopy(ee1_data)

    if ee_flag:
        platform_dict=extract_rpt_data(ee1_data, args.input_path, 'ee1')
        extract_rpt_data(eef_data, args.input_path, 'eef')
    if baseline_flag:
        platform_dict=extract_rpt_data(baseline_data,
                args.baseline_input_path)

    #gen max graph
    _gen_graph(args,ee_flag,baseline_flag,"resource_max",
            ee1_data,eef_data,baseline_data)
    #generate individual rsc graph
    #for rsc in rsc_names:
    #    _gen_graph(args,ee_flag,baseline_flag,rsc,
    #       ee1_data,eef_data,baseline_data)

    #create combined plot
    if baseline_flag and ee_flag:
        eef_exit_fraction_l = [0.1,0.2,0.25,0.33,0.5]
        for eef_frac in eef_exit_fraction_l:
            combined_data = combine_network_sections(args, ee1_data, eef_data,
                platform_dict, baseline_data, eef_exit_fraction=eef_frac)
            #print graph of combined vs baseline vs ee1
            rsc_str = "resource_max"
            fig, ax = plt.subplots()
            ax.scatter(ee1_data[rsc_str], ee1_data["throughput"], c="blue", label='EE1')
            ax.scatter(combined_data[rsc_str], combined_data["throughput"], c="green", label='COMB')
            ax.scatter(baseline_data[rsc_str], baseline_data["throughput"], c="red", label='BASE')
            #fix the title of the plot
            ax.set( xlabel='Fraction of {}s'.format(rsc_str),
                    ylabel='Throughput (sample/s)',
                    title='Exit resource vs throughput EEF: {}%\n({})'.format(
                        str(100*eef_frac),args.save_name))
            ax.legend(loc='best')
            ax.grid()
            plt.xlim(0.0,1.00)
            #save plot
            fig.savefig(os.path.join(args.output_path,"plot_{}_{}_{}eefp.png".format(
                args.save_name, rsc_str, str(int(100*eef_frac)))))
            print("Saved Combined {} Graph. EEF Frac: {}".format(rsc_str,str(100*eef_frac)))

###########################################################
#################        main         #####################
###########################################################

def main():
    parser = argparse.ArgumentParser(description="script for running experiments")
    parser.add_argument('--expr',
            choices=['parser','vis', 'out', 'out_brn', 'opt_brn', 'gen_graph'],
            help='for testing parser, vis or outputing network json')

    parser.add_argument('--save_name', type=str, help='save name for json file')

    parser.add_argument('-o','--output_path', metavar='PATH', required=True,
            help='Path to output directory')

    parser.add_argument('-i', '--input_path', metavar='PATH',
            help='folder location for report JSONs')
    parser.add_argument('-bi', '--baseline_input_path', metavar='PATH',
            help='folder location for baseline report JSONs')

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
    #platform path
    #platpath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/platforms/zc706.json"
    platpath = "/home/localadmin/phd/fpgaconvnet-optimiser/examples/platforms/xcvu440-flga2892-3-e.json"

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
        optim_expr(args, filepath, True, optpath, platpath)
    elif args.expr == 'gen_graph':
        gen_graph(args)
    else:
        raise NameError("Experiment doesn\'t exist")

if __name__ == "__main__":
    main()
