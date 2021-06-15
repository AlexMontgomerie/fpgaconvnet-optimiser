"""
A command line interface for running the optimiser for given networks
"""

import os
import yaml
import json
import argparse
import shutil
import random
import numpy as np
import time

from fpgaconvnet_optimiser.optimiser import SimulatedAnnealing
from fpgaconvnet_optimiser.optimiser import Improve

def main():
    parser = argparse.ArgumentParser(description="Optimiser Script")
    parser.add_argument('-n','--name',metavar='PATH',required=True,
        help='network name')
    parser.add_argument('-m','--model_path',metavar='PATH',required=True,
        help='Path to ONNX model')
    parser.add_argument('-p','--platform_path',metavar='PATH',required=True,
        help='Path to platform information')
    parser.add_argument('-c','--cluster_path',metavar='PATH',required=True,
        help='Path to cluster information')
    parser.add_argument('-o','--output_path',metavar='PATH',required=True,
        help='Path to output directory')
    parser.add_argument('-net','--network_load_path',metavar='PATH',required=True,
        help='Path to network load directory')
    parser.add_argument('-b','--batch_size',metavar='N',type=int,default=1,required=False,
        help='Batch size')
    parser.add_argument('--objective',choices=['throughput','latency'],required=True,
        help='Optimiser objective')
    parser.add_argument('--transforms',nargs="+", choices=['fine','coarse','weights_reloading','partition','group'],
        default=['fine','coarse','weights_reloading','partition','group'],
        help='network transforms')
    parser.add_argument('--optimiser',choices=['simulated_annealing','improve'],default='improve',
        help='Optimiser strategy')
    parser.add_argument('--optimiser_config_path',metavar='PATH',
        help='Configuration file (.yml) for optimiser')
    parser.add_argument('--seed',metavar='N',type=int,default=1234567890,
        help='Seed for the optimiser run')

    args = parser.parse_args()

    # setup seed
    seed=time.time()*(10**9)
    print("Seed:",int(seed)%1000000)
    random.seed(int(seed)%1000000)
    np.random.seed(int(seed)%1000000)

    # copy input files to the output path
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    #shutil.copy(args.model_path, os.path.join(args.output_path,os.path.basename(args.model_path)) )
    shutil.copy(args.platform_path, os.path.join(args.output_path,os.path.basename(args.platform_path)) )

    # load optimiser config
    with open(args.optimiser_config_path,"r") as f:
        optimiser_config = yaml.load(f)

    # load network
    if args.optimiser == "improve":
        # create network
        net = Improve(args.name,args.model_path,
                T=float(optimiser_config["annealing"]["T"]),
                T_min=float(optimiser_config["annealing"]["T_min"]),
                k=float(optimiser_config["annealing"]["k"]),
                cool=float(optimiser_config["annealing"]["cool"]),
                iterations=int(optimiser_config["annealing"]["iterations"]))
    elif args.optimiser == "simulated_annealing":
        # create network
        net = SimulatedAnnealing(args.name,args.model_path,
                T=float(optimiser_config["annealing"]["T"]),
                T_min=float(optimiser_config["annealing"]["T_min"]),
                k=float(optimiser_config["annealing"]["k"]),
                cool=float(optimiser_config["annealing"]["cool"]),
                iterations=int(optimiser_config["annealing"]["iterations"]))

    # turn on debugging
    net.DEBUG = True

    # get platform
    with open(args.platform_path,'r') as f:
        platform = json.load(f)

    # update platform information
    net.update_platform(args.platform_path)
    
    # update cluster information
    net.update_cluster(args.cluster_path)
    net.init_groups()
    #print(net.platform)
    #net.load_network(args.network_load_path)

    # specify optimiser objective
    if args.objective == "throughput":
        net.objective  = 1
    if args.objective == "latency":
        net.objective  = 0

    # specify batch size
    net.batch_size = args.batch_size

    # specify available transforms
    net.transforms = args.transforms
    if len(net.cluster)==1:
        net.transforms.remove("group")
    #print(net.transforms)
    if len(net.groups[0])==0:
        net.groups[0].append(0)

    # initialize graph
    # completely partition graph
    if bool(optimiser_config["transforms"]["partition"]["start_complete"]):
        net.split_complete()
#
    ## apply complete max weights reloading
    if bool(optimiser_config["transforms"]["weights_reloading"]["start_max"]):
        for partition_index in range(len(net.partitions)):
            net.partitions[partition_index].apply_max_weights_reloading()

    # run optimiser
    net.run_optimiser()

    # update all partitions
    net.update_partitions()

    # find the best batch_size
    #if args.objective == "throughput":
    #    net.get_optimal_batch_size()
    print(net.groups)
    # create_csv_report
    net.print_throughput()
    net.create_csv_report('/'.join(args.output_path.split('/run_')[0].split('/')[:-1]),args.output_path.split('/run_')[1])

    # visualise network
    net.visualise(os.path.join(args.output_path,"topology.png"))

    # create report
    net.create_report(os.path.join(args.output_path,"report.json"))

    # save all partitions
    net.save_all_partitions(args.output_path)

    # create scheduler
    net.get_schedule_csv(os.path.join(args.output_path,"scheduler.csv"))
    

