"""
A command line interface for running the optimiser for given networks
"""

import pickle
import logging
import os
import toml
import json
import argparse
import shutil
import random
import numpy as np
import wandb
import sys
import copy

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import from_cfg_type

from fpgaconvnet.optimiser.solvers import Improve
from fpgaconvnet.optimiser.solvers import SimulatedAnnealing
from fpgaconvnet.optimiser.solvers import GreedyPartition

import fpgaconvnet.optimiser.transforms.partition
import fpgaconvnet.optimiser.transforms.coarse
import fpgaconvnet.optimiser.transforms.fine
import fpgaconvnet.optimiser.transforms.skipping_windows

def main():
    parser = argparse.ArgumentParser(description="fpgaConvNet Optimiser Command Line Interface")
    parser.add_argument('-n','--name', metavar='PATH', required=True,
        help='network name')
    parser.add_argument('-m','--model_path', metavar='PATH', required=True,
        help='Path to ONNX model')
    parser.add_argument('-p','--platform_path', metavar='PATH', required=True,
        help='Path to platform information')
    parser.add_argument('-o','--output_path', metavar='PATH', required=True,
        help='Path to output directory')
    parser.add_argument('-b','--batch_size', metavar='N',type=int, default=1, required=False,
        help='Batch size')
    parser.add_argument('--objective', choices=['throughput','latency'], required=True,
        help='Optimiser objective')
    parser.add_argument('--optimiser', choices=['simulated_annealing', 'improve', 'greedy_partition'],
        default='improve', help='Optimiser strategy')
    parser.add_argument('--optimiser_config_path', metavar='PATH', required=True,
        help='Configuration file (.yml) for optimiser')
    parser.add_argument('--teacher_partition_path', metavar='PATH', required=False,
        help='Previously optimised partitions saved in JSON')
    parser.add_argument('--seed', metavar='n', type=int, default=random.randint(0,2**32-1),
        help='seed for the optimiser run')
    parser.add_argument('--enable-wandb', action="store_true", help='seed for the optimiser run')

    # parse the arguments
    args = parser.parse_args()

    # setup seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # make the output directory if it does not exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    shutil.copy(args.optimiser_config_path, os.path.join(args.output_path,os.path.basename(args.optimiser_config_path)) )
    shutil.copy(args.model_path, os.path.join(args.output_path,os.path.basename(args.model_path)) )
    shutil.copy(args.platform_path, os.path.join(args.output_path,os.path.basename(args.platform_path)) )

    # load optimiser configuration
    with open(args.optimiser_config_path, "r") as f:
        optimiser_config = toml.load(f)

    # Initialise logger
    if bool(optimiser_config["general"]["logging"]):
        FORMAT="%(asctime)s.%(msecs)03d %(levelname)s = (%(module)s) %(message)s"
        logging.basicConfig(level=logging.INFO, filename=os.path.join(args.output_path,"optimiser.log"),
                format=FORMAT, filemode="w", datefmt='%H:%M:%S')
    else:
        logging.getLogger().disabled = True

    # create the checkpoint directory
    if not os.path.exists(os.path.join(args.output_path,"checkpoint")):
        os.makedirs(os.path.join(args.output_path,"checkpoint"))

    # load platform configuration
    with open(args.platform_path, "r") as f:
        platform_config = toml.load(f)

    # enable wandb
    if args.enable_wandb:
        # project name
        wandb_name = f"fpgaconvnet-{args.name}-{args.objective}"
        # wandb config
        wandb_config = optimiser_config
        wandb_config |= platform_config
        # TODO: remove useless config
        # initialize wandb
        wandb.init(config=wandb_config,
                project=wandb_name,
                entity="alexmontgomerie") # or "fpgaconvnet", and can add "name"

    # # turn on debugging
    # net.DEBUG = True

    # parse the network
    fpgaconvnet_parser = Parser()

    # create network
    net = fpgaconvnet_parser.onnx_to_fpgaconvnet(args.model_path, args.platform_path)

    # update the resouce allocation
    net.rsc_allocation = float(optimiser_config["general"]["resource_allocation"])

    # load network
    if args.optimiser == "improve":
        opt = Improve(net, **optimiser_config["annealing"])
    elif args.optimiser == "simulated_annealing":
        opt = SimulatedAnnealing(net, **optimiser_config["annealing"])
    elif args.optimiser == "greedy_partition":
        opt = GreedyPartition(net)
    else:
        raise RuntimeError(f"optimiser {args.optimiser} not implmented")

    # specify optimiser objective
    if args.objective == "throughput":
        opt.objective  = 1
    if args.objective == "latency":
        opt.objective  = 0
        #opt.set_double_buffer_weights()

    # specify batch size
    opt.net.batch_size = args.batch_size

    # specify available transforms
    opt.transforms = list(optimiser_config["transforms"].keys())
    opt.transforms = []
    for transform in optimiser_config["transforms"]:
        if optimiser_config["transforms"][transform]["apply_transform"]:
            opt.transforms.append(transform)


    # initialize graph
    ## completely partition graph
    if bool(optimiser_config["transforms"]["partition"]["start_complete"]):
        # format the partition transform allowed partitions
        allowed_partitions = []
        for allowed_partition in optimiser_config["transforms"]["partition"]["allowed_partitions"]:
            allowed_partitions.append((from_cfg_type(allowed_partition[0]), from_cfg_type(allowed_partition[1])))        
        if len(allowed_partitions) == 0:
            allowed_partitions = None
        fpgaconvnet.optimiser.transforms.partition.split_complete(opt.net, allowed_partitions)


    if bool(optimiser_config["transforms"]["fine"]["start_complete"]):
        for partition in net.partitions:
            fpgaconvnet.optimiser.transforms.fine.apply_complete_fine(partition)

    ## apply max fine factor to the graph
    if bool(optimiser_config["transforms"]["skipping_windows"]["apply_transform"]):
        for partition in net.partitions:
            fpgaconvnet.optimiser.transforms.skipping_windows.apply_complete_skipping_windows(partition)

    ## apply complete max weights reloading
    if bool(optimiser_config["transforms"]["weights_reloading"]["start_max"]):
        for partition_index in range(len(opt.net.partitions)):
            fpgaconvnet.optimiser.transforms.weights_reloading.apply_max_weights_reloading(
                    opt.net.partitions[partition_index])

    if bool(optimiser_config["general"]["starting_point_distillation"]) and args.teacher_partition_path != None:
        net.update_partitions()
        net.starting_point_distillation(args.teacher_partition_path, not run_optimiser)
        net.update_partitions()

    # print("size: ", len(pickle.dumps(opt.net)))
    opt_onnx_model = copy.deepcopy(opt.net.model)
    opt.net.model = None

    # run optimiser
    opt.run_solver()

    # print("size: ", len(pickle.dumps(opt.net)))
    opt.net.model = opt_onnx_model

    # update all partitions
    opt.net.update_partitions()

    if args.optimiser == "greedy_partition":
        opt.merge_memory_bound_partitions()
        opt.net.update_partitions()

    # find the best batch_size
    #if args.objective == "throughput":
    #    net.get_optimal_batch_size()

    # create report
    opt.net.create_report(os.path.join(args.output_path,"report.json"))

    # save all partitions
    opt.net.save_all_partitions(os.path.join(args.output_path, "config.json"))

    # create scheduler
    opt.net.get_schedule_csv(os.path.join(args.output_path,"scheduler.csv"))

    # visualise network
    #opt.net.visualise(os.path.join(args.output_path, "topology.png"))

if __name__ == "__main__":
    main()
