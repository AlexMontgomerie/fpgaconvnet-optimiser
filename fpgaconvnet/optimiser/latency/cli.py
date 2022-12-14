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
from fpgaconvnet.tools.layer_enum import from_onnx_op_type
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.optimiser.latency.solvers import LatencySolver, LatencySimulatedAnnealing

import fpgaconvnet.optimiser.transforms.partition
import fpgaconvnet.optimiser.transforms.coarse
import fpgaconvnet.optimiser.transforms.fine

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

    shutil.copy(args.optimiser_config_path,
            os.path.join(args.output_path,os.path.basename(args.optimiser_config_path)) )
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
        wandb_name = f"fpgaconvnet-{args.name}-latency"
        # wandb config
        wandb_config = optimiser_config
        wandb_config |= platform_config
        # TODO: remove useless config
        # initialize wandb
        wandb.init(config=wandb_config,
                project=wandb_name,
                entity="fpgaconvnet") # or "fpgaconvnet", and can add "name"

    # # turn on debugging
    # net.DEBUG = True

    # parse the network
    fpgaconvnet_parser = Parser()

    # create network
    net = fpgaconvnet_parser.onnx_to_fpgaconvnet(args.model_path)

    # update platform information
    net.platform.update(args.platform_path)
    print(net.platform)

    # update the resouce allocation
    net.rsc_allocation = float(optimiser_config["general"]["resource_allocation"])

    # load network
    if args.optimiser == "simulated_annealing":
        opt = LatencySimulatedAnnealing(net, objective=0) # TODO: include optimiser_config["annealing"]
    else:
        raise NotImplementedError(f"optimiser {args.optimiser} not implmented")

    # # specify available transforms
    # opt.transforms = list(optimiser_config["transforms"].keys())
    # opt.transforms = []
    # for transform in optimiser_config["transforms"]:
    #     if optimiser_config["transforms"][transform]["apply_transform"]:
    #         opt.transforms.append(transform)

    # ## apply max fine factor to the graph
    # if bool(optimiser_config["transforms"]["fine"]["start_complete"]):
    #     for partition in net.partitions:
    #         fpgaconvnet.optimiser.transforms.fine.apply_complete_fine(partition)

    # run optimiser
    opt.run_solver(log=args.enable_wandb)

    # update all partitions
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

    # # visualise network
    # opt.net.visualise(os.path.join(args.output_path, "topology.png"))


