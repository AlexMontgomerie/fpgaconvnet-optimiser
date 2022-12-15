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
from fpgaconvnet.tools.layer_enum import LAYER_TYPE, from_onnx_op_type

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
        help='Configuration file (.toml) for optimiser')
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
        wandb_name = f"fpgaconvnet-{args.name}-{platform_config['device']['name']}-latency"
        # wandb config
        wandb_config = optimiser_config
        wandb_config |= platform_config
        # TODO: remove useless config
        # initialize wandb
        wandb.init(config=wandb_config,
                project=wandb_name,
                entity="fpgaconvnet") # or "fpgaconvnet", and can add "name"

    # parse the network
    fpgaconvnet_parser = Parser()

    # create network
    net = fpgaconvnet_parser.onnx_to_fpgaconvnet(args.model_path)

    # update platform information
    net.platform.update(args.platform_path)

    # load network
    if args.optimiser == "simulated_annealing":
        opt = LatencySimulatedAnnealing(net, objective=0,
                runtime_parameters=optimiser_config["general"]["runtime_parameters"],
                weight_storage=optimiser_config["general"]["weight_storage"],
                **optimiser_config["annealing"])
    else:
        raise NotImplementedError(f"optimiser {args.optimiser} not implmented")

    # specify available transforms
    opt.transforms = {}
    for transform in optimiser_config["transforms"]:
        if optimiser_config["transforms"][transform]["apply_transform"]:
            opt.transforms[transform] = \
                    optimiser_config["transforms"][transform]["probability"]

    # transform-specific config
    if "combine" in optimiser_config["transforms"]:
        opt.combine_nodes = optimiser_config["transforms"]["combine"]["num_nodes"]
        opt.combine_discriminate = optimiser_config["transforms"]["combine"]["discriminate"]

    if "seperate" in optimiser_config["transforms"]:
        opt.seperate_nodes = optimiser_config["transforms"]["seperate"]["num_nodes"]
        opt.seperate_allowed_types = [ from_onnx_op_type(_) for _ in \
                optimiser_config["transforms"]["seperate"]["allowed_types"] ]

    if "shape" in optimiser_config["transforms"]:
        opt.shape_method = optimiser_config["transforms"]["shape"]["method"]

    # combine all execution nodes
    if optimiser_config["transforms"]["combine"]["start_combine_all"]:

        # get all the layer types in the network
        layer_types = list(set([ opt.net.graph.nodes[node]["type"] \
                for node in opt.net.graph.nodes ]))

        # combine all the layer_types
        for layer_type in layer_types:
            opt.combine(layer_type, num_nodes=-1)

    # apply min shape to all building blocks
    if optimiser_config["transforms"]["shape"]["start_min"]:
        for hw_node in opt.building_blocks:
            opt.apply_min_shape(hw_node)

    ## apply max fine factor to the graph
    if optimiser_config["transforms"]["fine"]["start_complete"]:
        for hw_node in opt.building_blocks:
            opt.apply_random_fine_node(hw_node)

    # apply weight storage to building_blocks
    opt.apply_weight_storage()

    # run optimiser
    opt.run_solver(log=args.enable_wandb)

    # create report
    opt.net.create_report(os.path.join(args.output_path,"report.json"))

    # save all partitions
    opt.net.save_all_partitions(os.path.join(args.output_path, "config.json"))

    # create scheduler
    opt.net.get_schedule_csv(os.path.join(args.output_path,"scheduler.csv"))

    # # visualise network
    # opt.net.visualise(os.path.join(args.output_path, "topology.png"))


