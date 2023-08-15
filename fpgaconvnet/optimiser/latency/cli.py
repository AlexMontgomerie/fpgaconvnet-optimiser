"""
A command line interface for running the optimiser for given networks
"""

import logging
import os
import toml
import argparse
import shutil
import random
import numpy as np
import wandb
import yaml

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import LAYER_TYPE, from_onnx_op_type

from fpgaconvnet.optimiser.latency.solvers import LatencySolver, LatencySimulatedAnnealing

import fpgaconvnet.optimiser.transforms.partition
import fpgaconvnet.optimiser.transforms.coarse
import fpgaconvnet.optimiser.transforms.fine

from fpgaconvnet.platform.Platform import Platform

def parse_args():
    """
    Command line argument parser
    """
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
    parser.add_argument('--sweep_config_path', metavar='PATH', required=False,
        help='Wandb sweep configuration file (.yml) for optimiser')
    parser.add_argument('--seed', metavar='n', type=int, default=random.randint(0,2**32-1),
        help='seed for the optimiser run')
    parser.add_argument('--enable-wandb', action="store_true", help='whether to enable wandb logging')
    parser.add_argument('--sweep-wandb', action="store_true", help='whether to enable wandb sweep')

    return parser.parse_args()

def optimize():
    args = parse_args()

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
        if args.sweep_wandb:
            wandb.init()
            optimiser_config = wandb.config
            optimiser_config.update(platform_config)
        else:
            # project name
            project_name = f"harflow3d-{args.name}-latency"
            # wandb config
            wandb_config = optimiser_config
            wandb_config |= platform_config
            # remove useless config
            # wandb_config['general'].pop('logging')
            # wandb_config['annealing'].pop('warm_start_time_limit')
            # wandb_config['device'].pop('board')
            # wandb_config['system'].pop('reconfiguration_time')
            # initialize wandb
            wandb.init(config=wandb_config,
                    project=project_name,
                    entity="fpgaconvnet") # or "fpgaconvnet", and can add "name"
            optimiser_config = wandb.config

    # parse the network
    fpgaconvnet_parser = Parser(regression_model=optimiser_config["general"]["resource_model"],
            convert_gemm_to_conv=optimiser_config["general"]["convert_gemm_to_conv"])
    fpgaconvnet_parser.add_onnx_optimization_passes(optimiser_config["general"]["optimization_passes"])

    # create network
    net = fpgaconvnet_parser.onnx_to_fpgaconvnet(args.model_path)

    # update platform information
    platform = Platform()
    platform.update(args.platform_path)

    # load network
    if args.optimiser == "simulated_annealing":
        opt = LatencySimulatedAnnealing(net, platform, objective=0,
                runtime_parameters=optimiser_config["general"]["runtime_parameters"],
                weight_storage=optimiser_config["general"]["weight_storage"],
                channel_tiling=optimiser_config["general"]["channel_tiling"],
                filter_tiling=optimiser_config["general"]["filter_tiling"],
                **optimiser_config["annealing"])
    else:
        raise NotImplementedError(f"optimiser {args.optimiser} not implmented")

    # specify resource allocation
    opt.rsc_allocation = float(optimiser_config["general"]["resource_allocation"])

    # specify available transforms
    opt.transforms = {}
    for i, transform in enumerate(optimiser_config["transforms"]):
        if optimiser_config["transforms"][transform]["apply_transform"]:
            if args.sweep_wandb:
                probabilities = optimiser_config["transforms_probabilities"]
                opt.transforms[transform] = probabilities[i]
            else:
                opt.transforms[transform] = \
                        optimiser_config["transforms"][transform]["probability"]

    # transform-specific config
    if "combine" in optimiser_config["transforms"]:
        opt.combine_nodes = optimiser_config["transforms"]["combine"]["num_nodes"]
        opt.combine_discriminate = optimiser_config["transforms"]["combine"]["discriminate"]

    if "seperate" in optimiser_config["transforms"]:
        opt.seperate_nodes = optimiser_config["transforms"]["seperate"]["num_nodes"]
        opt.allowed_seperate_types = [ from_onnx_op_type(_) for _ in \
                optimiser_config["transforms"]["seperate"]["allowed_types"] ]

    if "shape" in optimiser_config["transforms"]:
        opt.shape_method = optimiser_config["transforms"]["shape"]["method"]
        if opt.shape_method in [ "random", "mixed" ]:
            opt.use_previous_shape = optimiser_config["transforms"]["shape"].get(
                    "use_previous_shape", True)
            if args.sweep_wandb:
                opt.rand_shape_range = [ int(optimiser_config["transforms"]["shape"]["rand_shape_range"]["rows"]),
                                         int(optimiser_config["transforms"]["shape"]["rand_shape_range"]["cols"]),
                                         int(optimiser_config["transforms"]["shape"]["rand_shape_range"]["depth"]),
                                         int(optimiser_config["transforms"]["shape"]["rand_shape_range"]["channels"]) ]
            else:
                opt.rand_shape_range = optimiser_config["transforms"]["shape"].get(
                        "rand_shape_range", [5, 5, 5, 5])

    # combine all execution nodes
    if optimiser_config["transforms"]["combine"]["start_combine_all"]:

        # get all the layer types in the network
        layer_types = list(set([ opt.net.graph.nodes[node]["type"] \
                for node in opt.net.graph.nodes ]))

        # combine all the layer_types
        for layer_type in layer_types:
            for _ in range(100): # hack to make sure it chooses all the discrimination groups
                opt.combine(layer_type, discriminate=opt.combine_discriminate, num_nodes=-1)

    # apply min shape to all building blocks
    if "starting_shape" in optimiser_config["transforms"]["shape"]:
        for hw_node in opt.building_blocks:
            match optimiser_config["transforms"]["shape"]["starting_shape"]:
                case "min":
                    shape_in, shape_out = opt.get_min_shape(hw_node)
                    opt.update_building_block_shape(hw_node, shape_in, shape_out)
                case "max":
                    shape_in, shape_out = opt.get_max_shape(hw_node)
                    opt.update_building_block_shape(hw_node, shape_in, shape_out)
                case "median":
                    shape_in, shape_out = opt.get_median_shape(hw_node)
                    opt.update_building_block_shape(hw_node, shape_in, shape_out)
                case "percentage":
                    shape_in, shape_out = opt.get_percentage_shape(hw_node,
                            percentage=optimiser_config["transforms"]["shape"].get(
                                "starting_shape_percentage",10))
                    opt.update_building_block_shape(hw_node, shape_in, shape_out)
                case "":
                    pass
                case _:
                    raise TypeError(optimiser_config["transforms"]["shape"]["starting_shape"])

    ## apply max fine factor to the graph
    if optimiser_config["transforms"]["fine"]["start_complete"]:
        for hw_node in opt.building_blocks:
            opt.apply_random_fine_node(hw_node)

    ## set the shapes for hw nodes
    for hw_node in opt.building_blocks:
        shape_in = opt.building_blocks[hw_node]["hw"].shape_in()
        shape_out = opt.building_blocks[hw_node]["hw"].shape_out()
        opt.update_building_block_shape(hw_node, shape_in, shape_out)

    # apply weight storage to building_blocks
    opt.apply_weight_storage()

    # run optimiser
    opt.run_solver(log=args.enable_wandb)

    # create report
    # opt.create_report(os.path.join(args.output_path,"report.json"))

    # save all partitions
    # opt.net.save_all_partitions(os.path.join(args.output_path, "config.json"))

    # create scheduler
    # opt.net.get_schedule_csv(os.path.join(args.output_path,"scheduler.csv"))

    # # visualise network
    # opt.net.visualise(os.path.join(args.output_path, "topology.png"))

def main():
    args = parse_args()

    if args.sweep_wandb:
        project_name = f"harflow3d-{args.name}-latency"
        # load wandb sweep configuration
        with open(args.sweep_config_path, "r") as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep_config, project=project_name, entity="fpgaconvnet")
        wandb.agent(sweep_id, function=optimize)
    else:
        optimize()


