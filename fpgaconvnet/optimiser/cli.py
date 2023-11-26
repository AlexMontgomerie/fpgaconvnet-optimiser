"""
A command line interface for running the optimiser for given networks
"""

import argparse
import copy
import logging
import os
import pickle
import random
import shutil

import numpy as np
import toml
import wandb
from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.platform.Platform import Platform
from fpgaconvnet.tools.layer_enum import from_cfg_type

import fpgaconvnet.optimiser.transforms.coarse
import fpgaconvnet.optimiser.transforms.fine
import fpgaconvnet.optimiser.transforms.partition
from fpgaconvnet.optimiser.solvers import (GreedyPartition, Improve,
                                           SimulatedAnnealing)


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
    parser.add_argument('-b','--batch_size', metavar='N',type=int, default=1, required=False,
        help='Batch size')
    parser.add_argument('--objective', choices=['throughput','latency'], required=True,
        help='Optimiser objective')
    parser.add_argument('--optimiser', choices=['simulated_annealing', 'improve', 'greedy_partition'],
        default='improve', help='Optimiser strategy')
    parser.add_argument('--optimiser_config_path', metavar='PATH', required=True,
        help='Configuration file (.toml) for optimiser')
    parser.add_argument('--seed', metavar='n', type=int, default=random.randint(0,2**32-1),
        help='seed for the optimiser run')
    parser.add_argument('--enable-wandb', action="store_true", help='whether to enable wandb logging')
    parser.add_argument('--ram_usage', default=None, type=float, help='ram utilization ratio')
    parser.add_argument('--custom_onnx', action="store_true", help='whether to enable custom onnx parsing on model Parser')

    return parser.parse_args()

def get_wandb_config(optimiser_config, platform_config, args):
    """
    Get the configuration for Weights & Biases (wandb) based on the optimiser_config, platform_config, and command line arguments.

    Args:
        optimiser_config (dict): The optimiser configuration.
        platform_config (dict): The platform configuration.
        args (argparse.Namespace): The command line arguments.

    Returns:
        dict: The wandb configuration.
    """
    wandb_config = copy.deepcopy(optimiser_config)
    wandb_config |= platform_config
    wandb_config["batch_size"] = args.batch_size
    wandb_config["optimiser"] = args.optimiser

    # remove useless config
    wandb_config["general"].pop("logging")
    wandb_config["general"].pop("checkpoints")
    if "ethernet" in wandb_config:
        wandb_config.pop("ethernet")
    if args.optimiser != "simulated_annealing":
        wandb_config.pop("annealing")
    if args.name == "unet":
        wandb_config["model_version"] = os.path.basename(args.model_path).split(".onnx")[0].split("unet_")[-1]

    return wandb_config

def main():
    args = parse_args()
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
        wandb_config = get_wandb_config(optimiser_config, platform_config, args)

        # initialize wandb
        wandb.init(config=wandb_config,
                project=wandb_name,
                entity="fpgaconvnet") # or "fpgaconvnet", and can add "name"

    # # turn on debugging
    # net.DEBUG = True

    # parse the network
    fpgaconvnet_parser = Parser(custom_onnx=args.custom_onnx)
    net = fpgaconvnet_parser.onnx_to_fpgaconvnet(args.model_path)

    # load platform
    platform = Platform()
    platform.update(args.platform_path)

    if args.optimiser == "improve":
        opt = Improve(net, platform, **optimiser_config["annealing"], wandb_enabled=args.enable_wandb)
    elif args.optimiser == "simulated_annealing":
        opt = SimulatedAnnealing(net, platform, **optimiser_config["annealing"], wandb_enabled=args.enable_wandb)
    elif args.optimiser == "greedy_partition":
        opt = GreedyPartition(net, platform, wandb_enabled=args.enable_wandb)
        #opt.multi_fpga = True
        #opt.constrain_port_width = False
    else:
        raise RuntimeError(f"optimiser {args.optimiser} not implmented")

    # specify resource allocation
    opt.rsc_allocation = float(optimiser_config["general"]["resource_allocation"])
    if args.ram_usage is not None:
        opt.ram_usage = args.ram_usage
    else:
        opt.ram_usage = opt.rsc_allocation

    # specify optimiser objective
    if args.objective == "throughput":
        opt.objective  = 1
    if args.objective == "latency":
        opt.objective  = 0
        #opt.set_double_buffer_weights()

    # specify batch size
    opt.net.batch_size = args.batch_size

    # specify available transforms
    opt.transforms = []
    opt.transforms_probs = []
    for transform in optimiser_config["transforms"]:
        if optimiser_config["transforms"][transform]["apply_transform"]:
            opt.transforms.append(transform)
            opt.transforms_probs.append(optimiser_config["transforms"][transform]["probability"])

    if "weights_reloading" not in opt.transforms:
        for partition in opt.net.partitions:
            partition.enable_wr = False

    # initialize graph
    ## completely partition graph
    if (optimiser_config["transforms"]["partition"]["apply_transform"] and bool(optimiser_config["transforms"]["partition"]["start_complete"])) or (args.optimiser == "greedy_partition" and bool(optimiser_config["transforms"]["partition"]["start_complete"])):
        # format the partition transform allowed partitions
        allowed_partitions = []
        for allowed_partition in optimiser_config["transforms"]["partition"]["allowed_partitions"]:
            allowed_partitions.append((from_cfg_type(allowed_partition[0]), from_cfg_type(allowed_partition[1])))
        if len(allowed_partitions) == 0:
            allowed_partitions = None
        vertical = optimiser_config["transforms"]["partition"]["vertical"]
        fpgaconvnet.optimiser.transforms.partition.split_complete(opt.net, allowed_partitions, vertical)
        allowed_single_layer_merge_pre = []
        for single_layer_merge in optimiser_config["transforms"]["partition"]["allowed_single_layer_merge_pre"]:
            allowed_single_layer_merge_pre.append(from_cfg_type(single_layer_merge))
        fpgaconvnet.optimiser.transforms.partition.merge_single_layer_partition_to_prev(opt.net, allowed_single_layer_merge_pre)
        allowed_single_layer_merge_post = []
        for single_layer_merge in optimiser_config["transforms"]["partition"]["allowed_single_layer_merge_post"]:
            allowed_single_layer_merge_post.append(from_cfg_type(single_layer_merge))
        fpgaconvnet.optimiser.transforms.partition.merge_single_layer_partition_to_next(opt.net, allowed_single_layer_merge_post)

    ## apply max fine factor to the graph
    if bool(optimiser_config["transforms"]["fine"]["start_complete"]):
        for partition in net.partitions:
            fpgaconvnet.optimiser.transforms.fine.apply_complete_fine(partition)

    ## apply complete max weights reloading
    if bool(optimiser_config["transforms"]["weights_reloading"]["start_max"]):
        for partition_index in range(len(opt.net.partitions)):
            fpgaconvnet.optimiser.transforms.weights_reloading.apply_max_weights_reloading(
                    opt.net.partitions[partition_index])

    opt_onnx_model = copy.deepcopy(opt.net.model)
    opt.net.model = None

    # run optimiser
    valid_solution = opt.run_solver()
    if not valid_solution:
        print(f"WARNING - Optimiser could not find a valid solution. The generated reports and configuration files are not valid.")
    opt.net.model = opt_onnx_model

    # update all partitions
    opt.update_partitions()

    if args.optimiser == "greedy_partition":
        if args.enable_wandb:
            opt.net.save_all_partitions(os.path.join(args.output_path, "pre_merge_config.json"))
            pre_merge_config_artifact = wandb.Artifact(f"{args.name}_pre_merge_config", type="json")
            pre_merge_config_artifact.add_file(os.path.join(args.output_path,"pre_merge_config.json"))
            wandb.log_artifact(pre_merge_config_artifact)

            opt.net.visualise_partitions_nx(os.path.join(args.output_path,
            "partitions_nx_graphs"))
            graph_paths = [os.path.join(args.output_path, "partitions_nx_graphs", graph) for graph in sorted(os.listdir(os.path.join(args.output_path, "partitions_nx_graphs")))]
            wandb.log({"pre_merge_partitions_nx_graphs": [wandb.Image(path) for path in graph_paths]})
        opt.merge_memory_bound_partitions()
        opt.update_partitions()

    # # find the best batch_size
    # if args.objective == "throughput":
    #    net.get_optimal_batch_size()

    #Write channel indices from model to optimised onnx path
    optimised_onnx_path = f"{args.model_path.split('.onnx')[0]}_optimized.onnx"
    opt.net.write_channel_indices_to_onnx(optimised_onnx_path)
    shutil.copy(optimised_onnx_path, os.path.join(args.output_path,os.path.basename(optimised_onnx_path)) )

    # save the model checkpoint
    if optimiser_config["general"]["checkpoints"]:
        opt.save_design_checkpoint(os.path.join(args.output_path,"checkpoint",f"{args.name}.pkl"))

    # create report
    opt.create_report(os.path.join(args.output_path,"report.json"))

    # save all partitions
    opt.net.save_all_partitions(os.path.join(args.output_path, "config.json"))

    # create scheduler
    # FIXME: This does not work correctly (at least for models with nested branches)
    opt.net.get_schedule_csv(os.path.join(args.output_path,"scheduler.csv"))

    # visualise partitions (nx graphs)
    opt.net.visualise_partitions_nx(os.path.join(args.output_path, "partitions_nx_graphs"))

    # visualise network
    # opt.net.visualise(os.path.join(args.output_path, "topology.png"))

    if args.enable_wandb:
        # log the report
        report_artifact = wandb.Artifact(f"{args.name}_report", type="json")
        report_artifact.add_file(os.path.join(args.output_path,"report.json"))
        wandb.log_artifact(report_artifact)

        # log the config
        config_artifact = wandb.Artifact(f"{args.name}_config", type="json")
        config_artifact.add_file(os.path.join(args.output_path,"config.json"))
        wandb.log_artifact(config_artifact)

        # log the scheduler
        scheduler_artifact = wandb.Artifact(f"{args.name}_scheduler", type="csv")
        scheduler_artifact.add_file(os.path.join(args.output_path,"scheduler.csv"))
        wandb.log_artifact(scheduler_artifact)

        # log the partitions nx graphs
        graph_paths = [os.path.join(args.output_path, "partitions_nx_graphs", graph) for graph in sorted(os.listdir(os.path.join(args.output_path, "partitions_nx_graphs")))]
        wandb.log({"final_partitions_nx_graphs": [wandb.Image(path) for path in graph_paths]})

if __name__ == "__main__":
    main()