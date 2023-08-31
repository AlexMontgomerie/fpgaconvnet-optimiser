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
import yaml
import onnx

from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.tools.layer_enum import from_cfg_type

from fpgaconvnet.optimiser.solvers import Improve
from fpgaconvnet.optimiser.solvers import SimulatedAnnealing
from fpgaconvnet.optimiser.solvers import GreedyPartition

import fpgaconvnet.optimiser.transforms.partition
import fpgaconvnet.optimiser.transforms.coarse
import fpgaconvnet.optimiser.transforms.fine
import fpgaconvnet.optimiser.transforms.skipping_windows

from fpgaconvnet.tools.layer_enum import LAYER_TYPE


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
    parser.add_argument('--teacher_partition_path', metavar='PATH', required=False,
        help='Previously optimised partitions saved in JSON')
    parser.add_argument('--sweep_config_path', metavar='PATH', required=False,
        help='Wandb sweep configuration file (.yml) for optimiser')
    parser.add_argument('--seed', metavar='n', type=int, default=random.randint(0,2**32-1),
        help='seed for the optimiser run')
    parser.add_argument('--rerun-optim', action="store_true", help='whether to run solver again at end of program')
    parser.add_argument('--gain', default=1.2, type = float, help='conservative fine gain')
    parser.add_argument('--enable-wandb', action="store_true", help='whether to enable wandb logging')
    parser.add_argument('--sweep-wandb', action="store_true", help='whether to enable wandb sweep')

    return parser.parse_args()

def write_channel_indices_to_onnx(net, onnx_path):
    onnx_model = onnx.load(onnx_path)


    for partition_index in range(len(net.partitions)):
        partition = net.partitions[partition_index]
        for node in partition.graph.nodes():
            # choose to apply to convolution node only
            if partition.graph.nodes[node]['type'] == LAYER_TYPE.Convolution:
                # choose max fine for convolution node
                onnx_node = next(filter(lambda x: x.name == node, onnx_model.graph.node))

                hw = partition.graph.nodes[node]['hw']
                if len(hw.sparsity):
                    channel_indices = hw.get_stream_sparsity()[2]
                    new_attr = onnx.helper.make_attribute("channel indices", channel_indices)
                    onnx_node.attribute.append(new_attr)

    onnx.save(onnx_model, onnx_path)


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
        if args.sweep_wandb:
            wandb.init()
            optimiser_config = wandb.config
            optimiser_config.update(platform_config)
        else:
            # project name
            wandb_name = f"fpgaconvnet-{args.name}-{args.objective}"
            # wandb config
            wandb_config = optimiser_config
            wandb_config |= platform_config
            # TODO: remove useless config
            # initialize wandb
            wandb.login()
            wandb.init(config=wandb_config,
                    project=wandb_name,
                    entity="krish-agrawal") # or "fpgaconvnet", and can add "name"
            optimiser_config = wandb.config


    # # turn on debugging
    # net.DEBUG = True

    # parse the network
    fpgaconvnet_parser = Parser(custom_onnx=True)

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

    if args.sweep_wandb:
        opt.transforms = {}
    else:
        opt.transforms = []

    for i, transform in enumerate(optimiser_config["transforms"]):
        if optimiser_config["transforms"][transform]["apply_transform"]:
            if args.sweep_wandb:
                probabilities = optimiser_config["transforms_probabilities"]
                opt.transforms[transform] = probabilities[i]
            else:
                opt.transforms.append(transform)


    # initialize graph
    ## completely partition graph
    if bool(optimiser_config["transforms"]["partition"]["start_complete"]):
        # format the partition transform allowed partitions
        if "start_fixed" in optimiser_config["transforms"]["partition"] and bool(optimiser_config["transforms"]["partition"]["start_fixed"]):
            allowed_partitions = []
            for allowed_partition in optimiser_config["transforms"]["partition"]["fixed_partitions"]:
                allowed_partitions.append((allowed_partition[0], allowed_partition[1]))
            if len(allowed_partitions) == 0:
                allowed_partitions = None
            fpgaconvnet.optimiser.transforms.partition.split_fixed(opt.net, allowed_partitions)
        else:
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
    if "skipping_windows" in opt.transforms:
        if bool(optimiser_config["transforms"]["skipping_windows"]["apply_transform"]):
            for partition in net.partitions:
                fpgaconvnet.optimiser.transforms.skipping_windows.apply_complete_skipping_windows(partition)
            net.update_partitions()        

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
    opt.run_solver(conservative_gain=args.gain)

    # print("size: ", len(pickle.dumps(opt.net)))
    opt.net.model = opt_onnx_model

    # update all partitions
    opt.net.update_partitions()

    if args.optimiser == "greedy_partition":
        opt.merge_memory_bound_partitions()
        opt.net.update_partitions()

    #If renrun, run solver again
    if args.rerun_optim:
        print("\nRerunning optimiser...")
        opt.run_solver(aggressive_fine = True)
        opt.net.update_partitions()

        if args.optimiser == "greedy_partition":
            opt.merge_memory_bound_partitions()
            opt.net.update_partitions()

    # find the best batch_size
    #if args.objective == "throughput":
    #    net.get_optimal_batch_size()
    
    #Write channel indices from model to optimised onnx path
    optimised_onnx_path = f"{args.model_path.split('.onnx')[0]}_optimized.onnx"


    write_channel_indices_to_onnx(opt.net, optimised_onnx_path)

    shutil.copy(optimised_onnx_path, os.path.join(args.output_path,os.path.basename(optimised_onnx_path)) )


    # create report
    opt.net.create_report(os.path.join(args.output_path,"report.json"))

    # save all partitions
    opt.net.save_all_partitions(os.path.join(args.output_path, "config.json"))

    # create scheduler
    opt.net.get_schedule_csv(os.path.join(args.output_path,"scheduler.csv"))

    # visualise network
    #opt.net.visualise(os.path.join(args.output_path, "topology.png"))

if __name__ == "__main__":
    args = parse_args()

    if args.sweep_wandb:
        project_name = f"fpgaconvnet-{args.name}-{args.objective}"
        # load wandb sweep configuration
        with open(args.sweep_config_path, "r") as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep_config, project=project_name, entity="krish-agrawal")
        wandb.agent(sweep_id, function=main)
    else:
        main()
