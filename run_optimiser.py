import os
import json
import argparse
import shutil

from optimiser.simulated_annealing import SimulatedAnnealing
from optimiser.improve import Improve

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimiser Script")
    parser.add_argument('-n','--name',metavar='PATH',required=True,
        help='network name')
    parser.add_argument('-m','--model_path',metavar='PATH',required=True,
        help='Path to ONNX model')
    parser.add_argument('-p','--platform_path',metavar='PATH',required=True,
        help='Path to platform information')
    parser.add_argument('-o','--output_path',metavar='PATH',required=True,
        help='Path to output directory')
    parser.add_argument('-b','--batch_size',metavar='N',type=int,default=1,required=False,
        help='Batch size')
    parser.add_argument('--objective',choices=['throughput','latency'],required=True,
        help='Optimiser objective')
    parser.add_argument('--transforms',nargs="+", choices=['fine','coarse','weights_reloading','partition'],
        default=['fine','coarse','weights_reloading','partition'],
        help='network transforms')
    parser.add_argument('--optimiser',choices=['simulated_annealing','improve'],default='improve',
        help='Optimiser strategy')

    args = parser.parse_args()

    # copy input files to the output path
    shutil.copy(args.model_path, os.path.join(args.output_path,os.path.basename(args.model_path)) )
    shutil.copy(args.platform_path, os.path.join(args.output_path,os.path.basename(args.platform_path)) )

    # load network
    net = Improve(args.name,args.model_path)
    net.DEBUG = True

    # get platform
    with open(args.platform_path,'r') as f:
        platform = json.load(f)

    # update platform information
    net.update_platform(args.platform_path)

    # specify optimiser objective
    if args.objective == "throughput":
        net.objective  = 1 
    if args.objective == "latency":
        net.objective  = 0

    # specify batch size
    net.batch_size = args.batch_size

    # specify available transforms
    net.transforms = args.transforms

    # initialize graph
    ## completely partition graph
    net.split_complete()
    ## apply complete max weights reloading
    for partition_index in range(len(net.partitions)):
        net.apply_max_weights_reloading(partition_index)

    # run optimiser
    net.run_optimiser()

    # update all partitions
    net.update_partitions()

    # find the best batch_size
    #if args.objective == "throughput":
    #    net.get_optimal_batch_size()

    # create report
    net.create_report(os.path.join(args.output_path,"report.json"))

    # save all partitions
    net.save_all_partitions(args.output_path)
    
    # create scheduler
    net.get_schedule_csv(os.path.join(args.output_path,"scheduler.csv"))
