import numpy as np
import os
import json
import pydot
import copy
import math

import tools.parser as parser
import tools.graphs as graphs
import tools.matrix as matrix

import transforms.helper as helper

from models.partition.Partition import Partition

from tools.layer_enum import LAYER_TYPE

class Network():

    def __init__(self, name, network_path, batch_size=1, freq=125, reconf_time=0.0):

        ## percentage resource allocation
        self.rsc_allocation = 0.7 

        ## bitwidths
        self.data_width     = 16
        self.weight_width   = 8
        self.acc_width      = 30 

        # network name
        self.name = name

        # initialise variables
        self.batch_size = batch_size
 
        # load network
        self.model, self.graph = parser.parse_net(network_path, view=False)

        # node and edge lists
        self.node_list = list(self.graph.nodes())
        self.edge_list = list(self.graph.edges())

        # matrices
        self.connections_matrix = matrix.get_connections_matrix(self.graph)
        self.workload_matrix    = matrix.get_workload_matrix(self.graph)

        # partitions
        self.partitions = [Partition(copy.deepcopy(self.graph))]
        """
        self.partitions = [{
            'input_nodes'   : graphs.get_input_nodes(self.graph),
            'output_nodes'  : graphs.get_output_nodes(self.graph),
            'graph'         : copy.deepcopy(self.graph),
            'nodes'         : self.node_list,
            'edges'         : self.edge_list,
            'ports_in'      : 1,
            'ports_out'     : 1,
            'streams_in'    : 1,
            'streams_out'   : 1,
            'size_in'       : 0,
            'size_out'      : 0,
            'size_wr'       : 0,
            'batch_size'    : int(batch_size),
            'wr_layer'      : None,
            'wr_factor'     : 1
        }]
        """

        # update wr layer
        self.partitions[0].wr_layer = self.get_wr_layer(0)
    
        # platform
        self.platform = {
            'name'          : 'platform',
            'freq'          : freq,
            'reconf_time'   : 0.0,
            'wr_time'       : 0.0,
            'ports'         : 4,
            'port_width'    : 64,
            'mem_capacity'  : 0,
            'constraints'   : {
                'FF'    : 0,
                'LUT'   : 0,
                'DSP'   : 0,
                'BRAM'  : 0
            }
        }

        # all types of layers
        self.conv_layers = helper.get_all_layers(self.graph, LAYER_TYPE.Convolution)
        self.pool_layers = helper.get_all_layers(self.graph, LAYER_TYPE.Pooling)
 
    # import transforms
    ## fine transform
    from transforms.fine import apply_random_fine_layer 
    from transforms.fine import apply_complete_fine_partition 
    from transforms.fine import apply_complete_fine 

    ## weights reloading transform
    from transforms.weights_reloading import get_wr_layer
    from transforms.weights_reloading import get_weights_reloading_factors
    from transforms.weights_reloading import apply_random_weights_reloading 
    from transforms.weights_reloading import apply_max_weights_reloading 
    from transforms.weights_reloading import fix_weights_reloading
    from transforms.weights_reloading import apply_weights_reloading_transform 

    ## coarse transform
    from transforms.coarse import apply_random_coarse_layer
    from transforms.coarse import apply_max_coarse
    from transforms.coarse import apply_max_coarse_layer
    from transforms.coarse import fix_coarse_partition

    ## partitioning transform
    from transforms.partition import check_parallel_block
    from transforms.partition import get_all_horizontal_splits
    from transforms.partition import get_all_vertical_splits
    from transforms.partition import get_all_horizontal_merges
    from transforms.partition import get_all_vertical_merges
    from transforms.partition import split_horizontal 
    from transforms.partition import split_vertical
    from transforms.partition import merge_horizontal 
    from transforms.partition import merge_vertical
    from transforms.partition import split_horizontal_complete 
    from transforms.partition import split_vertical_complete 
    from transforms.partition import split_complete 
    from transforms.partition import merge_horizontal_complete
    from transforms.partition import merge_vertical_complete
    from transforms.partition import merge_complete
    from transforms.partition import apply_random_partition

    # import reporting functions
    from models.network.report import get_partition_colours
    from models.network.report import layer_interval_plot
    from models.network.report import partition_interval_plot
    from models.network.report import create_markdown_report 

    # import scheduling functions
    from models.network.scheduler import get_partition_order
    from models.network.scheduler import get_input_base_addr
    from models.network.scheduler import get_output_base_addr
    from models.network.scheduler import get_partition_input_dependence
    from models.network.scheduler import get_partition_output_dependence
    from models.network.scheduler import get_scheduler
    from models.network.scheduler import get_schedule_csv
    from models.network.scheduler import check_scheduler

    # update
    from models.network.update import update_partitions
    from models.network.update import update_platform

    # represent
    from models.network.represent import get_model_input_node 
    from models.network.represent import get_model_output_node 
    from models.network.represent import save_all_partitions
    
    # validate
    from models.network.validate import check_ports
    from models.network.validate import check_resources
    from models.network.validate import check_workload
    from models.network.validate import check_streams
    from models.network.validate import check_partitions

    """

    """

    def get_memory_usage_estimate(self):

        # for sequential networks, our worst-case memory usage is
        # going to be both the largest input and output featuremap pair
        
        # get the largest input featuremap size
        max_input_size = 0
        max_output_size = 0
        for partition in self.partitions:
            input_node  = partition.input_nodes[0]
            output_node = partition.output_nodes[0]
            partition_input_size  = partition.graph.nodes[input_node]['hw'].workload_in(0)*partition.batch_size
            partition_output_size = partition.graph.nodes[output_node]['hw'].workload_out(0)*partition.batch_size*partition.wr_factor
            if partition_input_size > max_input_size:
                max_input_size = partition_input_size
            if partition_output_size > max_output_size:
                max_output_size = partition_output_size

        return math.ceil(((max_input_size + max_output_size)*self.data_width)/8)

    """

    """

    def get_latency(self):
        latency = 0
        # iterate over partitions:
        for partition in self.partitions:
            # accumulate latency for each partition
            latency += partition.get_latency(self.platform["freq"])
        # return the total latency as well as reconfiguration time
        return latency + (len(self.partitions)-1)*self.platform["reconf_time"]

    def get_throughput(self):
        # return the frames per second
        return float(self.batch_size)/self.get_latency()


