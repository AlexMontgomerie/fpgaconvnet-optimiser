import numpy as np
import os
import json
import pydot
import math
import copy
import random

import tools.parser as parser
import tools.graphs as graphs
import tools.matrix as matrix

import transforms.helper as helper

from models.layers.SqueezeLayer import SqueezeLayer

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

        # update model coefficients
        self.update_coefficients()

        # node and edge lists
        self.node_list = list(self.graph.nodes())
        self.edge_list = list(self.graph.edges())

        # matrices
        self.connections_matrix = matrix.get_connections_matrix(self.graph)
        self.workload_matrix    = matrix.get_workload_matrix(self.graph)

        # partitions
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
        
        # update wr layer
        self.partitions[0]['wr_layer'] = self.get_wr_layer(0)
    
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

    # auxiliary layer functions
    from models.network.auxiliary import add_squeeze
    from models.network.auxiliary import remove_squeeze
    from models.network.auxiliary import fix_split
    from models.network.auxiliary import fix_concat
    from models.network.auxiliary import remove_redundant_split
    from models.network.auxiliary import remove_redundant_concat
    from models.network.auxiliary import add_buffer
    from models.network.auxiliary import remove_buffer

    # update
    from models.network.update import update_partitions
    from models.network.update import update_modules
    from models.network.update import update_modules_partition
    from models.network.update import update_coefficients
    from models.network.update import update_platform
    from models.network.update import update_batch_size

    # metrics
    from models.network.metrics import get_pipeline_depth
    from models.network.metrics import get_interval
    from models.network.metrics import get_latency_partition
    from models.network.metrics import get_latency
    from models.network.metrics import get_throughput

    # usage
    from models.network.usage import get_resource_usage
    from models.network.usage import get_power_average_partition
    from models.network.usage import get_power_average
    from models.network.usage import get_memory_usage_estimate

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
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    NETWORK CHECK FUNCTIONS    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        check_ports : 

        check_resources :

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    """
    def check_ports(self):
        # check each partition
        for p in self.partitions:
            # verify that the number of input ports are not exceeded
            if len(graphs.get_nodes_in(p['graph'])) > self.platform['ports']:
                return False
            if len(graphs.get_nodes_out(p['graph'])) > self.platform['ports']:
                return False
        return True

    def check_resources(self):
        # iterate over partitions
        for partition_index in range(len(self.partitions)):
            # get the resource usage for the platform
            partition_resource_usage = self.get_resource_usage(partition_index)
            #assert partition_resource_usage['FF']   <= (self.platform['constraints']['FF'])
            #assert partition_resource_usage['LUT']  <= (self.platform['constraints']['LUT'])
            assert partition_resource_usage['DSP']  <= (self.rsc_allocation*self.platform['constraints']['DSP']) , "ERROR: DSP usage exceeded"
            assert partition_resource_usage['BRAM'] <= (self.rsc_allocation*self.platform['constraints']['BRAM']), "ERROR: BRAM usage exceeded"

    def check_workload(self):
        workload_total = np.zeros( shape=( len(self.edge_list),len(self.node_list) ) , dtype=float )
        # iterate over partitions
        for partition_index in range(len(self.partitions)):
            # get parttion workload matrix
            graph     = self.partitions[partition_index]['graph']
            layers    = self.partitions[partition_index]['layers']
            wr_factor = self.partitions[partition_index]['wr_factor']
            # get parttion workload matrix
            workload_total += matrix.get_workload_matrix(graph,layers,node_list=self.node_list,edge_list=self.edge_list)*wr_factor

    def check_partitions(self):
        # iterate over partitions
        for p in self.partitions:
            pass
    """

    """    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    REPRESENTATION 
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
        save_net : 
        
        visualise : 

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    """
    def save_all_partitions(self,filepath): # TODO: update

        info = [None]*len(self.partitions)
        for i in range(len(self.partitions)):
            # get layer info from nodes
            graph_out = graphs.to_json(self.partitions[i]['graph']) 
            layer_info_out = {}
            for node in graphs.ordered_node_list(self.partitions[i]['graph']):
                layer_info_out[node.replace("/","_")]   = self.partitions[i]['graph'].nodes[node]['hw'].layer_info()
            # information for partition
            #print(self.partitions[i])
            input_node  = self.partitions[i]['input_nodes'][0]
            output_node = self.partitions[i]['output_nodes'][0]
            info[i] = {
                'partition_info': {
                    'ports'         : 1, # TODO: be able to change ports
                    'streams_in'    : self.partitions[i]['graph'].nodes[input_node]['hw'].coarse_in, 
                    'streams_out'   : self.partitions[i]['graph'].nodes[output_node]['hw'].coarse_out, 
                    'input_node'    : input_node,
                    'output_node'   : output_node,
                    'batch_size'    : self.partitions[i]['batch_size'],
                    'weights_reloading_factor'  : self.partitions[i]['wr_factor'],
                    'weights_reloading_layer'   : self.partitions[i]['wr_layer'] 
                },
                'graph'         : graph_out,
                'layer_info'    : layer_info_out
            }
        # save node_info
        with open(os.path.join(filepath,self.name+'.json'),'w') as f:
            json.dump(info,f,indent=4)

    # TODO: 
    def visualise(self,partition_index):
        g = pydot.Dot(graph_type='digraph')
        # add clusters
        edge_labels = {}
        for node in self.partitions[partition_index]['graph']:
            cluster, label_in, label_out = self.node_info[node]['hw'].visualise(node)
            edge_labels[node] = {
                "label_in"      : label_in,
                "label_out"     : label_out
            }
            g.add_subgraph(cluster)
        # create edges
        for node in self.partitions[partition_index]['graph']:
            label_out = edge_labels[node]['label_out']
            for edge in self.partitions[partition_index]['graph'][node]:
                label_in = edge_labels[edge]['label_in']
                for i in range(self.node_info[node]['hw'].coarse_out):
                    g.add_edge(pydot.Edge( "_".join([label_out,str(i)]) , "_".join([label_in,str(i)]) ))
        # save graph
        g.write_png('outputs/images/'+self.name+'_module_level.png')
    """

if __name__=="__main__":
    #net = Network('lenet_test',"data/models/lenet_short.prototxt")
    net = Network('lenet_test',"data/models/multipath.prototxt")
    #net.node_info['conv1']['hw'].coarse_out = 2
    net.node_info['a_split']['hw'].coarse_out = 2
    net.add_squeeze()
    print(net.partitions[0]['graph'])
    net.remove_squeeze()
    print(net.partitions[0]['graph'])
    net.add_squeeze()
    print(net.partitions[0]['graph'])

