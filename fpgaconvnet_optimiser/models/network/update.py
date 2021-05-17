import json
import copy
import fpgaconvnet_optimiser.tools.graphs as graphs
from fpgaconvnet_optimiser.transforms.helper import get_factors

def update_partitions(self):

    # update modules
    for node in self.graph.nodes():
        self.graph.nodes[node]['hw'].update()
 
    # remove all auxiliary layers
    for partition_index in range(len(self.partitions)):

        ## remove squeeze layer
        self.partitions[partition_index].remove_squeeze()

    # remove all empty partitions
    for partition_index in range(len(self.partitions)):

        # remove empty partition
        if len(self.partitions[partition_index].graph.nodes) == 0:
            del self.partitions[partition_index]

    # update coarse in and out of partition to avoid mismatch
    # self.update_group_coarse_partition()
    # self.update_group_wr_partition()
    self.update_coarse_in_out_partition()
 
    # update partitions 
    for partition_index in range(len(self.partitions)):

        ## update streams in and out
        input_node  = graphs.get_input_nodes(self.partitions[partition_index].graph)[0]
        output_node = graphs.get_output_nodes(self.partitions[partition_index].graph)[0]

        self.partitions[partition_index].streams_in  = min(self.partitions[partition_index].max_streams_in,
                self.partitions[partition_index].graph.nodes[input_node]["hw"].streams_in())
        self.partitions[partition_index].streams_out = min(self.partitions[partition_index].max_streams_out,
                self.partitions[partition_index].graph.nodes[output_node]["hw"].streams_out())

        ## add auxiliary layers
        self.partitions[partition_index].add_squeeze()
        
        ## update partition info
        input_nodes  = graphs.get_input_nodes(self.partitions[partition_index].graph)
        output_nodes = graphs.get_output_nodes(self.partitions[partition_index].graph)
        self.partitions[partition_index].input_nodes  = input_nodes
        self.partitions[partition_index].output_nodes = output_nodes
        
        ## update batch size for partitions
        self.partitions[partition_index].batch_size = self.batch_size
        
        ## update sizes
        self.partitions[partition_index].size_in  = self.partitions[partition_index].graph.nodes[input_nodes[0]]['hw'].size_in()
        self.partitions[partition_index].size_out = self.partitions[partition_index].graph.nodes[input_nodes[0]]['hw'].size_out()
        if self.partitions[partition_index].wr_layer != None:
            wr_layer_info = self.partitions[partition_index].graph.nodes[self.partitions[partition_index].wr_layer]['hw']
            self.partitions[partition_index].size_wr = wr_layer_info.get_parameters_size()['weights']
        else:
            self.partitions[partition_index].size_wr = 0
        
        # update modules
        self.partitions[partition_index].update_modules()

    ## validate
    self.check_streams()
    self.check_workload()

def update_platform(self, platform_path):

    # get platform
    with open(platform_path,'r') as f:
        platform = json.load(f)

    # update platform information
    #self.platform['name']           = paltform['name']
    self.platform['ports']          = int(platform['ports'])
    #self.platform['port_width']     = int(platform['port_width'])
    #self.platform['freq']           = int(platform['freq'])
    self.platform['reconf_time']    = float(platform['reconf_time'])
    self.platform['mem_capacity']   = int(platform['mem_capacity'])
    self.platform['mem_bandwidth']  = float(platform['mem_bandwidth'])

    # update constraints
    self.platform['constraints']['FF']   = platform['FF']
    self.platform['constraints']['DSP']  = platform['DSP']
    self.platform['constraints']['LUT']  = platform['LUT']
    self.platform['constraints']['BRAM'] = platform['BRAM']

def update_coarse_in_out_partition(self):
    if len(self.partitions) > 1:
        # iterate over partitions
        for i in range(1,len(self.partitions)):
            # get input and output port between partitions
            input_node  = graphs.get_input_nodes(self.partitions[i].graph)[0] # TODO: support multi-port
            output_node = graphs.get_output_nodes(self.partitions[i-1].graph)[0] # TODO: support multi-port
            # update input node's coarse in with previous coarse out
            self.partitions[i].graph.nodes[input_node]['hw'].update_coarse_in(
                self.partitions[i-1].graph.nodes[output_node]['hw'].streams_out()
            )
