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

    # update coarse in and out of partition to avoid mismatch
    self.update_group_coarse_partition()
    self.update_group_wr_partition()
    #self.update_coarse_in_out_partition()
 
    # update partitions 
    for partition_index in range(len(self.partitions)):

        ## update streams in and out
        input_node  = graphs.get_input_nodes(self.partitions[partition_index].graph)[0]
        output_node = graphs.get_output_nodes(self.partitions[partition_index].graph)[0]
        self.partitions[partition_index].streams_in  = min(self.partitions[partition_index].max_streams_in,
                self.partitions[partition_index].graph.nodes[input_node]["hw"].coarse_in * self.partitions[partition_index].graph.nodes[input_node]["hw"].coarse_group)
        self.partitions[partition_index].streams_out = min(self.partitions[partition_index].max_streams_out,
                self.partitions[partition_index].graph.nodes[output_node]["hw"].coarse_out * self.partitions[partition_index].graph.nodes[input_node]["hw"].coarse_group)

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
            if self.partitions[i].graph.nodes[input_node]['hw'].groups == 1:
                self.partitions[i].graph.nodes[input_node]['hw'].update_coarse_in(
                    self.partitions[i-1].graph.nodes[output_node]['hw'].coarse_out * self.partitions[i-1].graph.nodes[output_node]['hw'].coarse_group 
                )
                self.partitions[i].graph.nodes[input_node]['hw'].update_coarse_group(1)
            else:
                self.partitions[i].graph.nodes[input_node]['hw'].update_coarse_in(1)
                self.partitions[i].graph.nodes[input_node]['hw'].update_coarse_group(
                    min(self.partitions[i-1].graph.nodes[output_node]['hw'].coarse_out * self.partitions[i-1].graph.nodes[output_node]['hw'].coarse_group,
                        self.partitions[i].graph.nodes[input_node]['hw'].get_coarse_group_feasible()[-1])
                )
                
def update_group_coarse_partition(self):
    if len(self.partitions) > 1:
        # iterate over partitions
        for i in range(1,len(self.partitions)):
            # get input and output port between partitions
            input_node  = graphs.get_input_nodes(self.partitions[i].graph)[0] # TODO: support multi-port
            output_node = graphs.get_output_nodes(self.partitions[i-1].graph)[0] # TODO: support multi-port
            # update input node's coarse in with previous coarse out
            if self.partitions[i].graph.nodes[input_node]['hw'].groups != 1:
                if self.partitions[i-1].graph.nodes[output_node]['hw'].coarse_out not in get_factors(self.partitions[i].graph.nodes[input_node]['hw'].groups):
                    coarse_out_feasible = self.partitions[i-1].graph.nodes[output_node]['hw'].get_coarse_out_feasible()
                    coarse_out_feasible = [ x for x in coarse_out_feasible if (x in get_factors(self.partitions[i].graph.nodes[input_node]['hw'].groups))]
                    coarse_out = min(coarse_out_feasible, key=lambda x:abs(x-self.partitions[i-1].graph.nodes[output_node]['hw'].coarse_out))
                    self.partitions[i-1].graph.nodes[output_node]['hw'].update_coarse_out(coarse_out)

            if self.partitions[i-1].graph.nodes[output_node]['hw'].groups != 1:
                if self.partitions[i].graph.nodes[input_node]['hw'].coarse_in not in get_factors(self.partitions[i-1].graph.nodes[output_node]['hw'].groups):
                    coarse_in_feasible = self.partitions[i].graph.nodes[input_node]['hw'].get_coarse_in_feasible()
                    coarse_in_feasible = [ x for x in coarse_in_feasible if (x in get_factors(self.partitions[i-1].graph.nodes[output_node]['hw'].groups))]
                    coarse_in = min(coarse_in_feasible, key=lambda x:abs(x-self.partitions[i].graph.nodes[input_node]['hw'].coarse_in))
                    self.partitions[i].graph.nodes[input_node]['hw'].update_coarse_in(coarse_in)

def update_group_wr_partition(self):
    if len(self.partitions) > 1:
        # iterate over partitions
        for i in range(1,len(self.partitions)):
            input_node  = graphs.get_input_nodes(self.partitions[i].graph)[0]
            if self.partitions[i].graph.nodes[input_node]['hw'].groups != 1:
                if self.partitions[i-1].wr_layer:
                    if self.partitions[i-1].wr_factor not in get_factors(self.partitions[i].graph.nodes[input_node]['hw'].groups):
                        old_wr_factor = self.partitions[i-1].wr_factor
                        self.partitions[i-1].remove_weights_reloading_transform()
                        wr_factor_feasible = self.partitions[i-1].graph.nodes[self.partitions[i-1].wr_layer]['hw'].get_weights_reloading_feasible()
                        wr_factor_feasible = [ x for x in wr_factor_feasible if (x in get_factors(self.partitions[i].graph.nodes[input_node]['hw'].groups))]
                        wr_factor = min(wr_factor_feasible, key=lambda x:abs(x-old_wr_factor))
                        # update partition weights reloading factor
                        self.partitions[i-1].wr_factor = wr_factor
                        self.partitions[i-1].apply_weights_reloading_transform()
