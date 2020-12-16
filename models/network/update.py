import json
import copy
import tools.graphs as graphs

def update_partitions(self):
    
    # update all modules
    self.update_modules()
    
    # update partitions 
    for partition_index in range(len(self.partitions)):

        ## update all modules again
        self.update_modules_partition(partition_index)
        
        ## update nodes in partition
        #for layer in self.partitions[partition_index]['graph'].nodes():
        #    if layer in self.graph.nodes():
        #        self.partitions[partition_index]['graph'].nodes[layer]['hw'] = copy.deepcopy(self.graph.nodes[layer]['hw'])
        
        ## remove auxiliary layers
        self.remove_squeeze(partition_index)
        
        ## fix weights reloading in each partition
        self.fix_weights_reloading(partition_index)
        
        ## apply weights reloading
        self.apply_weights_reloading_transform(partition_index)
        
        ## update all modules again
        self.update_modules_partition(partition_index)
       
        ## fix coarse in for all layers
        self.fix_coarse_partition(partition_index)

        ## update all modules again
        self.update_modules_partition(partition_index)
 
        ## add auxiliary layers
        self.add_squeeze(partition_index)
        # TODO: self.add_split(partition_index)
        
        ## update partition info
        input_nodes  = graphs.get_input_nodes(self.partitions[partition_index]['graph'])
        output_nodes = graphs.get_output_nodes(self.partitions[partition_index]['graph'])
        self.partitions[partition_index]['input_nodes']  = input_nodes
        self.partitions[partition_index]['output_nodes'] = output_nodes
        
        # update graph order 
        #self.partitions[partition_index]['graph'] = graphs.order_graph(self.partitions[partition_index]['graph'])
        
        ## update node and edge list
        # TODO
        
        ## update batch size for partitions
        self.partitions[partition_index]['batch_size'] = self.batch_size
        
        ## update sizes
        self.partitions[partition_index]['size_in']  = self.partitions[partition_index]['graph'].nodes[input_nodes[0]]['hw'].size_in()
        self.partitions[partition_index]['size_out'] = self.partitions[partition_index]['graph'].nodes[input_nodes[0]]['hw'].size_out()
        if self.partitions[partition_index]['wr_layer'] != None:
            wr_layer_info = self.partitions[partition_index]['graph'].nodes[self.partitions[partition_index]['wr_layer']]['hw']
            self.partitions[partition_index]['size_wr'] = wr_layer_info.get_parameters_size()['weights']
        else:
            self.partitions[partition_index]['size_wr'] = 0
        
        # update modules
        self.update_modules_partition(partition_index)

    ## validate
    self.check_streams()
    self.check_workload()

"""
def update_partition_streams(self,partition_index):
    # get the input and output node
    input_node  = graphs.get_input_nodes(self.partitions[partition_index]['graph'])[0]
    output_node = graphs.get_output_nodes(self.partitions[partition_index]['graph'])[0]
    # find the input and output channels
    input_channels  = self.partitions[partition_index]['graph'].nodes[input_node]['hw'].channels_in()
    output_channels = self.partitions[partition_index]['graph'].nodes[output_node]['hw'].channels_out()
    # update the streams in and out based on channels
"""

def update_batch_size(self):
    # update the batch size for each partition
    for partition_index in range(len(self.partitions)):
        self.partitions[partition_index]['batch_size'] = self.batch_size

def update_modules(self):
    for node in self.graph.nodes():
        self.graph.nodes[node]['hw'].update()

def update_modules_partition(self, partition_index):
    for layer in self.partitions[partition_index]['graph'].nodes():
        self.partitions[partition_index]['graph'].nodes[layer]['hw'].update()

def update_coefficients(self):
    for node in self.graph.nodes():
        self.graph.nodes[node]['hw'].load_coef()

def update_coefficients_partition(self, partition_index):
    for node in self.partitions[partition_index]['graph'].nodes():
        self.partitions[partition_index]['graph'].nodes[node]['hw'].load_coef()

def update_buffer_depths(self, partition_index):
    # iterative function to get the wait depth of each layer in branch
    def _buffer_depth_branch(layer):
        if graphs.get_graph_inv(self.partitions[partition_index]['graph'])[layer] > 1:
            return 0
        wait_depth = self.partitions[partition_index]['layers'][layer]['hw'].wait_depth()
        return wait_depth + _buffer_depth_branch(self.partitions[partition_index]['graph'][layer][0])
    # iterate over layers in partition
    for layer in self.partitions[partition_index]['graph']:
        # find split layers
        if len(self.partitions[partition_index]['graph'][layer]) > 1:
            buffer_depths = [ _buffer_depth_branch(arc) for arc in self.partitions[partition_index]['graph'][layer] ]
            for arc in self.partitions[partition_index]['graph'][layer]:
                self.node_info[arc]['hw'].buffer_depth = max(buffer_depths) - buffer_depths[self.partitions[partition_index]['graph'][layer].index(arc)]

def update_ports_in(self, partition_index): #TODO
    """
    ## choose max of input port and previous output port
    ports_out_prev = 0
    if (partition_index-1) in partition_order:
        ports_out_prev = self.partitions[partition_index-1]['ports_out']
    ports_in = self.partitions[partition_index]['ports_in']
    return max(ports_out_prev,ports_in)
    """
    return 1

def update_ports_out(self, partition_index): #TODO
    """
    ## choose max of input port and previous output port
    ports_in_next = 0
    if (partition_index+1) in partition_order:
        ports_in_next = self.partitions[partition_index+1]['ports_in']
    ports_out = self.partitions[partition_index]['ports_out']
    return max(ports_in_next,ports_out)
    """
    return 1

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

