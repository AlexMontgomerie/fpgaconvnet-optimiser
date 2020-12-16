import math

def get_resource_usage(self, partition_index):
    # initialise resource usage at 0
    resource_usage = { # TODO: initialise with partition resource usage
        'FF'    : 0,
        'LUT'   : 0,
        'DSP'   : 0,
        'BRAM'  : 0
    }
    # iterate over nodes in partition
    for node in self.partitions[partition_index]['graph'].nodes():
        # get the resource usage of the node
        #resource_usage_node = self.node_info[node]['hw'].resource()
        resource_usage_node = self.partitions[partition_index]['graph'].nodes[node]['hw'].resource()
        # update total resource usage for partition
        resource_usage['FF']    += resource_usage_node['FF'] 
        resource_usage['LUT']   += resource_usage_node['LUT'] 
        resource_usage['DSP']   += resource_usage_node['DSP'] 
        resource_usage['BRAM']  += resource_usage_node['BRAM']
    # return resource usage for partition
    return resource_usage

def get_power_average_partition(self, partition_index): #TODO
    return 0

def get_power_average(self): #TODO
    return 0



def get_memory_usage_estimate(self):

    # for sequential networks, our worst-case memory usage is
    # going to be both the largest input and output featuremap pair
    
    # get the largest input featuremap size
    max_input_size = 0
    max_output_size = 0
    for partition in self.partitions:
        input_node  = partition['input_nodes'][0]
        output_node = partition['output_nodes'][0]
        partition_input_size  = partition['graph'].nodes[input_node]['hw'].workload_in(0)*partition['batch_size']
        partition_output_size = partition['graph'].nodes[output_node]['hw'].workload_out(0)*partition['batch_size']*partition['wr_factor']
        if partition_input_size > max_input_size:
            max_input_size = partition_input_size
        if partition_output_size > max_output_size:
            max_output_size = partition_output_size

    data_width = 16

    return math.ceil(((max_input_size + max_output_size)*16)/8)
