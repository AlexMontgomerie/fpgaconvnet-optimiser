import json
import copy
import fpgaconvnet_optimiser.tools.graphs as graphs

def update_partitions(self):

    # update modules
    for node in self.graph.nodes():
        self.graph.nodes[node]['hw'].update()
   
    # update partitions 
    for partition_index in range(len(self.partitions)):
        self.partitions[partition_index].set_id(partition_index)
        self.update_partition_map()

        ## update all modules again
        self.partitions[partition_index].update_modules()
       
        ## remove auxiliary layers
        self.partitions[partition_index].remove_squeeze()
        
        ## fix weights reloading in each partition
        #self.fix_weights_reloading(partition_index)
        
        ## apply weights reloading
        #self.apply_weights_reloading_transform(partition_index)
        
        ## update all modules again
        self.partitions[partition_index].update_modules()
       
        ## fix coarse in for all layers
        self.partitions[partition_index].fix_coarse()

        ## update all modules again
        self.partitions[partition_index].update_modules()
 
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


def update_cluster(self, cluster_path):
    # get platform
    with open(cluster_path,'r') as f:
        cluster = json.load(f)

    for platform in cluster:
        with open(platform['platform'],'r') as f:
            platform_specification = json.load(f)

        temp_platform = {}
        temp_platform['id']                 = platform['id']
        temp_platform['connections_in']     = platform['connections_in']
        temp_platform['connections_out']    = platform['connections_out']

        temp_platform['name']               = platform_specification['name']+"_{id:03d}".format(id=temp_platform['id'])
        temp_platform['specification']      = platform_specification
        self.cluster[temp_platform['id']]    = copy.deepcopy(temp_platform)


def update_partition_map(self):
    self.partitionmap={partition.get_id():partition.get_id()%len(self.cluster)+1 for partition in self.partitions}