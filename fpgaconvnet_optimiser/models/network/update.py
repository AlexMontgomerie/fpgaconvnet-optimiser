from fpgaconvnet_optimiser.models.network.tools import get_group_id
import math
from fpgaconvnet_optimiser.transforms import group
import json
import copy
import fpgaconvnet_optimiser.tools.graphs as graphs

def update_partitions(self):

    # update modules
    for node in self.graph.nodes():
        self.graph.nodes[node]['hw'].update()
    # update partition platforms
    for i,partition in enumerate(self.partitions):
        partition.platform = self.cluster[i%len(self.cluster)]
    # remove all auxiliary layers
    for partition_index in range(len(self.partitions)):

        ## remove squeeze layer
        self.partitions[partition_index].remove_squeeze()

    # update coarse in and out of partition to avoid mismatch
    self.update_coarse_in_out_partition()
    self.update_partition_index()
    #self.update_partition_map()

    # update partitions 
    for partition_index in range(len(self.partitions)):

        ## update streams in and out
        input_node  = graphs.get_input_nodes(self.partitions[partition_index].graph)[0]
        output_node = graphs.get_output_nodes(self.partitions[partition_index].graph)[0]
        self.partitions[partition_index].streams_in  = min(self.partitions[partition_index].max_streams_in,
                self.partitions[partition_index].graph.nodes[input_node]["hw"].coarse_in)
        self.partitions[partition_index].streams_out = min(self.partitions[partition_index].max_streams_out,
                self.partitions[partition_index].graph.nodes[output_node]["hw"].coarse_out)

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
    self.platform['comm_bandwidth']  = float(platform['comm_bandwidth'])


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
                self.partitions[i-1].graph.nodes[output_node]['hw'].coarse_out 
            )

def update_cluster(self, cluster_path):
    # get platform
    self.cluster={}
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
        temp_platform['platform']           = platform_specification
        self.cluster[platform['id']]   = copy.deepcopy(temp_platform)


def update_partition_index(self):
    #find_merges_and_splits
    if len(self.cluster)>1:
        for id,partition,  in enumerate(self.partitions):
            if (id<partition.get_id()):
                for group_id,group in self.groups.items():
                    if group_id > self.get_group_id(id):
                        if len(group)!=0:
                            group.insert(0,group[0]-1)
                            group.pop()
                self.groups[self.get_group_id(id)].pop()
                break
            if (id>partition.get_id()):
                group = self.groups[self.get_group_id(partition.get_id())]
                group.append(group[len(group)-1]+1)
                for group_id,group in self.groups.items():
                    if group_id > self.get_group_id(id):
                        if len(group)!=0:
                            group.append(group[len(group)-1]+1)
                            group.pop(0)
                break 
        if len(self.partitions) < sum([len(group)for _, group in self.groups.items()]):
            #print("FOUND IT")
            self.groups[self.get_group_id(len(self.partitions))].pop()
    for id,partition,  in enumerate(self.partitions):
        #if (id!=partition.get_id()):
        #    print("Partition:{},{}".format(id,partition.get_id()))
            
        partition.set_id(id)
    #print(self.groups)


def init_groups(self):
    #print(len(self.partitions))
    #print(self.cluster)
    if len(self.groups)==1 or len(self.groups)==0:
        self.groups=dict()
        for cluster in range(len(self.cluster)):
            self.groups.setdefault(cluster, []) 
        #self.groups[0].append(0)
    else:
        for cluster in range(len(self.cluster)-len(self.groups)):
            self.groups.setdefault(cluster+len(self.groups), []) 
    #print(self.groups)