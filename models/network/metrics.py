import numpy as np
import tools.graphs as graphs
import tools.matrix as matrix

def get_pipeline_depth(self, partition_index, node): # TODO: change to longest path problem
    # find the pipeline depth of the current node
    pipeline_depth = self.partitions[partition_index]['graph'].nodes[node]['hw'].pipeline_depth()
    # find the longest path to end from this node
    if self.partitions[partition_index]['graph'].out_degree(node) == 0:
        return pipeline_depth
    else:
        return pipeline_depth + max([ 
            self.get_pipeline_depth(partition_index,edge) for edge in graphs.get_next_nodes(self.partitions[partition_index]['graph'],node) ])

def get_interval(self,partition_index):
    # get the interval matrix        
    interval_matrix = matrix.get_interval_matrix(self.partitions[partition_index]['graph'])
    # return the overall interval
    return np.max(np.absolute(interval_matrix))

def get_latency_partition(self, partition_index):
    # get the interval for the partition
    interval = self.get_interval(partition_index)
    # get pipeline depth of partition
    input_node = graphs.get_input_nodes(self.partitions[partition_index]['graph'])[0]
    pipeline_depth = self.get_pipeline_depth(partition_index, input_node) # TODO: find max of all input nodes
    # return the latency (in seconds)
    batch_size  = int(self.batch_size)
    wr_factor   = self.partitions[partition_index]['wr_factor']
    size_wr     = self.partitions[partition_index]['size_wr']
    freq        = self.platform['freq']
    return ( (interval*batch_size+pipeline_depth)*wr_factor + (wr_factor-1)*size_wr )/(freq*1000000) 

def get_latency(self):
    latency = 0
    # iterate over partitions:
    for partition_index in range(len(self.partitions)):
        # accumulate latency for each partition
        latency += self.get_latency_partition(partition_index)
    # return the total latency as well as reconfiguration time
    return latency + (len(self.partitions)-1)*self.platform['reconf_time']

def get_throughput(self):
    # return the frames per second
    return float(self.batch_size)/self.get_latency()

