import math
import numpy as np
import fpgaconvnet_optimiser.tools.graphs as graphs
import fpgaconvnet_optimiser.tools.matrix as matrix

def get_pipeline_depth(self, node): # TODO: change to longest path problem
    # find the pipeline depth of the current node
    pipeline_depth = self.graph.nodes[node]['hw'].pipeline_depth()
    # find the longest path to end from this node
    if self.graph.out_degree(node) == 0:
        return pipeline_depth
    else:
        return pipeline_depth + max([ 
            self.get_pipeline_depth(edge) for edge in graphs.get_next_nodes(self.graph,node) ])

def get_interval(self):
    # get the interval matrix        
    interval_matrix = matrix.get_interval_matrix(self.graph)
    # return the overall interval
    return np.max(np.absolute(interval_matrix))

def get_latency(self,freq):
    # get the interval for the partition
    interval = self.get_interval()
    # get pipeline depth of partition
    input_node = graphs.get_input_nodes(self.graph)[0]
    pipeline_depth = self.get_pipeline_depth(input_node) # TODO: find max of all input nodes
    # return the latency (in seconds)
    batch_size  = int(self.batch_size)
    wr_factor   = self.wr_factor
    size_wr     = self.size_wr
    return ( (interval*batch_size+pipeline_depth)*wr_factor + (wr_factor-1)*size_wr )/(freq*1000000) 

def get_bandwidth_in(self,freq):
    # get the interval for the partition
    interval = self.get_interval()
    # get workload and streams in 
    input_node = graphs.get_input_nodes(self.graph)[0]
    workload = self.graph.nodes[input_node]["hw"].workload_in(0)
    streams = self.streams_in
    # calculate rate from interval
    rate = workload / (interval*streams) 
    # get bandwidth (GB/s)
    return (rate*streams*self.data_width*freq)/8000

def get_bandwidth_out(self,freq):
    # get the interval for the partition
    interval = self.get_interval()
    # get workload and streams out 
    output_node = graphs.get_output_nodes(self.graph)[0]
    workload = self.graph.nodes[output_node]["hw"].workload_in(0)
    streams = self.streams_out
    # calculate rate from interval
    rate = workload / (interval*streams) 
    # get bandwidth (GB/s)
    return (rate*streams*self.data_width*freq)/8000

def get_comm_interval_in(self):
    # get the interval for the partition
    bandwidth = self.platform['platform']["comm_bandwidth"]
    # get workload and streams out 
    input_node = graphs.get_input_nodes(self.graph)[0]
    workload = self.graph.nodes[input_node]["hw"].workload_in(0)
    # calculate rate from interval
    interval =  math.ceil(workload*self.data_width*self.platform['platform']["freq"] / (bandwidth*1000000000))
    # get bandwidth (GB/s)
    return interval

def get_comm_interval_out(self):
    # get the interval for the partition
    bandwidth = self.platform['platform']["comm_bandwidth"]
    # get workload and streams out 
    output_node = graphs.get_output_nodes(self.graph)[0]
    workload = self.graph.nodes[output_node]["hw"].workload_in(0)
    # interval from bandwidth and workload
    interval = math.ceil(workload*self.data_width*self.platform['platform']["freq"] / (bandwidth*1000000000))
    return interval



def get_total_operations(self):
    return sum([self.graph.nodes[node]['hw'].get_operations() for node in self.graph.nodes])
    
