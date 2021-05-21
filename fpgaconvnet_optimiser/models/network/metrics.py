import fpgaconvnet_optimiser.tools.graphs as graphs
import math

def get_multi_fpga_throughput(self):
         # return the frames per second
        interval = 0
        for partition in self.partitions:
            if interval < partition.get_interval():
                interval = partition.get_interval()
        print("Throughput based on max interval= {}",1/interval*self.platform["freq"]*1000000)
        #no_weight_reloading = False
        no_weight_reloading = True
        if len(self.cluster)>=len(self.partitions):
            no_weight_reloading = True
            for partition in self.partitions:
                if partition.wr_factor > 1:
                    no_weight_reloading = False
        print("No weight reloading {}".format(no_weight_reloading))
        return 1/interval*self.platform["freq"]*1000000

def get_cluster_latency(self,cluster,freq):
    max_interval=0
    latency=0
    batch_size  = int(self.batch_size)
    for i,partition in enumerate(cluster):
            # get the interval for the partition
        interval = partition.get_interval()
        comm_interval = partition.get_comm_interval_out()
        partition.platform["connections_out"]
        # get pipeline depth of partition
        input_node = graphs.get_input_nodes(partition.graph)[0]
        pipeline_depth = partition.get_pipeline_depth(input_node) # TODO: find max of all input nodes
        # return the latency (in seconds)
        wr_factor   = partition.wr_factor
        size_wr     = partition.size_wr
        latency += (pipeline_depth*wr_factor + (wr_factor-1)*(size_wr+batch_size*interval))/(freq*1000000)
        max_interval = max(max_interval,interval,comm_interval) 
    latency += max_interval*batch_size/(freq*1000000)
    
    return latency

def get_latency(self):
    latency = 0
    # iterate over partitions:
    for i in range(math.ceil(len(self.partitions)/len(self.cluster))):
        # accumulate latency for each partition
        cluster = self.partitions[i*len(self.cluster):max((i+1)*len(self.cluster),len(self.partitions))]
        latency += self.get_cluster_latency(cluster,self.platform["freq"])
    # return the total latency as well as reconfiguration time
    return latency + (math.ceil(len(self.partitions)/len(self.cluster))-1)*self.platform["reconf_time"]

def get_throughput(self):
    return float(self.batch_size)/self.get_latency()