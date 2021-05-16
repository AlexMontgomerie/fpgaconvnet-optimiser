import numpy as np
import copy

import fpgaconvnet_optimiser.tools.graphs as graphs
import fpgaconvnet_optimiser.tools.matrix as matrix


def check_ports(self):
    # check each partition
    for p in self.partitions:
        # verify that the number of input ports are not exceeded
        if len(graphs.get_nodes_in(p.graph)) > self.platform['ports']:
            return False
        if len(graphs.get_nodes_out(p.graph)) > self.platform['ports']:
            return False
    return True

def check_resources(self):
    # iterate over partitions
    for partition in self.partitions:
        # get the resource usage for the platform
        partition_resource_usage = partition.get_resource_usage()
        assert partition_resource_usage['FF']   <= (self.platform['constraints']['FF'])
        assert partition_resource_usage['LUT']  <= (self.platform['constraints']['LUT'])
        assert partition_resource_usage['DSP']  <= (self.rsc_allocation*self.platform['constraints']['DSP']) , "ERROR: DSP usage exceeded"
        assert partition_resource_usage['BRAM'] <= (self.rsc_allocation*self.platform['constraints']['BRAM']), "ERROR: BRAM usage exceeded"

def get_resources_bad_partitions(self):
    bad_partitions = {}

    # iterate over partitions
    for partition_index, partition in enumerate(self.partitions):
        bad_resource = {}
        # get the resource usage for the platform
        partition_resource_usage = partition.get_resource_usage()
        if partition_resource_usage['FF']   > (self.platform['constraints']['FF']):
            bad_resource['FF'] = partition_resource_usage['FF']
        if partition_resource_usage['LUT']  > (self.platform['constraints']['LUT']):
            bad_resource['LUT'] = partition_resource_usage['LUT']
        if partition_resource_usage['DSP']  > (self.rsc_allocation*self.platform['constraints']['DSP']):
            bad_resource['DSP'] = partition_resource_usage['DSP']
        if partition_resource_usage['BRAM'] > (self.rsc_allocation*self.platform['constraints']['BRAM']):
            bad_resource['BRAM'] = partition_resource_usage['BRAM']

        if len(bad_resource) > 0:
            bad_partitions[partition_index] = bad_resource

    return bad_partitions

def check_workload(self):
    workload_total = np.zeros( shape=( len(self.edge_list),len(self.node_list) ) , dtype=float )
    # iterate over partitions
    for partition_index in range(len(self.partitions)):
        # get weights reloading factor
        wr_factor = self.partitions[partition_index].wr_factor
        # iterate over layers in partition
        for node in self.partitions[partition_index].graph:
            # check layer is in original graph
            if node not in self.graph:
                continue
            # check workload in
            workload_ref = self.graph.nodes[node]['hw'].workload_in(0)*self.graph.nodes[node]['hw'].streams_in()
            workload_actual = self.partitions[partition_index].graph.nodes[node]['hw'].workload_in(0)*\
                self.partitions[partition_index].graph.nodes[node]['hw'].streams_in()*wr_factor
            assert workload_actual >= workload_ref, f"({node}) workload in imbalance"
            # check workload out
            workload_ref = self.graph.nodes[node]['hw'].workload_out(0)*self.graph.nodes[node]['hw'].streams_out()
            workload_actual = self.partitions[partition_index].graph.nodes[node]['hw'].workload_out(0)*\
                self.partitions[partition_index].graph.nodes[node]['hw'].streams_out()*wr_factor
            assert workload_actual >= workload_ref, f"({node}) workload out imbalance"

def check_streams(self):
    for partition_index in range(len(self.partitions)):
        # get the streams matrix
        streams_matrix = matrix.get_streams_matrix(self.partitions[partition_index].graph) 
        # check that the streams cancel
        assert (np.sum(streams_matrix,axis=1) == 0).all(), ""

def check_partitions(self):
    # iterate over partitions
    for p in self.partitions:
        pass

