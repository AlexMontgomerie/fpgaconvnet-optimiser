def get_next_partition(self,partition_id):

    next_partition_id = partition_id + 1
    if next_partition_id == len(self.partitions):
        return -1 
    # TODO: Check if there is a connection between partition and next partition next_cluster_id=self.partitions[partition_id].platform["connections_in"]
    # print("Cluster {}, Partition {}".format(next_cluster_id,next_partition_id))
    return next_partition_id
    
def get_group_id(self,partition_id):
    for group_id,group_partitions in self.groups.items():
        if partition_id in group_partitions:
            return group_id
    return -1