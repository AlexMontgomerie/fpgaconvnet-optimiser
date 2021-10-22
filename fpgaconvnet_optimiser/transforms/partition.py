"""
Defines how the graph is split into subgraphs of the model for different reconfigurable components.
The partitioning method is described in terms of horizontal and vertical splits.

.. figure:: ../../figures/horizontal_vertical_split.png
"""

from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE, from_onnx_op_type
from fpgaconvnet_optimiser.transforms.helper import get_all_layers
from itertools import combinations, chain
import random
import fpgaconvnet_optimiser.tools.graphs as graphs
import fpgaconvnet_optimiser.tools.matrix as matrix
import copy
import math

def check_parallel_block(self, partition_index):
    """
    Checks whether the given partition is a parallel block. A parallel block is defined as a
    partition where the input node is a `fpgaconvnet_optimiser.models.layers.SplitLayer` and
    the output node is a `fpgaconvnet_optimiser.models.layers.ConcatLayer`.

    Parameters
    ----------
    partition_index: int
        index of the partition

    Returns
    -------
    bool
        whether the partition is a parallel block
    """
    # check the input and output nodes
    input_node  = graphs.get_input_nodes(self.partitions[partition_index].graph)[0]
    output_node = graphs.get_output_nodes(self.partitions[partition_index].graph)[0]
    # check if not a parallel block
    if not ((self.partitions[partition_index].graph.nodes[input_node]['type'] == LAYER_TYPE.Split) and \
            (self.partitions[partition_index].graph.nodes[output_node]['type'] == LAYER_TYPE.Concat)):
        return False
    # check the number of split and concat nodes
    n_split  = len(get_all_layers(self.partitions[partition_index].graph,LAYER_TYPE.Split))
    n_concat = len(get_all_layers(self.partitions[partition_index].graph,LAYER_TYPE.Concat))
    # break if there's too many parallel blocks
    if (n_split > 1) or (n_concat > 1):
        return False
    # is a standalone parallel block
    return True

def get_all_horizontal_splits(self, partition_index, allowed_partitions=[]):
    """
    Gets all the possible horizontal splits for a given partition.

    Parameters
    ----------
    partition_index: int
        index of the partition
    allowed_partitions: list of tuple of str, optional
        a list of pairs of layer types

    Returns
    -------
    list of tuple of str
        a list of all legal horizontal splits. This is a list of
        pairs of nodes in the graph.
    """
    # function to iterate over graph
    def _iterate_graph(edge_list,input_node,in_parallel_block):
        # check if end
        if self.partitions[partition_index].graph.out_degree(input_node) == 0:
            return edge_list
        # next node
        next_node = graphs.get_next_nodes(self.partitions[partition_index].graph,input_node)[0]
        # check if exiting parallel block
        if self.partitions[partition_index].graph.in_degree(input_node) > 1:
            in_parallel_block = False
        # check if entering parallel block
        if self.partitions[partition_index].graph.out_degree(input_node) > 1:
            return _iterate_graph(edge_list,next_node,True)
        # skip node - concat partition
        if self.partitions[partition_index].graph.in_degree(next_node) > 1:
            return _iterate_graph(edge_list,next_node,in_parallel_block)
        # append to partition list
        if not in_parallel_block:
            edge_list.append((input_node,next_node))
        return _iterate_graph(edge_list,next_node,in_parallel_block)
    # iterate over graph from start node
    input_node = graphs.get_input_nodes(self.partitions[partition_index].graph)[0]
    all_horizontal_splits = _iterate_graph([], input_node, False)
    # filter all the splits
    filtered_horizontal_splits = []
    # if there is a filter on the types of splits, apply this filter
    if allowed_partitions:
        for split in all_horizontal_splits:
            # get layer types
            layer_types = (
                self.partitions[partition_index].graph.nodes[split[0]]["type"],
                self.partitions[partition_index].graph.nodes[split[1]]["type"]
            )
            # see if it's in the allowed partitions
            if layer_types in allowed_partitions:
                filtered_horizontal_splits.append(split)
        # return the filtered splits
        return filtered_horizontal_splits
    else:
        # return the all splits
        return all_horizontal_splits

def get_all_vertical_splits(self, partition_index):
    """
    Gets all the possible vertical splits for a given partition.

    Parameters
    ----------
    partition_index: int
        index of the partition

    Returns
    -------
    list of tuple of str
        a list of all legal vertical splits. This is a list of
        pairs of nodes in the graph.
    """
    # check if the partition is a parallel block
    if not self.check_parallel_block(partition_index):
        return None
    vertical_splits = []
    # get the input node (a split layer)
    input_node = graphs.get_input_nodes(self.partitions[partition_index].graph)[0]
    assert self.partitions[partition_index].graph.nodes[input_node]["type"] == LAYER_TYPE.Split
    # get all the nodes after the the split layer
    split_nodes = graphs.get_next_nodes(self.partitions[partition_index].graph,input_node)
    # get all comibinations of the nodes after the split
    subsets = [v for a in range(len(split_nodes)) for v in combinations(split_nodes, a)]
    # filter the combinations of nodes
    for i in range(1,math.ceil(len(subsets)/2)):
        vertical_splits.append([list(chain(subsets[i])), [e for e in split_nodes if e not in subsets[i]]])
    # return all vertical splits
    return vertical_splits

def get_all_horizontal_merges(self,partition_index):
    """
    Gets all the possible horizontal merges for a given partition. There is at most
    two possible merges for a given partition

    Parameters
    ----------
    partition_index: int
        index of the partition

    Returns
    -------
    list of tuple of int
        a list of two tuples. These tuples are either empty or contain a pair
        of partition indices that can be merged.
    """
    # partition pairs
    partition_pairs = [(),()]
    # get the next node
    output_node = graphs.get_output_nodes(self.partitions[partition_index].graph)[0]
    def _find_next_partition():
        if self.graph.out_degree(output_node) > 0:
            next_node = graphs.get_next_nodes(self.graph,output_node)[0]
            # find the partition pair for the output
            for i in range(len(self.partitions)):
                if i == partition_index:
                    continue
                if next_node in graphs.get_input_nodes(self.partitions[i].graph):
                    # check that this is a complete block
                    if ( self.partitions[i].graph.out_degree(next_node) ==
                            self.graph.out_degree(next_node) ) or ( self.graph.out_degree(next_node) == 1 ):
                        return (partition_index,i)
        return ()

    # check that if it's a concat layer, it's complete
    if self.graph.in_degree(output_node) > 1:
        if ( self.partitions[partition_index].graph.in_degree(output_node) ==
                self.graph.in_degree(output_node) ) or ( self.graph.out_degree(output_node) == 1 ):
            partition_pairs[0] = _find_next_partition()
    else:
        partition_pairs[0] = _find_next_partition()

    # get the previous node
    input_node = graphs.get_input_nodes(self.partitions[partition_index].graph)[0]
    def _find_prev_partition():
        if self.graph.in_degree(input_node) > 0:
            prev_node = graphs.get_prev_nodes(self.graph,input_node)[0]
            # find the partition pair for the input
            for i in range(len(self.partitions)):
                if i == partition_index:
                    continue
                if prev_node in graphs.get_output_nodes(self.partitions[i].graph):
                    # check that this is a complete block
                    if ( self.partitions[i].graph.in_degree(prev_node) ==
                            self.graph.in_degree(prev_node) ) or ( self.graph.out_degree(prev_node) == 1 ):
                        return (i,partition_index)
        return ()

    # check that if it's a split layer, it's complete
    if self.graph.out_degree(input_node) > 1:
        if ( self.partitions[partition_index].graph.out_degree(input_node) ==
                self.graph.out_degree(input_node) ) or ( self.graph.out_degree(next_node) == 1 ):
            partition_pairs[1] = _find_prev_partition()
    else:
        partition_pairs[1] = _find_prev_partition()

    return partition_pairs

def get_all_vertical_merges(self,partition_index):
    """
    Gets all the possible vertical merges for a given partition.

    Parameters
    ----------
    partition_index: int
        index of the partition
    """
    # check if parallel block
    if not self.check_parallel_block(partition_index):
        return []
    # partition pairs
    partition_pairs = []
    # get input and output node
    input_node  = graphs.get_input_nodes(self.partitions[partition_index].graph)[0]
    output_node = graphs.get_output_nodes(self.partitions[partition_index].graph)[0]
    # find partitions with the same input,output pairs
    for i in range(len(self.partitions)):
        if partition_index == i:
            continue
        if (self.partitions[i].input_nodes[0] == input_node) and (self.partitions[i].output_nodes[0] == output_node):
            partition_pairs.append((i,partition_index))
    return partition_pairs

def split_horizontal(self, partition_index, edge):
    """
    Performs a horizontal split for a partition. The edge parameter defines
    where in the graph of that partition is going to be split.



    Parameters
    ----------
    partition_index: int
        index of the partition
    edge: list of str
        a pair of two adjacent nodes in the graph, where the split will
        take place
    """

    # remove weights reloading transform
    self.partitions[partition_index].remove_weights_reloading_transform()
    # create a new partition
    self.partitions.insert(partition_index,copy.deepcopy(self.partitions[partition_index]))
    # split graph
    partition_graphs = graphs.split_graph_horizontal(self.partitions[partition_index].graph,edge)
    self.partitions[partition_index].graph   = partition_graphs[0]
    self.partitions[partition_index+1].graph = partition_graphs[1]
    # apply max weights reloading to both
    wr_layer = self.partitions[partition_index].wr_layer
    if wr_layer in self.partitions[partition_index].graph.nodes:
        self.partitions[partition_index+1].apply_max_weights_reloading()
    else:
        self.partitions[partition_index].wr_layer  = self.partitions[partition_index].get_wr_layer()
        self.partitions[partition_index].wr_factor = 1
    self.partitions[partition_index].apply_weights_reloading_transform()
    self.partitions[partition_index+1].apply_weights_reloading_transform()
    #self.partitions[partition_index+1].apply_max_weights_reloading()

def split_vertical(self, partition_index, nodes):
    # remove weights reloading transform
    self.partitions[partition_index].remove_weights_reloading_transform()
     # create a new partition
    self.partitions.insert(partition_index,copy.deepcopy(self.partitions[partition_index]))
    # split the graph
    partition_graphs = graphs.split_graph_vertical(self.partitions[partition_index].graph,nodes)
    self.partitions[partition_index].graph   = partition_graphs[0]
    self.partitions[partition_index+1].graph = partition_graphs[1]
    # apply max weights reloading to both
    self.partitions[partition_index].apply_max_weights_reloading()
    self.partitions[partition_index+1].apply_max_weights_reloading()

def merge_horizontal(self,partition_index_a,partition_index_b):
    # remove weights reloading transform
    self.partitions[partition_index_a].remove_weights_reloading_transform()
    self.partitions[partition_index_b].remove_weights_reloading_transform()
    # merge graphs
    graph = graphs.merge_graphs_horizontal(self.partitions[partition_index_a].graph,self.partitions[partition_index_b].graph)
    self.partitions[partition_index_a].graph = graph
    # apply last partitions weights reloading
    #self.partitions[partition_index_a].wr_layer  = self.partitions[partition_index_b].wr_layer
    #self.partitions[partition_index_a].wr_factor = self.partitions[partition_index_b].wr_factor
    #self.partitions[partition_index_a].apply_weights_reloading_transform()
    self.partitions[partition_index_a].apply_max_weights_reloading()
    self.partitions[partition_index_b].apply_max_weights_reloading()
    # remove last partition
    del self.partitions[partition_index_b]

def merge_vertical(self, partition_index_a, partition_index_b):
    # remove weights reloading transform
    self.partitions[partition_index_a].remove_weights_reloading_transform()
    self.partitions[partition_index_b].remove_weights_reloading_transform()
    # merge graphs
    graph = graphs.merge_graphs_vertical(self.partitions[partition_index_a].graph,self.partitions[partition_index_b].graph)
    # update graphs
    self.partitions[partition_index_a].graph = graph
    # remove weights reloading
    self.partitions[partition_index_a].apply_max_weights_reloading()
    # remove last partition
    del self.partitions[partition_index_b]

def split_horizontal_complete(self):

    # get the allowed partitions
    allowed_partitions = []
    if "partition" in self.transforms_config:
        allowed_partitions = self.transforms_config["partition"]["allowed_partitions"]

    # function to find a horizontal split
    def _find_horizontal_split_partition():
        for i in range(len(self.partitions)):
            if self.get_all_horizontal_splits(i, allowed_partitions=allowed_partitions):
                return i
        return None

    partition_index = _find_horizontal_split_partition()

    # keep iterating until all horizontal splits done
    while partition_index != None:
        # apply first possible split
        horizontal_splits = self.get_all_horizontal_splits(partition_index, allowed_partitions=allowed_partitions)
        self.split_horizontal(partition_index,horizontal_splits[0])
        # find next partition
        partition_index = _find_horizontal_split_partition()

def split_vertical_complete(self):
    def _find_vertical_split_partition():
        for i in range(len(self.partitions)):
            if self.get_all_vertical_splits(i):
                return i
        return None
    partition_index = _find_vertical_split_partition()
    # keep iterating until all horizontal splits done
    while partition_index != None:
        # apply first possible split
        vertical_splits = self.get_all_vertical_splits(partition_index)
        self.split_vertical(partition_index,vertical_splits[0])
        # find next partition
        partition_index = _find_vertical_split_partition()

def split_complete(self): # TODO: this assumes no parallel blocks within parallel blocks
    self.split_horizontal_complete()
    self.split_vertical_complete()
    self.split_horizontal_complete()

def merge_horizontal_complete(self):
    def _find_horizontal_merge_partition():
        for i in range(len(self.partitions)):
            horizontal_merges = self.get_all_horizontal_merges(i)
            if horizontal_merges[0] or horizontal_merges[1]:
                return i
        return None
    # keep iterating until all horizontal splits done
    partition_index = _find_horizontal_merge_partition()
    while partition_index != None:
        # apply first possible split
        horizontal_merges = self.get_all_horizontal_merges(partition_index)
        if horizontal_merges[0]:
            self.merge_horizontal(*horizontal_merges[0])
        else:
            self.merge_horizontal(*horizontal_merges[1])
        # find next partition
        partition_index = _find_horizontal_merge_partition()

def merge_vertical_complete(self):
    def _find_vertical_merge_partition():
        for i in range(len(self.partitions)):
            if self.get_all_vertical_merges(i):
                return i
        return None
    partition_index = _find_vertical_merge_partition()
    # keep iterating until all horizontal splits done
    while partition_index != None:
        # apply first possible split
        vertical_merges = self.get_all_vertical_merges(partition_index)
        self.merge_vertical(*vertical_merges[0])
        # find next partition
        partition_index = _find_vertical_merge_partition()

def merge_complete(self):
    self.merge_horizontal_complete()
    self.merge_vertical_complete()
    self.merge_horizontal_complete()

def apply_random_partition(self, partition_index):
   # choose randomly between merge or split
    ## split partition
    transform_type = random.choice(["split", "merge"])
    allowed_partitions = []
    if "partition" in self.transforms_config:
        transform_type = random.choice(self.transforms_config["partition"]["allowed_type"])
        allowed_partitions = self.transforms_config["partition"]["allowed_partitions"]
    if transform_type == 'split':
        ## get all possible splits
        horizontal_splits = self.get_all_horizontal_splits(partition_index, allowed_partitions=allowed_partitions)
        vertical_splits   = self.get_all_vertical_splits(partition_index)
        ## split horizontally first
        if horizontal_splits:
            self.split_horizontal(partition_index,random.choice(horizontal_splits))
        ## vertically second choice
        elif vertical_splits:
            self.split_vertical(partition_index,random.choice(vertical_splits))
    ## merge partition
    if transform_type == 'merge':
        # get all possible merges
        horizontal_merges = self.get_all_horizontal_merges(partition_index)
        #print(horizontal_merges)
        #graphs.print_graph(self.partitions[partition_index].graph)
        horizontal_merges = list(filter(None,horizontal_merges))
        vertical_merges   = self.get_all_vertical_merges(partition_index)
        ## merge horizontally first
        if horizontal_merges:
            horizontal_merge = random.choice(horizontal_merges)
            self.merge_horizontal(*horizontal_merge)
        elif vertical_merges:
            self.merge_vertical(*random.choice(vertical_merges))

