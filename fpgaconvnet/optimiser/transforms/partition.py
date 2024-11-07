"""
Defines how the graph is split into subgraphs of the model for different reconfigurable components.
"""

from itertools import combinations, chain
import random
import copy
import math
import pickle
import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE
import fpgaconvnet.tools.matrix as matrix
from fpgaconvnet.optimiser.transforms.helper import get_all_layers
import fpgaconvnet.optimiser.transforms.off_chip_streaming as off_chip_streaming
import fpgaconvnet.optimiser.transforms.weights_reloading as weights_reloading

def check_parallel_block(net, partition_index):
    input_node = graphs.get_input_nodes(net.partitions[partition_index].graph)[0]
    return net.partitions[partition_index].graph.nodes[input_node]['type'] in [LAYER_TYPE.Split, LAYER_TYPE.Chop]

def check_config_allowed_partitions(allowed_partitions, node0, node1):
    # get the node types
    node0_type = node0['type']
    node1_type = node1['type']
    if node0_type == LAYER_TYPE.EltWise:
        node0_type = node0['hw']._op_type
    if node1_type == LAYER_TYPE.EltWise:
        node1_type = node1['hw']._op_type
    # check if allowed
    if allowed_partitions is not None:
        for allowed_split in allowed_partitions:
            if (allowed_split[0] == "*" or allowed_split[0] == node0_type) and \
               (allowed_split[1] == "*" or allowed_split[1] == node1_type):
                return True
        return False
    else:
        return True

def get_all_horizontal_splits(net, partition_index, allowed_partitions=None):
    # function to iterate over graph
    def _iterate_graph(edge_list,input_node,in_parallel_block):
        # check if end
        if net.partitions[partition_index].graph.out_degree(input_node) == 0:
            return edge_list
        # next node
        next_node = graphs.get_next_nodes(net.partitions[partition_index].graph,input_node)[0]
        # check if exiting parallel block
        if net.partitions[partition_index].graph.in_degree(input_node) > 1:
            in_parallel_block = False
        # check if entering parallel block
        if net.partitions[partition_index].graph.out_degree(input_node) > 1:
            in_parallel_block = True
            output_node = graphs.get_output_nodes(net.partitions[partition_index].graph)[0]
            if graphs.get_next_nodes(net.partitions[partition_index].graph,input_node)[1] != output_node:
                return _iterate_graph(edge_list,next_node,in_parallel_block)
        # skip node - concat partition
        if net.partitions[partition_index].graph.in_degree(next_node) > 1:
            return _iterate_graph(edge_list,next_node,in_parallel_block)
        # skip node - split position not valid
        if not check_config_allowed_partitions(allowed_partitions,
            net.partitions[partition_index].graph.nodes[input_node],
            net.partitions[partition_index].graph.nodes[next_node]):
            return _iterate_graph(edge_list,next_node,in_parallel_block)
        # append to partition list
        if not in_parallel_block:
            edge_list.append((input_node,next_node))
        return _iterate_graph(edge_list,next_node,in_parallel_block)
    # iterate over graph from start node
    input_node = graphs.get_input_nodes(net.partitions[partition_index].graph)[0] # TODO: assuming only one input
    edge_list = _iterate_graph([],input_node,False)
    if len(graphs.get_output_nodes(net.partitions[partition_index].graph)) > 1:
        # handle the case of multiple outputs
        for node in net.partitions[partition_index].graph.nodes:
            if net.partitions[partition_index].graph.in_degree(node) > 1:
                edge_list += _iterate_graph([],node,False)
                edge_list = list(sorted(set(edge_list)))
    return edge_list

def get_all_vertical_splits(net, partition_index): # TODO: improve to get all possible combinations
    # check if parallel block
    if not check_parallel_block(net, partition_index):
        return None
    vertical_splits = []
    input_node = graphs.get_input_nodes(net.partitions[partition_index].graph)[0]
    split_nodes = graphs.get_next_nodes(net.partitions[partition_index].graph,input_node)
    subsets = [v for a in range(len(split_nodes)) for v in combinations(split_nodes, a)]
    for i in range(1,math.ceil(len(subsets)/2)):
        vertical_splits.append([list(chain(subsets[i])), [e for e in split_nodes if e not in subsets[i]]])
    return vertical_splits

def get_all_horizontal_merges(net, partition_index):
    # partition pairs
    partition_pairs = [(),()]
    # get the next node
    output_node = graphs.get_output_nodes(net.partitions[partition_index].graph)[0]
    def _find_next_partition():
        if net.graph.out_degree(output_node) > 0:
            next_node = graphs.get_next_nodes(net.graph,output_node)[0]
            # find the partition pair for the output
            for i in range(len(net.partitions)):
                if next_node in graphs.get_input_nodes(net.partitions[i].graph):
                    # check that this is a complete block if it's a split
                    if partition_index + 1 == i:#net.partitions[i].graph.out_degree(next_node) == net.graph.out_degree(next_node):
                        partition_pairs[0] = (partition_index,i)

    # check that if it's a concat layer, it's complete
    if net.graph.in_degree(output_node) > 1:
        if 1:#net.partitions[partition_index].graph.in_degree(output_node) == net.graph.in_degree(output_node):
            _find_next_partition()
    else:
        _find_next_partition()

    # get the previous node
    input_node = graphs.get_input_nodes(net.partitions[partition_index].graph)[0]
    def _find_prev_partition():
        if net.graph.in_degree(input_node) > 0:
            prev_node = graphs.get_prev_nodes(net.graph,input_node)[0]
            # find the partition pair for the input
            for i in range(len(net.partitions)):
                if prev_node in graphs.get_output_nodes(net.partitions[i].graph):
                    # check that this is a complete block
                    if i + 1 == partition_index:#net.partitions[i].graph.in_degree(prev_node) == net.graph.in_degree(prev_node):
                        partition_pairs[1] = (i,partition_index)

    # check that if it's a split layer, it's complete
    if net.graph.out_degree(input_node) > 1:
        if 1:#net.partitions[partition_index].graph.out_degree(input_node) == net.graph.out_degree(input_node):
            _find_prev_partition()
    else:
        _find_prev_partition()

    return partition_pairs

def get_all_vertical_merges(net, partition_index):
    # check if parallel block
    if not check_parallel_block(net, partition_index):
        return []
    # partition pairs
    partition_pairs = []
    # get input and output node
    input_node  = graphs.get_input_nodes(net.partitions[partition_index].graph)[0]
    output_node = graphs.get_output_nodes(net.partitions[partition_index].graph)[0]
    # find partitions with the same input,output pairs
    for i in range(len(net.partitions)):
        if partition_index == i:
            continue
        if (net.partitions[i].input_nodes[0] == input_node) and (net.partitions[i].output_nodes[0] == output_node):
            partition_pairs.append((i,partition_index))
    return partition_pairs

def split_horizontal(net, partition_index, edge):
    # remove weights reloading transform
    weights_reloading.remove_weights_reloading_transform(net.partitions[partition_index])
    # create a new partition
    net.partitions.insert(partition_index,pickle.loads(pickle.dumps(net.partitions[partition_index])))
    # split graph
    partition_graphs = graphs.split_graph_horizontal(net.partitions[partition_index].graph,edge)
    net.partitions[partition_index].graph   = partition_graphs[0]
    net.partitions[partition_index+1].graph = partition_graphs[1]
    # apply max weights reloading to both
    weights_reloading.apply_max_weights_reloading(net.partitions[partition_index])
    weights_reloading.apply_max_weights_reloading(net.partitions[partition_index+1])

def split_vertical(net, partition_index, nodes):
    # remove weights reloading transform
    weights_reloading.remove_weights_reloading_transform(net.partitions[partition_index])
     # create a new partition
    net.partitions.insert(partition_index,pickle.loads(pickle.dumps(net.partitions[partition_index])))
    # split the graph
    partition_graphs = graphs.split_graph_vertical(net.partitions[partition_index].graph,nodes)
    net.partitions[partition_index].graph   = partition_graphs[0]
    net.partitions[partition_index+1].graph = partition_graphs[1]
    # apply max weights reloading to both
    weights_reloading.apply_max_weights_reloading(net.partitions[partition_index])
    weights_reloading.apply_max_weights_reloading(net.partitions[partition_index+1])

def merge_horizontal(net, partition_index_a, partition_index_b):
    # remove weights reloading transform
    weights_reloading.remove_weights_reloading_transform(net.partitions[partition_index_a])
    weights_reloading.remove_weights_reloading_transform(net.partitions[partition_index_b])

    # fix streaming
    off_chip_streaming.fix_streaming(net, partition_index_a, partition_index_b)

    # merge graphs
    graph = graphs.merge_graphs_horizontal(
            net.partitions[partition_index_a].graph,
            net.partitions[partition_index_b].graph,
            net.network_branch_edges)

    net.partitions[partition_index_a].graph = graph
    # apply max weights reloading
    weights_reloading.apply_max_weights_reloading(net.partitions[partition_index_a])

    # remove last partition
    del net.partitions[partition_index_b]

def merge_vertical(net, partition_index_a, partition_index_b):
    # remove weights reloading transform
    weights_reloading.remove_weights_reloading_transform(net.partitions[partition_index_a])
    weights_reloading.remove_weights_reloading_transform(net.partitions[partition_index_b])
    # merge graphs
    graph = graphs.merge_graphs_vertical(net.partitions[partition_index_a].graph,net.partitions[partition_index_b].graph)
    # update graphs
    net.partitions[partition_index_a].graph = graph
    # remove weights reloading
    weights_reloading.apply_max_weights_reloading(net.partitions[partition_index_a])
    # remove last partition
    del net.partitions[partition_index_b]

def split_horizontal_complete(net, allowed_partitions):
    # function to find a horizontal split
    def _find_horizontal_split_partition():
        for i in range(len(net.partitions)):
            if get_all_horizontal_splits(net, i, allowed_partitions):
                return i
        return None
    partition_index = _find_horizontal_split_partition()
    # keep iterating until all horizontal splits done
    while partition_index != None:
        # apply first possible split
        horizontal_splits = get_all_horizontal_splits(net, partition_index, allowed_partitions)
        split_horizontal(net, partition_index, horizontal_splits[0])
        # find next partition
        partition_index = _find_horizontal_split_partition()

def split_vertical_complete(net):
    def _find_vertical_split_partition():
        for i in range(len(net.partitions)):
            if get_all_vertical_splits(net, i):
                return i
        return None
    partition_index = _find_vertical_split_partition()
    # keep iterating until all horizontal splits done
    while partition_index != None:
        # apply first possible split
        vertical_splits = get_all_vertical_splits(net, partition_index)
        split_vertical(net, partition_index,vertical_splits[0])
        # find next partition
        partition_index = _find_vertical_split_partition()

def split_complete(net, allowed_partitions, vertical=False):
    partition_num = len(net.partitions)
    while True:
        split_horizontal_complete(net, allowed_partitions)
        if vertical:
            split_vertical_complete(net)
        if len(net.partitions) == partition_num:
            break
        partition_num = len(net.partitions)

def merge_horizontal_complete(net):
    def _find_horizontal_merge_partition():
        for i in range(len(net.partitions)):
            horizontal_merges = net.get_all_horizontal_merges(i)
            if horizontal_merges[0] or horizontal_merges[1]:
                return i
        return None
    # keep iterating until all horizontal splits done
    partition_index = _find_horizontal_merge_partition()
    while partition_index != None:
        # apply first possible split
        horizontal_merges = net.get_all_horizontal_merges(partition_index)
        if horizontal_merges[0]:
            net.merge_horizontal(*horizontal_merges[0])
        else:
            net.merge_horizontal(*horizontal_merges[1])
        # find next partition
        partition_index = _find_horizontal_merge_partition()

def merge_vertical_complete(net):
    def _find_vertical_merge_partition():
        for i in range(len(net.partitions)):
            if net.get_all_vertical_merges(i):
                return i
        return None
    partition_index = _find_vertical_merge_partition()
    # keep iterating until all horizontal splits done
    while partition_index != None:
        # apply first possible split
        vertical_merges = net.get_all_vertical_merges(partition_index)
        net.merge_vertical(*vertical_merges[0])
        # find next partition
        partition_index = _find_vertical_merge_partition()

def merge_complete(net):
    net.merge_horizontal_complete()
    net.merge_vertical_complete()
    net.merge_horizontal_complete()

def merge_single_layer_partition_to_prev(net, allowed_layers):
    def _find_single_layer_partition():
        for i in range(len(net.partitions)):
            if len(net.partitions[i].graph.nodes) == 1:
                input_node = graphs.get_input_nodes(net.partitions[i].graph)[0]
                if net.partitions[i].graph.nodes[input_node]['type'] in allowed_layers:
                    return i, input_node
        return None, None
    partition_index, partition_node = _find_single_layer_partition()
    # keep iterating until all single layer partitions are merged
    while partition_index != None:
        # get the previous connected nodes of the partition node
        prev_nodes = graphs.get_prev_nodes(net.graph, partition_node)
        # find the partition(s) that feed into this node
        prev_partitions = [i for i in range(len(
            net.partitions)) for prev_node in prev_nodes if prev_node in graphs.get_output_nodes(net.partitions[i].graph)]
        assert len(prev_partitions) == 1, "WARNING: multiple partitions feeding into single layer partition. Currently not handled"
        # merge partition to previous
        merge_horizontal(net, prev_partitions[0], partition_index)
        # find next partition to merge
        partition_index, partition_node = _find_single_layer_partition()

def merge_single_layer_partition_to_next(net, allowed_layers):
    def _find_single_layer_partition():
        for i in range(len(net.partitions)):
            if len(net.partitions[i].graph.nodes) == 1:
                input_node = graphs.get_input_nodes(net.partitions[i].graph)[0]
                if net.partitions[i].graph.nodes[input_node]['type'] in allowed_layers or (hasattr(net.partitions[i].graph.nodes[input_node]['hw'], 'op_type') and net.partitions[i].graph.nodes[input_node]['hw'].op_type in allowed_layers):
                    return i, input_node
        return None, None
    partition_index, partition_node = _find_single_layer_partition()
    # keep iterating until all single layer partitions are merged
    while partition_index != None:
        # get the next connected nodes of the partition node
        next_nodes = graphs.get_next_nodes(net.graph, partition_node)
        # find the partition(s) that this node feeds
        next_partitions = [i for i in range(len(
            net.partitions)) for next_node in next_nodes if next_node in graphs.get_input_nodes(net.partitions[i].graph)]
        assert len(next_partitions) == 1, "WARNING: single layer partition feeding multiple partitions. Currently not handled"
        # merge partition to previous
        merge_horizontal(net, partition_index, next_partitions[0])
        # find next partition to merge
        partition_index, partition_node = _find_single_layer_partition()

def apply_random_partition(net, partition_index):
   # choose randomly between merge or split
    ## split partition
    transform_type = random.choice(['split','merge'])
    if transform_type == 'split':
        ## get all possible splits
        horizontal_splits = get_all_horizontal_splits(net, partition_index)
        vertical_splits   = get_all_vertical_splits(net, partition_index)
        ## split horizontally first
        if horizontal_splits:
            split_horizontal(net, partition_index, random.choice(horizontal_splits))
        ## vertically second choice
        elif vertical_splits:
            split_vertical(net, partition_index, random.choice(vertical_splits))
    ## merge partition
    if transform_type == 'merge':
        # get all possible merges
        horizontal_merges = get_all_horizontal_merges(net, partition_index)
        vertical_merges   = get_all_vertical_merges(net, partition_index)
        ## merge horizontally first
        if horizontal_merges[0]:
            merge_horizontal(net, *horizontal_merges[0])
        elif horizontal_merges[1]:
            merge_horizontal(net, *horizontal_merges[1])
        elif vertical_merges:
            merge_vertical(net, *random.choice(vertical_merges))

