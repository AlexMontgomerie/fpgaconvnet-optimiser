import random
import fpgaconvnet.tools.graphs as graphs

from fpgaconvnet.tools.layer_enum import LAYER_TYPE


def apply_random_off_chip_streaming(partition, layer):
    if partition.graph.nodes[layer]['type'] in \
        [ LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct ]:
        stream_type = random.choice(['weights','inputs'])
    elif partition.graph.nodes[layer]['type'] in \
        [ LAYER_TYPE.Concat, LAYER_TYPE.EltWise ]:
        stream_type = 'inputs'
    else:
        return

    hw_node = partition.graph.nodes[layer]['hw']
    if stream_type == 'weights':
        # randomly choose a ratio
        weight_step_size = random.choice([0.1, 0.2, 0.3, 0.4, 0.5,
                                        0.6, 0.7, 0.8, 0.9, 1.0])
        hw_node.stream_weights = hw_node.stream_unit() * \
            hw_node.stream_step(weight_step_size)

    elif stream_type == 'inputs':
        # case cannot handle
        if partition.graph.nodes[layer]['type'] in [ LAYER_TYPE.Concat, LAYER_TYPE.EltWise ] \
            and hw_node.ports_in > partition.graph.in_degree(layer):
            return

        # randomly choose an input
        stream_index = random.randint(0, 
            len(partition.graph.nodes[layer]['hw'].stream_inputs)-1)
        is_streamed = random.choice([True, False])
        
        partition.remove_squeeze()
        hw_node.stream_inputs[stream_index] = is_streamed
        if layer not in graphs.get_input_nodes(partition.graph, allow_multiport=True):
            prev_layer = graphs.get_prev_nodes(partition.graph,layer)[stream_index]
            prev_hw_node = partition.graph.nodes[prev_layer]["hw"]
            for j, l in enumerate(graphs.get_next_nodes(partition.graph,prev_layer)):
                if l == layer:
                    break
            prev_hw_node.stream_outputs[j] = is_streamed
        partition.add_squeeze()

def fix_streaming(net, partition_index_a, partition_index_b):
    prev_output_nodes = graphs.get_output_nodes(net.partitions[partition_index_a].graph)
    next_input_nodes  = graphs.get_input_nodes(net.partitions[partition_index_b].graph)

    for node in prev_output_nodes:
        net.partitions[partition_index_a].graph.nodes[node]['hw'].stream_outputs = [False] * \
            len(net.partitions[partition_index_a].graph.nodes[node]['hw'].stream_outputs)

    for node in next_input_nodes:
        net.partitions[partition_index_b].graph.nodes[node]['hw'].stream_inputs = [False] * \
            len(net.partitions[partition_index_b].graph.nodes[node]['hw'].stream_inputs)
