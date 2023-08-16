from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.models.layers import ConvolutionLayer, ConvolutionLayer3D
from fpgaconvnet.models.layers import PoolingLayer, PoolingLayer3D
from fpgaconvnet.models.layers import GlobalPoolingLayer, GlobalPoolingLayer3D
from fpgaconvnet.models.layers import InnerProductLayer, InnerProductLayer3D
from fpgaconvnet.models.layers import EltWiseLayer, EltWiseLayer3D
from fpgaconvnet.models.layers import ReLULayer, ReLULayer3D
from fpgaconvnet.models.layers import ThresholdedReLULayer
from fpgaconvnet.models.layers import ActivationLayer3D

def get_convolution_from_dict(param, dimensionality):
    if dimensionality == 2:
        return ConvolutionLayer(
                param["filters"],  param["rows"],
                param["cols"], param["channels"],
                groups=param["groups"],
                fine=param["fine"],
                coarse_in=param["coarse_in"],
                coarse_out=param["coarse_out"],
                coarse_group=param["coarse_group"],
                kernel_rows=param["kernel_rows"],
                kernel_cols=param["kernel_cols"],
                stride_rows=param["stride_rows"],
                stride_cols=param["stride_cols"],
                pad_left=param["pad_left"],
                pad_right=param["pad_right"],
                pad_top=param["pad_top"],
                pad_bottom=param["pad_bottom"])
    elif dimensionality == 3:
        return ConvolutionLayer3D(
                param["filters"],  param["rows"],
                param["cols"], param["depth"],
                param["channels"],
                groups=param["groups"],
                fine=param["fine"],
                coarse_in=param["coarse_in"],
                coarse_out=param["coarse_out"],
                coarse_group=param["coarse_group"],
                kernel_rows=param["kernel_rows"],
                kernel_cols=param["kernel_cols"],
                kernel_depth=param["kernel_depth"],
                stride_rows=param["stride_rows"],
                stride_cols=param["stride_cols"],
                stride_depth=param["stride_depth"],
                pad_left=param["pad_left"],
                pad_right=param["pad_right"],
                pad_top=param["pad_top"],
                pad_bottom=param["pad_bottom"],
                pad_front=param["pad_front"],
                pad_back=param["pad_back"])
    else:
        raise NotImplementedError

def get_inner_product_from_dict(param, dimensionality):
    if dimensionality == 2:
        return InnerProductLayer(
                param["filters"],  1, 1,
                param["rows"]*param["cols"]*param["channels"],
                coarse_in=param["coarse_in"],
                coarse_out=param["coarse_out"]
            )
    elif dimensionality == 3:
        return InnerProductLayer3D(
                param["filters"],  1, 1, 1,
                param["rows"]*param["cols"]*param["depth"]*param["channels"],
                coarse_in=param["coarse_in"],
                coarse_out=param["coarse_out"],
            )
    else:
        raise NotImplementedError

def get_pooling_from_dict(param, dimensionality):
    if dimensionality == 2:
        return PoolingLayer(
                param["rows"], param["cols"],
                param["channels"],
                coarse=param["coarse"],
                kernel_rows=param["kernel_rows"],
                kernel_cols=param["kernel_cols"],
                stride_rows=param["stride_rows"],
                stride_cols=param["stride_cols"],
                pad_left=param["pad_left"],
                pad_right=param["pad_right"],
                pad_top=param["pad_top"],
                pad_bottom=param["pad_bottom"])
    elif dimensionality == 3:
        return PoolingLayer3D(
                param["rows"], param["cols"],
                param["depth"],
                param["channels"],
                coarse=param["coarse"],
                kernel_rows=param["kernel_rows"],
                kernel_cols=param["kernel_cols"],
                kernel_depth=param["kernel_depth"],
                stride_rows=param["stride_rows"],
                stride_cols=param["stride_cols"],
                stride_depth=param["stride_depth"],
                pad_left=param["pad_left"],
                pad_right=param["pad_right"],
                pad_top=param["pad_top"],
                pad_bottom=param["pad_bottom"],
                pad_front=param["pad_front"],
                pad_back=param["pad_back"])
    else:
        raise NotImplementedError

def get_eltwise_from_dict(param, dimensionality):
    if dimensionality == 2:
        return EltWiseLayer(
                param["rows"],  param["cols"], param["channels"],
                ports_in=param["ports_in"],
                coarse=param["coarse"],
                op_type=param["op_type"],
                broadcast=param["broadcast"],
            )
    elif dimensionality == 3:
        return EltWiseLayer3D(
                param["rows"],  param["cols"],
                param["depth"], param["channels"],
                ports_in=param["ports_in"],
                coarse=param["coarse"],
                op_type=param["op_type"],
                broadcast=param["broadcast"],
            )
    else:
        raise NotImplementedError

def get_global_pooling_from_dict(param, dimensionality):
    if dimensionality == 2:
        return GlobalPoolingLayer(
                param["rows"],
                param["cols"],
                param["channels"],
                coarse=param["coarse"],
            )
    elif dimensionality == 3:
        return GlobalPoolingLayer3D(
                param["rows"],
                param["cols"],
                param["depth"],
                param["channels"],
                coarse=param["coarse"],
            )
    else:
        raise NotImplementedError

def get_relu_from_dict(param, dimensionality):
    if dimensionality == 2:
        return ReLULayer(
                param["rows"],
                param["cols"],
                param["channels"],
                coarse=param["coarse"],
            )
    elif dimensionality == 3:
        return ReLULayer3D(
                param["rows"],
                param["cols"],
                param["depth"],
                param["channels"],
                coarse=param["coarse"],
            )
    else:
        raise NotImplementedError

def get_thresholded_relu_from_dict(param, dimensionality):

    return ThresholdedReLULayer(
            param["rows"],
            param["cols"],
            param["channels"],
            coarse=param["coarse"],
            threshold=param["threshold"]
        )
   
def get_activation_from_dict(param, dimensionality):
    if dimensionality == 2:
        raise NotImplementedError("Activation layer not implemented for 2D")
    elif dimensionality == 3:
        return ActivationLayer3D(
                param["rows"],
                param["cols"],
                param["depth"],
                param["channels"],
                param["op_type"],
                coarse=param["coarse"],
            )
    else:
        raise NotImplementedError

def get_hw_from_dict(layer_type, param, dimensionality):
    match layer_type:
        case LAYER_TYPE.Convolution:
            return get_convolution_from_dict(param, dimensionality)
        case LAYER_TYPE.InnerProduct:
            return get_inner_product_from_dict(param, dimensionality)
        case LAYER_TYPE.Pooling:
            return get_pooling_from_dict(param, dimensionality)
        case LAYER_TYPE.EltWise:
            return get_eltwise_from_dict(param, dimensionality)
        case LAYER_TYPE.ReLU:
            return get_relu_from_dict(param, dimensionality)
        case LAYER_TYPE.ThresholdedReLU:
            return get_thresholded_relu_from_dict(param, dimensionality)
        case LAYER_TYPE.GlobalPooling:
            return get_global_pooling_from_dict(param, dimensionality)
        case LAYER_TYPE.Sigmoid | LAYER_TYPE.SiLU:
            return get_activation_from_dict(param, dimensionality)
        case _:
            raise NotImplementedError(f"layer type {layer_type} not implemented")

def update_node_param(layer_type, node, param, dimensionality):
    match layer_type:
        case LAYER_TYPE.Convolution:
            node.rows = param["rows"]
            node.cols = param["cols"]
            node.channels = param["channels"]
            node.filters = param["filters"]
            node.groups = param["groups"]
            node.fine = param["fine"]
            node.coarse_in = param["coarse_in"]
            node.coarse_out = param["coarse_out"]
            node.coarse_group = param["coarse_group"]
            node.kernel_cols = param["kernel_cols"]
            node.kernel_rows = param["kernel_rows"]
            node.stride_cols = param["stride_cols"]
            node.stride_rows = param["stride_rows"]
            node.pad_left = param["pad_left"]
            node.pad_right = param["pad_right"]
            node.pad_top = param["pad_top"]
            node.pad_bottom = param["pad_bottom"]
            if dimensionality == 3:
                node.depth = param["depth"]
                node.kernel_depth = param["kernel_depth"]
                node.stride_depth = param["stride_depth"]
                node.pad_front = param["pad_front"]
                node.pad_back = param["pad_back"]
        case LAYER_TYPE.InnerProduct:
            node.rows = param["rows"]
            node.cols = param["cols"]
            node.channels = param["channels"]
            node.filters = param["filters"]
            node.coarse_in = param["coarse_in"]
            node.coarse_out = param["coarse_out"]
            if dimensionality == 3:
                node.depth = param["depth"]
        case LAYER_TYPE.Pooling:
            node.rows = param["rows"]
            node.cols = param["cols"]
            node.channels = param["channels"]
            node.coarse = param["coarse"]
            node.kernel_cols = param["kernel_cols"]
            node.kernel_rows = param["kernel_rows"]
            node.stride_cols = param["stride_cols"]
            node.stride_rows = param["stride_rows"]
            node.pad_left = param["pad_left"]
            node.pad_right = param["pad_right"]
            node.pad_top = param["pad_top"]
            node.pad_bottom = param["pad_bottom"]
            if dimensionality == 3:
                node.depth = param["depth"]
                node.kernel_depth = param["kernel_depth"]
                node.stride_depth = param["stride_depth"]
                node.pad_front = param["pad_front"]
                node.pad_back = param["pad_back"]
        case LAYER_TYPE.EltWise:
            node.rows = [param["rows"]] * param["ports_in"]
            node.cols = [param["cols"]] * param["ports_in"]
            node.channels = [param["channels"]] * param["ports_in"]
            node.coarse = param["coarse"]
            if dimensionality == 3:
                node.depth = [param["depth"]] * param["ports_in"]
        case LAYER_TYPE.ReLU:
            node.rows = param["rows"]
            node.cols = param["cols"]
            node.channels = param["channels"]
            node.coarse = param["coarse"]
            if dimensionality == 3:
                node.depth = param["depth"]
        case LAYER_TYPE.GlobalPooling:
            node.rows = param["rows"]
            node.cols = param["cols"]
            node.channels = param["channels"]
            node.coarse = param["coarse"]
            if dimensionality == 3:
                node.depth = param["depth"]
        case LAYER_TYPE.ReLU | LAYER_TYPE.Sigmoid | LAYER_TYPE.SiLU:
            node.rows = param["rows"]
            node.cols = param["cols"]
            node.channels = param["channels"]
            node.coarse = param["coarse"]
            if dimensionality == 3:
                node.depth = param["depth"]
        case _:
            raise NotImplementedError(f"layer type {layer_type} not implemented")
    node.update()

def get_runtime_latency(layer_type, node, param, dimensionality):

    # get all the previous parameters from the attributes
    prev_param = node.__dict__
    prev_param = { key.lstrip("_") : (val if not isinstance(val, list) else val[0]) for key, val in prev_param.items() }

    # update the node and get the latency
    update_node_param(layer_type, node, param, dimensionality)

    # get the latency
    latency = node.latency()

    # give back previous parameters
    update_node_param(layer_type, node, prev_param, dimensionality)

    # return latency
    return latency

def apply_mem_bw_limitations(graph, building_blocks, total_mem_bw, channel_tiling=False):

    # split the memory bandwidth equally between the in and out
    mem_bw_in = total_mem_bw / 2
    mem_bw_out = total_mem_bw / 2

    # apply memory bandwidth limitations to the graph execution nodes
    for node in graph.nodes:
        match graph.nodes[node]["type"]:
            case LAYER_TYPE.EltWise:
                graph.nodes[node]["hw"].mem_bw_in = [mem_bw_in/graph.nodes[node]["hw"].ports_in] * graph.nodes[node]["hw"].ports_in
                graph.nodes[node]["hw"].mem_bw_out = [mem_bw_out/graph.nodes[node]["hw"].ports_out] * graph.nodes[node]["hw"].ports_out

            # FIXME: hack to get channel tiling working for the moment
            case LAYER_TYPE.Convolution:
                if channel_tiling:
                    graph.nodes[node]["hw"].mem_bw_in = mem_bw_in/2
                else:
                    graph.nodes[node]["hw"].mem_bw_in = mem_bw_in
                graph.nodes[node]["hw"].mem_bw_out = mem_bw_out
            case LAYER_TYPE.Convolution:
                if channel_tiling:
                    graph.nodes[node]["hw"].mem_bw_in = mem_bw_in/2
                else:
                    graph.nodes[node]["hw"].mem_bw_in = mem_bw_in
                graph.nodes[node]["hw"].mem_bw_out = mem_bw_out
            case _:
                graph.nodes[node]["hw"].mem_bw_in = mem_bw_in
                graph.nodes[node]["hw"].mem_bw_out = mem_bw_out

    # apply memory bandwidth limitations to the hardware building blocks
    for hw_node in building_blocks:
        match building_blocks[hw_node]["type"]:
            case LAYER_TYPE.EltWise:
                building_blocks[hw_node]["hw"].mem_bw_in = [mem_bw_in/building_blocks[hw_node]["hw"].ports_in] * building_blocks[hw_node]["hw"].ports_in
                building_blocks[hw_node]["hw"].mem_bw_out = [mem_bw_out/building_blocks[hw_node]["hw"].ports_out] * building_blocks[hw_node]["hw"].ports_out
            case _:
                building_blocks[hw_node]["hw"].mem_bw_in = mem_bw_in
                building_blocks[hw_node]["hw"].mem_bw_out = mem_bw_out
