from fpgaconvnet.tools.layer_enum import LAYER_TYPE
from fpgaconvnet.models.layers import ConvolutionLayer, InnerProductLayer, ReLULayer, EltWiseLayer

def get_convolution_from_dict(param):
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

# TODO: need to do this for all layer types




def get_hw_from_dict(layer_type, param):
    match layer_type:
        case LAYER_TYPE.Convolution:
            return get_convolution_from_dict(param)
        case _:
            raise NotImplementedError(f"layer type {layer_type} not implemented")


