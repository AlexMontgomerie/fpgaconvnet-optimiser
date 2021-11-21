from enum import Enum
import fpgaconvnet_optimiser.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2

# Get enumeration from:
#   https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto
class LAYER_TYPE(Enum):
    Concat       =3
    Convolution  =4
    Dropout      =6
    InnerProduct =14
    LRN          =15
    Pooling      =17
    ReLU         =18
    Sigmoid      =19
    Softmax      =20
    Eltwise      =25
    # Not Enumerated
    BatchNorm = 40
    Scale     = 41
    Split     = 42
    Merge     = 43
    Squeeze   = 44
    Transpose = 45
    Flatten   = 46
    Cast      = 47
    Clip      = 48
    Shape     = 49
    #EE Layers - arbitrarily assigned
    If        = 50
    ReduceMax = 51
    Greater   = 52
    Identity  = 53
    #Not an ONNX op in this case
    Buffer    = 54

    @classmethod
    def get_type(cls, t):
        if type(t) is str:
            return cls[t]
        elif type(t) is int:
            return cls(t)

def to_proto_layer_type(layer_type):
    layer_types = {
        LAYER_TYPE.Convolution  : fpgaconvnet_pb2.layer.layer_type.CONVOLUTION,
        LAYER_TYPE.InnerProduct : fpgaconvnet_pb2.layer.layer_type.INNER_PRODUCT,
        LAYER_TYPE.Pooling      : fpgaconvnet_pb2.layer.layer_type.POOLING,
        LAYER_TYPE.ReLU         : fpgaconvnet_pb2.layer.layer_type.RELU,
        LAYER_TYPE.Squeeze      : fpgaconvnet_pb2.layer.layer_type.SQUEEZE,
        LAYER_TYPE.Concat       : fpgaconvnet_pb2.layer.layer_type.CONCAT,
        LAYER_TYPE.BatchNorm    : fpgaconvnet_pb2.layer.layer_type.BATCH_NORM
    }
    return layer_types.get(layer_type, lambda: "Invalid Layer Type")

def from_proto_layer_type(layer_type):
    layer_types = {
        fpgaconvnet_pb2.layer.layer_type.CONVOLUTION   : LAYER_TYPE.Convolution,
        fpgaconvnet_pb2.layer.layer_type.INNER_PRODUCT : LAYER_TYPE.InnerProduct,
        fpgaconvnet_pb2.layer.layer_type.POOLING       : LAYER_TYPE.Pooling,
        fpgaconvnet_pb2.layer.layer_type.RELU          : LAYER_TYPE.ReLU,
        fpgaconvnet_pb2.layer.layer_type.SQUEEZE       : LAYER_TYPE.Squeeze,
        fpgaconvnet_pb2.layer.layer_type.CONCAT        : LAYER_TYPE.Concat,
        fpgaconvnet_pb2.layer.layer_type.BATCH_NORM    : LAYER_TYPE.BatchNorm,
    }
    return layer_types.get(layer_type, lambda: "Invalid Layer Type")

def from_onnx_op_type(op_type):
    layer_types = {
        "Conv"      : LAYER_TYPE.Convolution,
        "Gemm"      : LAYER_TYPE.InnerProduct,
        "MatMul"    : LAYER_TYPE.InnerProduct,
        "Relu"      : LAYER_TYPE.ReLU,
        "MaxPool"   : LAYER_TYPE.Pooling,
        "LRN"       : LAYER_TYPE.LRN,
        "Reshape"   : LAYER_TYPE.Transpose,
        "Softmax"   : LAYER_TYPE.Softmax,
        "Dropout"   : LAYER_TYPE.Dropout,
        "Flatten"   : LAYER_TYPE.Flatten,
        "BatchNormalization" : LAYER_TYPE.BatchNorm,
        "GlobalAveragePool"  : LAYER_TYPE.Pooling,
        "AveragePool"        : LAYER_TYPE.Pooling,
        "Add"       : LAYER_TYPE.Eltwise,
        "Cast"      : LAYER_TYPE.Cast,
        "Clip"      : LAYER_TYPE.Clip,
        "Shape"     : LAYER_TYPE.Shape,
        "Squeeze"   : LAYER_TYPE.Squeeze,
        "Transpose" : LAYER_TYPE.Transpose,
        #placeholder layers for branching + exit decision
        "If"        : LAYER_TYPE.If,
        "ReduceMax" : LAYER_TYPE.ReduceMax,
        "Greater"   : LAYER_TYPE.Greater,
        "Identity"  : LAYER_TYPE.Identity,
        #hw layer to help split dataflow
        "Split"     : LAYER_TYPE.Split,
        #flexble buffer point for intermediate results
        "Buffer"    : LAYER_TYPE.Buffer,
    }
    return layer_types.get(op_type, lambda: TypeError)
