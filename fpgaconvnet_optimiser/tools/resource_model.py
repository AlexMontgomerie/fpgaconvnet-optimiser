import bisect
import numpy as np
import math

BRAM_CONF_DEPTH={16384:1,8192:2,4096:4,2048:9,1024:18,512:36}

def bram_array_resource_model(depth, width, array_type, force_bram_pragma=False):
    # based on xilinx forum post: https://forums.xilinx.com/t5/High-Level-Synthesis-HLS/BRAM-usage-large-for-FIFO/m-p/1247118

    assert width > 0, "width must be greater than zero"
    assert width <= 36, "width must be less than 36"
    assert array_type in ['stream', 'memory']

    # based on vivado behaviour, hls prediction may differ
    if (depth == 0) or \
        (array_type == 'stream' and not force_bram_pragma and width * depth <= 1024) or \
        (array_type == 'memory' and not force_bram_pragma and width * depth < 1024):
        return 0

    # find the closest depth from the BRAM configuration
    if depth in list(BRAM_CONF_DEPTH.keys()):
        bram_depth = depth
    elif depth > sorted(list(BRAM_CONF_DEPTH.keys()))[-1]:
        bram_depth = sorted(list(BRAM_CONF_DEPTH.keys()))[-1]
    else:
        bram_depth = sorted(list(BRAM_CONF_DEPTH.keys()))[
                bisect.bisect_right(sorted(list(BRAM_CONF_DEPTH.keys())), depth)]

    # get the depth for the bram
    bram_width = BRAM_CONF_DEPTH[bram_depth]

    # return the ceiling
    return math.ceil(width/bram_width)

def dsp_multiplier_resource_model(multiplicand_width, multiplier_width, dsp_type="DSP48E1"):
    #https://github.com/Xilinx/finn/blob/4fee6ffd8e13f91314ec9086e9ce9b2ea9de15c7/src/finn/custom_op/fpgadataflow/streamingfclayer_batch.py#L368,
    return math.ceil((multiplicand_width+multiplier_width)/48)


if __name__ == "__main__":
    print(bram_array_resource_model(512,4,'stream'))
    print(bram_array_resource_model(1024,4,'stream'))
    print(bram_array_resource_model(2048,4,'stream'))
    print(bram_array_resource_model(4096,4,'stream'))

    print(bram_array_resource_model(512,8,'stream'))
    print(bram_array_resource_model(1024,8,'stream'))
    print(bram_array_resource_model(2048,8,'stream'))
    print(bram_array_resource_model(4096,8,'stream'))

    print(bram_array_resource_model(512,16,'stream'))
    print(bram_array_resource_model(1024,16,'stream'))
    print(bram_array_resource_model(2048,16,'stream'))
    print(bram_array_resource_model(4096,16,'stream'))
