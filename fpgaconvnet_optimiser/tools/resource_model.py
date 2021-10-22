import bisect
import math

# BRAM configurations from https://www.xilinx.com/support/documentation/user_guides/ug473_7Series_Memory_Resources.pdf
BRAM_CONF_WIDTH={1:16384,2:8192,4:4096,9:2048,18:1024,36:512}
BRAM_CONF_DEPTH={v: k for k, v in BRAM_CONF_WIDTH.items()}

def bram_resource_model(depth, width):
    # based on xilinx forum post: https://forums.xilinx.com/t5/High-Level-Synthesis-HLS/BRAM-usage-large-for-FIFO/m-p/1247118

    assert width > 0, "width must be greater than zero"
    assert width <= 36, "width must be less than 36"

    # if there is zero depth, return no BRAM usage
    if depth == 0:
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

def bram_stream_resource_model(depth, width):
    return bram_resource_model(depth, width)

def bram_memory_resource_model(depth, width):
    return bram_resource_model(depth, width)

def dsp_multiplier_resource_model(multiplicand_width, multiplier_width, dsp_type="DSP48E1"):
    #https://github.com/Xilinx/finn/blob/4fee6ffd8e13f91314ec9086e9ce9b2ea9de15c7/src/finn/custom_op/fpgadataflow/streamingfclayer_batch.py#L368,
    return math.ceil((multiplicand_width+multiplier_width)/48)

if __name__ == "__main__":
    print(bram_stream_resource_model(512,4))
    print(bram_stream_resource_model(1024,4))
    print(bram_stream_resource_model(2048,4))
    print(bram_stream_resource_model(4096,4))

    print(bram_stream_resource_model(512,8))
    print(bram_stream_resource_model(1024,8))
    print(bram_stream_resource_model(2048,8))
    print(bram_stream_resource_model(4096,8))

    print(bram_stream_resource_model(512,16))
    print(bram_stream_resource_model(1024,16))
    print(bram_stream_resource_model(2048,16))
    print(bram_stream_resource_model(4096,16))
