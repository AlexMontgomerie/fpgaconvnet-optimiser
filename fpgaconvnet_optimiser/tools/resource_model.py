import bisect
import math

BRAM_CONF={1:16000,2:8000,4:4000,9:2000,18:1000,36:512}

def bram_stream_resource_model(depth, width):
    assert width > 0, "width must be greater than zero"
    assert width <= 36, "width must be less than 36"

    # find the closest width from the BRAM configuration
    bram_width = BRAM_CONF.keys()[bisect.bisect_right(BRAM_CONF.keys(), width)-1]

    # get the depth for the bram
    bram_depth = BRAM_CONF[bram_width]

    # get the address width
    addr_width = math.ceil(math.log(depth,2))

    # return the ceiling
    return max(addr_width,math.ceil(depth/bram_depth))

def bram_memory_resource_model(depth, width):
    assert width > 0, "width must be greater than zero"
    assert width <= 36, "width must be less than 36"

    # find the closest width from the BRAM configuration
    bram_width = BRAM_CONF.keys()[bisect.bisect_right(BRAM_CONF.keys(), width)-1]

    # get the depth for the bram
    bram_depth = BRAM_CONF[bram_width]

    # return the ceiling
    return math.ceil(depth/bram_depth)


def dsp_multiplier_resource_model(multiplicand_width, multiplier_width, dsp_type="DSP48E1"):
    return math.ceil((multiplicand_width+multiplier_width)/48)
