import math
import numpy as np

def bram18k_depth(data_width):
    if data_width == 1:
        return 16384
    elif data_width == 2:
        return 8192
    elif data_width <= 4:
        return 4096
    elif data_width <= 9:
        return 2048
    elif data_width <= 18:
        return 1024
    elif data_width <= 36:
        return 512
    else:
        assert False, "Unreachable"

def stable_array_rsc(data_width, array_size, resource_type=None):
    if data_width*array_size >= 1024:
        resource_type = "BRAM"

    array_bram = 0
    array_lut = 0
    if resource_type == "BRAM":
        max_bram_depth = bram18k_depth(min(data_width,36))
        array_bram = math.ceil(data_width/36)

        # this while loop is based on hls behaviour, vivado may decide to use more BRAMs to avoid large multiplexer
        while array_size > max_bram_depth: 
            array_size = math.ceil(array_size/2)
            array_bram = int(array_bram*2)
    else:
        array_lut = math.ceil(data_width*array_size/64)
    
    return {
        "LUT"  : array_lut,
        "BRAM" : array_bram,
        "DSP"  : 0,
        "FF"   : 0,
    }

def stream_rsc(data_width, buffer_depth, resource_type=None):
    if data_width*buffer_depth > 1024: # hls use 512, vivado use 1024 
        resource_type = "BRAM"

    stream_bram = 0
    if resource_type == "BRAM":
        max_bram_depth = bram18k_depth(min(data_width,36)) # hls use 18, vivado use 36 
        stream_bram = math.ceil(data_width/36) # hls use 18, vivado use 36  

        # this while loop is based on hls behaviour, vivado may decide to use more BRAMs to avoid large multiplexer
        while buffer_depth > max_bram_depth:
            buffer_depth = math.ceil(buffer_depth/2)
            stream_bram = int(stream_bram*2)

    return {
        "LUT"  : 0,
        "BRAM" : stream_bram,
        "DSP"  : 0,
        "FF"   : 0,
    }