import random
import statistics
import numpy as np

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def validate_in_out_shapes(self, hw_node, shape_in, shape_out):
    assert shape_in[0] == shape_in[1], "Row and column dimensions must be equal for input"
    assert shape_out[0] == shape_out[1], "Row and column dimensions must be equal for output"
    match self.building_blocks[hw_node]['type']:
        case LAYER_TYPE.Convolution:
            # we assume that the output shape of a convolution will always reduce the spatio-temporal dimensions and increase the channel dimension
            assert shape_in[0] >= shape_out[0], f"(CONV) Input row dimension must be greater than or equal to output row dimension in:{shape_in} out:{shape_out}"
            # assert shape_in[-1] <= shape_out[-1], f"(CONV) Input channel dimension must be less than or equal to output channel dimension in:{shape_in} out:{shape_out}"
        case LAYER_TYPE.Pooling:
            assert shape_in[0] >= shape_out[0], f"(POOL) Input row dimension must be greater than or equal to output row dimension in:{shape_in} out:{shape_out}"
            assert shape_in[-1] == shape_out[-1], f"(POOL) Input and output channel dimensions must be equal in:{shape_in} out:{shape_out}"
        case LAYER_TYPE.InnerProduct:
            # do nothing
            pass
        case LAYER_TYPE.GlobalPooling:
            if self.dimensionality == 2:
                assert shape_out[0] == 1 and shape_out[1] == 1, f"(GLOBALPOOL) Output shape must be 1x1x1 in:{shape_in} out:{shape_out}"
            else:
                assert shape_out[0] == 1 and shape_out[1] == 1 and shape_out[2] == 1, "(GLOBALPOOL) Output shape must be 1x1x1 in:{shape_in} out:{shape_out}"
            assert shape_in[-1] == shape_out[-1], f"(GLOBALPOOL) Input and output channel dimensions must be equal in:{shape_in} out:{shape_out}"
        case LAYER_TYPE.EltWise | LAYER_TYPE.ReLU | LAYER_TYPE.Sigmoid | LAYER_TYPE.SiLU:
            assert shape_in == shape_out, f"(ELTWISE) Input and output shapes must be equal in:{shape_in} out:{shape_out}"
        case _:
            raise Exception(f"Unknown layer type {self.building_blocks[hw_node]['type']}")

def get_max_input_shape(self, hw_node):
    return [ max([ self.net.graph.nodes[exec_node]["hw"].shape_in()[i] \
            for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]) for \
            i in range(self.dimensionality+1) ]

def get_max_output_shape(self, hw_node):
    return [ max([ self.net.graph.nodes[exec_node]["hw"].shape_out()[i] \
            for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]) for \
            i in range(self.dimensionality+1) ]

def get_random_shape(self, hw_node, rand_shape_range = [15, 15, 5, 20], use_previous_shape = True):
    """
    get a random shape for executing the featuremap.
    """

    # get the previous input shape
    prev_input_shape = self.building_blocks[hw_node]["hw"].shape_in()
    prev_output_shape = self.building_blocks[hw_node]["hw"].shape_out()

    # get the max shape for the input and output
    max_input_shape = self.get_max_input_shape(hw_node)
    max_output_shape = self.get_max_output_shape(hw_node)

    if use_previous_shape:
        # constraint shapes to be less than max
        prev_input_shape = [ min(prev_input_shape[i], max_input_shape[i]) for \
                i in range(len(prev_input_shape)) ]
        prev_output_shape = [ min(prev_output_shape[i], max_output_shape[i]) for \
                i in range(len(prev_output_shape)) ]
        # get a random shape based on the previous (within a range)
        next_input_shape = [ random.randint(
                max(1, prev_input_shape[i]-rand_shape_range[i]),
                min(prev_input_shape[i]+rand_shape_range[i], max_input_shape[i])) for \
                        i in range(len(prev_input_shape)) ]
        next_output_shape = [ random.randint(
                max(1, prev_output_shape[i]-rand_shape_range[i]),
                min(prev_output_shape[i]+rand_shape_range[i], max_output_shape[i])) for \
                        i in range(len(prev_output_shape)) ]
        match self.building_blocks[hw_node]['type']:
            case LAYER_TYPE.Convolution:
                # we assume that the output shape of a convolution will always reduce the spatio-temporal dimensions and increase the channel dimension
                next_output_shape[:-1] = [ random.randint(
                    max(1, prev_output_shape[i]-rand_shape_range[i]),
                    min(prev_output_shape[i]+rand_shape_range[i], next_input_shape[i])) for \
                            i in range(len(prev_output_shape)-1) ]
                if self.building_blocks[hw_node]['hw'].depthwise:
                    next_output_shape[-1] = next_input_shape[-1]
                    max_output_shape[-1] = max_input_shape[-1]
            case LAYER_TYPE.Pooling:
                next_output_shape[:-1] = [ random.randint(
                    max(1, prev_output_shape[i]-rand_shape_range[i]),
                    min(prev_output_shape[i]+rand_shape_range[i], next_input_shape[i])) for \
                            i in range(len(prev_output_shape)-1) ]
                next_output_shape[-1] = next_input_shape[-1]
            case LAYER_TYPE.InnerProduct:
                # do nothing
                pass
            case LAYER_TYPE.GlobalPooling:
                next_output_shape[0], next_output_shape[1], next_output_shape[2] = 1, 1, 1
                next_output_shape[-1] = next_input_shape[-1]
            case LAYER_TYPE.EltWise | LAYER_TYPE.ReLU | LAYER_TYPE.Sigmoid | LAYER_TYPE.SiLU:
                next_output_shape = next_input_shape
            case _:
                raise Exception(f"Unknown layer type {self.building_blocks[hw_node]['type']}")
    else:
        # get a random shape
        next_input_shape = [ random.randint(1, max_dim) for max_dim in max_input_shape ]
        next_output_shape = [ random.randint(1, max_dim) for max_dim in max_output_shape ]
        match self.building_blocks[hw_node]['type']:
            case LAYER_TYPE.Convolution:
                # we assume that the output shape of a convolution will always reduce the spatio-temporal dimensions and increase the channel dimension
                next_output_shape[:-1] = [ random.randint(1, max_dim) for max_dim in next_input_shape[:-1] ]
                if self.building_blocks[hw_node]['hw'].depthwise:
                    next_output_shape[-1] = next_input_shape[-1]
                    max_output_shape[-1] = max_input_shape[-1]
            case LAYER_TYPE.Pooling:
                next_output_shape[:-1] = [ random.randint(1, max_dim) for max_dim in next_input_shape[:-1] ]
                next_output_shape[-1] = next_input_shape[-1]
            case LAYER_TYPE.InnerProduct:
                # do nothing
                pass
            case LAYER_TYPE.GlobalPooling:
                next_output_shape[0], next_output_shape[1], next_output_shape[2] = 1, 1, 1
                next_output_shape[-1] = next_input_shape[-1]
            case LAYER_TYPE.EltWise | LAYER_TYPE.ReLU | LAYER_TYPE.Sigmoid | LAYER_TYPE.SiLU:
                next_output_shape = next_input_shape
            case _:
                raise Exception(f"Unknown layer type {self.building_blocks[hw_node]['type']}")

    # make sure the input and output channel dimension are greater than the minimum for the ports
    next_input_shape[-1] = max(self.min_channels_in, next_input_shape[-1])
    next_output_shape[-1] = max(self.min_channels_out, next_output_shape[-1])

    # make sure the row and column dimensions are equal for the input and output shapes
    next_input_shape[1] = next_input_shape[0]
    next_output_shape[1] = next_output_shape[0]

    # validate the produced shapes based on the layer type
    self.validate_in_out_shapes(hw_node, next_input_shape, next_output_shape)

    # return next shapes
    return next_input_shape, next_output_shape

def get_mixed_shape(self, hw_node, rand_shape_range = [15, 15, 5, 20], use_previous_shape = True):

    # get both random and inherited shapes
    random_input_shape, random_output_shape = self.get_random_shape(hw_node, rand_shape_range, use_previous_shape)
    inherit_input_shape, inherit_output_shape = self.get_inherited_shape(hw_node)

    # make the spatial shapes random, and the channels inherited
    next_input_shape = [ *random_input_shape[:-1], inherit_input_shape[-1] ]
    next_output_shape = [ *random_output_shape[:-1], inherit_output_shape[-1] ]

    # validate the produced shapes based on the layer type
    self.validate_in_out_shapes(hw_node, next_input_shape, next_output_shape)

    # return next shapes
    return next_input_shape, next_output_shape

def get_inherited_shape(self, hw_node):
    """
    get a shape from the execution nodes, as well as factors of those shapes.
    """

    # get dimensions of shape in and out
    size = self.dimensionality+1

    # get the max shape for the input and output
    max_input_shape = self.get_max_input_shape(hw_node)
    max_output_shape = self.get_max_output_shape(hw_node)

    # helper function to get factors of a number
    def get_factors(num):
        return [n for n in range(1, num + 1) if num % n == 0]

    # get all the factors of the shapes in and out
    all_input_shapes = [[1] for _ in range(size)]
    all_output_shapes = [[1] for _ in range(size)]
    for exec_node in self.building_blocks[hw_node]["exec_nodes"]:
        for i in range(size):
            all_input_shapes[i].extend(get_factors(
                self.net.graph.nodes[exec_node]["hw"].shape_in()[i]))
            all_output_shapes[i].extend(get_factors(
                self.net.graph.nodes[exec_node]["hw"].shape_out()[i]))

    # reduce to the set of these values
    all_input_shapes = [ list(set(shape)) for shape in all_input_shapes ]
    all_output_shapes = [ list(set(shape)) for shape in all_output_shapes ]

    # filter out the min channels in and out from all the shapes
    all_input_shapes[-1] = list(filter(lambda s: s >= self.min_channels_in, all_input_shapes[-1]))
    all_output_shapes[-1] = list(filter(lambda s: s >= self.min_channels_out, all_output_shapes[-1]))

    # needs to at least be the minimum as an option
    if all_input_shapes[-1] == []:
        all_input_shapes[-1] = [self.min_channels_in]
    if all_output_shapes[-1] == []:
        all_output_shapes[-1] = [self.min_channels_out]

    # choose a random shape from these shapes
    next_input_shape = [ random.choice(shape) for shape in all_input_shapes ]
    next_output_shape = [ random.choice(shape) for shape in all_output_shapes ]

    # Fix input and output shapes based on the layer type
    match self.building_blocks[hw_node]['type']:
        case LAYER_TYPE.Convolution:
            # we assume that the output shape of a convolution will always reduce the spatio-temporal dimensions and increase the channel dimension
            next_output_shape[:-1] = [random.choice(list(filter(lambda f: f <= next_input_shape[i], all_output_shapes[i]))) for i in range(size - 1)]
            if self.building_blocks[hw_node]['hw'].depthwise:
                next_output_shape[-1] = next_input_shape[-1]
                max_output_shape[-1] = max_input_shape[-1]
        case LAYER_TYPE.Pooling:
            next_output_shape[:-1] = [random.choice(list(filter(lambda f: f <= next_input_shape[i], all_output_shapes[i]))) for i in range(size - 1)]
            next_output_shape[-1] = next_input_shape[-1]
        case LAYER_TYPE.InnerProduct:
            # do nothing
            pass
        case LAYER_TYPE.GlobalPooling:
            next_output_shape[0], next_output_shape[1], next_output_shape[2] = 1, 1, 1
            next_output_shape[-1] = next_input_shape[-1]
        case LAYER_TYPE.EltWise | LAYER_TYPE.ReLU | LAYER_TYPE.Sigmoid | LAYER_TYPE.SiLU:
            next_output_shape = next_input_shape
        case _:
            raise Exception(f"Unknown layer type {self.building_blocks[hw_node]['type']}")

    # make sure the input and output channel dimension are greater than the minimum for the ports
    next_input_shape[-1] = max(self.min_channels_in, next_input_shape[-1])
    next_output_shape[-1] = max(self.min_channels_out, next_output_shape[-1])

    # make sure the row and column dimensions are equal for the input and output shapes
    next_input_shape[1] = next_input_shape[0]
    next_output_shape[1] = next_output_shape[0]

    # validate the produced shapes based on the layer type
    self.validate_in_out_shapes(hw_node, next_input_shape, next_output_shape)

    # return next shapes
    return next_input_shape, next_output_shape

def get_min_shape(self, hw_node):
    """
    get the min shape for executing the featuremap.
    """

    # get dimensions of shape in and out
    size = self.dimensionality + 1

    # set the min shape for both input and output to be 1 for all dimensions
    min_input_shape = [ 1 for _ in range(size) ]
    min_output_shape = [ 1 for _ in range(size) ]

    # make sure the input and output channel dimension are greater than the minimum for the ports
    min_input_shape[-1] = max(self.min_channels_in, min_input_shape[-1])
    min_output_shape[-1] = max(self.min_channels_out, min_output_shape[-1])

    # validate the produced shapes based on the layer type
    self.validate_in_out_shapes(hw_node, min_input_shape, min_output_shape)

    # return next shapes
    return min_input_shape, min_output_shape

def get_max_shape(self, hw_node):
    """
    get the max shape for executing the featuremap.
    """

    # get dimensions of shape in and out
    size = self.dimensionality + 1

    # get the max shape for the input and output
    max_input_shape = self.get_max_input_shape(hw_node)
    max_output_shape = self.get_max_output_shape(hw_node)

    # make sure the row and column dimensions are equal for the input and output shapes
    max_input_shape[1] = max_input_shape[0]
    max_output_shape[1] = max_output_shape[0]

    # validate the produced shapes based on the layer type
    self.validate_in_out_shapes(hw_node, max_input_shape, max_output_shape)

    # return next shapes
    return max_input_shape, max_output_shape

def get_median_shape(self, hw_node):
    """
    get the max shape for executing the featuremap.
    """

    # get dimensions of shape in and out
    size = self.dimensionality + 1

    # get all input and output shapes
    all_input_shapes = [ [self.net.graph.nodes[exec_node]["hw"].shape_in()[i] \
            for exec_node in self.building_blocks[hw_node]["exec_nodes"] ] for i in range(size) ]
    all_output_shapes = [ [self.net.graph.nodes[exec_node]["hw"].shape_out()[i] \
            for exec_node in self.building_blocks[hw_node]["exec_nodes"] ] for i in range(size) ]

    # get the median shape for each dimension
    next_input_shape = [ statistics.median(all_input_shapes[i]) for i in range(size) ]
    next_output_shape = [ statistics.median(all_output_shapes[i]) for i in range(size) ]

    # make sure the row and column dimensions are equal for the input and output shapes
    next_input_shape[1] = next_input_shape[0]
    next_output_shape[1] = next_output_shape[0]

    # validate the produced shapes based on the layer type
    self.validate_in_out_shapes(hw_node, next_input_shape, next_output_shape)

    # return next shapes
    return next_input_shape, next_output_shape

def get_percentage_shape(self, hw_node, percentage=10):
    """
    get the max shape for executing the featuremap.
    """

    # get dimensions of shape in and out
    size = self.dimensionality + 1

    # get the max shape for the input and output
    max_input_shape = self.get_max_input_shape(hw_node)
    max_output_shape = self.get_max_output_shape(hw_node)

    # get the median shape for each dimension
    next_input_shape = [ max(1, int(max_input_shape[i]*(percentage/100))) for i in range(size) ]
    next_output_shape = [ max(1, int(max_output_shape[i]*(percentage/100))) for i in range(size) ]

    # make sure the input and output channel dimension are greater than the minimum for the ports
    next_input_shape[-1] = max(self.min_channels_in, next_input_shape[-1])
    next_output_shape[-1] = max(self.min_channels_out, next_output_shape[-1])

    # make sure the row and column dimensions are equal for the input and output shapes
    next_input_shape[1] = next_input_shape[0]
    next_output_shape[1] = next_output_shape[0]

    # validate the produced shapes based on the layer type
    self.validate_in_out_shapes(hw_node, next_input_shape, next_output_shape)

    # return next shapes
    return next_input_shape, next_output_shape

def update_building_block_shape(self, hw_node, next_input_shape, next_output_shape):

    # get the max shape for the input and output
    max_input_shape = self.get_max_input_shape(hw_node)
    max_output_shape = self.get_max_output_shape(hw_node)

    # update the next shape for specific hardware types
    match self.building_blocks[hw_node]["type"]:
        case LAYER_TYPE.Convolution:
            # get the max kernel size
            max_kernel_rows = max([ self.net.graph.nodes[exec_node]["hw"].kernel_rows \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            max_kernel_cols = max([ self.net.graph.nodes[exec_node]["hw"].kernel_cols \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            max_stride_rows = max([ self.net.graph.nodes[exec_node]["hw"].stride_rows \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            max_stride_cols = max([ self.net.graph.nodes[exec_node]["hw"].stride_cols \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            if self.dimensionality == 3:
                max_kernel_depth = max([ self.net.graph.nodes[exec_node]["hw"].kernel_depth \
                        for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
                max_stride_depth = max([ self.net.graph.nodes[exec_node]["hw"].stride_depth \
                        for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            # make sure rows are greater than the kernel size
            # TODO: get the actual min shape
            self.building_blocks[hw_node]["hw"].rows = max(max_kernel_rows+max_stride_rows+1, max_input_shape[0])
            self.building_blocks[hw_node]["hw"].cols = max(max_kernel_cols+max_stride_cols+1, next_input_shape[1])
            if self.dimensionality == 3:
                self.building_blocks[hw_node]["hw"].depth = max(max_kernel_depth+max_stride_depth+1, next_input_shape[2])
            # channel and filter dimensions
            self.building_blocks[hw_node]["hw"].channels = \
                    next_input_shape[-1] if self.channel_tiling else max_input_shape[-1]
            self.building_blocks[hw_node]["hw"].filters = \
                    next_output_shape[-1] if self.filter_tiling else max_output_shape[-1]
            if self.building_blocks[hw_node]["hw"].depthwise:
                assert self.building_blocks[hw_node]["hw"].channels == \
                        self.building_blocks[hw_node]["hw"].filters
                self.building_blocks[hw_node]["hw"].groups = \
                        self.building_blocks[hw_node]["hw"].channels
        case LAYER_TYPE.InnerProduct:
            self.building_blocks[hw_node]["hw"].rows = next_input_shape[0]
            self.building_blocks[hw_node]["hw"].cols = next_input_shape[1]
            if self.dimensionality == 3:
                self.building_blocks[hw_node]["hw"].depth = next_input_shape[2]
            # channel and filter dimensions
            self.building_blocks[hw_node]["hw"].channels = \
                    next_input_shape[-1] if self.channel_tiling else max_input_shape[-1]
            self.building_blocks[hw_node]["hw"].filters = \
                    next_output_shape[-1] if self.filter_tiling else max_output_shape[-1]
        case LAYER_TYPE.Pooling:
            # get the max kernel size
            max_kernel_rows = max([ self.net.graph.nodes[exec_node]["hw"].kernel_rows \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            max_kernel_cols = max([ self.net.graph.nodes[exec_node]["hw"].kernel_cols \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            max_stride_rows = max([ self.net.graph.nodes[exec_node]["hw"].stride_rows \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            max_stride_cols = max([ self.net.graph.nodes[exec_node]["hw"].stride_cols \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            if self.dimensionality == 3:
                max_kernel_depth = max([ self.net.graph.nodes[exec_node]["hw"].kernel_depth \
                        for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
                max_stride_depth = max([ self.net.graph.nodes[exec_node]["hw"].stride_depth \
                        for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            # make sure rows are greater than the kernel size
            # TODO: get the actual min shape
            self.building_blocks[hw_node]["hw"].rows = max(max_kernel_rows+max_stride_rows+1, max_input_shape[0])
            self.building_blocks[hw_node]["hw"].cols = max(max_kernel_cols+max_stride_cols+1, next_input_shape[1])
            if self.dimensionality == 3:
                self.building_blocks[hw_node]["hw"].depth = max(max_kernel_depth+max_stride_depth+1, next_input_shape[2])
            # update the channel dimension
            self.building_blocks[hw_node]["hw"].channels = next_output_shape[-1]
        # TODO: handle the other layer types
        case LAYER_TYPE.EltWise:
            self.building_blocks[hw_node]["hw"].rows = [max_output_shape[0]] * self.building_blocks[hw_node]["hw"].ports_in
            self.building_blocks[hw_node]["hw"].cols = [max_output_shape[1]] * self.building_blocks[hw_node]["hw"].ports_in
            if self.dimensionality == 3:
                self.building_blocks[hw_node]["hw"].depth = [max_output_shape[2]] * self.building_blocks[hw_node]["hw"].ports_in
            self.building_blocks[hw_node]["hw"].channels = [max_output_shape[-1]] * self.building_blocks[hw_node]["hw"].ports_in
        case _:
            self.building_blocks[hw_node]["hw"].rows = max_output_shape[0]
            self.building_blocks[hw_node]["hw"].cols = max_output_shape[1]
            if self.dimensionality == 3:
                self.building_blocks[hw_node]["hw"].depth = max_output_shape[2]
            self.building_blocks[hw_node]["hw"].channels = max_output_shape[-1]
    # update the hw node
    self.building_blocks[hw_node]["hw"].update()

