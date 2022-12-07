import random
import numpy as np

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def apply_random_shape(self, hw_node, rand_shape_range = [10, 10, 10], use_previous_shape = False):
    """
    get a random shape for executing the featuremap.
    """

    # get the previous input shape
    prev_input_shape = self.building_blocks[hw_node]["hw"].shape_in()
    prev_output_shape = self.building_blocks[hw_node]["hw"].shape_out()

    # get the max shape for the input and output
    max_input_shape = [ max([ self.net.graph.nodes[exec_node]["hw"].shape_in()[i] \
                for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]) for \
                i in range(len(prev_input_shape)) ]
    max_output_shape = [ max([ self.net.graph.nodes[exec_node]["hw"].shape_out()[i] \
                for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]) for \
                i in range(len(prev_output_shape)) ]

    if use_previous_shape:
        # get a random shape based on the previous (within a range)
        next_input_shape = [ random.randint(
                max(1, prev_input_shape[i]-rand_shape_range[i]),
                min(prev_input_shape[i]+rand_shape_range[i], max_input_shape[i])) for \
                        i in range(len(prev_input_shape)) ]
        next_output_shape = [ random.randint(
                max(1, prev_output_shape[i]-rand_shape_range[i]),
                min(prev_output_shape[i]+rand_shape_range[i], max_output_shape[i])) for \
                        i in range(len(prev_output_shape)) ]
    else:
        # get a random shape
        next_input_shape = [ random.randint(1, max_dim) for max_dim in max_input_shape ]
        next_output_shape = [ random.randint(1, max_dim) for max_dim in max_output_shape ]

    # update the next shape for specific hardware types
    self.update_building_block_shape(hw_node,
            next_input_shape, max_input_shape,
            next_output_shape, max_output_shape)

def apply_inherited_shape(self, hw_node):
    """
    get a shape from the execution nodes, as well as factors of those shapes.
    """

    # get dimensions of shape in and out
    size = self.dimensionality+1

    # get the max shape for the input and output
    max_input_shape = [ max([ self.net.graph.nodes[exec_node]["hw"].shape_in()[i] \
                for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]) for \
                i in range(size) ]
    max_output_shape = [ max([ self.net.graph.nodes[exec_node]["hw"].shape_out()[i] \
                for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]) for \
                i in range(size) ]

    # helper function to get factors of a number
    def get_factors(num):
        return [n for n in range(1, num + 1) if num % n == 0]

    # get all the factors of the shapes in and out
    all_input_shapes = [[1]]*size
    all_output_shapes =[[1]]*size
    for exec_node in self.building_blocks[hw_node]["exec_nodes"]:
        for i in range(size):
            all_input_shapes[i].extend(get_factors(
                self.net.graph.nodes[exec_node]["hw"].shape_in()[i]))
            all_output_shapes[i].extend(get_factors(
                self.net.graph.nodes[exec_node]["hw"].shape_out()[i]))

    # reduce to the set of these values
    all_input_shapes = [ list(set(shape)) for shape in all_input_shapes ]
    all_output_shapes = [ list(set(shape)) for shape in all_output_shapes ]

    # choose a random shape from these shapes
    next_input_shape = [ random.choice(shape) for shape in all_input_shapes ]
    next_output_shape = [ random.choice(shape) for shape in all_output_shapes ]

    # update the next shape for specific hardware types
    self.update_building_block_shape(hw_node,
            next_input_shape, max_input_shape,
            next_output_shape, max_output_shape)

def apply_min_shape(self, hw_node):
    """
    get a random shape for executing the featuremap.
    """

    # get dimensions of shape in and out
    size = self.dimensionality + 1

    # get the max shape for the input and output
    max_input_shape = [ max([ self.net.graph.nodes[exec_node]["hw"].shape_in()[i] \
                for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]) for \
                i in range(size) ]
    max_output_shape = [ max([ self.net.graph.nodes[exec_node]["hw"].shape_out()[i] \
                for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]) for \
                i in range(size) ]

    # get the min shape for the input and output
    min_input_shape = [ max([ self.net.graph.nodes[exec_node]["hw"].shape_in()[i] \
                for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]) for \
                i in range(size) ]
    min_output_shape = [ max([ self.net.graph.nodes[exec_node]["hw"].shape_out()[i] \
                for exec_node in self.building_blocks[hw_node]["exec_nodes"] ]) for \
                i in range(size) ]

    # give channels a dimension of 1
    # min_input_shape[-1] = 1
    min_output_shape[-1] = 1

    # update the next shape for specific hardware types
    self.update_building_block_shape(hw_node,
            min_input_shape, max_input_shape,
            min_output_shape, max_output_shape)


def update_building_block_shape(self, hw_node, next_input_shape,
        max_input_shape, next_output_shape, max_output_shape):

    # update the next shape for specific hardware types
    match self.building_blocks[hw_node]["type"]:
        case LAYER_TYPE.Convolution:
            # get the max kernel size
            max_kernel_rows = max([ self.net.graph.nodes[exec_node]["hw"].kernel_size[0] \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            max_kernel_cols = max([ self.net.graph.nodes[exec_node]["hw"].kernel_size[1] \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            if self.dimensionality == 3:
                max_kernel_depth = max([ self.net.graph.nodes[exec_node]["hw"].kernel_size[2] \
                        for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            # make sure rows are greater than the kernel size
            # TODO: get the actual min shape
            self.building_blocks[hw_node]["hw"].rows = max(max_kernel_rows+5, next_input_shape[0])
            self.building_blocks[hw_node]["hw"].cols = max(max_kernel_cols+5, next_input_shape[1])
            if self.dimensionality == 3:
                self.building_blocks[hw_node]["hw"].depth = max(max_kernel_depth+5, next_input_shape[2])
            # fix channels to be max TODO: do we want to have runtime channels?
            self.building_blocks[hw_node]["hw"].channels = max_input_shape[-1]
            # set a random filter dimension TODO
            self.building_blocks[hw_node]["hw"].filters = next_output_shape[-1]
        case LAYER_TYPE.InnerProduct:
            self.building_blocks[hw_node]["hw"].rows = next_input_shape[0]
            self.building_blocks[hw_node]["hw"].cols = next_input_shape[1]
            if self.dimensionality == 3:
                self.building_blocks[hw_node]["hw"].depth = next_input_shape[2]
            # fix channels to be max TODO: do we want to have runtime channels?
            self.building_blocks[hw_node]["hw"].channels = max_input_shape[-1]
            # set a random filter dimension TODO
            self.building_blocks[hw_node]["hw"].filters = next_output_shape[-1]
        case LAYER_TYPE.Pooling:
            # get the max kernel size
            max_kernel_rows = max([ self.net.graph.nodes[exec_node]["hw"].kernel_size[0] \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            max_kernel_cols = max([ self.net.graph.nodes[exec_node]["hw"].kernel_size[1] \
                    for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            if self.dimensionality == 3:
                max_kernel_depth = max([ self.net.graph.nodes[exec_node]["hw"].kernel_size[2] \
                        for exec_node in self.building_blocks[hw_node]["exec_nodes"] ])
            # make sure rows are greater than the kernel size
            # TODO: get the actual min shape
            self.building_blocks[hw_node]["hw"].rows = max(max_kernel_rows+5, next_input_shape[0])
            self.building_blocks[hw_node]["hw"].cols = max(max_kernel_cols+5, next_input_shape[1])
            if self.dimensionality == 3:
                self.building_blocks[hw_node]["hw"].depth = max(max_kernel_depth+5, next_input_shape[2])
            # update the channel dimension
            self.building_blocks[hw_node]["hw"].channels = next_input_shape[-1]
        # TODO: handle the other layer types
        case _:
            self.building_blocks[hw_node]["hw"].rows = next_input_shape[0]
            self.building_blocks[hw_node]["hw"].cols = next_input_shape[1]
            if self.dimensionality == 3:
                self.building_blocks[hw_node]["hw"].depth = next_input_shape[2]
            self.building_blocks[hw_node]["hw"].channels = next_input_shape[-1]
    # update the hw node
    self.building_blocks[hw_node]["hw"].update()

