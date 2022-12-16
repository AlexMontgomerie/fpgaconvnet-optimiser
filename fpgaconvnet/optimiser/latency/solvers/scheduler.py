import math
import copy
import numpy as np

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def get_convolution_schedule(self, hw_node, exec_node):

    # helper function to get factors of a number
    def get_factors(num):
        return [n for n in range(1, num + 1) if num % n == 0]

    # initialise the schedule
    schedule = []

    # get the parameters for the exec node
    base_param = self.net.graph.nodes[exec_node]["hw"].layer_info_dict()

    # choose the largest factor for fine that's below the hardware's fine
    fine = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].fine,
            self.net.graph.nodes[exec_node]["hw"].get_fine_feasible()))
    if not self.channel_tiling:
        coarse_in = max(filter(lambda f: f <= \
            self.building_blocks[hw_node]["hw"].coarse_in and \
            f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(),
            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))
    if not self.filter_tiling:
        coarse_out = max(filter(lambda f: f <= \
            self.building_blocks[hw_node]["hw"].coarse_out and \
            f in self.building_blocks[hw_node]["hw"].get_coarse_out_feasible(),
            self.net.graph.nodes[exec_node]["hw"].get_coarse_out_feasible()))
    coarse_group = max(filter(lambda f: f <= \
            self.building_blocks[hw_node]["hw"].coarse_group and \
            f in self.building_blocks[hw_node]["hw"].get_coarse_group_feasible(),
            self.net.graph.nodes[exec_node]["hw"].get_coarse_group_feasible()))

    # get the repetition of each dimension
    row_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].rows_in() / \
                self.building_blocks[hw_node]["hw"].rows_in())
    col_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].cols_in() / \
                self.building_blocks[hw_node]["hw"].cols_in())
    if self.dimensionality == 3:
        depth_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].depth_in() / \
                    self.building_blocks[hw_node]["hw"].depth_in())
    if self.channel_tiling:
        channel_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].channels_in() / \
                    self.building_blocks[hw_node]["hw"].channels_in())
    if self.filter_tiling:
        filter_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].channels_out() / \
                    self.building_blocks[hw_node]["hw"].channels_out())

    # get the iteration space
    iteration_space = [ row_repetition, col_repetition ]
    if self.dimensionality == 3:
        iteration_space = [ row_repetition, col_repetition, depth_repetition ]
    if self.channel_tiling:
        iteration_space.append(channel_repetition)
    if self.filter_tiling:
        iteration_space.append(filter_repetition)

    # calculate the channel and filter offset
    if self.channel_tiling and self.filter_tiling:
        channel_offset = -2
        filter_offset = -1
    elif self.channel_tiling:
        channel_offset = -1
    elif self.filter_tiling:
        filter_offset = -1

    # get the parameters to be updated for each execution
    new_param = copy.copy(base_param)
    new_param["fine"] = fine
    new_param["coarse_group"] = coarse_group

    if not self.channel_tiling:
        new_param["coarse_in"] = coarse_in
    if not self.filter_tiling:
        new_param["coarse_out"] = coarse_out

    # iterate over the tiled dimensions
    for index in np.ndindex(*iteration_space):

        # get the greatest spatial dimensions for each execution
        new_param["rows_in"] = min(self.building_blocks[hw_node]["hw"].rows_in(),
                base_param["rows_in"]-index[0]*self.building_blocks[hw_node]["hw"].rows_in())
        new_param["cols_in"] = min(self.building_blocks[hw_node]["hw"].cols_in(),
                base_param["cols_in"]-index[1]*self.building_blocks[hw_node]["hw"].cols_in())
        if self.dimensionality == 3:
            new_param["depth_in"] = min(self.building_blocks[hw_node]["hw"].depth_in(),
                    base_param["depth_in"]-index[2]*self.building_blocks[hw_node]["hw"].depth_in())

        # add the required overlap for spatial dimensions
        if row_repetition > 1:
            new_param["rows_in"] += base_param["kernel_rows"] - 1
        if col_repetition > 1:
            new_param["cols_in"] += base_param["kernel_cols"] - 1
        if self.dimensionality == 3:
            if depth_repetition > 1:
                new_param["depth_in"] += base_param["kernel_depth"] - 1

        # greedy channel dimension
        if self.channel_tiling:
            new_param["channels_in"] = min(self.building_blocks[hw_node]["hw"].channels_in(),
                    base_param["channels_in"]-index[channel_offset]*self.building_blocks[hw_node]["hw"].channels_in())
            # choose coarse in as a factor of the channels dimension
            new_param["coarse_out"] = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in and \
                    f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(new_param["channels_in"])))

        if self.filter_tiling:
            # greedy filter dimension
            new_param["channels_out"] = min(self.building_blocks[hw_node]["hw"].channels_out(),
                    base_param["filters"]-index[filter_offset]*self.building_blocks[hw_node]["hw"].channels_out())
            # choose coarse out as a factor of the filters dimension
            new_param["coarse_out"] = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_out and \
                    f in self.building_blocks[hw_node]["hw"].get_coarse_out_feasible(), get_factors(new_param["channels_out"])))
            new_param["filters"] = new_param["channels_out"]

        # only add padding parameters at the edge
        new_param["pad_top"]    = base_param["pad_top"]     if index[0] else 0
        new_param["pad_bottom"] = base_param["pad_bottom"]  if index[0] == (row_repetition-1) else 0
        new_param["pad_left"]   = base_param["pad_left"]    if index[1] else 0
        new_param["pad_right"]  = base_param["pad_right"]   if index[1] == (col_repetition-1) else 0
        if self.dimensionality == 3:
            new_param["pad_front"]  = base_param["pad_front"]   if index[2] else 0
            new_param["pad_back"]   = base_param["pad_back"]    if index[2] == (depth_repetition-1) else 0

        # append to the schedule
        schedule.append(new_param.copy())

    # return the schedule
    return schedule, iteration_space

def get_inner_product_schedule(self, hw_node, exec_node):

    # helper function to get factors of a number
    def get_factors(num):
        return [n for n in range(1, num + 1) if num % n == 0]

    # initialise the schedule
    schedule = []

    # get the parameters for the exec node
    base_param = self.net.graph.nodes[exec_node]["hw"].layer_info_dict()

    if not self.channel_tiling:
        coarse_in = max(filter(lambda f: f <= \
        self.building_blocks[hw_node]["hw"].coarse_in and \
        f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(),
        self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))
    if not self.filter_tiling:
        coarse_out = max(filter(lambda f: f <= \
        self.building_blocks[hw_node]["hw"].coarse_out and \
        f in self.building_blocks[hw_node]["hw"].get_coarse_out_feasible(),
        self.net.graph.nodes[exec_node]["hw"].get_coarse_out_feasible()))

    if self.channel_tiling:
        # number of times to repeat channel dimension
        channel_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].channels_in() / \
                    self.building_blocks[hw_node]["hw"].channels_in())
    if self.filter_tiling:
        # number of times to repeat filter dimension
        filter_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].channels_out() / \
                    self.building_blocks[hw_node]["hw"].channels_out())

    # get the iteration space
    iteration_space = []
    if self.channel_tiling:
        iteration_space.append(channel_repetition)
    if self.filter_tiling:
        iteration_space.append(filter_repetition)

    # calculate the channel and filter offset
    if self.channel_tiling and self.filter_tiling:
        channel_offset = -2
        filter_offset = -1
    elif self.channel_tiling:
        channel_offset = -1
    elif self.filter_tiling:
        filter_offset = -1

    # get the parameters to be updated for each execution
    new_param = copy.copy(base_param)

    if not self.channel_tiling:
        new_param["coarse_in"] = coarse_in
    if not self.filter_tiling:
        new_param["coarse_out"] = coarse_out

    # iterate over the tiled dimensions
    for index in np.ndindex(*iteration_space):

        # greedy channel dimension
        if self.channel_tiling:
            new_param["channels_in"] = min(self.building_blocks[hw_node]["hw"].channels_in(),
                    base_param["channels_in"]-index[channel_offset]*self.building_blocks[hw_node]["hw"].channels_in())
            # choose coarse in as a factor of the channels dimension
            new_param["coarse_out"] = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in and \
                    f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(new_param["channels_in"])))

        if self.filter_tiling:
            # greedy filter dimension
            new_param["channels_out"] = min(self.building_blocks[hw_node]["hw"].channels_out(),
                    base_param["filters"]-index[filter_offset]*self.building_blocks[hw_node]["hw"].channels_out())
            # choose coarse out as a factor of the filters dimension
            new_param["coarse_out"] = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_out and \
                    f in self.building_blocks[hw_node]["hw"].get_coarse_out_feasible(), get_factors(new_param["channels_out"])))
            new_param["filters"] = new_param["channels_out"]

        # append to the schedule
        schedule.append(new_param.copy())

    if not iteration_space:
        param = copy.copy(base_param)
        param["coarse_in"] = coarse_in
        param["coarse_out"] = coarse_out
        schedule.append(param)

    # return the schedule
    return schedule, iteration_space

def get_pooling_schedule(self, hw_node, exec_node):

    # helper function to get factors of a number
    def get_factors(num):
        return [n for n in range(1, num + 1) if num % n == 0]

    # initialise the schedule
    schedule = []

    # get the parameters for the exec node
    base_param = self.net.graph.nodes[exec_node]["hw"].layer_info_dict()

    # get the repetition of each dimension
    row_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].rows_in() / \
                self.building_blocks[hw_node]["hw"].rows_in())
    col_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].cols_in() / \
                self.building_blocks[hw_node]["hw"].cols_in())
    if self.dimensionality == 3:
        depth_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].depth_in() / \
                    self.building_blocks[hw_node]["hw"].depth_in())
    channel_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].channels_in() / \
                self.building_blocks[hw_node]["hw"].channels_in())

    # get the iteration space
    iteration_space = [ row_repetition, col_repetition, channel_repetition ]
    if self.dimensionality == 3:
        iteration_space = [ row_repetition, col_repetition, depth_repetition, channel_repetition ]

    # get the parameters to be updated for each execution
    new_param = copy.copy(base_param)

    # iterate over the tiled dimensions
    for index in  np.ndindex(*iteration_space):

        # get the greatest spatial dimensions for each execution
        new_param["rows_in"] = min(self.building_blocks[hw_node]["hw"].rows_in(),
                base_param["rows_in"]-index[0]*self.building_blocks[hw_node]["hw"].rows_in())
        new_param["cols_in"] = min(self.building_blocks[hw_node]["hw"].cols_in(),
                base_param["cols_in"]-index[1]*self.building_blocks[hw_node]["hw"].cols_in())
        if self.dimensionality == 3:
            new_param["depth_in"] = min(self.building_blocks[hw_node]["hw"].depth_in(),
                    base_param["depth_in"]-index[2]*self.building_blocks[hw_node]["hw"].depth_in())
        new_param["channels_in"] = min(self.building_blocks[hw_node]["hw"].channels_in(),
                base_param["channels_in"]-index[-1]*self.building_blocks[hw_node]["hw"].channels_in())

        # choose coarse in as a factor of the channels dimension
        new_param["coarse"] = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in and \
                f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(new_param["channels_in"])))

        # add the required overlap for spatial dimensions
        if row_repetition > 1:
            new_param["rows_in"] += base_param["kernel_rows"] - 1
        if col_repetition > 1:
            new_param["cols_in"] += base_param["kernel_cols"] - 1
        if self.dimensionality == 3:
            if depth_repetition > 1:
                new_param["depth_in"] += base_param["kernel_depth"] - 1

        # only add padding parameters at the edge
        new_param["pad_top"]    = base_param["pad_top"]     if index[0] else 0
        new_param["pad_bottom"] = base_param["pad_bottom"]  if index[0] == (row_repetition-1) else 0
        new_param["pad_left"]   = base_param["pad_left"]    if index[1] else 0
        new_param["pad_right"]  = base_param["pad_right"]   if index[1] == (col_repetition-1) else 0
        if self.dimensionality == 3:
            new_param["pad_front"]  = base_param["pad_front"]   if index[2] else 0
            new_param["pad_back"]   = base_param["pad_back"]    if index[2] == (depth_repetition-1) else 0

        # append to the schedule
        schedule.append(new_param.copy())

    # return the schedule
    return schedule, iteration_space

def get_basic_schedule(self, hw_node, exec_node):

    # helper function to get factors of a number
    def get_factors(num):
        return [n for n in range(1, num + 1) if num % n == 0]

    # initialise the schedule
    schedule = []

    # get the parameters for the exec node
    base_param = self.net.graph.nodes[exec_node]["hw"].layer_info_dict()

    # get the repetition of each dimension
    row_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].rows_in() / \
                self.building_blocks[hw_node]["hw"].rows_in())
    col_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].cols_in() / \
                self.building_blocks[hw_node]["hw"].cols_in())
    if self.dimensionality == 3:
        depth_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].depth_in() / \
                    self.building_blocks[hw_node]["hw"].depth_in())
    channel_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].channels_in() / \
                self.building_blocks[hw_node]["hw"].channels_in())

    # get the iteration space
    iteration_space = [ row_repetition, col_repetition, channel_repetition ]
    if self.dimensionality == 3:
        iteration_space = [ row_repetition, col_repetition, depth_repetition, channel_repetition ]

    # get the parameters to be updated for each execution
    new_param = copy.copy(base_param)

    # iterate over the tiled dimensions
    for index in  np.ndindex(*iteration_space):

        # get the greatest spatial dimensions for each execution
        new_param["rows_in"] = min(self.building_blocks[hw_node]["hw"].rows_in(),
                base_param["rows_in"]-index[0]*self.building_blocks[hw_node]["hw"].rows_in())
        new_param["cols_in"] = min(self.building_blocks[hw_node]["hw"].cols_in(),
                base_param["cols_in"]-index[1]*self.building_blocks[hw_node]["hw"].cols_in())
        if self.dimensionality == 3:
            new_param["depth_in"] = min(self.building_blocks[hw_node]["hw"].depth_in(),
                    base_param["depth_in"]-index[2]*self.building_blocks[hw_node]["hw"].depth_in())
        new_param["channels_in"] = min(self.building_blocks[hw_node]["hw"].channels_in(),
                base_param["channels_in"]-index[-1]*self.building_blocks[hw_node]["hw"].channels_in())

        # choose coarse in as a factor of the channels dimension
        new_param["coarse"] = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in and \
                f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(new_param["channels_in"])))

        if self.net.graph.nodes[exec_node]["type"].name in ["ReLU", "Sigmoid", "SiLU"]:
            new_param["op_type"] = self.net.graph.nodes[exec_node]["type"].name.lower()
        if self.net.graph.nodes[exec_node]["type"].name == "EltWise":
            new_param["op_type"] = self.net.graph.nodes[exec_node]["hw"].op_type
            new_param["broadcast"] = self.net.graph.nodes[exec_node]["hw"].broadcast

        # append to the schedule
        schedule.append(new_param.copy())

    # return the schedule
    return schedule, iteration_space

def get_schedule(self):
    """
    returns a (unoptimised) schedule for the execution of the hardware for
    `self.net.graph`. Need to choose the configuration of the input shapes,
    the choice in coarse factors, and the fine factor. We will want to use
    as much of the hardware as possible for each run
    """
    # create a schedule for each node of the execution graph
    schedule = {}
    iteration_space = {}

    # iterate over nodes in the execution graph
    for exec_node in self.net.graph.nodes:

        # find the hardware node
        hw_node = self.get_building_block(exec_node)

        # handle different hardware types
        match self.net.graph.nodes[exec_node]["type"]:
            case LAYER_TYPE.Convolution:
                schedule[exec_node], iteration_space[exec_node] = \
                        self.get_convolution_schedule(hw_node, exec_node)
            case LAYER_TYPE.InnerProduct:
                schedule[exec_node], iteration_space[exec_node] = \
                        self.get_inner_product_schedule(hw_node, exec_node)
            case LAYER_TYPE.Pooling:
                schedule[exec_node], iteration_space[exec_node] = \
                        self.get_pooling_schedule(hw_node, exec_node)
            case LAYER_TYPE.ReLU | LAYER_TYPE.Sigmoid | LAYER_TYPE.SiLU:
                schedule[exec_node], iteration_space[exec_node] = \
                        self.get_basic_schedule(hw_node, exec_node)
            case LAYER_TYPE.EltWise:
                schedule[exec_node], iteration_space[exec_node] = \
                        self.get_basic_schedule(hw_node, exec_node)
            case LAYER_TYPE.GlobalPooling:
                schedule[exec_node], iteration_space[exec_node] = \
                        self.get_basic_schedule(hw_node, exec_node)
            case _:
                raise NotImplementedError(self.net.graph.nodes[exec_node]["type"],
                        "schedule not implemented")

        # change rows_in, cols_in, depth_in, etc... to rows, cols, depth, ...
        for i in range(len(schedule[exec_node])):
            schedule[exec_node][i]["rows"] = schedule[exec_node][i]["rows_in"]
            schedule[exec_node][i]["cols"] = schedule[exec_node][i]["cols_in"]
            schedule[exec_node][i]["channels"] = schedule[exec_node][i]["channels_in"]
            if "depth_in" in schedule[exec_node][i]:
                schedule[exec_node][i]["depth"] = schedule[exec_node][i]["depth_in"]

    # return the schedule
    return schedule, iteration_space


def validate_schedule(self):
    pass

