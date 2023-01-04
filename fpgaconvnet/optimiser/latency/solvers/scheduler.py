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
    fine = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].fine and \
            f in self.building_blocks[hw_node]["hw"].get_fine_feasible(),
            self.net.graph.nodes[exec_node]["hw"].get_fine_feasible()))
    if not self.building_blocks[hw_node]["hw"].depthwise:
        coarse_group = max(filter(lambda f: f <= \
                self.building_blocks[hw_node]["hw"].coarse_group and \
                f in self.building_blocks[hw_node]["hw"].get_coarse_group_feasible(),
                self.net.graph.nodes[exec_node]["hw"].get_coarse_group_feasible()))

    # get the repetition of each dimension
    if self.building_blocks[hw_node]["hw"].rows_in() >= \
            self.net.graph.nodes[exec_node]["hw"].rows_in():
        row_repetition = 1
    else:
        row_repetition = math.ceil(
                self.net.graph.nodes[exec_node]["hw"].rows_in() / \
                    (self.building_blocks[hw_node]["hw"].rows_in() - \
                    (base_param["kernel_rows"]-1)))
    if self.building_blocks[hw_node]["hw"].cols_in() >= \
            self.net.graph.nodes[exec_node]["hw"].cols_in():
        col_repetition = 1
    else:
        col_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].cols_in() / \
                    (self.building_blocks[hw_node]["hw"].cols_in() - \
                    (base_param["kernel_cols"]-1)))
    if self.dimensionality == 3:
        if self.building_blocks[hw_node]["hw"].depth_in() >= \
                self.net.graph.nodes[exec_node]["hw"].depth_in():
            depth_repetition = 1
        else:
            depth_repetition = math.ceil(
                self.net.graph.nodes[exec_node]["hw"].depth_in() / \
                        (self.building_blocks[hw_node]["hw"].depth_in() - \
                        (base_param["kernel_depth"]-1)))
    channel_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].channels_in() / \
                self.building_blocks[hw_node]["hw"].channels_in())
    filter_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].channels_out() / \
                self.building_blocks[hw_node]["hw"].channels_out())

    # get the iteration space
    iteration_space = [ row_repetition, col_repetition,
            channel_repetition, filter_repetition ]
    if self.dimensionality == 3:
        iteration_space = [ row_repetition, col_repetition,
                depth_repetition, channel_repetition, filter_repetition ]
    if self.building_blocks[hw_node]["hw"].depthwise:
        iteration_space[-1] = 1

    # get the parameters to be updated for each execution
    new_param = copy.copy(base_param)
    new_param["fine"] = fine
    if not self.building_blocks[hw_node]["hw"].depthwise:
        new_param["coarse_group"] = coarse_group

    # get the max parameters
    if row_repetition == 1:
        rows_in_max = self.building_blocks[hw_node]["hw"].rows_in()
    else:
        rows_in_max = self.building_blocks[hw_node]["hw"].rows_in()-(base_param["kernel_rows"]-1)
    if col_repetition == 1:
        cols_in_max = self.building_blocks[hw_node]["hw"].cols_in()
    else:
        cols_in_max = self.building_blocks[hw_node]["hw"].cols_in()-(base_param["kernel_cols"]-1)
    if self.dimensionality == 3:
        if depth_repetition == 1:
            depth_in_max = self.building_blocks[hw_node]["hw"].depth_in()
        else:
            depth_in_max = self.building_blocks[hw_node]["hw"].depth_in()-(base_param["kernel_depth"]-1)
    channels_in_max = self.building_blocks[hw_node]["hw"].channels_in()
    channels_out_max = self.building_blocks[hw_node]["hw"].channels_out()
    if self.building_blocks[hw_node]["hw"].depthwise:
        channels_out_max = channels_in_max
    coarse_in_max = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in and \
        f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(channels_in_max)))
    coarse_out_max = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_out and \
        f in self.building_blocks[hw_node]["hw"].get_coarse_out_feasible(), get_factors(channels_out_max)))
    if self.building_blocks[hw_node]["hw"].depthwise:
        coarse_group_max = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_group and \
            f in self.building_blocks[hw_node]["hw"].get_coarse_group_feasible(), get_factors(channels_out_max)))

    # get the edge parameters
    rows_in_edge = base_param["rows_in"]-(row_repetition-1)*rows_in_max
    cols_in_edge = base_param["cols_in"]-(col_repetition-1)*cols_in_max
    if self.dimensionality == 3:
        depth_in_edge = base_param["depth_in"]-(depth_repetition-1)*depth_in_max
    channels_in_edge = base_param["channels_in"]-(channel_repetition-1)*channels_in_max
    channels_out_edge = base_param["channels_out"]-(filter_repetition-1)*channels_out_max
    if self.building_blocks[hw_node]["hw"].depthwise:
        channels_out_edge = channels_in_edge
    coarse_in_edge = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in and \
        f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(channels_in_edge)))
    coarse_out_edge = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_out and \
        f in self.building_blocks[hw_node]["hw"].get_coarse_out_feasible(), get_factors(channels_out_edge)))
    if self.building_blocks[hw_node]["hw"].depthwise:
        coarse_group_edge = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_group and \
            f in self.building_blocks[hw_node]["hw"].get_coarse_group_feasible(), get_factors(channels_out_edge)))

    # get the schedule
    schedule_iteration_space = [ min(3,_) for _ in iteration_space ]
    for index in np.ndindex(*schedule_iteration_space):

        # get the offsets for repetition
        row_rep_offset = schedule_iteration_space[0]-1
        col_rep_offset = schedule_iteration_space[1]-1
        if self.dimensionality == 3:
            depth_rep_offset  = schedule_iteration_space[2]-1
        channel_rep_offset = schedule_iteration_space[-2]-1
        filter_rep_offset = schedule_iteration_space[-1]-1

        # get the parameter repetition
        param_repetition = \
                ( (row_repetition-row_rep_offset) if index[0] == 1 else 1 ) *\
                ( (col_repetition-col_rep_offset) if index[1] == 1 else 1 ) *\
                ( (channel_repetition-channel_rep_offset) if index[-2] == 1 else 1 ) *\
                ( (filter_repetition-filter_rep_offset) if index[-1] == 1 else 1 )
        if self.dimensionality == 3:
            param_repetition *= ( (depth_repetition-depth_rep_offset) if index[2] == 1 else 1 )

        # get the new parameters
        new_param["rows_in"] = rows_in_max if index[0] > 0 else rows_in_edge
        new_param["cols_in"] = cols_in_max if index[1] > 0 else cols_in_edge
        if self.dimensionality == 3:
            new_param["depth_in"] = depth_in_max if index[2] > 0 else depth_in_edge

        new_param["channels_in"] = channels_in_max if index[-2] > 0 else channels_in_edge
        new_param["channels_out"] = channels_out_max if index[-1] > 0 else channels_out_edge
        new_param["filters"] = new_param["channels_out"]

        new_param["coarse_in"] = coarse_in_max if index[-2] > 0 else coarse_in_edge
        new_param["coarse_out"] = coarse_out_max if index[-1] > 0 else coarse_out_edge

        if self.building_blocks[hw_node]["hw"].depthwise:
            new_param["coarse_group"] = coarse_group_max if index[-2] > 0 else coarse_group_edge
            new_param["channels_out"] = new_param["channels_in"]
            new_param["filters"] = new_param["channels_out"]
            new_param["coarse_out"] = new_param["coarse_in"]
            assert new_param["channels_in"] == new_param["channels_out"], \
                    "Depthwise convolution must have the same number of input and output channels"
            new_param["groups"] = new_param["channels_in"]

        # set all the padding to zero
        new_param["pad_top"] = 0
        new_param["pad_bottom"] = 0
        new_param["pad_left"] = 0
        new_param["pad_right"] = 0

        if self.dimensionality == 3:
            new_param["pad_front"] = 0
            new_param["pad_back"] = 0

        # only add padding parameters at the edge
        if index[0] == 0:
            new_param["pad_top"] = base_param["pad_top"]
        if index[0] == (schedule_iteration_space[0]-1):
            new_param["pad_bottom"] = base_param["pad_bottom"]
        if index[1] == 0:
            new_param["pad_left"] = base_param["pad_left"]
        if index[1] == (schedule_iteration_space[1]-1):
            new_param["pad_right"] = base_param["pad_right"]
        if self.dimensionality == 3:
            if index[2] == 0:
                new_param["pad_front"] = base_param["pad_front"]
            if index[2] == (schedule_iteration_space[2]-1):
                new_param["pad_back"] = base_param["pad_back"]

        # add the required overlap for spatial dimensions
        if row_repetition > 1:
            new_param["rows_in"] += base_param["kernel_rows"] - 1 \
                    - new_param["pad_top"] - new_param["pad_bottom"]
        if col_repetition > 1:
            new_param["cols_in"] += base_param["kernel_cols"] - 1 \
                    - new_param["pad_right"] - new_param["pad_left"]
        if self.dimensionality == 3:
            if depth_repetition > 1:
                new_param["depth_in"] += base_param["kernel_depth"] - 1 \
                        - new_param["pad_front"] - new_param["pad_back"]

        # append to the schedule
        schedule_param = (new_param.copy(), param_repetition)


        # append scheduled parameters to schedule
        schedule.append(schedule_param)
        # schedule.extend(schedule_param)

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

    # number of times to repeat channel and filter dimension
    channel_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].channels_in() / \
                self.building_blocks[hw_node]["hw"].channels_in())
    filter_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].channels_out() / \
                self.building_blocks[hw_node]["hw"].channels_out())

    # get the iteration space
    iteration_space = [ channel_repetition, filter_repetition ]

    # get the parameters to be updated for each execution
    new_param = copy.copy(base_param)

    # get the max parameters
    channels_in_max = self.building_blocks[hw_node]["hw"].channels_in()
    channels_out_max = self.building_blocks[hw_node]["hw"].channels_out()
    coarse_in_max = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in and \
        f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(channels_in_max)))
    coarse_out_max = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_out and \
        f in self.building_blocks[hw_node]["hw"].get_coarse_out_feasible(), get_factors(channels_out_max)))

    # get the edge parameters
    channels_in_edge = base_param["channels_in"]-(channel_repetition-1)*\
            self.building_blocks[hw_node]["hw"].channels_in()
    channels_out_edge = base_param["channels_out"]-(filter_repetition-1)*\
            self.building_blocks[hw_node]["hw"].channels_out()
    coarse_in_edge = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in and \
        f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(channels_in_edge)))
    coarse_out_edge = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_out and \
        f in self.building_blocks[hw_node]["hw"].get_coarse_out_feasible(), get_factors(channels_out_edge)))

    # get the schedule
    schedule_iteration_space = [ min(2,_) for _ in iteration_space ]
    for index in np.ndindex(*schedule_iteration_space):
        # get the offsets for repetition
        channel_rep_offset = schedule_iteration_space[0]-1
        filter_rep_offset = schedule_iteration_space[1]-1
        # get the parameter repetition
        param_repetition = \
                ( (channel_repetition-channel_rep_offset) if index[0] else 1 ) *\
                ( (filter_repetition-filter_rep_offset) if index[1] else 1 )
        # get the new parameters
        new_param["channels_in"] = channels_in_max if index[0] else channels_in_edge
        new_param["channels_out"] = channels_out_max if index[1] else channels_out_edge
        new_param["coarse_in"] = coarse_in_max if index[0] else coarse_in_edge
        new_param["coarse_out"] = coarse_out_max if index[1] else coarse_out_edge
        new_param["filters"] = new_param["channels_out"]
        # append to the schedule
        # schedule.extend([new_param.copy()]*param_repetition)
        schedule.append((new_param.copy(), param_repetition))

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
    if self.building_blocks[hw_node]["hw"].rows_in() >= \
            self.net.graph.nodes[exec_node]["hw"].rows_in():
        row_repetition = 1
    else:
        row_repetition = math.ceil(
                self.net.graph.nodes[exec_node]["hw"].rows_in() / \
                    (self.building_blocks[hw_node]["hw"].rows_in() - \
                    (base_param["kernel_rows"]-1)))
    if self.building_blocks[hw_node]["hw"].cols_in() >= \
            self.net.graph.nodes[exec_node]["hw"].cols_in():
        col_repetition = 1
    else:
        col_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].cols_in() / \
                    (self.building_blocks[hw_node]["hw"].cols_in() - \
                    (base_param["kernel_cols"]-1)))
    if self.dimensionality == 3:
        if self.building_blocks[hw_node]["hw"].depth_in() >= \
                self.net.graph.nodes[exec_node]["hw"].depth_in():
            depth_repetition = 1
        else:
            depth_repetition = math.ceil(
                self.net.graph.nodes[exec_node]["hw"].depth_in() / \
                        (self.building_blocks[hw_node]["hw"].depth_in() - \
                        (base_param["kernel_depth"]-1)))
    channel_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].channels_in() / \
                self.building_blocks[hw_node]["hw"].channels_in())

    # get the iteration space
    iteration_space = [ row_repetition, col_repetition,
            channel_repetition ]
    if self.dimensionality == 3:
        iteration_space = [ row_repetition, col_repetition,
                depth_repetition, channel_repetition ]

    # get the parameters to be updated for each execution
    new_param = copy.copy(base_param)

    # get the max parameters
    if row_repetition == 1:
        rows_in_max = self.building_blocks[hw_node]["hw"].rows_in()
    else:
        rows_in_max = self.building_blocks[hw_node]["hw"].rows_in()-(base_param["kernel_rows"]-1)
    if col_repetition == 1:
        cols_in_max = self.building_blocks[hw_node]["hw"].cols_in()
    else:
        cols_in_max = self.building_blocks[hw_node]["hw"].cols_in()-(base_param["kernel_cols"]-1)
    if self.dimensionality == 3:
        if depth_repetition == 1:
            depth_in_max = self.building_blocks[hw_node]["hw"].depth_in()
        else:
            depth_in_max = self.building_blocks[hw_node]["hw"].depth_in()-(base_param["kernel_depth"]-1)
    channels_in_max = self.building_blocks[hw_node]["hw"].channels_in()
    coarse_max = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse and \
        f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(channels_in_max)))

    # get the edge parameters
    rows_in_edge = base_param["rows_in"]-(row_repetition-1)*rows_in_max
    cols_in_edge = base_param["cols_in"]-(col_repetition-1)*cols_in_max
    if self.dimensionality == 3:
        depth_in_edge = base_param["depth_in"]-(depth_repetition-1)*depth_in_max
    channels_in_edge = base_param["channels_in"]-(channel_repetition-1)*channels_in_max
    coarse_edge = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse and \
        f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(channels_in_edge)))

    # get the schedule
    schedule_iteration_space = [ min(3,_) for _ in iteration_space ]
    for index in np.ndindex(*schedule_iteration_space):

        # get the offsets for repetition
        row_rep_offset = schedule_iteration_space[0]-1
        col_rep_offset = schedule_iteration_space[1]-1
        if self.dimensionality == 3:
            depth_rep_offset  = schedule_iteration_space[2]-1
        channel_rep_offset = schedule_iteration_space[-1]-1

        # get the parameter repetition
        param_repetition = \
                ( (row_repetition-row_rep_offset) if index[0] == 1 else 1 ) *\
                ( (col_repetition-col_rep_offset) if index[1] == 1 else 1 ) *\
                ( (channel_repetition-channel_rep_offset) if index[-1] == 1 else 1 )
        if self.dimensionality == 3:
            param_repetition *= ( (depth_repetition-depth_rep_offset) if index[2] == 1 else 1 )

        # get the new parameters
        new_param["rows_in"] = rows_in_max if index[0] > 0 else rows_in_edge
        new_param["cols_in"] = cols_in_max if index[1] > 0 else cols_in_edge
        if self.dimensionality == 3:
            new_param["depth_in"] = depth_in_max if index[2] > 0 else depth_in_edge

        new_param["channels_in"] = channels_in_max if index[-1] > 0 else channels_in_edge

        new_param["coarse"] = coarse_max if index[-1] > 0 else coarse_edge

        # set all the padding to zero
        new_param["pad_top"] = 0
        new_param["pad_bottom"] = 0
        new_param["pad_left"] = 0
        new_param["pad_right"] = 0

        if self.dimensionality == 3:
            new_param["pad_front"] = 0
            new_param["pad_back"] = 0

        # only add padding parameters at the edge
        if index[0] == 0:
            new_param["pad_top"] = base_param["pad_top"]
        if index[0] == (schedule_iteration_space[0]-1):
            new_param["pad_bottom"] = base_param["pad_bottom"]
        if index[1] == 0:
            new_param["pad_left"] = base_param["pad_left"]
        if index[1] == (schedule_iteration_space[1]-1):
            new_param["pad_right"] = base_param["pad_right"]
        if self.dimensionality == 3:
            if index[2] == 0:
                new_param["pad_front"] = base_param["pad_front"]
            if index[2] == (schedule_iteration_space[2]-1):
                new_param["pad_back"] = base_param["pad_back"]

        # add the required overlap for spatial dimensions
        if row_repetition > 1:
            new_param["rows_in"] += base_param["kernel_rows"] - 1 \
                    - new_param["pad_top"] - new_param["pad_bottom"]
        if col_repetition > 1:
            new_param["cols_in"] += base_param["kernel_cols"] - 1 \
                    - new_param["pad_right"] - new_param["pad_left"]
        if self.dimensionality == 3:
            if depth_repetition > 1:
                new_param["depth_in"] += base_param["kernel_depth"] - 1 \
                        - new_param["pad_front"] - new_param["pad_back"]

        # append to the schedule
        schedule_param = (new_param.copy(), param_repetition)


        # append scheduled parameters to schedule
        schedule.append(schedule_param)
        # schedule.extend(schedule_param)

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

    # set some parameters based on hardware type
    if self.net.graph.nodes[exec_node]["type"].name in ["ReLU", "Sigmoid", "SiLU"]:
        new_param["op_type"] = self.net.graph.nodes[exec_node]["type"].name.lower()
    if self.net.graph.nodes[exec_node]["type"].name == "EltWise":
        new_param["op_type"] = self.net.graph.nodes[exec_node]["hw"].op_type
        new_param["broadcast"] = self.net.graph.nodes[exec_node]["hw"].broadcast

    # get the max parameters
    rows_in_max = self.building_blocks[hw_node]["hw"].rows_in()
    cols_in_max = self.building_blocks[hw_node]["hw"].cols_in()
    if self.dimensionality == 3:
        depth_in_max = self.building_blocks[hw_node]["hw"].depth_in()
    channels_in_max = self.building_blocks[hw_node]["hw"].channels_in()
    match self.building_blocks[hw_node]['type']:
        case LAYER_TYPE.EltWise:
            coarse_max = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].streams_in() and \
                f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(channels_in_max)))
        case _:
            coarse_max = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in and \
                f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(channels_in_max)))

    # get the edge parameters
    rows_in_edge = base_param["rows_in"]-(row_repetition-1)*\
            self.building_blocks[hw_node]["hw"].rows_in()
    cols_in_edge = base_param["cols_in"]-(col_repetition-1)*\
            self.building_blocks[hw_node]["hw"].cols_in()
    if self.dimensionality == 3:
        depth_in_edge = base_param["depth_in"]-(depth_repetition-1)*\
            self.building_blocks[hw_node]["hw"].depth_in()
    channels_in_edge = base_param["channels_in"]-(channel_repetition-1)*\
            self.building_blocks[hw_node]["hw"].channels_in()
    match self.building_blocks[hw_node]['type']:
        case LAYER_TYPE.EltWise:
            coarse_edge = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].streams_in() and \
                f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(channels_in_edge)))
        case _:
            coarse_edge = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in and \
                f in self.building_blocks[hw_node]["hw"].get_coarse_in_feasible(), get_factors(channels_in_edge)))

    # get the schedule
    schedule_iteration_space = [ min(2,_) for _ in iteration_space ]
    for index in np.ndindex(*schedule_iteration_space):

        # get the offsets for repetition
        row_rep_offset = schedule_iteration_space[0]-1
        col_rep_offset = schedule_iteration_space[1]-1
        if self.dimensionality == 3:
            depth_rep_offset  = schedule_iteration_space[2]-1
        channel_rep_offset = schedule_iteration_space[-1]-1

        # get the parameter repetition
        param_repetition = \
                ( (row_repetition-row_rep_offset) if index[0] else 1 ) *\
                ( (col_repetition-col_rep_offset) if index[1] else 1 ) *\
                ( (channel_repetition-channel_rep_offset) if index[-1] else 1 )
        if self.dimensionality == 3:
            param_repetition *= ( (depth_repetition-depth_rep_offset) if index[2] else 1 )

        # get the new parameters
        new_param["rows_in"] = rows_in_max if index[0] else rows_in_edge
        new_param["cols_in"] = cols_in_max if index[1] else cols_in_edge
        if self.dimensionality == 3:
            new_param["depth_in"] = depth_in_max if index[2] else depth_in_edge

        new_param["channels_in"] = channels_in_max if index[-1] else channels_in_edge
        new_param["coarse"] = coarse_max if index[-1] else coarse_edge

        # append to the schedule
        # schedule_param = [new_param.copy()]*param_repetition
        schedule_param = (new_param.copy(), param_repetition)

        # append scheduled parameters to schedule
        # schedule.extend(schedule_param)
        schedule.append(schedule_param)

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
            schedule[exec_node][i][0]["rows"] = schedule[exec_node][i][0]["rows_in"]
            schedule[exec_node][i][0]["cols"] = schedule[exec_node][i][0]["cols_in"]
            schedule[exec_node][i][0]["channels"] = schedule[exec_node][i][0]["channels_in"]
            if "depth_in" in schedule[exec_node][i][0]:
                schedule[exec_node][i][0]["depth"] = schedule[exec_node][i][0]["depth_in"]

    # validate the schedule
    self.validate_schedule(schedule, iteration_space)

    # return the schedule
    return schedule, iteration_space


def validate_schedule(self, schedule, iteration_space):

    # check all the exec nodes are in the schedule
    exec_nodes = set(self.net.graph.nodes())
    schedule_nodes = set(schedule.keys())
    assert exec_nodes.difference(schedule_nodes) == set(), "not all execution nodes are scheduled"

    # iterate over exec nodes in schedule
    for exec_node in self.net.graph.nodes:

        # get the hw_node and check the schedule parameters fit within the hw node
        hw_node = self.get_building_block(exec_node)
        hw_node_type = self.building_blocks[hw_node]["type"]
        hw_node = self.building_blocks[hw_node]["hw"]

        for conf in schedule[exec_node]:
            for param, val in conf[0].items():
                # assertion message
                assertion_message = f"[{exec_node}] {param} too large! ({val})"
                # get the hw_node param
                match param:
                    case "batch_size":
                        pass
                    case "rows_in":
                        assert val <= hw_node.rows_in(), assertion_message
                    case "cols_in":
                        assert val <= hw_node.cols_in(), assertion_message
                    case "depth_in":
                        assert val <= hw_node.depth_in(), assertion_message
                    case "channels_in":
                        assert val <= hw_node.channels_in(), assertion_message
                    case "coarse_in":
                        if hw_node_type == LAYER_TYPE.EltWise:
                            assert val <= hw_node.streams_in(), assertion_message
                        else:
                            assert val <= hw_node.coarse_in, assertion_message
                    case "coarse_out":
                        if hw_node_type == LAYER_TYPE.EltWise:
                            assert val <= hw_node.streams_out(), assertion_message
                        else:
                            assert val <= hw_node.coarse_out, assertion_message
                    case "mem_bw_in" | "mem_bw_out":
                        assert val <= getattr(hw_node, param) + 1e-4, assertion_message
                    case "rows" | "cols" | "depth" | "channels":
                        pass
                    case "rows_out" | "cols_out" | "depth_out" | "channels_out":
                        pass
                    case "data_t" | "weight_t" | "acc_t" | "input_t" | "output_t":
                        pass
                    case "mem_bw_in_array" | "mem_bw_out_array":
                        pass
                    case "has_bias":
                        pass
                    case "op_type":
                        pass
                    case _:
                        assert val <= getattr(hw_node, param), assertion_message

