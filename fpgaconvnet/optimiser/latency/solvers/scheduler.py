import math
import copy
import numpy as np

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def get_convolution_schedule(self, hw_node, exec_node):

    # initialise the schedule
    schedule = []

    # get the parameters for the exec node
    base_param = self.net.graph.nodes[exec_node]["hw"].layer_info_dict()

    # choose the largest factor for fine that's below the hardware's fine
    fine = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].fine,
            self.net.graph.nodes[exec_node]["hw"].get_fine_feasible()))[-1]
    # do the same for the coarse factors TODO: improve the channel, coarse trade-off
    coarse_in = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in,
            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))[-1]
    coarse_out = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_out,
            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))[-1]
    coarse_group = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_group,
            self.net.graph.nodes[exec_node]["hw"].get_coarse_group_feasible()))[-1]

    # get the repetition of each dimension
    row_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].rows_out() / \
                self.building_blocks[hw_node]["hw"].rows_out())
    col_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].cols_out() / \
                self.building_blocks[hw_node]["hw"].cols_out())
    if self.dimensionality == 3:
        depth_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].depth_out() / \
                    self.building_blocks[hw_node]["hw"].depth_())

    # channel_repetition = math.ceil(
    #     self.net.graph.nodes[exec_node]["hw"].channels_in() / \
    #             self.building_blocks[hw_node]["hw"].channels_in())
    # filter_repetition = math.ceil(
    #     self.net.graph.nodes[exec_node]["hw"].filters / \
    #             self.building_blocks[hw_node]["hw"].filters)
    # TODO: at the moment, assume filters and channels always fit
    # iterate across each dimension to be repeated

    # get the iteration space
    iteration_space = [ row_repetition, col_repetition ]
    if self.dimensionality == 3:
        iteration_space = [ row_repetition, col_repetition, depth_repetition ]

    # iterate over the tiled dimensions
    for index in  np.ndindex(*iteration_space):

         # get the greatest spatial dimensions for each execution
        rows_out = min(self.building_blocks[hw_node]["hw"].rows_out(),
                base_param["rows_out"]-index[0]*self.building_blocks[hw_node]["hw"].rows_out())
        cols_out = min(self.building_blocks[hw_node]["hw"].cols_out(),
                base_param["cols_out"]-index[1]*self.building_blocks[hw_node]["hw"].cols_out())
        if self.dimensionality == 3:
            depth_out = min(self.building_blocks[hw_node]["hw"].depth_out(),
                    base_param["cols_out"]-index[2]*self.building_blocks[hw_node]["hw"].depth_out())

        # convert the output dimensions to input dimensions
        rows_in = (rows_out*base_param["stride"][0]) + base_param["kernel_size"][0] \
                - base_param["pad_bottom"] - base_param["pad_top"] - 1
        cols_in = (cols_out*base_param["stride"][1]) + base_param["kernel_size"][1] \
                - base_param["pad_left"] - base_param["pad_right"] - 1
        if self.dimensionality == 3:
            depth_in = (depth_out*base_param["stride"][2]) + base_param["kernel_size"][2] \
                    - base_param["pad_front"] - base_param["pad_back"] - 1

        # add the parameters to the schedule
        param = copy.deepcopy(base_param)
        param["rows_in"] = rows_in
        param["cols_in"] = cols_in
        if self.dimensionality == 3:
            param["depth_in"] = depth_in
        param["fine"] = fine
        param["coarse_in"] = coarse_in
        param["coarse_out"] = coarse_out
        param["coarse_group"] = coarse_group

        # append to the schedule
        schedule.append(param)

    # return the schedule
    return schedule

def get_inner_product_schedule(self, hw_node, exec_node):

    # initialise the schedule
    schedule = []

    # get the parameters for the exec node
    base_param = self.net.graph.nodes[exec_node]["hw"].layer_info_dict()

    # do the same for the coarse factors TODO: improve the channel, coarse trade-off
    coarse_in = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in,
            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))[-1]
    coarse_out = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_out,
            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))[-1]
    coarse_group = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_group,
            self.net.graph.nodes[exec_node]["hw"].get_coarse_group_feasible()))[-1]

    # add the parameters to the schedule
    param = copy.deepcopy(base_param)
    param["coarse_in"] = coarse_in
    param["coarse_out"] = coarse_out
    param["coarse_group"] = coarse_group

    # append to the schedule
    schedule.append(param)

    # return the schedule
    return schedule

def get_pooling_schedule(self, hw_node, exec_node):

    # initialise the schedule
    schedule = []

    # get the parameters for the exec node
    base_param = self.net.graph.nodes[exec_node]["hw"].layer_info_dict()

    # do the same for the coarse factors TODO: improve the channel, coarse trade-off
    coarse = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse,
            self.net.graph.nodes[exec_node]["hw"].get_coarse_feasible()))[-1]

    # get the repetition of each dimension
    row_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].rows_out() / \
                self.building_blocks[hw_node]["hw"].rows_out())
    col_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].cols_out() / \
                self.building_blocks[hw_node]["hw"].cols_out())
    if self.dimensionality == 3:
        depth_repetition = math.ceil(
            self.net.graph.nodes[exec_node]["hw"].depth_out() / \
                    self.building_blocks[hw_node]["hw"].depth_())
    channel_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].channels_in() / \
                self.building_blocks[hw_node]["hw"].channels_in())

    # get the iteration space
    iteration_space = [ row_repetition, col_repetition, channel_repetition ]
    if self.dimensionality == 3:
        iteration_space = [ row_repetition, col_repetition, depth_repetition, channel_repetition ]

    # iterate over the tiled dimensions
    for index in  np.ndindex(*iteration_space):

         # get the greatest spatial dimensions for each execution
        rows_out = min(self.building_blocks[hw_node]["hw"].rows_out(),
                base_param["rows_out"]-index[0]*self.building_blocks[hw_node]["hw"].rows_out())
        cols_out = min(self.building_blocks[hw_node]["hw"].cols_out(),
                base_param["cols_out"]-index[1]*self.building_blocks[hw_node]["hw"].cols_out())
        if self.dimensionality == 3:
            depth_out = min(self.building_blocks[hw_node]["hw"].depth_out(),
                    base_param["cols_out"]-index[2]*self.building_blocks[hw_node]["hw"].depth_out())
        channels_in = min(self.building_blocks[hw_node]["hw"].depth_out(),
                base_param["channels_in"]-index[3]*self.building_blocks[hw_node]["hw"].channels_in())

        # convert the output dimensions to input dimensions
        rows_in = (rows_out*base_param["stride"][0]) + base_param["kernel_size"][0] \
                - base_param["pad_bottom"] - base_param["pad_top"] - 1
        cols_in = (cols_out*base_param["stride"][1]) + base_param["kernel_size"][1] \
                - base_param["pad_left"] - base_param["pad_right"] - 1
        if self.dimensionality == 3:
            depth_in = (depth_out*base_param["stride"][2]) + base_param["kernel_size"][2] \
                    - base_param["pad_front"] - base_param["pad_back"] - 1

        # add the parameters to the schedule
        param = copy.deepcopy(base_param)
        param["rows_in"] = rows_in
        param["cols_in"] = cols_in
        if self.dimensionality == 3:
            param["depth_in"] = depth_in
        param["channels_in"] = channels_in
        param["coarse"] = coarse

        # append to the schedule
        schedule.append(param)

    # return the schedule
    return schedule

def get_basic_schedule(self, hw_node, exec_node):

    # initialise the schedule
    schedule = []

    # get the parameters for the exec node
    base_param = self.net.graph.nodes[exec_node]["hw"].layer_info_dict()

    # do the same for the coarse factors TODO: improve the channel, coarse trade-off
    coarse = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse,
            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))[-1]

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

    # iterate over the tiled dimensions
    for index in  np.ndindex(*iteration_space):

         # get the greatest spatial dimensions for each execution
        rows_in = min(self.building_blocks[hw_node]["hw"].rows_in(),
                base_param["rows_in"]-index[0]*self.building_blocks[hw_node]["hw"].rows_in())
        cols_in = min(self.building_blocks[hw_node]["hw"].cols_in(),
                base_param["cols_in"]-index[1]*self.building_blocks[hw_node]["hw"].cols_in())
        if self.dimensionality == 3:
            depth_in = min(self.building_blocks[hw_node]["hw"].depth_in(),
                    base_param["cols_in"]-index[2]*self.building_blocks[hw_node]["hw"].depth_in())
        channels_in = min(self.building_blocks[hw_node]["hw"].depth_out(),
                base_param["channels_in"]-index[3]*self.building_blocks[hw_node]["hw"].channels_in())

        # add the parameters to the schedule
        param = copy.deepcopy(base_param)
        param["rows_in"] = rows_in
        param["cols_in"] = cols_in
        if self.dimensionality == 3:
            param["depth_in"] = depth_in
        param["channels_in"] = channels_in
        param["coarse"] = coarse

        # append to the schedule
        schedule.append(param)

    # return the schedule
    return schedule

def get_schedule(self):
    """
    returns a (unoptimised) schedule for the execution of the hardware for
    `self.net.graph`. Need to choose the configuration of the input shapes,
    the choice in coarse factors, and the fine factor. We will want to use
    as much of the hardware as possible for each run
    """
    # create a schedule for each node of the execution graph
    schedule = {}

    # iterate over nodes in the execution graph
    for exec_node in self.net.graph.nodes:

        # find the hardware node
        hw_node = self.get_building_block(exec_node)

        # handle different hardware types
        match self.net.graph.nodes[exec_node]["type"]:
            case LAYER_TYPE.Convolution:
                schedule[exec_node] = self.get_convolution_schedule(hw_node, exec_node)
            case LAYER_TYPE.InnerProduct:
                schedule[exec_node] = self.get_inner_product_schedule(hw_node, exec_node)
            case LAYER_TYPE.Pooling:
                schedule[exec_node] = self.get_pooling_schedule(hw_node, exec_node)
            case LAYER_TYPE.ReLU:
                schedule[exec_node] = self.get_basic_schedule(hw_node, exec_node)
            case LAYER_TYPE.EltWise:
                schedule[exec_node] = self.get_basic_schedule(hw_node, exec_node)
            case _:
                raise NotImplementedError(self.net.graph.nodes[exec_node]["type"], "schedule not implemented")

        # change rows_in, cols_in, depth_in, etc... to rows, cols, depth, ...
        for i in range(len(schedule[exec_node])):
            schedule[exec_node][i]["rows"] = schedule[exec_node][i]["rows_in"]
            schedule[exec_node][i]["cols"] = schedule[exec_node][i]["cols_in"]
            schedule[exec_node][i]["channels"] = schedule[exec_node][i]["channels_in"]
            if "depth_in" in schedule[exec_node][i]:
                schedule[exec_node][i]["depth"] = schedule[exec_node][i]["depth_in"]

    # return the schedule
    return schedule


def validate_schedule(self):
    pass

