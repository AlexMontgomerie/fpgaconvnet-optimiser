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
    coarse_in = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in,
            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))
    # coarse_out = list(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_out,
    #         self.net.graph.nodes[exec_node]["hw"].get_coarse_out_feasible()))[-1]
    coarse_group = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_group,
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

    # TODO: at the moment, assume channels always fit
    # iterate across each dimension to be repeated
    # channel_repetition = math.ceil(
    #     self.net.graph.nodes[exec_node]["hw"].channels_in() / \
    #             self.building_blocks[hw_node]["hw"].channels_in())
    filter_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].channels_out() / \
                self.building_blocks[hw_node]["hw"].channels_out())

    # get the iteration space
    iteration_space = [ row_repetition, filter_repetition ]
    if self.dimensionality == 3:
        iteration_space = [ row_repetition, depth_repetition, filter_repetition ]

    # iterate over the tiled dimensions
    for index in np.ndindex(*iteration_space):

        # get the greatest spatial dimensions for each execution
        rows_in = min(self.building_blocks[hw_node]["hw"].rows_in(),
                base_param["rows_in"]-index[0]*self.building_blocks[hw_node]["hw"].rows_in())
        cols_in = min(self.building_blocks[hw_node]["hw"].cols_in(),
                base_param["cols_in"]-index[0]*self.building_blocks[hw_node]["hw"].cols_in())
        if self.dimensionality == 3:
            depth_in = min(self.building_blocks[hw_node]["hw"].depth_in(),
                    base_param["depth_in"]-index[1]*self.building_blocks[hw_node]["hw"].depth_in())

        # add the required overlap for spatial dimensions
        if row_repetition > 1:
            rows_in += base_param["kernel_rows"] - 1
        if col_repetition > 1:
            cols_in += base_param["kernel_cols"] - 1
        if self.dimensionality == 3:
            if depth_repetition > 1:
                depth_in += base_param["kernel_depth"] - 1

        # greedy filter dimension
        filters = min(self.building_blocks[hw_node]["hw"].channels_out(),
                base_param["filters"]-index[-1]*self.building_blocks[hw_node]["hw"].channels_out())

        # choose coarse out as a factor of the filter dimension
        coarse_out = max(filter(lambda f: f <= \
                self.building_blocks[hw_node]["hw"].coarse_out, get_factors(filters)))

        # add the parameters to the schedule
        param = copy.copy(base_param)
        assert param["filters"] == param["channels_out"], f"filters must equal channels out for {exec_node} and {hw_node}"
        assert param["channels_in"] % coarse_in == 0, f"coarse in must be a factor of channels in for {exec_node} and {hw_node}"
        assert filters % coarse_out == 0, f"coarse out must be a factor of channels out (filters) for {exec_node} and {hw_node}"
        assert param["groups"] % coarse_group == 0, f"coarse group must be a factor of groups for {exec_node} and {hw_node}"
        param["rows_in"] = rows_in
        param["cols_in"] = cols_in
        if self.dimensionality == 3:
            param["depth_in"] = depth_in
        param["fine"] = fine
        param["filters"] = filters
        param["channels_out"] = filters
        param["coarse_in"] = coarse_in
        param["coarse_out"] = coarse_out
        param["coarse_group"] = coarse_group

        # only add padding parameters at the edge
        param["pad_top"] = param["pad_top"] if index[0] else 0
        param["pad_bottom"] = param["pad_bottom"] if index[0] == (row_repetition-1) else 0
        param["pad_left"] = param["pad_left"] if index[0] else 0
        param["pad_right"] = param["pad_right"] if index[0] == (col_repetition-1) else 0
        if self.dimensionality == 3:
            param["pad_front"] = param["pad_front"] if index[1] else 0
            param["pad_back"] = param["pad_back"] if index[1] == (depth_repetition-1) else 0

        # append to the schedule
        schedule.append(param)

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

    # do the same for the coarse factors TODO: improve the channel, coarse trade-off
    coarse_in = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_in,
            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))
    # coarse_out = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse_out,
    #         self.net.graph.nodes[exec_node]["hw"].get_coarse_out_feasible()))

    # number of times to repeat filter dimension
    filter_repetition = math.ceil(
        self.net.graph.nodes[exec_node]["hw"].channels_out() / \
                self.building_blocks[hw_node]["hw"].channels_out())

    # get the iteration space
    iteration_space = [  filter_repetition ]

    # iterate over the tiled dimensions
    for index in np.ndindex(*iteration_space):

        # greedy filter dimension
        filters = min(self.building_blocks[hw_node]["hw"].channels_out(),
                base_param["filters"]-index[-1]*self.building_blocks[hw_node]["hw"].channels_out())

        # choose coarse out as a factor of the filter dimension
        coarse_out = max(filter(lambda f: f <= \
                self.building_blocks[hw_node]["hw"].coarse_out, get_factors(filters)))

        # add the parameters to the schedule
        param = copy.copy(base_param)
        assert param["channels_in"] % coarse_in == 0, f"coarse in must be a factor of channels in for {exec_node} and {hw_node}"
        assert filters % coarse_out == 0, f"coarse out must be a factor of channels out (filters) for {exec_node} and {hw_node}"
        param["filters"] = filters
        param["channels_out"] = filters
        param["coarse_in"] = coarse_in
        param["coarse_out"] = coarse_out

        # append to the schedule
        schedule.append(param)

    # return the schedule
    return schedule, iteration_space

def get_pooling_schedule(self, hw_node, exec_node):

    # initialise the schedule
    schedule = []

    # get the parameters for the exec node
    base_param = self.net.graph.nodes[exec_node]["hw"].layer_info_dict()

    # do the same for the coarse factors TODO: improve the channel, coarse trade-off
    coarse = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse,
            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))

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
    iteration_space = [ row_repetition, channel_repetition ]
    if self.dimensionality == 3:
        iteration_space = [ row_repetition, depth_repetition, channel_repetition ]

    # iterate over the tiled dimensions
    for index in  np.ndindex(*iteration_space):

        # get the greatest spatial dimensions for each execution
        rows_in = min(self.building_blocks[hw_node]["hw"].rows_in(),
                base_param["rows_in"]-index[0]*self.building_blocks[hw_node]["hw"].rows_in())
        cols_in = min(self.building_blocks[hw_node]["hw"].cols_in(),
                base_param["cols_in"]-index[0]*self.building_blocks[hw_node]["hw"].cols_in())
        if self.dimensionality == 3:
            depth_in = min(self.building_blocks[hw_node]["hw"].depth_in(),
                    base_param["depth_in"]-index[1]*self.building_blocks[hw_node]["hw"].depth_in())
        channels_in = min(self.building_blocks[hw_node]["hw"].channels_in(),
                base_param["channels_in"]-index[-1]*self.building_blocks[hw_node]["hw"].channels_in())

        # add the required overlap for spatial dimensions
        if row_repetition > 1:
            rows_in += base_param["kernel_rows"] - 1
        if col_repetition > 1:
            cols_in += base_param["kernel_cols"] - 1
        if self.dimensionality == 3:
            if depth_repetition > 1:
                depth_in += base_param["kernel_depth"] - 1

        # add the parameters to the schedule
        param = copy.copy(base_param)
        param["rows_in"] = rows_in
        param["cols_in"] = cols_in
        if self.dimensionality == 3:
            param["depth_in"] = depth_in
        param["channels_in"] = channels_in
        param["coarse"] = coarse

        # only add padding parameters at the edge
        param["pad_top"] = param["pad_top"] if index[0] else 0
        param["pad_bottom"] = param["pad_bottom"] if index[0] == (row_repetition-1) else 0
        param["pad_left"] = param["pad_left"] if index[0] else 0
        param["pad_right"] = param["pad_right"] if index[0] == (col_repetition-1) else 0
        if self.dimensionality == 3:
            param["pad_front"] = param["pad_front"] if index[1] else 0
            param["pad_back"] = param["pad_back"] if index[1] == (depth_repetition-1) else 0

        # append to the schedule
        schedule.append(param)

    # return the schedule
    return schedule, iteration_space

def get_basic_schedule(self, hw_node, exec_node):

    # initialise the schedule
    schedule = []

    # get the parameters for the exec node
    base_param = self.net.graph.nodes[exec_node]["hw"].layer_info_dict()

    # do the same for the coarse factors TODO: improve the channel, coarse trade-off
    coarse = max(filter(lambda f: f <= self.building_blocks[hw_node]["hw"].coarse,
            self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible()))

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
    iteration_space = [ row_repetition, channel_repetition ]
    if self.dimensionality == 3:
        iteration_space = [ row_repetition, depth_repetition, channel_repetition ]

    # iterate over the tiled dimensions
    for index in  np.ndindex(*iteration_space):

         # get the greatest spatial dimensions for each execution
        rows_in = min(self.building_blocks[hw_node]["hw"].rows_in(),
                base_param["rows_in"]-index[0]*self.building_blocks[hw_node]["hw"].rows_in())
        cols_in = min(self.building_blocks[hw_node]["hw"].cols_in(),
                base_param["cols_in"]-index[0]*self.building_blocks[hw_node]["hw"].cols_in())
        if self.dimensionality == 3:
            depth_in = min(self.building_blocks[hw_node]["hw"].depth_in(),
                    base_param["cols_in"]-index[1]*self.building_blocks[hw_node]["hw"].depth_in())
        channels_in = min(self.building_blocks[hw_node]["hw"].channels_in(),
                base_param["channels_in"]-index[-1]*self.building_blocks[hw_node]["hw"].channels_in())

        # add the parameters to the schedule
        param = copy.copy(base_param)
        param["rows_in"] = rows_in
        param["cols_in"] = cols_in
        if self.dimensionality == 3:
            param["depth_in"] = depth_in
        param["channels_in"] = channels_in
        param["coarse"] = coarse
        if self.net.graph.nodes[exec_node]["type"].name in ["ReLU", "Sigmoid", "SiLU"]:
            param["op_type"] = self.net.graph.nodes[exec_node]["type"].name.lower()
        if self.net.graph.nodes[exec_node]["type"].name == "EltWise":
            param["op_type"] = self.net.graph.nodes[exec_node]["hw"].op_type
            param["broadcast"] = self.net.graph.nodes[exec_node]["hw"].broadcast

        # append to the schedule
        schedule.append(param)

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

