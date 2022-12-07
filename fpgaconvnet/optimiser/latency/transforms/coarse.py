import random
import math

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def apply_random_coarse_node(self, hw_node):

    # list of possible coarse types to apply
    if self.building_blocks[hw_node]["type"] == LAYER_TYPE.Convolution:
        coarse_types = [ "coarse_in", "coarse_out", "coarse_group" ]
    elif self.building_blocks[hw_node]["type"] == LAYER_TYPE.InnerProduct:
        coarse_types = [ "coarse_in", "coarse_out" ]
    else:
        coarse_types = [ "coarse" ]

    # choose a random coarse type
    coarse_type = random.choice(coarse_types)

    # get all the factors for that coarse type
    coarse_factors = []
    for exec_node in self.building_blocks[hw_node]["exec_nodes"]:
        if coarse_type == "coarse":
            coarse_factors.extend(self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible())
        if coarse_type == "coarse_in":
            coarse_factors.extend(self.net.graph.nodes[exec_node]["hw"].get_coarse_in_feasible())
        if coarse_type == "coarse_out":
            coarse_factors.extend(self.net.graph.nodes[exec_node]["hw"].get_coarse_out_feasible())
        if coarse_type == "coarse_group":
            coarse_factors.extend(self.net.graph.nodes[exec_node]["hw"].get_coarse_group_feasible())

    # get the hardware coarse factors
    if coarse_type == "coarse":
        hw_coarse_factor = self.building_blocks[hw_node]["hw"].get_coarse_in_feasible()
    if coarse_type == "coarse_in":
        hw_coarse_factor = self.building_blocks[hw_node]["hw"].get_coarse_in_feasible()
    if coarse_type == "coarse_out":
        hw_coarse_factor = self.building_blocks[hw_node]["hw"].get_coarse_out_feasible()
    if coarse_type == "coarse_group":
        hw_coarse_factor = self.building_blocks[hw_node]["hw"].get_coarse_group_feasible()

    # choose a random coarse factor
    coarse_factor = random.choice(list(set(coarse_factors).intersection(set(hw_coarse_factor))))

    # apply the coarse factor
    if coarse_type == "coarse":
        self.building_blocks[hw_node]["hw"].coarse = coarse_factor
    if coarse_type == "coarse_in":
        self.building_blocks[hw_node]["hw"].coarse_in = coarse_factor
    if coarse_type == "coarse_out":
        self.building_blocks[hw_node]["hw"].coarse_out = coarse_factor
    if coarse_type == "coarse_group":
        self.building_blocks[hw_node]["hw"].coarse_group = coarse_factor

    # update the hardware
    self.building_blocks[hw_node]["hw"].update()

def fix_coarse_node(self, hw_node):

    def get_max_coarse(coarse, factors):
        return list(filter(lambda f: f <= coarse, factors))[-1]

    match self.building_blocks[hw_node]["type"]:
        case LAYER_TYPE.Convolution:
            self.building_blocks[hw_node]["hw"].coarse_in = get_max_coarse(
                self.building_blocks[hw_node]["hw"].coarse_in,
                self.building_blocks[hw_node]["hw"].get_coarse_in_feasible())
            self.building_blocks[hw_node]["hw"].coarse_out = get_max_coarse(
                self.building_blocks[hw_node]["hw"].coarse_out,
                self.building_blocks[hw_node]["hw"].get_coarse_out_feasible())
            self.building_blocks[hw_node]["hw"].coarse_group = get_max_coarse(
                self.building_blocks[hw_node]["hw"].coarse_group,
                self.building_blocks[hw_node]["hw"].get_coarse_group_feasible())
        case LAYER_TYPE.InnerProduct:
            self.building_blocks[hw_node]["hw"].coarse_in = get_max_coarse(
                self.building_blocks[hw_node]["hw"].coarse_in,
                self.building_blocks[hw_node]["hw"].get_coarse_in_feasible())
            self.building_blocks[hw_node]["hw"].coarse_out = get_max_coarse(
                self.building_blocks[hw_node]["hw"].coarse_out,
                self.building_blocks[hw_node]["hw"].get_coarse_out_feasible())
        case _:
            self.building_blocks[hw_node]["hw"].coarse = get_max_coarse(
                self.building_blocks[hw_node]["hw"].coarse,
                self.building_blocks[hw_node]["hw"].get_coarse_in_feasible())

    # update the hardware
    self.building_blocks[hw_node]["hw"].update()
