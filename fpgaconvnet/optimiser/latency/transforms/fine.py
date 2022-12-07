import random
import math

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def apply_random_fine_node(self, hw_node):

    # feasible nodes
    feasible_nodes = self.get_layers_of_type(LAYER_TYPE.Convolution)

    # check node can have fine transform applied
    if hw_node in feasible_nodes:

        # choose random fine
        # fine = random.choice(self.building_blocks[hw_node]['hw'].get_fine_feasible())
        fine = random.randint(1, math.prod(
            self.building_blocks[hw_node]['hw'].kernel_size))

        # update modules fine grain folding factor
        self.building_blocks[hw_node]['hw'].fine = fine
        self.building_blocks[hw_node]['hw'].update()

