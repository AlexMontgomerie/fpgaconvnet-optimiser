import copy
import random

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def seperate(self, hw_node, num_nodes=1):
    """
    method to seperate out hardware nodes in `self.building_blocks`
    """
    # get all exec nodes
    exec_nodes = self.building_blocks[hw_node]["exec_nodes"]

    if len(exec_nodes) > num_nodes:
        # sample num nodes from the exec nodes
        split_exec_nodes = random.sample(exec_nodes, num_nodes)
    else:
        # seperate all nodes
        split_exec_nodes = exec_nodes

    # iterate over exec_nodes
    for exec_node in split_exec_nodes:
        # remove exec_node from the building block exec nodes
        self.building_blocks[hw_node]["exec_nodes"].remove(exec_node)
        # add hardware of exec_node to the latency nodes
        self.building_blocks[exec_node] = copy.deepcopy(self.net.graph.nodes[exec_node])
        self.building_blocks[exec_node]["exec_nodes"] = [ exec_node ]
        # # copy over performance parameters
        # match self.building_blocks[hw_node]["type"]:
        #     case LAYER_TYPE.Convolution:
        #         self.building_blocks[exec_node]["hw"].fine = \
        #             self.building_blocks[hw_node]["hw"].fine
        #         self.building_blocks[exec_node]["hw"].coarse_in = \
        #             self.building_blocks[hw_node]["hw"].coarse_in
        #         self.building_blocks[exec_node]["hw"].coarse_out = \
        #             self.building_blocks[hw_node]["hw"].coarse_out
        #         self.building_blocks[exec_node]["hw"].coarse_group = \
        #             self.building_blocks[hw_node]["hw"].coarse_group
        #     case LAYER_TYPE.InnerProduct:
        #         self.building_blocks[exec_node]["hw"].coarse_in = \
        #             self.building_blocks[hw_node]["hw"].coarse_in
        #         self.building_blocks[exec_node]["hw"].coarse_out = \
        #             self.building_blocks[hw_node]["hw"].coarse_out
        #     case _:
        #         self.building_blocks[exec_node]["hw"].coarse = \
        #             self.building_blocks[hw_node]["hw"].coarse

        # delete the original node if it has no exec nodes
        if self.building_blocks[hw_node]["exec_nodes"] == []:
            del self.building_blocks[hw_node]


