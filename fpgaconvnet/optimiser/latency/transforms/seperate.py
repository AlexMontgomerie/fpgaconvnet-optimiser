import copy
import random
import pickle
from fpgaconvnet.tools.layer_enum import LAYER_TYPE

def seperate(self, hw_node, num_nodes=1):
    """
    method to seperate out hardware nodes in `self.building_blocks`
    """
    # get all exec nodes
    exec_nodes = pickle.loads(pickle.dumps(self.building_blocks[hw_node]["exec_nodes"]))

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
        self.building_blocks[exec_node] = pickle.loads(pickle.dumps(self.net.graph.nodes[exec_node]))
        self.building_blocks[exec_node]["exec_nodes"] = [ exec_node ]

        # delete the original node if it has no exec nodes
        if self.building_blocks[hw_node]["exec_nodes"] == []:
            del self.building_blocks[hw_node]

    # return the split exec nodes
    return split_exec_nodes
