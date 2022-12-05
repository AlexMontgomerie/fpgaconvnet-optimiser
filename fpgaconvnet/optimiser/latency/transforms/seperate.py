import copy

def seperate(self, hw_node):
    """
    method to seperate out hardware nodes in `self.building_blocks`
    """
    # iterate over exec_nodes
    for exec_node in self.building_blocks[hw_node]["exec_nodes"]:
        # add hardware of exec_node to the latency nodes
        self.building_blocks[exec_node] = copy.deepcopy(self.net.graph.nodes[exec_node])
        self.building_blocks[exec_node]["exec_nodes"] = [ exec_node ]
        # keep performance parameters the same (coarse, fine, ...)
        # TODO

    # delete the original node from the latency nodes
    del self.building_blocks[hw_node]


