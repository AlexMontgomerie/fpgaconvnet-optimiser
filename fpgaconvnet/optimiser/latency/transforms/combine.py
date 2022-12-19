import itertools
import secrets
import random

from fpgaconvnet.tools.layer_enum import LAYER_TYPE, from_onnx_op_type

from fpgaconvnet.optimiser.latency.solvers.utils import get_hw_from_dict, apply_mem_bw_limitations

def combine(self, layer_type, discriminate=[], num_nodes=2):

    # get the layers of the given type
    nodes_of_type = self.get_hw_nodes_of_type(layer_type)

    if len(nodes_of_type) < 1:
        return None # nothing to combine

    # split the nodes into different groups
    discrimination_groups = []
    for d in discriminate:
        if not d:
            continue
        # find all nodes of this group
        group = []
        for hw_node in nodes_of_type:
            # flag to indicate if hw_node will be added to group
            add_to_group = True
            # iterate over discrimnation parameters
            if layer_type != from_onnx_op_type(d["layer_type"]):
                add_to_group = False
                continue
            for param, val in d.items():
                # check it has the correct layer type
                if param != "layer_type":
                    # get the hw_node parameters and see if they match
                    key_node_param = getattr(self.building_blocks[hw_node]["hw"], param)
                    if key_node_param != val:
                        add_to_group = False
            # add the node to the new group
            if add_to_group:
                group.append(hw_node)
        # add the new group and remove from nodes of type
        if group != []:
            discrimination_groups.append(group)
            nodes_of_type = list(filter(lambda v: v not in group, nodes_of_type))

    # create a group for the left over nodes
    discrimination_groups.append(nodes_of_type)

    # choose a random discrimnation group to combine
    nodes_to_combine = random.choice(discrimination_groups)

    # select a subset of the nodes to combine
    if num_nodes > 0:
        if len(nodes_to_combine) > num_nodes:
            nodes_to_combine = random.sample(nodes_to_combine, num_nodes)
        # combine all
        else:
            nodes_to_combine = nodes_to_combine

    # create a new layer name by combining
    new_layer_name = f"{layer_type.name}_{secrets.token_hex(2)}"
    # TODO: need to ensure the name is unique, in case we create
    # several building blocks of the same type

    # parameters to create new hardware node
    parameters = None

    # get the superset layer for the given layer type:
    match layer_type:
        case LAYER_TYPE.Convolution:

            # get all the parameter keys
            max_param_keys = [ "channels", "kernel_rows", "fine",
                    "kernel_cols", "stride_rows", "stride_cols", "pad_top",
                    "pad_bottom", "pad_left", "pad_right" ]
            min_param_keys = [ "rows", "cols", "filters", "groups",
                    "coarse_in", "coarse_out", "coarse_group" ]
            # Hack to deal with the depthwise convolution case
            if self.building_blocks[random.choice(nodes_to_combine)]["hw"].depthwise:
                max_param_keys.remove("channels")
                min_param_keys.append("channels")

            # add 3D specific parameters
            if self.dimensionality == 3:
                max_param_keys.extend(["kernel_depth",
                    "stride_depth", "pad_front", "pad_back"])
                min_param_keys.extend([ "depth" ])

            # get the parameters
            parameters = { key: self.get_max_attr_of_hw_nodes(
                nodes_to_combine, key) for key in max_param_keys }
            parameters.update({ key: self.get_min_attr_of_hw_nodes(
                nodes_to_combine, key) for key in min_param_keys })

        case LAYER_TYPE.InnerProduct:

            # get all the parameter keys
            max_param_keys = [ "rows", "cols", "channels" ]
            min_param_keys = [ "filters", "coarse_in", "coarse_out" ]

            # add 3D specific parameters
            if self.dimensionality == 3:
                max_param_keys.append("depth")

            # get the parameters
            #TODO: There is an issue here with DEPTHWISE_CONVOLUTION.
            # Shoud specifically handle this case and separate out the
            # depthwise convolution
            parameters = { key: self.get_max_attr_of_hw_nodes(
                nodes_to_combine, key) for key in max_param_keys }
            parameters.update({ key: self.get_min_attr_of_hw_nodes(
                nodes_to_combine, key) for key in min_param_keys })

        case LAYER_TYPE.Pooling:

            # get all the parameter keys
            max_param_keys = [ "kernel_rows", "kernel_cols",
                    "stride_rows", "stride_cols", "pad_top", "pad_bottom",
                    "pad_left", "pad_right" ]
            min_param_keys = [ "rows", "cols", "channels", "coarse" ]

            # add 3D specific parameters
            if self.dimensionality == 3:
                max_param_keys.extend(["kernel_depth",
                    "stride_depth", "pad_front", "pad_back"])
                min_param_keys.append("depth")

            # get the parameters
            parameters = { key: self.get_max_attr_of_hw_nodes(
                nodes_to_combine, key) for key in max_param_keys }
            parameters.update({ key: self.get_min_attr_of_hw_nodes(
                nodes_to_combine, key) for key in min_param_keys })

        case LAYER_TYPE.ReLU | LAYER_TYPE.Sigmoid | LAYER_TYPE.SiLU:

            max_param_keys = [ "rows", "cols", "channels", "coarse" ]

            # add 3D specific parameters
            if self.dimensionality == 3:
                max_param_keys.append("depth")

            # get the parameters
            parameters = { key: self.get_max_attr_of_hw_nodes(
                nodes_to_combine, key) for key in max_param_keys }

            parameters["op_type"] = layer_type.name.lower()

        case LAYER_TYPE.EltWise:

            min_param_keys = [ "rows", "cols", "channels" ]
            max_param_keys = [ "ports_in" ]

            # add 3D specific parameters
            if self.dimensionality == 3:
                min_param_keys.append("depth")

            # get the parameters
            parameters = { key: self.get_min_attr_of_hw_nodes_multi(
                nodes_to_combine, key) for key in min_param_keys }
            parameters.update({ key: self.get_min_attr_of_hw_nodes(
                nodes_to_combine, key) for key in ['coarse'] })
            parameters.update({ key: self.get_max_attr_of_hw_nodes(
                nodes_to_combine, key) for key in max_param_keys })

            # TODO: decide on op type and broadcast
            parameters["op_type"] = "mul"
            parameters["broadcast"] = True

        case LAYER_TYPE.GlobalPooling:

            max_param_keys = [ "rows", "cols", "channels", "coarse" ]

            # add 3D specific parameters
            if self.dimensionality == 3:
                max_param_keys.append("depth")

            # get the parameters
            parameters = { key: self.get_max_attr_of_hw_nodes(
                nodes_to_combine, key) for key in max_param_keys }

        case _:
            raise NotImplementedError(layer_type)

    # get all the execution nodes from the layers to combine
    exec_nodes = list(itertools.chain(*[
            self.building_blocks[hw_node]["exec_nodes"] \
                    for hw_node in nodes_to_combine ]))

    # create a new layer from these parameters
    self.building_blocks[new_layer_name] = {
        "type": layer_type,
        "hw": get_hw_from_dict(layer_type,
            parameters, self.dimensionality),
        "exec_nodes": exec_nodes,
    }

    # remove the combined layers
    for layer in nodes_to_combine:
        del self.building_blocks[layer]

    # apply memory bandwidth limitations
    apply_mem_bw_limitations(self.net.graph, self.building_blocks,
            self.net.platform.mem_bw_wpc, channel_tiling=self.channel_tiling)

    # return the key for the new layer generated
    return new_layer_name

def get_max_attr_of_hw_nodes(self, hw_nodes, attr):
    return max([ getattr(self.building_blocks[hw_node]["hw"], attr) \
            for hw_node in hw_nodes ])

def get_min_attr_of_hw_nodes(self, hw_nodes, attr):
    return min([ getattr(self.building_blocks[hw_node]["hw"], attr) \
            for hw_node in hw_nodes ])

def get_max_attr_of_hw_nodes_multi(self, hw_nodes, attr):
    return max([ getattr(self.building_blocks[hw_node]["hw"], attr)[0] \
            for hw_node in hw_nodes ])

def get_min_attr_of_hw_nodes_multi(self, hw_nodes, attr):
    return min([ getattr(self.building_blocks[hw_node]["hw"], attr)[0] \
            for hw_node in hw_nodes ])


