import itertools
import secrets
import random

from fpgaconvnet.tools.layer_enum import LAYER_TYPE

from fpgaconvnet.optimiser.latency.solvers.utils import get_hw_from_dict

def combine(self, layer_type, discriminate=[], num_nodes=2):

    # get the layers of the given type
    nodes_of_type = self.get_layers_of_type(layer_type)

    # further discriminate the layers to combine TODO
    nodes_to_combine = nodes_of_type

    # select a subset of the nodes to combine
    if len(nodes_to_combine) > num_nodes:
        nodes_to_combine = random.sample(nodes_to_combine, num_nodes)
    # combine all
    elif len(nodes_to_combine) > 1:
        nodes_to_combine = nodes_to_combine
    # escape if there are no layers to combine
    else:
        return

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
            max_param_keys = [ "rows", "cols", "filters", "channels", "groups", "kernel_rows",
                    "kernel_cols", "stride_rows", "stride_cols", "pad_top",
                    "pad_bottom", "pad_left", "pad_right" ]
            min_param_keys = [ "fine", "coarse_in", "coarse_out", "coarse_group" ]

            # add 3D specific parameters
            if self.dimensionality == 3:
                max_param_keys.extend(["kernel_depth", "depth",
                    "stride_depth", "pad_front", "pad_back"])

            # get the parameters
            parameters = { key: self.get_max_attr_of_hw_nodes(
                nodes_to_combine, key) for key in max_param_keys }
            parameters.update({ key: self.get_min_attr_of_hw_nodes(
                nodes_to_combine, key) for key in min_param_keys })

        case LAYER_TYPE.InnerProduct:

            # get all the parameter keys
            max_param_keys = [ "rows", "cols", "filters", "channels" ]
            min_param_keys = [ "coarse_in", "coarse_out" ]

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
    if len(nodes_to_combine) > 1:
        for layer in nodes_to_combine:
            del self.building_blocks[layer]

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


