import numpy as np
import json
import copy
import random
import math
import logging
from google.protobuf import json_format

import fpgaconvnet_optimiser.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
import fpgaconvnet_optimiser.tools.graphs as graphs
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE, from_proto_layer_type

def starting_point_distillation(self, teacher_partition_path):
    print("load starting point from:", teacher_partition_path)
    teacher_partitions = fpgaconvnet_pb2.partitions()
    with open(teacher_partition_path,'r') as f:
        json_format.Parse(f.read(), teacher_partitions)

    def _lcm(a,b):
        return int((int(a)*int(b))/math.gcd(int(a),int(b)))

    def _iterate_current_teacher_partition_until_conv(partition, current_layer_index, factors):
        for layer_index,layer in enumerate(partition.layers):
            if layer_index <= current_layer_index:
                continue

            if from_proto_layer_type(layer.type) in [LAYER_TYPE.Convolution, LAYER_TYPE.Pooling, LAYER_TYPE.ReLU, LAYER_TYPE.InnerProduct, LAYER_TYPE.BatchNorm]:
                teacher_parameters = json_format.MessageToDict(layer.parameters, preserving_proto_field_name=True)
                if "groups" in teacher_parameters.keys():
                    groups = teacher_parameters["groups"]
                else:
                    groups = 1
                if "coarse_in" in teacher_parameters.keys():
                    coarse_in = teacher_parameters["coarse_in"]
                    factors.append(coarse_in*groups)
                if "coarse_group" in teacher_parameters.keys():
                    coarse_group = teacher_parameters["coarse_group"]
                    factors.append(coarse_group)
                if from_proto_layer_type(layer.type) in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
                    return True

        return False

    def _iterate_next_teacher_partition_until_conv(partition, factors):
        for layer_index,layer in enumerate(partition.layers):
            if from_proto_layer_type(layer.type) in [LAYER_TYPE.Convolution, LAYER_TYPE.Pooling, LAYER_TYPE.ReLU, LAYER_TYPE.InnerProduct, LAYER_TYPE.BatchNorm]:
                teacher_parameters = json_format.MessageToDict(layer.parameters, preserving_proto_field_name=True)
                if "groups" in teacher_parameters.keys():
                    groups = teacher_parameters["groups"]
                else:
                    groups = 1
                if "coarse_in" in teacher_parameters.keys():
                    coarse_in = teacher_parameters["coarse_in"]
                    factors.append(coarse_in*groups)
                if "coarse_group" in teacher_parameters.keys():
                    coarse_group = teacher_parameters["coarse_group"]
                    factors.append(coarse_group)
                if from_proto_layer_type(layer.type) in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
                    return True

        return False


    def _iterate_current_student_partition_until_conv(partition, input_node, padded_channels):
        while partition.graph.out_degree(input_node) != 0:
            input_node = graphs.get_next_nodes(partition.graph,input_node)[0]
            print("padding channels of ", input_node, partition.graph.nodes[input_node]["hw"].channels, "-->")
            print(padded_channels)
            if partition.graph.nodes[input_node]["hw"].channels == partition.graph.nodes[input_node]["hw"].groups:
                print("padding groups of ", input_node, partition.graph.nodes[input_node]["hw"].groups, "-->")
                print(padded_channels)
                partition.graph.nodes[input_node]["hw"].groups = padded_channels
            partition.graph.nodes[input_node]["hw"].channels = padded_channels
            if partition.graph.nodes[input_node]["type"] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
                return True
        return False

    def _iterate_next_student_partition_until_conv(partition, input_node, padded_channels):
        print("padding channels of ", input_node, partition.graph.nodes[input_node]["hw"].channels, "-->")
        print(padded_channels)
        if partition.graph.nodes[input_node]["hw"].channels == partition.graph.nodes[input_node]["hw"].groups:
            print("padding groups of ", input_node, partition.graph.nodes[input_node]["hw"].groups, "-->")
            print(padded_channels)
            partition.graph.nodes[input_node]["hw"].groups = padded_channels
        partition.graph.nodes[input_node]["hw"].channels = padded_channels
        if partition.graph.nodes[input_node]["type"] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
            return True

        while partition.graph.out_degree(input_node) != 0:
            input_node = graphs.get_next_nodes(partition.graph,input_node)[0]
            print("padding channels of ", input_node, partition.graph.nodes[input_node]["hw"].channels, "-->")
            print(padded_channels)
            if partition.graph.nodes[input_node]["hw"].channels == partition.graph.nodes[input_node]["hw"].groups:
                partition.graph.nodes[input_node]["hw"].groups = padded_channels
            partition.graph.nodes[input_node]["hw"].channels = padded_channels
            if partition.graph.nodes[input_node]["type"] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
                return True
        return False

    for partition_index, teacher_partition in enumerate(teacher_partitions.partition):
        student_partition = self.partitions[partition_index]
        student_partition.remove_weights_reloading_transform()

    self.update_partitions()

    for partition_index, teacher_partition in enumerate(teacher_partitions.partition):
        student_partition = self.partitions[partition_index]
        student_node_index = 0
        for layer_index,layer in enumerate(teacher_partition.layers):

            if from_proto_layer_type(layer.type) in [LAYER_TYPE.Convolution, LAYER_TYPE.Pooling, LAYER_TYPE.ReLU, LAYER_TYPE.InnerProduct, LAYER_TYPE.BatchNorm]:
                node = graphs.ordered_node_list(student_partition.graph)[student_node_index]
                student_node_index += 1
                #node = layer.name
                teacher_parameters = json_format.MessageToDict(layer.parameters, preserving_proto_field_name=True)

                if student_partition.graph.nodes[node]["type"] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
                    padded_channels = student_partition.graph.nodes[node]['hw'].filters
                    factors = []

                    if node == student_partition.get_wr_layer():
                    #if node == teacher_partition.weights_reloading_layer:
                        wr_factor = teacher_partition.weights_reloading_factor
                        factors.append(wr_factor)
                    else:
                        wr_factor = 1
                    if "groups" in teacher_parameters.keys():
                        groups = teacher_parameters["groups"]
                    else:
                        groups = 1
                    if "coarse_out" in teacher_parameters.keys():
                        coarse_out = teacher_parameters["coarse_out"]
                        factors.append(coarse_out*groups*wr_factor)
                    if "coarse_group" in teacher_parameters.keys():
                        coarse_group = teacher_parameters["coarse_group"]
                        factors.append(coarse_group)

                    reach_conv = _iterate_current_teacher_partition_until_conv(teacher_partition, layer_index, factors)
                    if not reach_conv:
                        for next_parition_index in range(partition_index+1, len(self.partitions)):
                            reach_conv = _iterate_next_teacher_partition_until_conv(teacher_partitions.partition[next_parition_index], factors)
                            if reach_conv:
                                break

                    lcm = 1
                    for a in factors:
                        lcm = _lcm(lcm, a)
                    padded_channels = math.ceil(padded_channels/lcm)*lcm
                    student_partition.graph.nodes[node]['hw'].lcm = lcm

                    if padded_channels != student_partition.graph.nodes[node]['hw'].filters:
                        print("padding filters of ", node, student_partition.graph.nodes[node]['hw'].filters, "-->")
                        print(padded_channels)

                        if student_partition.graph.nodes[node]["hw"].filters == student_partition.graph.nodes[node]["hw"].groups:
                            print("padding groups of ", node, student_partition.graph.nodes[node]["hw"].groups, "-->")
                            print(padded_channels)
                            student_partition.graph.nodes[node]["hw"].groups = padded_channels
                        student_partition.graph.nodes[node]['hw'].filters = padded_channels
                        if not _iterate_current_student_partition_until_conv(student_partition, node, padded_channels):
                            for next_parition_index in range(partition_index+1, len(self.partitions)):
                                if _iterate_next_student_partition_until_conv(self.partitions[next_parition_index], self.partitions[next_parition_index].input_nodes[0], padded_channels):
                                    break

                if "coarse_in" in teacher_parameters.keys():
                    coarse_in = teacher_parameters["coarse_in"]
                    assert coarse_in in student_partition.graph.nodes[node]["hw"].get_coarse_in_feasible(), "padding required"
                    student_partition.graph.nodes[node]["hw"].update_coarse_in(coarse_in)
                if "coarse_out" in teacher_parameters.keys():
                    coarse_out = teacher_parameters["coarse_out"]
                    assert coarse_out in student_partition.graph.nodes[node]["hw"].get_coarse_out_feasible(), "padding required"
                    student_partition.graph.nodes[node]["hw"].update_coarse_out(coarse_out)
                if "coarse_group" in teacher_parameters.keys():
                    coarse_group = teacher_parameters["coarse_group"]
                    assert coarse_group in student_partition.graph.nodes[node]["hw"].get_coarse_group_feasible(), "padding required"
                    student_partition.graph.nodes[node]["hw"].update_coarse_group(coarse_group)
                if "fine" in teacher_parameters.keys():
                    fine = teacher_parameters["fine"]
                    assert fine in student_partition.graph.nodes[node]["hw"].get_fine_feasible(), "padding required"
                    student_partition.graph.nodes[node]["hw"].fine = fine


        if teacher_partition.weights_reloading_layer != "None":
            wr_layer = student_partition.get_wr_layer()
            #wr_layer = teacher_partition.weights_reloading_layer
            wr_factor = teacher_partition.weights_reloading_factor
            assert wr_factor in student_partition.graph.nodes[wr_layer]['hw'].get_weights_reloading_feasible(), "padding required"
            student_partition.wr_layer = wr_layer
            student_partition.wr_factor = wr_factor
            student_partition.apply_weights_reloading_transform()

def merge_memory_bound_partitions(self):
    print("resolving memory bound partitions")
    for _ in range(50):
        partitions = copy.deepcopy(self.partitions)

        self.update_partitions()
        input_memory_bound = []
        output_memory_bound = []

        for partition_index, partition in enumerate(self.partitions):
            if partition.is_input_memory_bound():
                input_memory_bound.append(partition_index)
            if partition.is_output_memory_bound():
                output_memory_bound.append(partition_index)
        memory_bound = input_memory_bound + output_memory_bound

        if len(memory_bound) == 0:
            self.partitions = partitions
            break

        for i in range(len(self.partitions)):
            self.partitions[i].remove_squeeze()

        ## Choose slowest partition
        partition_latencys = [ self.partitions[partition_index].get_latency(self.platform["freq"]) for partition_index in memory_bound]
        partition_index    = np.random.choice(memory_bound, 1, p=(partition_latencys/sum(partition_latencys)))[0]

        horizontal_merges = self.get_all_horizontal_merges(partition_index)

        if horizontal_merges[0] and partition_index in output_memory_bound:
            self.partitions[horizontal_merges[0][0]].reset()
            self.partitions[horizontal_merges[0][1]].reset()
            self.merge_horizontal(*horizontal_merges[0])
        elif horizontal_merges[1] and partition_index in input_memory_bound:
            self.partitions[horizontal_merges[1][0]].reset()
            self.partitions[horizontal_merges[1][1]].reset()
            self.merge_horizontal(*horizontal_merges[1])


        self.update_partitions()

        try:
            self.check_resources()
            self.check_constraints()
        except AssertionError:
            # revert to previous state
            self.partitions = partitions
            #continue
