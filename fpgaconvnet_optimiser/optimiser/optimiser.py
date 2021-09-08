import numpy as np
import json
import copy
import random
import math

from fpgaconvnet_optimiser.models.network.Network import Network
import fpgaconvnet_optimiser.tools.graphs as graphs

from google.protobuf import json_format
import fpgaconvnet_optimiser.proto.fpgaconvnet_pb2 as fpgaconvnet_pb2
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE, from_proto_layer_type 

LATENCY   =0
THROUGHPUT=1
POWER     =2

class Optimiser(Network):
    """
    Base class for all optimisation strategies. This inherits the `Network` class. 
    """

    def __init__(self,name,network_path,transforms_config={},fix_starting_point_config={},data_width=16,weight_width=8,acc_width=30,fuse_bn=True):
        """
        Parameters
        ----------
        name: str
            name of network
        network_path: str
            path to network model .onnx file


        Attributes
        ----------
        objective: int
            Objective for the optimiser. One of `LATENCY`, `THROUGHPUT` and `POWER`.
        constraints: dict 
            dictionary containing constraints for `latency`, `throughput` and `power`.
        transforms: list
            list of transforms that can be applied to the network. Allowed transforms
            are `['coarse','fine','partition','weights_reloading']`
        """
        # Initialise Network
        Network.__init__(self,name,network_path,data_width=data_width,weight_width=weight_width,acc_width=acc_width,fuse_bn=fuse_bn)

        self.objective   = 0
        self.constraints = {
            'latency'    : float("inf"),
            'throughput' : 0.0,
            'power'      : float("inf")
        }

        self.transforms = ['coarse','fine','partition','weights_reloading']

        self.transforms_config = transforms_config
        if len(fix_starting_point_config) == 0:
            self.fix_starting_point_config = transforms_config
        else:
            self.fix_starting_point_config = fix_starting_point_config

    def get_transforms(self):
        self.transforms = []
        for transform_type, attr in self.transforms_config.items():
            if bool(attr["apply_transform"]):
                self.transforms.append(transform_type)

    def get_cost(self, partition_list=None):
        """
        calculates the cost function of the optimisation strategy at it's current state. 
        This cost is based on the objective of the optimiser. There are three objectives
        that can be chosen: 
        
        - `LATENCY (0)`
        - `THROUGHPUT (1)`
        - `POWER (2)`

        Returns: 
        --------
        float
        """
        if partition_list == None:
            partition_list = list(range(len(self.partitions)))
        # Latency objective
        if   self.objective == LATENCY:
            return self.get_latency(partition_list)
        # Throughput objective
        elif self.objective == THROUGHPUT:
            return -self.get_throughput(partition_list)
        # Power objective
        elif self.objective == POWER:
            return self.get_power_average(partition_list)    
    
    def check_constraints(self):
        """
        function to check the performance constraints of the network. Checks
        `latency` and `throughput`.

        Raises
        ------
        AssertionError
            If not within performance constraints
        """
        assert self.get_latency()       <= self.constraints['latency']  , "ERROR : (constraint violation) Latency constraint exceeded"
        assert self.get_throughput()    >= self.constraints['throughput'], "ERROR : (constraint violation) Throughput constraint exceeded"

    def apply_transform(self, transform, partition_index=None, node=None):
        """
        function to apply chosen transform to the network. Partition index
        and node can be specified. If not, a random partition and node is 
        chosen.

        Parameters
        ----------
        transform: str
            string corresponding to chosen transform. Must be within the
            following `["coarse","fine","weights_reloading","partition"]`
        partition_index: int default: None
            index for partition to apply transform. Must be within
            `len(self.partitions)`
        node: str
            name of node to apply transform. Must be within
            `self.partitions[partition_index].graph`
        """

        # choose random partition index if not given
        if partition_index == None:
            partition_index = random.randint(0,len(self.partitions)-1)

        # choose random node in partition if given
        if node == None:
            node = random.choice(graphs.ordered_node_list(self.partitions[partition_index].graph))

        # Apply a random transform
        ## Coarse transform (node_info transform)
        if transform == 'coarse':
            self.partitions[partition_index].apply_random_coarse_layer(node)
            return

        ## Fine transform (node_info transform)
        if transform == 'fine':
            self.partitions[partition_index].apply_random_fine_layer(node)
            return

        ## Weights-Reloading transform (partition transform)
        if transform == 'weights_reloading':
            ### apply random weights reloading
            self.partitions[partition_index].apply_random_weights_reloading()
            return

        ## Partition transform (partition transform)
        if transform == 'partition':
            ### apply random partition
            # remove squeeze layers prior to partitioning
            self.partitions[partition_index].remove_squeeze()
            self.apply_random_partition(partition_index)
            return

    def optimiser_status(self):
        """
        prints out the current status of the optimiser.
        """
        # objective
        objectives = [ 'latency', 'throughput','power']
        objective  = objectives[self.objective]
        # cost 
        cost = self.get_cost()
        # Resources
        resources = [ self.get_resource_usage(i) for i in range(len(self.partitions)) ]
        BRAM = max([ resource['BRAM'] for resource in resources ])
        DSP  = max([ resource['DSP']  for resource in resources ])
        LUT  = max([ resource['LUT']  for resource in resources ])
        FF   = max([ resource['FF']   for resource in resources ])
        print("COST:\t {cost} ({objective}), RESOURCE:\t {BRAM}\t{DSP}\t{LUT}\t{FF}\t(BRAM|DSP|LUT|FF)".format(
            cost=cost,objective=objective,BRAM=int(BRAM),DSP=int(DSP),LUT=int(LUT),FF=int(FF)), end='\r')

    def get_optimal_batch_size(self):
        """
        gets an approximation of the optimal (largest) batch size for throughput-based 
        applications. This is dependant on the platform's memory capacity. Updates the
        `self.batch_size` variable. 
        """
 
        # change the batch size to zero
        self.batch_size = 1
        # update each partitions batch size
        self.update_batch_size()
        # calculate the maximum memory usage at batch 1
        max_mem = self.get_memory_usage_estimate()
        # update batch size to max
        self.batch_size = max(1,math.floor(self.platform['mem_capacity']/max_mem))
        self.update_batch_size()

    def run_optimiser(self, log=True):
        """
        template for running the optimiser.
        """
        pass


    def starting_point_distillation(self, teacher_partition_path, load_wr):
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

###############################
        assert len(self.partitions) >= len(teacher_partitions.partition)

        for i in range(len(self.partitions)):
            self.partitions[i].remove_squeeze()

        for partition_index, teacher_partition in enumerate(teacher_partitions.partition):
            num_of_conv = 0
            for layer in teacher_partition.layers:
                if from_proto_layer_type(layer.type) == LAYER_TYPE.Convolution:
                    num_of_conv += 1

            for _ in range(0, num_of_conv-1):
                horizontal_merges = self.get_all_horizontal_merges(partition_index)
                self.merge_horizontal(*horizontal_merges[0])

        self.update_partitions()   
###############################
            
        assert len(self.partitions) == len(teacher_partitions.partition)

        for partition_index, teacher_partition in enumerate(teacher_partitions.partition):
            student_partition = self.partitions[partition_index]
            student_partition.remove_weights_reloading_transform()
        
        self.update_partitions()

###############################

        for partition_index, teacher_partition in enumerate(teacher_partitions.partition):
            student_partition = self.partitions[partition_index]
            student_node_index = 0
            for layer_index,layer in enumerate(teacher_partition.layers):

                if from_proto_layer_type(layer.type) == LAYER_TYPE.Squeeze:
                    continue

                node = graphs.ordered_node_list(student_partition.graph)[student_node_index]
                student_node_index += 1
                #node = layer.name
                teacher_parameters = json_format.MessageToDict(layer.parameters, preserving_proto_field_name=True)

                if student_partition.graph.nodes[node]["type"] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
                    padded_channels = student_partition.graph.nodes[node]['hw'].filters
                    factors = []
                    
                    if "groups" in teacher_parameters.keys():
                        groups = teacher_parameters["groups"]
                    else:
                        groups = 1
                    if "coarse_out" in teacher_parameters.keys():
                        coarse_out = teacher_parameters["coarse_out"]
                        factors.append(coarse_out*groups)
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
                if load_wr:
                    wr_factor = teacher_partition.weights_reloading_factor
                    assert wr_factor in student_partition.graph.nodes[wr_layer]['hw'].get_weights_reloading_feasible(), "padding required"  
                else:
                    wr_factor = max(student_partition.graph.nodes[wr_layer]['hw'].get_weights_reloading_feasible())
                                  
                student_partition.wr_layer = wr_layer
                student_partition.wr_factor = wr_factor               
                student_partition.apply_weights_reloading_transform()
###############################

    def merge_memory_bound_partitions(self):

        print("resolving memory bound partitions")
        # todo: start from the end, while loop
        for _ in range(50):
            partitions = copy.deepcopy(self.partitions)

            self.update_partitions()
            input_memory_bound = []
            output_memory_bound = []

            for partition_index, partition in enumerate(self.partitions):
                for i in range(len(self.partitions)):
                    self.partitions[i].remove_squeeze()
                horizontal_merges = self.get_all_horizontal_merges(partition_index)
                self.update_partitions()

                if partition.is_input_memory_bound() and horizontal_merges[1] and self.partitions[horizontal_merges[1][0]].wr_factor == 1 \
                    or horizontal_merges[1] and partition.get_latency(self.platform["freq"]) < self.platform["reconf_time"]:
                    input_memory_bound.append(partition_index)

                if partition.is_output_memory_bound() and horizontal_merges[0] and self.partitions[horizontal_merges[0][0]].wr_factor == 1 \
                    or horizontal_merges[0] and partition.get_latency(self.platform["freq"]) < self.platform["reconf_time"]: 
                    output_memory_bound.append(partition_index)
  
            memory_bound = input_memory_bound + output_memory_bound
            if len(memory_bound) == 0:
                self.partitions = partitions
                break

            # remove all auxiliary layers
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