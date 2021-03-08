import numpy as np
import json
import copy
import random
import math

from fpgaconvnet_optimiser.models.network.Network import Network
import fpgaconvnet_optimiser.tools.graphs as graphs

LATENCY   =0
THROUGHPUT=1
POWER     =2

class Optimiser(Network):
    """
    Base class for all optimisation strategies. This inherits the `Network` class. 
    """

    def __init__(self,name,network_path):
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
        Network.__init__(self,name,network_path)

        self.objective   = 0
        self.constraints = {
            'latency'    : float("inf"),
            'throughput' : 0.0,
            'power'      : float("inf")
        }

        self.transforms = ['coarse','fine','partition','weights_reloading']

    def get_cost(self):
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
        # Latency objective
        if   self.objective == LATENCY:
            return self.get_latency()
        # Throughput objective
        elif self.objective == THROUGHPUT:
            return -self.get_throughput()
        # Power objective
        elif self.objective == POWER:
            return self.get_power_average()    
    
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
