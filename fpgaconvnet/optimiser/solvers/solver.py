import json
import copy
import random
import math
import numpy as np
from dataclasses import dataclass, field

LATENCY   =0
THROUGHPUT=1

from fpgaconvnet.models.network import Network

import fpgaconvnet.optimiser.transforms.weights_reloading as weights_reloading
import fpgaconvnet.optimiser.transforms.partition as partition
import fpgaconvnet.optimiser.transforms.coarse as coarse
import fpgaconvnet.optimiser.transforms.fine as fine

@dataclass
class Solver:
    net: Network
    objective: int = THROUGHPUT
    constraints: dict = field(default_factory=lambda: {
        'latency'    : float("inf"), 'throughput' : 0.0})
    transforms: list = field(default_factory=lambda:[
        'coarse','fine','partition', 'weights_reloading'])

    """
    Base class for all optimisation strategies. This inherits the `Network` class.

    Attributes
    ----------
    objective: int
        Objective for the solver. One of `LATENCY`, `THROUGHPUT` and `POWER`.
    constraints: dict
        dictionary containing constraints for `latency`, `throughput` and `power`.
    transforms: list
        list of transforms that can be applied to the network. Allowed transforms
        are `['coarse','fine','partition','weights_reloading']`
    """

        # self.transforms_config = transforms_config
        # if len(fix_starting_point_config) == 0:
        #     self.fix_starting_point_config = transforms_config
        # else:
        #     self.fix_starting_point_config = fix_starting_point_config

    # import optimiser utilities
    from fpgaconvnet.optimiser.solvers.utils import starting_point_distillation
    from fpgaconvnet.optimiser.solvers.utils import merge_memory_bound_partitions

    def get_transforms(self):
        self.transforms = []
        for transform_type, attr in self.transforms_config.items():
            if bool(attr["apply_transform"]):
                self.transforms.append(transform_type)

    def get_cost(self, partition_list=None):
        """
        calculates the cost function of the optimisation strategy at it's current state.
        This cost is based on the objective of the solver. There are three objectives
        that can be chosen:

        - `LATENCY (0)`
        - `THROUGHPUT (1)`
        - `POWER (2)`

        Returns:
        --------
        float
        """
        if partition_list == None:
            partition_list = list(range(len(self.net.partitions)))
        # Latency objective
        if   self.objective == LATENCY:
            return self.net.get_latency()
        # Throughput objective
        elif self.objective == THROUGHPUT:
            return -self.net.get_throughput()

    def check_resources(self):
        self.net.check_resources()

    def check_constraints(self):
        """
        function to check the performance constraints of the network. Checks
        `latency` and `throughput`.

        Raises
        ------
        AssertionError
            If not within performance constraints
        """
        assert self.net.get_latency() <= self.constraints['latency'], \
                "ERROR : (constraint violation) Latency constraint exceeded"
        assert self.net.get_throughput() >= self.constraints['throughput'], \
                "ERROR : (constraint violation) Throughput constraint exceeded"

    def apply_transform(self, transform, partition_index=None, node=None,iteration=None,cooltimes=None):
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
            partition_index = random.randint(0,len(self.net.partitions)-1)

        # choose random node in partition if given
        if node == None:
            node = random.choice(graphs.ordered_node_list(
                self.net.partitions[partition_index].graph))

        # Apply a random transform
        ## Coarse transform (node_info transform)
        if transform == 'coarse':
            coarse.apply_random_coarse_node(
                self.net.partitions[partition_index], node)
            return

        ## Fine transform (node_info transform)
        if transform == 'fine':
            fine.apply_random_fine_node(
                self.net.partitions[partition_index], node)
            return

        ## Weights-Reloading transform (partition transform)
        if transform == 'weights_reloading':
            ### apply random weights reloading
            weights_reloading.apply_random_weights_reloading(
                self.net, partition_index)
            return

        ## Partition transform (partition transform)
        if transform == 'partition':
            ### apply random partition
            # remove squeeze layers prior to partitioning
            self.net.partitions[partition_index].remove_squeeze()
            partition.apply_random_partition(
                self.net, partition_index)
            return

    def solver_status(self):
        """
        prints out the current status of the solver.
        """
        # objective
        objectives = [ 'latency', 'throughput','power']
        objective  = objectives[self.objective]
        # cost
        cost = self.get_cost()
        # Resources
        resources = [ partition.get_resource_usage() for partition in self.net.partitions ]
        BRAM = max([ resource['BRAM'] for resource in resources ])
        DSP  = max([ resource['DSP']  for resource in resources ])
        LUT  = max([ resource['LUT']  for resource in resources ])
        FF   = max([ resource['FF']   for resource in resources ])
        print("COST:\t {cost} ({objective}), RESOURCE:\t {BRAM}\t{DSP}\t{LUT}\t{FF}\t(BRAM|DSP|LUT|FF)".format(
            cost=cost,objective=objective,BRAM=int(BRAM),DSP=int(DSP),LUT=int(LUT),FF=int(FF)), end='\r')

    def save_design_checkpoint(self, output_path):
        # get the current state of the optimiser
        state = {
            "time" : str(datetime.now()),
            "self" : self
        }
        # pickle the current optimiser state
        checkpoint = pickle.dumps(self)
        # save to output path
        with open(output_path, "wb") as f:
            f.write(checkpoint)

    def get_optimal_batch_size(self):
        """
        gets an approximation of the optimal (largest) batch size for throughput-based
        applications. This is dependant on the platform's memory capacity. Updates the
        `self.batch_size` variable.
        """
        # change the batch size to zero
        self.net.batch_size = 1
        # update each partitions batch size
        self.net.update_batch_size()
        # calculate the maximum memory usage at batch 1
        max_mem = self.net.get_memory_usage_estimate()
        # update batch size to max
        self.net.batch_size = max(1,math.floor(self.net.platform['mem_capacity']/max_mem))
        self.net.update_batch_size()

    def run_solver(self, log=True):
        """
        template for running the solver.
        """
        raise RuntimeError("solver not implemented!")
