import copy
import random
import math
import sys
from numpy.random import choice
from dataclasses import dataclass
import numpy as np
import pickle
from fpgaconvnet.tools.graphs import ordered_node_list
from fpgaconvnet.optimiser.solvers import Solver

LATENCY   =0
THROUGHPUT=1
START_LOOP=1000

@dataclass
class Improve(Solver):
    T: float = 10.0
    k: float = 0.001
    T_min: float = 0.0001
    cool: float = 0.97
    iterations: int = 10

    """
    Chooses the hardware component causing a bottleneck and performs the same decision as simulated annealing
    """

    # def solver_status(self):
    #     # objective
    #     objectives = ['latency','throughput']
    #     objective  = objectives[self.objective]
    #     # cost
    #     cost = self.get_cost()
    #     # Resources
    #     resources = [ self.get_partition_resource(partition) for partition in self.partitions ]
    #     BRAM = max([ resource['BRAM'] for resource in resources ])
    #     DSP  = max([ resource['DSP']  for resource in resources ])
    #     LUT  = max([ resource['LUT']  for resource in resources ])
    #     FF   = max([ resource['FF']   for resource in resources ])
    #     sys.stdout.write("\033[K")
    #     print("TEMP:\t {temp}, COST:\t {cost} ({objective}), RESOURCE:\t {BRAM}\t{DSP}\t{LUT}\t{FF}\t(BRAM|DSP|LUT|FF)".format(
    #         temp=self.T,cost=cost,objective=objective,BRAM=int(BRAM),DSP=int(DSP),LUT=int(LUT),FF=int(FF)),end='\n')#,end='\r')

    def run_solver(self, log=True) -> bool:

        # update all partitions
        self.update_partitions()

        # Setup
        cost = self.get_cost()

        start = False

        try:
            self.check_resources()
            self.check_constraints()
            start = True
        except AssertionError as error:
            print(f"ERROR: Exceeds resource usage (trying to find valid starting point):\n{error}")
            bad_partitions = self.get_resources_bad_partitions()

        # Attempt to find a good starting point
        if not start:
            transforms_config = self.transforms_config
            self.get_transforms()

            for i in range(START_LOOP):
                transform = choice(self.transforms, p=self.transforms_probs)
                partition_index = list(bad_partitions.keys())[-1]
                self.apply_transform(transform, partition_index)
                self.update_partitions()

                try:
                    self.check_resources()
                    self.check_constraints()
                    self.transforms_config = transforms_config
                    self.get_transforms()
                    break
                except AssertionError as error:
                    pass

        try:
            self.check_resources()
            self.check_constraints()
        except AssertionError as error:
            print(f"ERROR: Exceeds resource usage:\n{error}")
            return False

        # Cooling Loop
        while self.T_min < self.T:

            # update partitions
            self.update_partitions()

            # get the current cost
            cost = self.get_cost()

            # Save previous iteration
            net_partitions = pickle.loads(pickle.dumps(self.net.partitions))

            # several iterations per cool down
            for _ in range(self.iterations):

                # update partitions
                self.update_partitions()

                # remove all auxiliary layers
                for partition in self.net.partitions:
                    partition.remove_squeeze()

                # Apply a transform
                ## Choose a random transform
                transform = choice(self.transforms, p=self.transforms_probs)

                ## Choose slowest partition
                partition_latencys = [ partition.get_latency(self.platform.board_freq) for partition in self.net.partitions ]
                partition_index    = np.random.choice(np.arange(len(self.net.partitions)), 1, p=(partition_latencys/sum(partition_latencys)))[0]

                ## Choose slowest node in partition
                node_latencys = np.array([ self.net.partitions[partition_index].graph.nodes[layer]['hw'].latency() \
                        for layer in ordered_node_list(self.net.partitions[partition_index].graph) ])
                node = np.random.choice(ordered_node_list(self.net.partitions[partition_index].graph), 1, p=(node_latencys/sum(node_latencys)))[0]

                ## Apply the transform
                self.apply_transform(transform, partition_index, node)

                ## Update partitions
                self.update_partitions()

            # Check resources
            try:
                self.check_resources()
                self.check_constraints()
            except AssertionError:
                # revert to previous state
                self.net.partitions = net_partitions
                continue

            # Simulated annealing descision
            if math.exp(min(0,(cost - self.get_cost())/(self.k*self.T))) < random.uniform(0,1):
                # revert to previous state
                self.net.partitions = net_partitions

            # print out solver status
            self.solver_status()

            # reduce temperature
            self.T *= self.cool

        return True
