import sys
import numpy as np
import json
import copy
import random
import math
from dataclasses import dataclass
import wandb

from fpgaconvnet.optimiser.latency.solvers.solver import LatencySolver

LATENCY   =0
THROUGHPUT=1

START_LOOP=1000

@dataclass
class LatencySimulatedAnnealing(LatencySolver):
    T: float = 10.0
    k: float = 0.001
    T_min: float = 0.0001
    cool: float = 0.97
    iterations: int = 10
    """
Randomly chooses a transform and hardware component to change. The change is accepted based on a probability-based decision function
    """

    def run_solver(self, log=True):

        # get all the layer types in the network
        layer_types = list(set([ self.net.graph.nodes[node]["type"] \
                for node in self.net.graph.nodes ]))

        # combine all the layer_types
        for layer_type in layer_types:
            self.combine(layer_type)

        # check the intial design is within constraints
        try:
            self.check_resources()
            self.check_building_blocks()
        except AssertionError as error:
            print("ERROR: Exceeds resource usage")
            return

        # Cooling Loop
        while self.T_min < self.T:

            # get the current cost
            cost = self.get_cost()

            # wandb logging and checkpoint
            if log:
                self.wandb_log(temperature=self.T,
                    num_blocks=len(self.building_blocks),
                    latency=self.evaluate_latency())
            # self.wandb_checkpoint()

            # Save previous building blocks
            building_blocks = copy.deepcopy(self.building_blocks)

            # several iterations per cool down
            for _ in range(self.iterations):

                # Apply a transform
                ## Choose a random transform
                transform = random.choice(self.transforms)

                ## Choose a random building block
                hw_node = random.choice(list(self.building_blocks.keys()))

                ## Choose a random execution node
                exec_node = random.choice(list(self.net.graph.nodes()))

                ## Apply the transform
                self.apply_transform(transform, hw_node, exec_node)

            # Check resources
            try:
                self.check_resources()
                self.check_building_blocks()
            except AssertionError:
                # revert to previous state
                self.building_blocks = building_blocks
                continue

            # Simulated annealing descision
            if math.exp(min(0,(cost - self.get_cost())/(self.k*self.T))) < random.uniform(0,1):
                # revert to previous state
                self.building_blocks = building_blocks

            # print solver status
            self.solver_status()

            # reduce temperature
            self.T *= self.cool

        # # store dataframe of
        # # https://docs.wandb.ai/guides/data-vis/log-tables
        # table = wandb.Table(columns=[])
        # for i, partition in enumerate(self.net.partitions):
        #     table.add_data([])
        # wandb.log({"partitions": table})

        # store image
        # wandb.log({"image": wandb.Image(path_to_image)})
        # wandb.log("plot": plt)
