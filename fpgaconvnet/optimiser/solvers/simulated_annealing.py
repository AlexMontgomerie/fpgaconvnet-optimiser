import sys
import numpy as np
import json
import copy
import random
import os
import math
from dataclasses import dataclass
import wandb

from fpgaconvnet.optimiser.solvers import Solver

LATENCY   =0
THROUGHPUT=1

START_LOOP=1000

@dataclass
class SimulatedAnnealing(Solver):
    T: float = 10.0
    k: float = 0.001
    T_min: float = 0.0001
    cool: float = 0.97
    iterations: int = 10
    """
Randomly chooses a transform and hardware component to change. The change is accepted based on a probability-based decision function
    """

    def run_solver(self, log=True):

        # update all partitions
        self.net.update_partitions()

        # Setup
        cost = self.get_cost()

        start = False

        try:
            self.check_resources()
            self.check_constraints()
            start = True
        except AssertionError as error:
            print("WARNING: Exceeds resource usage (trying to find valid starting point)")

        # Attempt to find a good starting point
        if not start:
            for i in range(START_LOOP):
                transform = random.choice(list(self.transforms.keys()))
                self.apply_transform(transform)
                self.net.update_partitions()

                try:
                    self.check_resources()
                    self.check_constraints()
                    break
                except AssertionError as error:
                    pass

        try:
            self.check_resources()
            self.check_constraints()
        except AssertionError as error:
            print("ERROR: Exceeds resource usage")
            return

        # Cooling Loop
        while self.T_min < self.T:

            # update partitions
            self.net.update_partitions()

            # get the current cost
            cost = self.get_cost()

            # wandb logging and checkpoint
            self.wandb_log(temperature=self.T)
            # self.wandb_checkpoint()

            # Save previous iteration
            net = copy.deepcopy(self.net)

            # several iterations per cool down
            for _ in range(self.iterations):

                # update partitions
                self.net.update_partitions()

                # remove all auxiliary layers
                for i in range(len(self.net.partitions)):
                    self.net.partitions[i].remove_squeeze()

                # Apply a transform
                ## Choose a random transform
                transform = random.choice(list(self.transforms.keys()))

                ## Choose a random partition
                partition_index = random.randint(0,len(self.net.partitions)-1)

                ## Choose a random node in partition
                node = random.choice(list(self.net.partitions[partition_index].graph))

                ## Apply the transform
                self.apply_transform(transform, partition_index, node)

                ## Update partitions
                self.net.update_partitions()

            # Check resources
            try:
                self.check_resources()
                self.check_constraints()
            except AssertionError:
                # revert to previous state
                self.net = net
                continue

            # Simulated annealing descision
            if math.exp(min(0,(cost - self.get_cost())/(self.k*self.T))) < random.uniform(0,1):
                # revert to previous state
                self.net = net

            # print solver status
            self.solver_status()

            # reduce temperature
            self.T *= self.cool

            # get config and report
            config = self.config()
            report = self.report()

            # save report and config
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            with open("tmp/config.json", "w") as f:
                json.dump(config, f, indent=2)
            with open("tmp/report.json", "w") as f:
                json.dump(report, f, indent=2)

            # store the design point
            artifact = wandb.Artifact('outputs', type='json')
            artifact.add_file("tmp/config.json")
            artifact.add_file("tmp/report.json")
            wandb.log_artifact(artifact)

        # # store dataframe of
        # # https://docs.wandb.ai/guides/data-vis/log-tables
        # table = wandb.Table(columns=[])
        # for i, partition in enumerate(self.net.partitions):
        #     table.add_data([])
        # wandb.log({"partitions": table})

        # store image
        # wandb.log({"image": wandb.Image(path_to_image)})
        # wandb.log("plot": plt)
