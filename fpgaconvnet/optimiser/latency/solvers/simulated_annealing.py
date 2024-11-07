import os
import numpy as np
import json
import copy
import random
import math
from dataclasses import dataclass
import wandb
import pandas as pd
import time
import pickle

from fpgaconvnet.optimiser.latency.solvers.solver import LatencySolver

LATENCY     =   0
THROUGHPUT  =   1
START_LOOP  =   10

@dataclass
class LatencySimulatedAnnealing(LatencySolver):
    T: float = 10.0
    k: float = 10.0
    T_min: float = 0.00001
    cool: float = 0.99
    transform_iterations: int = 15
    warm_start: bool = True
    warm_start_time_limit: int = 90
    """
    Randomly chooses a transform and hardware component to change.
    The change is accepted based on a probability-based decision function
    """

    def warm_start_solution(self):
        start_time = time.time()
        while not self.check_resources() and (time.time() - start_time) < self.warm_start_time_limit:
            # Choose a random transform
            transform = random.choice(list(self.transforms.keys()))

            # Choose a random building block
            hw_node = random.choice(list(self.building_blocks.keys()))

            # Choose a random execution node
            exec_node = random.choice(list(self.net.graph.nodes()))

            # Apply the transform
            self.apply_transform(transform, hw_node, exec_node, warm_start=True)

        if time.time() - start_time >= self.warm_start_time_limit:
            raise Exception("Warm start failed to find a solution within the time limit")

        # perform a few iterations of the solver to improve the initial solution
        for _ in range(START_LOOP):
            # get the current cost
            cost = self.get_cost()

            # Save previous building blocks
            building_blocks = pickle.loads(pickle.dumps(self.building_blocks))

            # several transform iterations per cool down
            for _ in range(self.transform_iterations):

                # Choose a random transform
                transform = random.choice(list(self.transforms.keys()))

                # Choose a random building block
                hw_node = random.choice(list(self.building_blocks.keys()))

                # Choose a random execution node
                exec_node = random.choice(list(self.net.graph.nodes()))

                # Apply the transform
                self.apply_transform(transform, hw_node, exec_node, warm_start=True)

            # Check resources
            try:
                assert self.check_resources()
                self.check_building_blocks()
            except AssertionError:
                # revert to previous state
                self.building_blocks = building_blocks
                continue

            # Simulated annealing descision
            curr_cost = self.get_cost()
            if curr_cost < cost:
                # accept new state
                pass
            else:
                # revert to previous state
                self.building_blocks = building_blocks

    def run_solver(self, log=True) -> bool:

        if self.warm_start:
            # warm start the solver
            self.warm_start_solution()

        # check the intial design is within constraints
        try:
            assert self.check_resources()
            self.check_building_blocks()
        except AssertionError as error:
            print("Initial design exceeded resource usage")
            return False

        # Cooling Loop
        while self.T_min < self.T:

            # get the current cost
            cost = self.get_cost()
            resources = self.get_resources()

            # Save previous building blocks
            building_blocks = pickle.loads(pickle.dumps(self.building_blocks))

            # several transform iterations per cool down
            # transform_iterations = random.randint(1, self.transform_iterations)
            for _ in range(self.transform_iterations):
            # for _ in range(transform_iterations):

                # Choose a random transform
                transform = np.random.choice(list(self.transforms.keys()),
                        p=list(self.transforms.values()))

                # Choose a random building block
                hw_node = random.choice(list(self.building_blocks.keys()))

                # Choose a random execution node
                exec_node = random.choice(list(self.net.graph.nodes()))

                # Apply the transform
                self.apply_transform(transform, hw_node, exec_node)

            # Check resources
            try:
                assert self.check_resources()
                self.check_building_blocks()
            except AssertionError:
                # revert to previous state
                self.building_blocks = building_blocks
                continue

            # Simulated annealing descision
            curr_cost = self.get_cost()
            status_cost = curr_cost
            if curr_cost < cost:
                # accept new state
                pass
            elif curr_cost == cost:
                curr_resources = self.get_resources()
                if sum(curr_resources.values()) < sum(resources.values()):
                    # accept new state
                    pass
            else:
                if math.exp((cost - curr_cost)/(self.k*self.T)) > random.uniform(0,1):
                    # accept new state
                    pass
                else:
                    # revert to previous state
                    self.building_blocks = building_blocks
                    status_cost = cost

            # print solver status
            self.solver_status(self.T, cost=status_cost)

            # wandb logging and checkpoint
            if log:
                self.wandb_log(temperature=self.T,
                    num_blocks=len(self.building_blocks),
                    latency=status_cost,
                    **self.get_resources_util())

            # reduce temperature
            self.T *= self.cool

        if log:

            # get config and report
            config = self.config()
            report = self.report()
            schedule = self.get_schedule()

            # get per layer table
            per_layer_table = {
                "exec_node": [],
                "hw_node": [],
                "type": [],
                "latency": [],
                "repetitions": [],
                "iteration_space": [],
            }
            for exec_node, per_layer_report in report["per_layer"].items():
                per_layer_table["exec_node"].append(exec_node)
                per_layer_table["hw_node"].append(per_layer_report["hw_node"])
                per_layer_table["type"].append(per_layer_report["type"])
                per_layer_table["latency"].append(per_layer_report["latency"])
                per_layer_table["repetitions"].append(per_layer_report["repetitions"])
                per_layer_table["iteration_space"].append(per_layer_report["iteration_space"])

            self.wandb_log(per_layer=wandb.Table(data=pd.DataFrame(per_layer_table)))

            # save report and config
            if not os.path.exists("tmp"):
                os.makedirs("tmp")
            with open("tmp/config.json", "w") as f:
                json.dump(config, f, indent=2)
            with open("tmp/report.json", "w") as f:
                json.dump(report, f, indent=2)
            with open("tmp/schedule.json", "w") as f:
                json.dump(schedule, f, indent=2)

            # save them as artifacts
            artifact = wandb.Artifact('outputs', type='json')
            artifact.add_file("tmp/config.json")
            artifact.add_file("tmp/report.json")
            artifact.add_file("tmp/schedule.json")
            wandb.log_artifact(artifact)
            # self.wandb_checkpoint()

        print(f"Final cost: {self.get_cost():.4f}")
        print(f"Final resources: {self.get_resources_util()}")
        print(f"Final building blocks: {list(self.building_blocks.keys())}")

        return True

        # # store dataframe of
        # # https://docs.wandb.ai/guides/data-vis/log-tables
        # table = wandb.Table(columns=[])
        # for i, partition in enumerate(self.net.partitions):
        #     table.add_data([])
        # wandb.log({"partitions": table})

        # store image
        # wandb.log({"image": wandb.Image(path_to_image)})
        # wandb.log("plot": plt)
