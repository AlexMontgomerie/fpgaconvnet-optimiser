import copy
import math
import pickle
import random
import time
from dataclasses import dataclass

from numpy.random import choice
from tabulate import tabulate

from fpgaconvnet.optimiser.solvers import Solver

LATENCY   =0
THROUGHPUT=1
START_LOOP=1000

@dataclass
class SimulatedAnnealing(Solver):
    T: float = 10.0
    k: float = 0.001
    T_min: float = 0.007
    cool: float = 0.97
    iterations: int = 10
    """
Randomly chooses a transform and hardware component to change. The change is accepted based on a probability-based decision function
    """

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
            print("WARNING: Exceeds resource usage (trying to find valid starting point)")

        # Attempt to find a good starting point
        if not start:
            warmup_loop_start_time = time.perf_counter()
            for i in range(START_LOOP):
                transform = choice(self.transforms, p=self.transforms_probs)
                self.apply_transform(transform)
                self.update_partitions()

                try:
                    self.check_resources()
                    self.check_constraints()
                    break
                except AssertionError as error:
                    pass
            self.total_opt_time += (time.perf_counter() - warmup_loop_start_time)
        try:
            self.check_resources()
            self.check_constraints()
        except AssertionError as error:
            print(f"ERROR: Exceeds resource usage:\n{error}")
            return False

        best_solution = pickle.loads(pickle.dumps(self.net))
        cooling_loop_start_time = time.perf_counter()
        # Cooling Loop
        while self.T_min < self.T:

            # get the current cost
            cost = self.get_cost()

            # Save previous iteration
            net_partitions = pickle.loads(pickle.dumps(self.net.partitions))

            # several iterations per cool down
            for _ in range(self.iterations):

                # remove all auxiliary layers
                for i in range(len(self.net.partitions)):
                    self.net.partitions[i].remove_squeeze()

                # Apply a transform
                ## Choose a random transform
                transform = choice(self.transforms, p=self.transforms_probs)

                ## Choose a random partition
                partition_index = random.randint(0,len(self.net.partitions)-1)

                ## Choose a random node in partition
                node = random.choice(list(self.net.partitions[partition_index].graph))

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
                self.update_partitions()
                continue

            # Simulated annealing descision
            new_cost = self.get_cost()
            if new_cost < cost:
                # accept new state
                pass
            else:
                if math.exp(min(0,(cost - new_cost)/(self.k*self.T))) < random.uniform(0,1):
                    # revert to previous state
                    self.net.partitions = net_partitions
                    self.update_partitions()

            data = [[f"temperature:",
                     f"{self.T:.6f}",
                     f"Min temperature:",
                     f"{self.T_min:.6f}",
                     "",
                     "number of partitions:",
                     len(self.net.partitions)]]
            data_table = tabulate(data, tablefmt="double_outline")
            print(data_table)

            # print solver status
            self.solver_status()

            if self.wandb_enabled:
                # wandb logging and checkpoint
                self.wandb_log(**{"temperature": self.T,
                                  "exp": math.exp(min(0,(cost - new_cost)/(self.k*self.T))),
                                  "delta": cost - new_cost})
                # self.wandb_checkpoint()

            # update best solution
            if self.get_cost() < self.get_cost(net=best_solution):
                best_solution = pickle.loads(pickle.dumps(self.net))

            # reduce temperature
            self.T *= self.cool

        # update partitions
        self.net = best_solution

        self.total_opt_time += (time.perf_counter() - cooling_loop_start_time)

        # # store dataframe of
        # # https://docs.wandb.ai/guides/data-vis/log-tables
        # table = wandb.Table(columns=[])
        # for i, partition in enumerate(self.net.partitions):
        #     table.add_data([])
        # wandb.log({"partitions": table})

        # store image
        # wandb.log({"image": wandb.Image(path_to_image)})
        # wandb.log("plot": plt)

        return True