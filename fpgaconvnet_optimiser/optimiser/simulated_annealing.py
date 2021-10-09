import os
import sys
import copy
import random
import math

from decimal import *
from fpgaconvnet_optimiser.optimiser.optimiser import Optimiser
from multiprocessing import Pool

LATENCY = 0
THROUGHPUT = 1

START_LOOP = 1000


class SimulatedAnnealing(Optimiser):
    """
Randomly chooses a transform and hardware component to change. The change is accepted based on a probability-based decision function
    """

    def __init__(self, name, network_path, T, k, T_min, cool, iterations):

        # Initialise Network
        super().__init__(name, network_path)

        self.T_min = float(T_min)
        self.cool = float(cool)
        self.iterations = int(iterations)

        # Simulated Annealing Variables
        if T == "auto":
            self.T = "auto"
            self.k = 1
        else:
            self.T = float(T)
            self.k = float(k)

    def optimiser_status(self):
        # objective
        objectives = ['latency', 'throughput']
        objective = objectives[self.objective]
        # cost
        cost = self.get_cost()
        # Resources
        resources = [partition.get_resource_usage() for partition in self.partitions]
        BRAM = max([resource['BRAM'] for resource in resources])
        DSP = max([resource['DSP'] for resource in resources])
        LUT = max([resource['LUT'] for resource in resources])
        FF = max([resource['FF'] for resource in resources])
        sys.stdout.write("\033[K")
        print(
            "TEMP:\t {temp}, COST:\t {cost} ({objective}), RESOURCE:\t {BRAM}\t{DSP}\t{LUT}\t{FF}\t(BRAM|DSP|LUT|FF)".format(
                temp=self.T, cost=cost, objective=objective, BRAM=int(BRAM), DSP=int(DSP), LUT=int(LUT), FF=int(FF)),
            end='\n')  # ,end='\r')

    def run_optimiser(self, log=True):

        # update all partitions
        self.update_partitions()

        # Setup
        cost = self.get_cost()

        # Initialize to valid partition within constraints
        start = False

        try:
            self.check_resources()
            self.check_constraints()
            start = True
        except AssertionError as error:
            print("ERROR: Exceeds resource usage (trying to find valid starting point)")

        # Attempt to find a good starting point
        if not start:
            for i in range(START_LOOP):
                transform = random.choice(self.transforms)
                self.apply_transform(transform)
                self.update_partitions()

                try:
                    self.check_resources()
                    self.check_constraints()
                    break
                except AssertionError as error:
                    pass

        # Cooling Loop
        while self.T_min < self.T:

            # update partitions
            self.update_partitions()

            # get the current cost
            cost = self.get_cost()

            # Save previous iteration
            partitions = copy.deepcopy(self.partitions)

            # several iterations per cool down
            for _ in range(self.iterations):

                # update partitions
                self.update_partitions()

                # remove all auxiliary layers
                for i in range(len(self.partitions)):
                    self.partitions[i].remove_squeeze()

                # Apply a transform
                ## Choose a random transform
                transform = random.choice(self.transforms)

                ## Choose a random partition
                partition_index = random.randint(0, len(self.partitions) - 1)

                ## Choose a random node in partition
                node = random.choice(list(self.partitions[partition_index].graph))

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
                self.partitions = partitions
                continue

            # Simulated annealing descision
            if math.exp(min(0, (cost - self.get_cost()) / (self.k * self.T))) < random.uniform(0, 1):
                # revert to previous state
                self.partitions = partitions

            # update cost
            if self.DEBUG:
                self.optimiser_status()

            # reduce temperature
            self.T *= self.cool

    def estimate_starting_temperature(self, chi_target=0.8, chi_tolerance=0.01, p=1, sample_target=100, threads=0):
        """
        An implementation of the algorithm from:
            Ben-Ameur, Walid. (2004). Computing the Initial Temperature of Simulated Annealing.
            Computational Optimization and Applications. 29. 369-385. 10.1023/B:COAP.0000044187.23143.bd.

        The function generates a number of random transitions and estimates a good value of temperature.
        I was not able to figure out why the algorithm seems to get stuck in loops while the temperature is estimated, so it simply restarts.
        """

        # Initialize list to store energy information of positive transitions

        # All transitions generated in single thread
        if threads == 1:
            transitions = self.generate_transitions(sample_target)
            sample_list = transitions[0]
            self.partitions = transitions[1]

        # Generation of transition spread across multiple cores
        else:

            # auto (0) in threads sets maximum number of threads
            if threads == 0:
                threads = os.cpu_count()

            # Per core number of transitions to be generated
            core_sample_targets = [sample_target // threads for _ in range(threads)]

            # Distribute remainder
            for i in range(sample_target % threads):
                core_sample_targets[i] += 1

            # Multiprocessing map function
            with Pool(threads) as p:
                transitions = p.map(self.generate_transitions, core_sample_targets)

            # Flatten transition list
            sample_list = []
            for core in transitions:
                sample_list.extend(core[0])

                # Load best performing partition from sampling
                if self.get_cost() > core[2]:
                    self.partitions = core[1]

        # Arbitrary starting temperature
        T = Decimal(7.5)
        chi_estimate = Decimal(0)
        chi_target = Decimal(chi_target)
        chi_tolerance = Decimal(chi_tolerance)

        # Algorithm from paper implemented
        i = 0
        while abs((chi_target - chi_estimate) / chi_target) > chi_tolerance:

            #Calculating exponentially weighted sum of transitions
            Emax_sum = Decimal(0)
            Emin_sum = Decimal(0)

            for j in sample_list:
                Emin_sum += (Decimal(-j[0]) / T).exp()
                Emax_sum += (Decimal(-j[1]) / T).exp()

            # Estimate of acceptance probability
            chi_estimate = Emax_sum / Emin_sum

            #Estimating temperature for next iteration
            T = T * Decimal(chi_estimate).ln() / Decimal(chi_target).ln()

            #In case algorithm fails to converge in reasonable time, some data points are replaced
            i += 1
            if i > 1000:
                print("Failed to converge, adding sample and restarting ")
                five_percent = int(len(sample_list) / 20)
                sample_list.extend(self.generate_transitions(five_percent)[0])

                for _ in range(five_percent):
                    sample_list.pop(0)
                i = 0
                T = Decimal(7.5)

        print("Starting temperature: " + str(T))
        self.T = float(T)

    def generate_transitions(self, sample_target=100, positive=False, negative=True):

        # Variable initialization
        min_partitions = []
        min_cost = float('inf')
        sample_list = []
        pid = os.getpid()

        #Main sample collection loop
        while len(sample_list) < sample_target:

            # update partitions
            self.update_partitions()

            # get the current cost
            cost = self.get_cost()

            # Save previous iteration
            partitions = copy.deepcopy(self.partitions)

            #Check for valid starting point
            start = False

            try:
                self.check_resources()
                self.check_constraints()
                start = True
            except AssertionError as error:
                print("ERROR: Exceeds resource usage (trying to find valid starting point)")

            # Attempt to find a good starting point
            if not start:
                for i in range(START_LOOP):
                    transform = random.choice(self.transforms)
                    self.apply_transform(transform)
                    self.update_partitions()

                    try:
                        self.check_resources()
                        self.check_constraints()
                        break
                    except AssertionError as error:
                        pass

            # several iterations per cool down
            for t in range(self.iterations):

                # update partitions
                self.update_partitions()

                # remove all auxiliary layers
                for i in range(len(self.partitions)):
                    self.partitions[i].remove_squeeze()

                # Apply a transform
                # Choose a random transform
                transform = random.choice(self.transforms)

                # Choose a random partition
                partition_index = random.randint(0, len(self.partitions) - 1)

                # Choose a random node in partition
                node = random.choice(list(self.partitions[partition_index].graph))

                # Apply the transform
                self.apply_transform(transform, partition_index, node)

                # Update partitions
                self.update_partitions()

            # Check resources
            try:
                self.check_resources()
                self.check_constraints()
            except AssertionError:
                # revert to previous state
                self.partitions = partitions

            # Add transition to sample list based on Negative and Positive flags
            if (self.get_cost() > cost) and negative:
                sample_list.append((cost, self.get_cost()))
                print(
                    f"Negative transition, diffference: {self.get_cost() - cost}, Sample point {len(sample_list)}/{sample_target}, PID:{pid}")

            elif (self.get_cost() < cost) and positive:
                sample_list.append((cost, self.get_cost()))
                print(
                    f"Positive transition, diffference: {self.get_cost() - cost}, Sample point {len(sample_list)}/{sample_target}, PID:{pid}")

                # Update the record of the best performing partition
            if self.get_cost() < min_cost:
                min_cost = self.get_cost()
                min_partitions = partitions

        return sample_list, min_partitions, min_cost
