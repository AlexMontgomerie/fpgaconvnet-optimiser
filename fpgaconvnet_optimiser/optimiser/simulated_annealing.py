import decimal
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

    def __init__(self, name, network_path, t, k, t_min, cool, iterations, csv_path=None, seed="123456789"):

        # Initialise Network
        super().__init__(name, network_path)
        self.DEBUG = False
        self.t_min = float(t_min)
        self.cool = float(cool)
        self.iterations = int(iterations)
        self.seed = seed
        self.network_path = network_path

        # Simulated Annealing Variables
        if t == "auto":
            self.t = "auto"
            self.k = 1
        else:
            self.t = float(t)
            self.k = float(k)

        # Set up debug dictionary for counting transform usage
        self.csv_path = csv_path
        self.transform_count = {}
        for transform in self.transforms:
            self.transform_count[transform] = 0

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
                temp=self.t, cost=cost, objective=objective, BRAM=int(BRAM), DSP=int(DSP), LUT=int(LUT), FF=int(FF)),
            end='\n')  # ,end='\r')

        # More internal information written to csv
        if self.csv_path is not None:
            # Resource averages
            BRAM_AVG = sum([resource['BRAM'] for resource in resources]) / len(resources)
            DSP_AVG = sum([resource['DSP'] for resource in resources]) / len(resources)
            LUT_AVG = sum([resource['LUT'] for resource in resources]) / len(resources)
            FF_AVG = sum([resource['FF'] for resource in resources]) / len(resources)

            # Performance metrics regardless of objective
            latency = self.get_latency()
            throughput = self.get_throughput()

            # Process transform counts
            count_string = ""
            for value in self.transform_count.values():
                count_string += f", {value}"

            csv_file = open(self.csv_path, 'a')

            # CSV output format
            csv_file.write(f"{self.t}, {cost}, {BRAM}, {BRAM_AVG}, {DSP}, {DSP_AVG}, {LUT}, {LUT_AVG}, {FF}, {FF_AVG},"
                           f" {latency}, {throughput}{count_string}\n")
            csv_file.close()

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

                # Increment transform counter
                if self.DEBUG:
                    self.transform_count[transform] += 1

                self.apply_transform(transform)
                self.update_partitions()

                try:
                    self.check_resources()
                    self.check_constraints()
                    break
                except AssertionError as error:
                    pass
        # Write debug csv column info
        if self.csv_path is not None:
            self.write_csv_header()

        # Cooling Loop
        while self.t_min < self.t:

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
                # Choose a random transform
                transform = random.choice(self.transforms)

                # Increment transform counter
                if self.DEBUG:
                    self.transform_count[transform] += 1

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
                continue

            # Simulated annealing descision
            if math.exp(min(0, (cost - self.get_cost()) / (self.k * self.t))) < random.uniform(0, 1):
                # revert to previous state
                self.partitions = partitions

            # update cost
            if self.DEBUG:
                self.optimiser_status()

            # reduce temperature
            self.t *= self.cool
        print(self.transform_count)

    def estimate_starting_temperature(self, chi_target=0.8, chi_tolerance=0.01, p=1, sample_target=100, threads=0):
        """
        An implementation of the algorithm from:
            Ben-Ameur, Walid. (2004). Computing the Initial Temperature of Simulated Annealing.
            Computational Optimization and Applications. 29. 369-385. 10.1023/B:COAP.0000044187.23143.bd.

        The function generates a number of random transitions and estimates a good value of temperature.
        I was not able to figure out why the algorithm seems to get stuck in loops while the temperature is estimated, so it simply restarts.
        """
        decimal.getcontext().prec = 1000

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
        t = Decimal(7.5)
        chi_estimate = Decimal(0)
        chi_target = Decimal(chi_target)
        chi_tolerance = Decimal(chi_tolerance)

        # Algorithm from paper implemented
        i = 0
        while abs((chi_target - chi_estimate) / chi_target) > chi_tolerance:

            # Calculating exponentially weighted sum of transitions
            e_max_sum = Decimal(0)
            e_min_sum = Decimal(0)

            for j in sample_list:
                e_min_sum += (Decimal(-j[0]) / t).exp()
                e_max_sum += (Decimal(-j[1]) / t).exp()

            # Estimate of acceptance probability
            chi_estimate = e_max_sum / e_min_sum

            # Estimating temperature for next iteration
            t = t * Decimal(chi_estimate).ln() / Decimal(chi_target).ln()

            # In case algorithm fails to converge in reasonable time, some data points are replaced
            i += 1
            if i > 1000:
                print("Failed to converge, adding sample and restarting ")
                five_percent = int(len(sample_list) / 20)
                sample_list.extend(self.generate_transitions(five_percent)[0])

                for _ in range(five_percent):
                    sample_list.pop(0)
                i = 0
                t = Decimal(7.5)

        print("Starting temperature: " + str(t))
        self.t = float(t)

    def generate_transitions(self, sample_target=100, positive=False, negative=True):

        # Variable initialization
        min_partitions = []
        min_cost = float('inf')
        sample_list = []
        pid = os.getpid()

        # Main sample collection loop
        while len(sample_list) < sample_target:

            # update partitions
            self.update_partitions()

            # get the current cost
            cost = self.get_cost()

            # Save previous iteration
            partitions = copy.deepcopy(self.partitions)

            # Check for valid starting point
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

                # Choose a random transform
                transform = random.choice(self.transforms)

                # Increment transform counter
                if self.DEBUG:
                    self.transform_count[transform] += 1

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
                    f"Negative transition, difference: {self.get_cost() - cost},"
                    f" Sample point {len(sample_list)}/{sample_target}, PID:{pid}")

            elif (self.get_cost() < cost) and positive:
                sample_list.append((cost, self.get_cost()))
                print(
                    f"Positive transition, difference: {self.get_cost() - cost},"
                    f" Sample point {len(sample_list)}/{sample_target}, PID:{pid}")

            # Update the record of the best performing partition
            if self.get_cost() < min_cost:
                min_cost = self.get_cost()
                min_partitions = partitions

        return sample_list, min_partitions, min_cost

    def write_csv_header(self):
        # Generate column header for transform counter
        key_header = ""
        for key in self.transform_count.keys():
            key_header += f", {key}"
        key_header += "\n"

        # Write to csv
        csv_file = open(self.csv_path, 'a')

        # CSV header format
        csv_file.write("TEMP, COST, BRAM_MAX, BRAM_AVG, DSP_MAX, DSP_AVG, LUT_MAX, LUT_AVG, FF_MAX, FF_AVG, LATENCY,"
                       " THROUGHPUT" + key_header)

        # Quick annealing parameter reference
        csv_file.write(f"Running with t:{self.t}, t_min:{self.t_min}, k:{self.k}, cool:{self.cool},"
                       f" iterations:{self.iterations}, seed, {self.seed}, {self.network_path}")
        csv_file.close()
