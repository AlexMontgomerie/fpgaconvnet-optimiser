import decimal
import sys
import copy
import random
import math
import logging
import uuid
from datetime import datetime

from decimal import *
from fpgaconvnet_optimiser.optimiser.optimiser import Optimiser
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE
import fpgaconvnet_optimiser.tools.graphs as graphs

LATENCY = 0
THROUGHPUT = 1

START_LOOP = 1000


class SimulatedAnnealing(Optimiser):
    """
    Randomly chooses a transform and hardware component to change. The
    change is accepted based on a probability-based decision function
    """

    def __init__(self, name, network_path, t, k, t_min, cool, iterations, csv_path=None, seed="123456789"):

        # Initialise Network
        super().__init__(name, network_path)

        # Simulated Annealing Variables
        self.DEBUG = False
        self.t_min = float(t_min)
        self.cool = float(cool)
        self.seed = seed
        self.network_path = network_path

        # Simulated Annealing auto variables
        if t == "auto" or k == "auto":
            self.t = "auto"
            self.k = 1
        else:
            self.t = float(t)
            self.k = float(k)

        if iterations == "auto":
            self.iterations = 3
        else:
            self.iterations = int(iterations)


        # Set up debug dictionary and transform counter
        self.csv_path = csv_path
        self.transform_count = {}
        for transform in self.transforms:
            self.transform_count[transform] = 0

    def optimiser_status(self):
        """Function prints network status information between cooling iterations and saves additional information into
        the debug csv file, if a path is set"""

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
            "TEMP:\t {temp}, COST:\t {cost} ({objective}), RESOURCE:\t {BRAM}\t{DSP}\t{LUT}\t{FF}\t(BRAM|DSP|LUT|FF)\t"
            "{transforms}".format(
                temp=self.t, cost=cost, objective=objective, BRAM=int(BRAM), DSP=int(DSP), LUT=int(LUT), FF=int(FF),
                transforms="\t".join(self.transforms)), end='\n')  # ,end='\r')

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
            transform_count_string = ", ".join([str(self.transform_count[key]) for key in self.transform_count.keys()])

            csv_file = open(self.csv_path, 'a')

            # CSV output format
            csv_file.write(f"{self.t}, {cost}, {BRAM}, {BRAM_AVG}, {DSP}, {DSP_AVG}, {LUT}, {LUT_AVG}, {FF}, {FF_AVG},"
                           f" {latency}, {throughput}, {transform_count_string}\n")
            csv_file.close()

    def run_optimiser(self, log=True):
        """Runs based on platform and transform information set separately of constructor. Upon completion
        self.partitions is set to the result of optimization"""

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
            print("WARNING: Exceeds resource usage (trying to find valid starting point)")
            bad_partitions = self.get_resources_bad_partitions()

        # Attempt to find a good starting point
        print(START_LOOP)
        if not start:
            transforms_config = self.transforms_config
            self.transforms_config = self.fix_starting_point_config
            self.get_transforms()

            for i in range(START_LOOP):
                print('The %dth iteration' %(i))
                transform = random.choice(self.transforms)

                # Increment transform counter
                if self.DEBUG:
                    self.transform_count[transform] += 1

                self.apply_transform(transform)
                self.update_partitions()

                try:
                    self.check_resources()
                    self.check_constraints()
                    self.transforms_config = transforms_config
                    self.get_transforms()
                    break
                except AssertionError as error:
                    bad_partitions = self.get_resources_bad_partitions()

        # Write debug csv column info
        if self.csv_path is not None:
            self.write_csv_header()

        # Following proper initialization estimate temperature
        if self.t == "auto":
            self.t = self.estimate_starting_temperature()

        # print the header for the optimiser status
        objectives = ['latency (s)','throughput (fps)']
        objective  = objectives[self.objective]
        print(f"Temperature\t{objective}\t  BRAM | DSP  | LUT    | FF    ")

        # Cooling Loop
        while self.t_min < self.t:

            # update partitions
            self.update_partitions()

            # get the current cost
            cost = self.get_cost()

            # Save previous iteration
            partitions = copy.deepcopy(self.partitions)

            # create a design checkpoint
            if self.checkpoint:
                self.save_design_checkpoint(os.path.join(self.checkpoint_path,f"{str(uuid.uuid4().hex)}.dcp"))

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
        if self.DEBUG:
            print(self.transform_count)

    def estimate_starting_temperature(self, chi_target=0.94, chi_tolerance=0.01, sample_target=100, mode="Ameur"):
        """
        The function uses the generate_transitions function to generate a sample of state transitions and then applies
        an algorithm to estimate the starting temperature. It both sets the network t annealing parameter and returns this temperature.

        There are two algorithms implemented:
            1) Ben-Ameur, Walid. (2004). Computing the Initial Temperature of Simulated Annealing.
            Computational Optimization and Applications. 29. 369-385. 10.1023/B:COAP.0000044187.23143.bd.
            This has additional parameters implemented as **kwargs:
                -chi-target: Target acceptance probability. Should be close to 1 as the goal is to accept all
                    transitions initially
                -chi-tolerance: Precision to which determine starting temperature and match chi-target
            2) Maximum transition - Set initial temperature to maximum observed difference as all transitions should be
            accepted initially

        Algorithm 1 may fail to converge. If that occurs, some transitions are replaced and another attempt is made.

        Smallest sample list for which a failure to converge has been observed (in case more digging is to be done)
                    sample_list = [(-5.8307128554903995, -5.640806442180019), (-11.918836347797749, -11.371805455155148),
                           (-11.371805455155148, -9.204115697648776), (-9.204115697648776, -4.309705653654608),
                           (-11.994668226037508, -6.035553174382293), (-3.0169132743870315, -2.405264519831464),
                           (-6.035553174382293, -5.948002938979628), (-5.948002938979628, -4.032819797286603),
                           (-11.183263539331135, -2.9867880267935436), (-2.9867880267935436, -2.9582999459601433),
                           (-2.9582999459601433, -2.957873422468494), (-3.9015029681854285, -1.0329898958713128),
                           (-4.01625381130438, -4.0073207659265915), (-4.0073207659265915, -3.770179622516949),
                           (-3.770179622516949, -2.4365893223042), (-2.4367352379557734, -2.4129557112982267),
                           (-11.813303310531769, -5.830639694407311), (-6.026676965052264, -5.882455365105181),
                           (-2804.639996410061, -12.074495969484948), (-12.074495969484948, -4.032841655731214),
                           (-4.032841655731214, -4.02756906467377), (-4.02756906467377, -3.0206963372027005),
                           (-4.017757846841786, -3.0169132743870315)]
        """

        # All transitions generated in single thread

        transitions = self.generate_transitions(sample_target)
        sample_list = transitions[0]
        self.partitions = transitions[1]


        #Select temperature estimation algorithm
        if mode == "Ameur":
            # Ben-Ameur algorithm from paper implemented

            #Set precision
            decimal.getcontext().prec = 28

            # Arbitrary starting temperature
            t = Decimal(7.5)
            chi_estimate = Decimal(0)
            chi_target = Decimal(chi_target)
            chi_tolerance = Decimal(chi_tolerance)

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
                    if self.DEBUG:
                        print("Failed to converge, adding sample and restarting ")

                    print(sample_list)
                    five_percent = int(len(sample_list) / 20)
                    sample_list.extend(self.generate_transitions(five_percent)[0])

                    for _ in range(five_percent):
                        sample_list.pop(0)
                    i = 0
                    t = Decimal(7.5)

            if self.DEBUG:
                print("Starting temperature: " + str(t))
            return float(t)

        #Estimation of starting temperature as maximum difference
        elif mode == "Max":
            max = 0
            for j in sample_list:
                if j[1]-j[0] > max:
                    max = j[1]-j[0]

            if self.DEBUG:
                print("Starting temperature: " + str(max))
            return float(max)

    def generate_transitions(self, sample_target, positive=False, negative=True):
        """Function generates a specified number (sample_target) of random transitions for its network in either the
        positive direction (performance increases afer change) or the negative direction (performance decreases after change)
        The function returns a tuple of three elements:
            1)List of tuples of two elements, the first being the cost function before specified transition and the
             second after
            2)A copy of the best performing partitions encountered during random search. Suggested as a good starting
            point for the optimiser
            3) The cost function value for the best performing partitions
        """

        # Variable initialization
        min_partitions = []
        min_cost = float('inf')
        sample_list = []

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

                    # Increment transition counter
                    if self.DEBUG:
                        self.transform_count[transform] += 1

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

            # Add transition to sample list based on Negative (perf. decreases) and Positive(perf. increases) flags
            if (self.get_cost() > cost) and negative:
                sample_list.append((cost, self.get_cost()))
                if self.DEBUG:
                    print(
                        f"Negative transition added, difference: {self.get_cost() - cost},"
                        f" Sample point {len(sample_list)}/{sample_target}")

            elif (self.get_cost() < cost) and positive:
                sample_list.append((cost, self.get_cost()))
                if self.DEBUG:
                    print(
                        f"Positive transition added, difference: {self.get_cost() - cost},"
                        f" Sample point {len(sample_list)}/{sample_target}")

            # Update the record of the best performing partition
            if self.get_cost() < min_cost:
                min_cost = self.get_cost()
                min_partitions = partitions

        return sample_list, min_partitions, min_cost

    def write_csv_header(self):
        """Function appends the data columns and human readable information on annealer settings into the csv file at
        self.csv_path"""

        # Generate column header for transform counter
        key_header = ", ".join(self.transform_count.keys())

        # Write to csv
        csv_file = open(self.csv_path, 'a')

        # CSV header format
        csv_file.write("TEMP, COST, BRAM_MAX, BRAM_AVG, DSP_MAX, DSP_AVG, LUT_MAX, LUT_AVG, FF_MAX, FF_AVG, LATENCY,"
                       " THROUGHPUT" + key_header)

        # Quick annealing parameter reference
        csv_file.write(f"Running with t:{self.t}, t_min:{self.t_min}, k:{self.k}, cool:{self.cool},"
                       f" iterations:{self.iterations}, seed, {self.seed}, {self.network_path}")
        csv_file.close()
