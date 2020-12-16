import numpy as np
import json
import copy
import random
import math

from optimiser.optimiser import Optimiser

LATENCY   =0
THROUGHPUT=1
POWER     =2

class SimulatedAnnealing(Optimiser):
    def __init__(self,name,network_path):
        # Initialise Network
        Optimiser.__init__(self,name,network_path)

        # Simulate Annealing Variables
        self.T          = 10.0
        self.k          = 0.01
        self.T_min      = 0.0001
        self.cool       = 0.01 
        self.iterations = 10

    """    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    OPTIMISER
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def optimiser_status(self):
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
        print("TEMP:\t {temp}, COST:\t {cost} ({objective}), RESOURCE:\t {BRAM}\t{DSP}\t{LUT}\t{FF}\t(BRAM|DSP|LUT|FF)".format(
            temp=self.T,cost=cost,objective=objective,BRAM=int(BRAM),DSP=int(DSP),LUT=int(LUT),FF=int(FF)))

    def run_optimiser(self, log=True):

        # Setup
        cost = self.get_cost()       

        # Check within resources 
        if not self.check_resources():
            print("ERROR: Exceeds resource usage")
            return

        # Check within constraints
        if not self.check_constraints():
            print("ERROR: Outside of constraints")
            return

        # Cooling Loop
        while self.T_min < self.T:

            # several iterations per cool down
            for _ in range(self.iterations):

                #print(self.partitions)

                self.update_modules()
                self.get_buffer_depths()

                # get the current cost
                cost = self.get_cost()
               
                # Save previous iteration
                partitions  = copy.deepcopy(self.partitions)
                node_info   = copy.deepcopy(self.node_info)

                # Apply a transform
                ## Choose a random transform
                transform = random.choice(self.transforms)

                ## Choose a random partition
                partition_index = random.randint(0,len(self.partitions)-1)
 
                ## Choose a random node in partition
                node = random.choice(list(self.partitions[partition_index]['graph']))
                
                ## Apply the transform
                self.apply_transform(transform, partition_index, node)

                # Check resources
                if not self.check_resources():
                    # revert to previous state
                    self.partitions = partitions
                    self.node_info  = node_info
                    continue

                # Check ports
                if not self.check_ports():
                    # revert to previous state
                    self.partitions = partitions
                    self.node_info  = node_info
                    continue

                # Check Constraints
                if not self.check_constraints():
                    # revert to previous state
                    self.partitions = partitions
                    self.node_info  = node_info
                    continue

                # Simulated annealing descision
                #print(math.exp(min(0,(cost - self.get_cost())/(self.k*self.T))))
                if math.exp(min(0,(cost - self.get_cost())/(self.k*self.T))) < random.uniform(0,1):
                    # revert to previous state
                    self.partitions = partitions
                    self.node_info  = node_info

                # update cost
                if self.DEBUG:
                    self.optimiser_status()
                cost = self.get_cost() 

            # reduce temperature
            self.T *= self.cool
