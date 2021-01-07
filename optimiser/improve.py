import numpy as np
import json
import copy
import random
import math
import sys

from optimiser.optimiser import Optimiser
from operator import itemgetter

LATENCY   =0
THROUGHPUT=1
POWER     =2

START_LOOP=1000

class Improve(Optimiser):
    def __init__(self,name,network_path):
        # Initialise Network
        Optimiser.__init__(self,name,network_path)

        # Simulate Annealing Variables
        self.T          = 10.0
        self.k          = 0.001
        self.T_min      = 0.0001
        self.cool       = 0.97
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
        resources = [ partition.get_resource_usage() for partition in self.partitions ]
        BRAM = max([ resource['BRAM'] for resource in resources ])
        DSP  = max([ resource['DSP']  for resource in resources ])
        LUT  = max([ resource['LUT']  for resource in resources ])
        FF   = max([ resource['FF']   for resource in resources ])
        sys.stdout.write("\033[K")
        print("TEMP:\t {temp}, COST:\t {cost} ({objective}), RESOURCE:\t {BRAM}\t{DSP}\t{LUT}\t{FF}\t(BRAM|DSP|LUT|FF)".format(
            temp=self.T,cost=cost,objective=objective,BRAM=int(BRAM),DSP=int(DSP),LUT=int(LUT),FF=int(FF)),end='\n')#,end='\r')

    def run_optimiser(self, log=True):
       
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
            print("ERROR: Exceeds resource usage")
            #print(error)
            #return
        
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

        try: 
            self.check_resources()
            self.check_constraints()
        except AssertionError as error:
            print("ERROR: Exceeds resource usage")
            #print(error)
            return
 
        # Cooling Loop
        while self.T_min < self.T:

            # several iterations per cool down
            for _ in range(self.iterations):
                #print(self.partitions)

                # update partitions
                self.update_partitions()

                # get the current cost
                cost = self.get_cost()
            
                # Save previous iteration
                partitions = copy.deepcopy(self.partitions)

                # remove all auxiliary layers
                for i in range(len(self.partitions)):
                    self.partitions[i].remove_squeeze()

                # Apply a transform
                ## Choose a random transform
                transform = random.choice(self.transforms)

                ## Choose slowest partition
                partition_latencys = [ partition.get_latency(self.platform["freq"]) for partition in self.partitions ]
                partition_index    = np.random.choice(np.arange(len(self.partitions)), 1, p=(partition_latencys/sum(partition_latencys)))[0]

                ## Choose slowest node in partition
                node_latencys = np.array([ self.partitions[partition_index].graph.nodes[layer]['hw'].get_latency() \
                        for layer in self.partitions[partition_index].graph.nodes() ])
                node = np.random.choice(list(self.partitions[partition_index].graph.nodes()), 1, p=(node_latencys/sum(node_latencys)))[0]
                
                ## Apply the transform
                self.apply_transform(transform, partition_index, node)
            
                ## Update partitions
                self.update_partitions()

                # Check resources
                try: 
                    #self.check_ports() # TODO
                    self.check_resources()
                    self.check_constraints()
                except AssertionError:
                    # revert to previous state
                    self.partitions = partitions
                    continue

                # Simulated annealing descision
                if math.exp(min(0,(cost - self.get_cost())/(self.k*self.T))) < random.uniform(0,1):
                    # revert to previous state
                    self.partitions = partitions

                # update cost
                if self.DEBUG:
                    self.optimiser_status()

            # reduce temperature
            self.T *= self.cool
