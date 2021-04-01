import sys
import numpy as np
import json
import copy
import random
import math

from fpgaconvnet_optimiser.optimiser.optimiser import Optimiser
import fpgaconvnet_optimiser.tools.graphs as graphs

LATENCY   =0
THROUGHPUT=1

START_LOOP=1000

class GreedyPartition(Optimiser):
    """
Randomly chooses a transform and hardware component to change. The change is accepted based on a probability-based decision function
    """
    
    def __init__(self,name,network_path,T=10.0,k=0.001,T_min=0.0001,cool=0.97,iterations=10,transforms_config={},fix_starting_point_config={},data_width=16,weight_width=8,acc_width=30):

        # Initialise Network
        Optimiser.__init__(self,name,network_path,transforms_config,fix_starting_point_config,data_width,weight_width,acc_width)

        # Simulate Annealing Variables
        self.T          = T
        self.k          = k
        self.T_min      = T_min
        self.cool       = cool
        self.iterations = iterations

    def optimiser_status(self):
        # objective
        objectives = ['latency','throughput']
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
            print("ERROR: Exceeds resource usage (trying to find valid starting point)")
            bad_partitions = self.get_resources_bad_partitions()
        
        # Attempt to find a good starting point
        if not start:
            transforms_config = self.transforms_config
            self.transforms_config = self.fix_starting_point_config
            self.get_transforms()

            for i in range(START_LOOP):  
                transform = random.choice(self.transforms)
                partition_index = list(bad_partitions.keys())[-1]
                self.apply_transform(transform,partition_index)
                self.update_partitions()

                try:
                    self.check_resources()
                    self.check_constraints()
                    self.transforms_config = transforms_config
                    self.get_transforms()
                    break
                except AssertionError as error:
                    bad_partitions = self.get_resources_bad_partitions()

        try: 
            self.check_resources()
            self.check_constraints()
        except AssertionError as error:
            print("ERROR: Exceeds resource usage")
            return
        
        #return

        assert "partition" not in self.transforms
        

        # Cooling Loop
        T_start = self.T
        for partition_index in range(len(self.partitions)):
            self.T = T_start
            while self.T_min < self.T:
                
                # update partitions
                self.update_partitions()

                # get the current cost
                cost = self.get_cost([partition_index])
                
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
    
                    ## Choose a random node in partition
                    node = random.choice(graphs.ordered_node_list(self.partitions[partition_index].graph))
                    
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
                if math.exp(min(0,(cost - self.get_cost([partition_index]))/(self.k*self.T))) < random.uniform(0,1):
                    # revert to previous state
                    self.partitions = partitions

                # reduce temperature
                self.T *= self.cool

            # update cost
            if self.DEBUG:
                print(partition_index,"throughput:",self.get_cost([partition_index]))
                self.optimiser_status()
