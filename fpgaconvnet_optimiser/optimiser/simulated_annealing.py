import os
import sys
import numpy as np
import json
import copy
import random
import math
import logging
import uuid
from datetime import datetime

from fpgaconvnet_optimiser.optimiser.optimiser import Optimiser
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE
import fpgaconvnet_optimiser.tools.graphs as graphs

LATENCY   =0
THROUGHPUT=1

START_LOOP=1000

class SimulatedAnnealing(Optimiser):
    """
    Randomly chooses a transform and hardware component to change. The
    change is accepted based on a probability-based decision function
    """

    def __init__(self,name,network_path,T=10.0,k=0.001,T_min=0.0001,cool=0.97,
            iterations=10,transforms_config={},fix_starting_point_config={},
            data_width=16,weight_width=8,acc_width=30,fuse_bn=True,checkpoint=False,
            checkpoint_path="."):

        # Initialise Network
        Optimiser.__init__(self,name,network_path,transforms_config,
                fix_starting_point_config,data_width,weight_width,acc_width,fuse_bn)

        # Simulate Annealing Variables
        self.T          = T
        self.k          = k
        self.T_min      = T_min
        self.cool       = cool
        self.iterations = iterations

        # checkpoint directory routes
        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path

    def optimiser_status(self, return_char='\r'):
        # cost
        cost = self.get_cost()
        # resources
        resources = [ partition.get_resource_usage() for partition in self.partitions ]
        BRAM = max([ resource['BRAM'] for resource in resources ])
        DSP  = max([ resource['DSP']  for resource in resources ])
        LUT  = max([ resource['LUT']  for resource in resources ])
        FF   = max([ resource['FF']   for resource in resources ])
        # print the current status of the optimiser
        print(f"{self.T:.5e}\t{abs(self.get_cost()):.5e}\t\t  {int(BRAM):4d} | {int(DSP):4d} | {int(LUT):6d} | {int(FF):6d}",end=return_char)

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

        # print the header for the optimiser status
        objectives = ['latency (s)','throughput (fps)']
        objective  = objectives[self.objective]
        print(f"Temperature\t{objective}\t  BRAM | DSP  | LUT    | FF    ")

        # Cooling Loop
        cooltimes=0
        while self.T_min < self.T:

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
                ## Choose a random transform
                transform = random.choice(self.transforms)

                ## Choose a random partition
                partition_index = random.randint(0,len(self.partitions)-1)

                ## Choose a random node in partition
                node = random.choice(graphs.ordered_node_list(self.partitions[partition_index].graph))

                ## Apply the transform
                logging.info(f"applying {transform} to {node} in partition {partition_index}")
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
            if math.exp(min(0,(cost - self.get_cost())/(self.k*self.T))) < random.uniform(0,1):
                # revert to previous state
                self.partitions = partitions

            # update cost
            if self.DEBUG:
                self.optimiser_status()

            # reduce temperature
            self.T *= self.cool

        # end optimisation loop
        self.optimiser_status(return_char='\n')
        print("optimiser complete!")
