import sys
import numpy as np
import json
import copy
import random
import math

from fpgaconvnet_optimiser.optimiser.optimiser import Optimiser
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE
LATENCY   =0
THROUGHPUT=1

START_LOOP=1000

class SimulatedAnnealing(Optimiser):
    """
Randomly chooses a transform and hardware component to change. The change is accepted based on a probability-based decision function
    """
    
    def __init__(self,name,network_path,T=10.0,k=0.001,T_min=0.0001,cool=0.97,iterations=10,wordlength=16):

        # Initialise Network
        Optimiser.__init__(self,name,network_path)

        # Simulate Annealing Variables
        self.T          = T
        self.k          = k
        self.T_min      = T_min
        self.cool       = cool
        self.iterations = iterations
        self.wordlength = wordlength

    def optimiser_status(self):
        # objective
        objectives = ['latency','throughput']
        objective  = objectives[self.objective]
        # cost 
        cost = self.get_cost()
        #for partition_index in range(len(self.partitions)):    
            #for layer in self.partitions[partition_index].graph.nodes():  
                  #print(self.partitions[partition_index].graph.nodes[layer]['hw'].data_width)
        # Resources        
        resources = [ partition.get_resource_usage() for partition in self.partitions ]
        BRAM = max([ resource['BRAM'] for resource in resources ])
        DSP  = max([ resource['DSP']  for resource in resources ])
        LUT  = max([ resource['LUT']  for resource in resources ])
        FF   = max([ resource['FF']   for resource in resources ])
        sys.stdout.write("\033[K")
        print("TEMP:\t {temp}, COST:\t {cost} ({objective}), RESOURCE:\t {BRAM}\t{DSP}\t{LUT}\t{FF}\t(BRAM|DSP|LUT|FF) wordlength:\t{wordlength}".format(
            temp=self.T,cost=cost,objective=objective,BRAM=int(BRAM),DSP=int(DSP),LUT=int(LUT),FF=int(FF),wordlength=int(self.wordlength)),end='\n')#,end='\r')

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
        
        # Attempt to find a good starting point
        print(START_LOOP)
        if not start:
            for i in range(START_LOOP):
                print('The %dth iteration' %(i))
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
            return         
        # Cooling Loop
        cooltimes=0
        while self.T_min < self.T:
            
            # update partitions
            self.update_partitions()

            # get the current cost
            cost = self.get_cost()
          
            # Save previous iteration
            partitions = copy.deepcopy(self.partitions)
            wordlength=self.wordlength
            
            iteration=0
            # several iterations per cool down
            for _ in range(self.iterations):

                # update partitions
                self.update_partitions()

                # remove all auxiliary layers
                for i in range(len(self.partitions)):
                    self.partitions[i].remove_squeeze()

                # Apply a transform
                ## Choose a random transform
                while 1:
                      transform = random.choice(self.transforms)
                      if iteration==0:
                          break
                      if transform!='wordlength':
                          break

                ## Choose a random partition
                partition_index = random.randint(0,len(self.partitions)-1)
 
                ## Choose a random node in partition
                node = random.choice(list(self.partitions[partition_index].graph))
                
                ## Apply the transform
                self.apply_transform(transform, partition_index, node,iteration,cooltimes)
                
                ## Update partitions
                self.update_partitions()
                if transform == 'wordlength':
                      iteration+=1
            
            for partition_index in range(len(self.partitions)):    
                  for layer in self.partitions[partition_index].graph.nodes():   
                      self.partitions[partition_index].graph.nodes[layer]['hw'].data_width=self.wordlength  
                      if self.partitions[partition_index].graph.nodes[layer]['type'] == LAYER_TYPE.Convolution or self.partitions[partition_index].graph.nodes[layer]['type'] == LAYER_TYPE.InnerProduct:   
                          self.partitions[partition_index].graph.nodes[layer]['hw'].weight_width=self.wordlength
                          self.partitions[partition_index].graph.nodes[layer]['hw'].acc_width=2*self.wordlength 
                      self.partitions[partition_index].graph.nodes[layer]['hw'].update()  
            # Check resources            
            try:        
                self.check_resources()               
                self.check_constraints()
            except AssertionError:
                # revert to previous state
                self.partitions = partitions
                self.wordlength = wordlength
                #for partition_index in range(len(self.partitions)):    
                  #for layer in self.partitions[partition_index].graph.nodes():   
                      #self.partitions[partition_index].graph.nodes[layer]['hw'].data_width=self.wordlength  
                      #if self.partitions[partition_index].graph.nodes[layer]['type'] == LAYER_TYPE.Convolution or self.partitions[partition_index].graph.nodes[layer]['type'] == LAYER_TYPE.InnerProduct:   
                          #self.partitions[partition_index].graph.nodes[layer]['hw'].weight_width=self.wordlength
                          #self.partitions[partition_index].graph.nodes[layer]['hw'].acc_width=2*self.wordlength 
                      #self.partitions[partition_index].graph.nodes[layer]['hw'].update()             
                continue
                  

            # Simulated annealing descision
            if math.exp(min(0,(cost - self.get_cost())/(self.k*self.T))) < random.uniform(0,1):
                # revert to previous state
                self.partitions = partitions
                self.wordlength = wordlength
                #for partition_index in range(len(self.partitions)):    
                  #for layer in self.partitions[partition_index].graph.nodes():   
                      #self.partitions[partition_index].graph.nodes[layer]['hw'].data_width=self.wordlength  
                      #if self.partitions[partition_index].graph.nodes[layer]['type'] == LAYER_TYPE.Convolution or self.partitions[partition_index].graph.nodes[layer]['type'] == LAYER_TYPE.InnerProduct:   
                          #self.partitions[partition_index].graph.nodes[layer]['hw'].weight_width=self.wordlength
                          #self.partitions[partition_index].graph.nodes[layer]['hw'].acc_width=2*self.wordlength 
                      #self.partitions[partition_index].graph.nodes[layer]['hw'].update()                  

            # update cost
            if self.DEBUG:
                self.optimiser_status()
                
            # reduce temperature
            self.T *= self.cool
            cooltimes+=1
