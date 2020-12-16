import numpy as np
import json
import copy
import random
import math

from models.Network import Network

LATENCY   =0
THROUGHPUT=1
POWER     =2

class Optimiser(Network):
    def __init__(self,name,network_path):
        # Initialise Network
        Network.__init__(self,name,network_path)

        self.objective   = 0
        self.constraints = {
            'latency'    : float("inf"),
            'throughput' : 0.0,
            'power'      : float("inf")
        }

        self.transforms = ['coarse','fine','partition','weights_reloading']

    """    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    METRICS    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def get_cost(self):
        # Latency objective
        if   self.objective == LATENCY:
            return self.get_latency()
        # Throughput objective
        elif self.objective == THROUGHPUT:
            return -self.get_throughput()
        # Power objective
        elif self.objective == POWER:
            return self.get_power_average()    
    
    """    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    CONSTRAINTS    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def check_constraints(self):
        assert self.get_latency()       <=  self.constraints['latency']  , "ERROR : (constraint violation) Latency constraint exceeded"
        assert self.get_throughput()    >= self.constraints['throughput'], "ERROR : (constraint violation) Throughput constraint exceeded"
        assert self.get_power_average() <= self.constraints['power']     , "ERROR : (constraint violation) Power constraint exceeded"

    """    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    TRANSFORMS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def apply_transform(self, transform, partition_index=None, node=None):

        # choose random partition index if not given
        if partition_index == None:
            partition_index = random.randint(0,len(self.partitions)-1)

        # choose random node in partition if given
        if node == None:
            node = random.choice([*self.partitions[partition_index]['graph']])

        # Apply a random transform
        ## Coarse transform (node_info transform)
        if transform == 'coarse':
            self.apply_random_coarse_layer(partition_index, node)
            return

        ## Fine transform (node_info transform)
        if transform == 'fine':
            self.apply_random_fine_layer(partition_index,node)
            return

        ## Weights-Reloading transform (partition transform)
        if transform == 'weights_reloading':
            ### apply random weights reloading
            self.apply_random_weights_reloading(partition_index)
            return

        ## Partition transform (partition transform)
        if transform == 'partition':
            ### apply random partition
            self.apply_random_partition(partition_index)
            return

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
        print("COST:\t {cost} ({objective}), RESOURCE:\t {BRAM}\t{DSP}\t{LUT}\t{FF}\t(BRAM|DSP|LUT|FF)".format(
            cost=cost,objective=objective,BRAM=int(BRAM),DSP=int(DSP),LUT=int(LUT),FF=int(FF)), end='\r')

    def get_optimal_batch_size(self):
        # change the batch size to zero
        self.batch_size = 1
        # update each partitions batch size
        self.update_batch_size()
        # calculate the maximum memory usage at batch 1
        max_mem = self.get_memory_usage_estimate()
        # update batch size to max
        self.batch_size = max(1,math.floor(self.platform['mem_capacity']/max_mem))
        self.update_batch_size()

    def run_optimiser(self, log=True):
        pass
