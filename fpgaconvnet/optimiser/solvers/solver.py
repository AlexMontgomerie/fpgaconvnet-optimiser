import json
import copy
import random
import math
import numpy as np
from dataclasses import dataclass, field
import wandb
import uuid
import pickle
from datetime import datetime
from fpgaconvnet.tools import graphs

LATENCY   =0
THROUGHPUT=1

import fpgaconvnet.tools.graphs as graphs

from fpgaconvnet.models.network import Network
from fpgaconvnet.platform.Platform import Platform

import fpgaconvnet.optimiser.transforms.weights_reloading as weights_reloading
import fpgaconvnet.optimiser.transforms.partition as partition
import fpgaconvnet.optimiser.transforms.coarse as coarse
import fpgaconvnet.optimiser.transforms.fine as fine
import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE


@dataclass
class Solver:
    net: Network
    platform: Platform
    objective: int = THROUGHPUT
    constraints: dict = field(default_factory=lambda: {
        'latency'    : float("inf"), 'throughput' : 0.0})
    transforms: list = field(default_factory=lambda:[
        'coarse','fine','partition', 'weights_reloading'])
    rsc_allocation: float = 1.0
    multi_fpga: bool = False
    constrain_port_width: bool = True

    """
    Base class for all optimisation strategies. This inherits the `Network` class.

    Attributes
    ----------
    objective: int
        Objective for the solver. One of `LATENCY`, `THROUGHPUT` and `POWER`.
    constraints: dict
        dictionary containing constraints for `latency`, `throughput` and `power`.
    transforms: list
        list of transforms that can be applied to the network. Allowed transforms
        are `['coarse','fine','partition','weights_reloading']`
    """


    def get_transforms(self):
        self.transforms = []
        for transform_type, attr in self.transforms_config.items():
            if bool(attr["apply_transform"]):
                self.transforms.append(transform_type)

    def get_cost(self, partition_list=None):
        """
        calculates the cost function of the optimisation strategy at it's current state.
        This cost is based on the objective of the solver. There are three objectives
        that can be chosen:

        - `LATENCY (0)`
        - `THROUGHPUT (1)`
        - `POWER (2)`

        Returns:
        --------
        float
        """
        if partition_list == None:
            partition_list = list(range(len(self.net.partitions)))
        # Latency objective
        if   self.objective == LATENCY:
            return self.net.get_latency(self.platform.board_freq, self.multi_fpga, self.get_inter_delay(), partition_list)
        # Throughput objective
        elif self.objective == THROUGHPUT:
            return -self.net.get_throughput(self.platform.board_freq, self.multi_fpga, self.get_inter_delay(), partition_list)
        
    def get_inter_delay(self):
        """
        Calculates the interconnect delay between partitions.
        """
        if self.multi_fpga:
            return self.platform.eth_delay
        else:
            return self.platform.reconf_time

    def get_partition_resource(self, partition, bram_to_lut=True):
        lut_to_bram_ratio = 288 # BRAM: 18Kbits, LUT: 64bits
        partition_resource_usage = partition.get_resource_usage()
        if bram_to_lut:
            bram_shortage = math.ceil(partition_resource_usage['BRAM'] - self.ram_usage*self.platform.get_bram())
            lut_surplus = int((self.rsc_allocation*self.platform.get_lut() - partition_resource_usage['LUT'])/lut_to_bram_ratio)
            if bram_shortage > 0 and lut_surplus > 0:
                partition_resource_usage['BRAM'] -= min(bram_shortage, lut_surplus)
        return partition_resource_usage

    def check_partition_resources(self, partition):
        partition_resource_usage = self.get_partition_resource(partition)
        assert partition_resource_usage['FF']   <= \
                (self.rsc_allocation*self.platform.get_ff()), "ERROR: FF usage exceeded, partition: {partition_index}" 
        assert partition_resource_usage['LUT']  <= \
                (self.rsc_allocation*self.platform.get_lut()), "ERROR: LUT usage exceeded, partition: {partition_index}" 
        assert partition_resource_usage['DSP']  <= \
                (self.rsc_allocation*self.platform.get_dsp()) , "ERROR: DSP usage exceeded, partition: {partition_index}" 
        assert partition_resource_usage['BRAM'] <= \
                (self.ram_usage*self.platform.get_bram()), "ERROR: BRAM usage exceeded, partition: {partition_index}" 
        assert partition_resource_usage['URAM'] <= \
                (self.ram_usage*self.platform.get_uram()), "ERROR: URAM usage exceeded, partition: {partition_index}"

        bandwidth_total = partition.get_total_bandwidth(self.platform.board_freq)
        assert bandwidth_total <= self.rsc_allocation*self.platform.get_mem_bw(), "ERROR: Memory bandwidth exceeded, partition: {partition_index}"

    def check_resources(self):
        # iterate over partitions
        for partition_index, partition in enumerate(self.net.partitions):
            # get the resource usage for the platform
            self.check_partition_resources(partition)

    def check_constraints(self):
        """
        function to check the performance constraints of the network. Checks
        `latency` and `throughput`.

        Raises
        ------
        AssertionError
            If not within performance constraints
        """
        assert self.net.get_latency(self.platform.board_freq, self.multi_fpga, self.get_inter_delay()) <= self.constraints['latency'], \
                "ERROR : (constraint violation) Latency constraint exceeded"
        assert self.net.get_throughput(self.platform.board_freq, self.multi_fpga, self.get_inter_delay()) >= self.constraints['throughput'], \
                "ERROR : (constraint violation) Throughput constraint exceeded"

    def apply_transform(self, transform, partition_index=None, node=None,
            iteration=None, cooltimes=None):
        """
        function to apply chosen transform to the network. Partition index
        and node can be specified. If not, a random partition and node is
        chosen.

        Parameters
        ----------
        transform: str
            string corresponding to chosen transform. Must be within the
            following `["coarse","fine","weights_reloading","partition"]`
        partition_index: int default: None
            index for partition to apply transform. Must be within
            `len(self.partitions)`
        node: str
            name of node to apply transform. Must be within
            `self.partitions[partition_index].graph`
        """
        # choose random partition index if not given
        if partition_index == None:
            partition_index = random.randint(0,len(self.net.partitions)-1)

        # choose random node in partition if given
        if node == None:
            node = random.choice(graphs.ordered_node_list(
                self.net.partitions[partition_index].graph))

        # Apply a random transform
        ## Coarse transform (node_info transform)
        if transform == 'coarse':
            coarse.apply_random_coarse_node(
                self.net.partitions[partition_index], node)
            return

        ## Fine transform (node_info transform)
        if transform == 'fine':
            fine.apply_random_fine_node(
                self.net.partitions[partition_index], node)
            return

        ## Weights-Reloading transform (partition transform)
        if transform == 'weights_reloading':
            ### apply random weights reloading
            weights_reloading.apply_random_weights_reloading(
                self.net.partitions[partition_index])
            return

        ## Partition transform (partition transform)
        if transform == 'partition':
            ### apply random partition
            # remove squeeze layers prior to partitioning
            self.net.partitions[partition_index].remove_squeeze()
            partition.apply_random_partition(
                self.net, partition_index)
            return

    def solver_status(self):
        """
        prints out the current status of the solver.
        """
        # objective
        objectives = [ 'latency', 'throughput','power']
        objective  = objectives[self.objective]
        # cost
        cost = self.get_cost()
        # Resources
        resources = [ partition.get_resource_usage() for partition in self.net.partitions ]
        BRAM = max([ resource['BRAM'] for resource in resources ])
        DSP  = max([ resource['DSP']  for resource in resources ])
        LUT  = max([ resource['LUT']  for resource in resources ])
        FF   = max([ resource['FF']   for resource in resources ])
        BW   = max([ partition.get_total_bandwidth(self.platform.board_freq) for partition in self.net.partitions ])
        if self.platform.get_uram() > 0:
            URAM = max([ resource['URAM'] for resource in resources ])
            print("COST:\t {cost} ({objective}), RESOURCE:\t {URAM}\t{BRAM}\t{DSP}\t{LUT}\t{FF}\t<>\t{BW:.2f}\t(URAM|BRAM|DSP|LUT|FF<>Bandwidth)".format(
                cost=cost,objective=objective,URAM=int(URAM),BRAM=int(BRAM),DSP=int(DSP),LUT=int(LUT),FF=int(FF),BW=BW), end='\n')
        else:
            print("COST:\t {cost} ({objective}), RESOURCE:\t {BRAM}\t{DSP}\t{LUT}\t{FF}\t<>\t{BW:.2f}\t(BRAM|DSP|LUT|FF<>Bandwidth)".format(
                cost=cost,objective=objective,BRAM=int(BRAM),DSP=int(DSP),LUT=int(LUT),FF=int(FF),BW=BW), end='\n')

    def save_design_checkpoint(self, output_path):
        # pickle the current optimiser state
        checkpoint = pickle.dumps(self)
        # save to output path
        with open(output_path, "wb") as f:
            f.write(checkpoint)

    def wandb_checkpoint(self):
        # pickle optimiser object
        checkpoint = pickle.dumps(self)
        checkpoint_path = f"checkpoint/{uuid.uuid4().hex}.dcp"
        with open(checkpoint_path,"wb") as f:
            f.write(checkpoint)
        # create a wandb artifact
        artifact_timestamp = datetime.now().strftime("%m-%d-%Y-%H.%M.%S.%f")
        # artifact_name = f"checkpoint-{artifact_timestamp}"
        artifact_name = f"checkpoint"
        artifact = wandb.Artifact(artifact_name, type="dcp")
        artifact.add_file(checkpoint_path)
        # log the artifact
        wandb.log_artifact(artifact)

    def wandb_log(self, **kwargs):
        # get common log values
        wandb_log = {
            "latency": self.net.get_latency(self.platform.board_freq, self.multi_fpga, self.get_inter_delay()),
            "throughput": self.net.get_throughput(self.platform.board_freq, self.multi_fpga, self.get_inter_delay()),
            "num_partitions": len(self.net.partitions),
        }
        # add extra log values
        wandb_log.update(kwargs)
        # update wandb log
        wandb.log(wandb_log)

    def get_optimal_batch_size(self):
        """
        gets an approximation of the optimal (largest) batch size for throughput-based
        applications. This is dependant on the platform's memory capacity. Updates the
        `self.batch_size` variable.
        """
        # change the batch size to zero
        self.net.batch_size = 1
        # update each partitions batch size
        self.net.update_batch_size()
        # calculate the maximum memory usage at batch 1
        max_mem = self.net.get_memory_usage_estimate()
        # update batch size to max
        self.net.batch_size = max(1,math.floor(self.platform['mem_capacity']/max_mem))
        self.net.update_batch_size()

    def run_solver(self, log=True):
        """
        template for running the solver.
        """
        raise RuntimeError("solver not implemented!")

    def set_double_buffer_weights(self):
        for partition in self.net.partitions:
            for node in partition.graph.nodes():
                if partition.graph.nodes[node]['type'] in [LAYER_TYPE.Convolution, LAYER_TYPE.InnerProduct]:
                    partition.graph.nodes[node]['hw'].double_buffered = True

    def update_io_port_width(self):
        if not self.constrain_port_width:
            return

        port_width = self.platform.eth_port_width if self.multi_fpga else self.platform.port_width
        for partition in self.net.partitions:
            ## remove auxiliary layers
            partition.remove_squeeze()

            max_streams_in = partition.ports_in*int(port_width/partition.data_width)
            max_streams_out = partition.ports_out*int(port_width/partition.data_width)

            ## update streams in
            partition.streams_in = []
            inputs = graphs.get_input_nodes(partition.graph)
            for i, input_node in enumerate(inputs):
                ## get valid streams in
                streams_in_valid = partition.graph.nodes[input_node]["hw"].get_coarse_in_feasible()
                # get the max stream values in
                streams_in_max = min(max_streams_in//len(inputs), partition.graph.nodes[input_node]["hw"].streams_in())
                # choose the max of all the valid stream values, below the max
                partition.streams_in.append(max([ s for s in streams_in_valid if s <= streams_in_max ]))
                
            ## update streams out
            partition.streams_out = []
            outputs = graphs.get_output_nodes(partition.graph)
            for i, output_node in enumerate(outputs):
                ## get valid streams out
                streams_out_valid = partition.graph.nodes[output_node]["hw"].get_coarse_out_feasible()
                # get the max stream values out
                streams_out_max = min(max_streams_out//len(outputs), partition.graph.nodes[output_node]["hw"].streams_out())
                # choose the max of all the valid stream values, below the max
                partition.streams_out.append(max([ s for s in streams_out_valid if s <= streams_out_max ]))    

            ## add auxiliary layers
            partition.add_squeeze()


    def update_partitions(self):
        self.update_io_port_width()
        self.net.update_partitions()


    from fpgaconvnet.optimiser.solvers.report import create_report