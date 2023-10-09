import sys
import numpy as np
import json
import copy
import random
import math

from fpgaconvnet.optimiser.solvers import Solver
from ortools.sat.python import cp_model

LATENCY   =0
THROUGHPUT=1

@dataclass
class IntegerLinearProgramming(Solver):
    def __post_init__(self):
        assert False, "this solver is not yet fully implemented" # TODO

        # initialise the solver
        # self.solver = pywraplp.Solver.CreateSolver('SCIP')
        self.sat_model = cp_model.CpModel()

        # create the variables for the solver
        self.nodes = [ node for node in self.net.partitions[0].graph ]
        self.channels_in  = [self.net.partitions[0].graph.nodes[node]['hw'].channels_in() for node in self.nodes]
        self.channels_out = [self.net.partitions[0].graph.nodes[node]['hw'].channels_out() for node in self.nodes]

        ## create the streams (coarse_in, coarse_out) variables
        self.streams_in = [ self.sat_model.NewIntVar(0, self.channels_in[i], "s_in_{i}") for i in range(len(self.channels_in)) ]
        self.streams_out = [ self.sat_model.NewIntVar(0, self.channels_out[i], "s_out_{i}") for i in range(len(self.channels_out)) ]


        # create the constraints for the solver

        ## streams constraint
        for i in range(len(self.streams_in)):
            self.sat_model.AddModuloEquality(self.channels_in[i], self.streams_in[i], 0)
        for i in range(len(self.streams_out)):
            self.sat_model.AddModuloEquality(self.channels_out[i], self.streams_out[i], 0)

        ## add an extra constraint on the input and output nodes so that they fit within the interface's number of streams
        # self.sat_model.Add(self.streams_in[0] <= self.net.partitions[0].streams_in)
        # self.sat_model.Add(self.streams_out[-1] <= self.net.partitions[0].streams_out)

        ## add the resource constraint
        # TODO

        # define the objective
        self.sat_model.Minimize(self.get_objective())

    def update_with_variables(self):
        # update the streams
        for i in range(len(self.streams_in)):
            self.net.partitions[0].graph.nodes[self.nodes[i]]["hw"].coarse_in = self.streams_in[i]
        for i in range(len(self.streams_out)):
            self.net.partitions[0].graph.nodes[self.nodes[i]]["hw"].coarse_out = self.streams_out[i]

        # update the partition
        self.update_partitions()

    def get_objective(self):

        # update with variables
        self.update_with_variables()

        # return throughput
        return -self.get_throughput()

    def optimiser_status(self):
        print("variables: ", self.solver.NumVariables())
        print("constraints: ", self.solver.NumConstraints())

    def run_optimiser(self, log=True):
        pass