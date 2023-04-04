"""
Optimisations schemes are used to explore the transform design space and find an optimal mapping for a given hardware platform
"""

from .solver import Solver
from .improve import Improve
from .simulated_annealing import SimulatedAnnealing
from .greedy_partition import GreedyPartition
