import glob
import unittest
import ddt
import fpgaconvnet_optimiser.tools.matrix as matrix
import fpgaconvnet_optimiser.tools.parser as parser
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

from numpy.linalg import matrix_rank
import scipy
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CONNECTION MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
@ddt.ddt
class TestConnectionMatrix(unittest.TestCase):

    # @ddt.data(*NETWORKS)
    @ddt.data(*glob.glob("tests/models/*.onnx"))
    def test_net(self,model_path):
        # graph definition
        _, graph = parser.parse_net(model_path,view=False)

        # get matrix and expected dimensions
        n_nodes             = len(matrix.get_node_list_matrix(graph))
        n_edges             = len(matrix.get_edge_list_matrix(graph))
        connections_matrix  = matrix.get_connections_matrix(graph)

        # check dimension of matrix
        self.assertEqual(connections_matrix.shape[0],n_edges)
        self.assertEqual(connections_matrix.shape[1],n_nodes)

        # check rank of matrix
        self.assertEqual(matrix_rank(connections_matrix),n_nodes-1)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
STREAMS MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

@ddt.ddt
class TestStreamsMatrix(unittest.TestCase):

    @ddt.data(*glob.glob("tests/models/*.onnx"))
    def test_net(self,model_path):

        # graph definition
        _, graph = parser.parse_net(model_path,view=False)

        # get matrix and expected dimensions
        n_nodes         = len(matrix.get_node_list_matrix(graph))
        n_edges         = len(matrix.get_edge_list_matrix(graph))
        streams_matrix  = matrix.get_streams_matrix(graph)

        # check dimension of matrix
        self.assertEqual(streams_matrix.shape[0],n_edges)
        self.assertEqual(streams_matrix.shape[1],n_nodes)

        # check rank of matrix
        self.assertEqual(matrix_rank(streams_matrix),n_nodes-1)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RATES MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
@ddt.ddt
class TestRatesMatrix(unittest.TestCase):

    @ddt.data(*glob.glob("tests/models/*.onnx"))
    def test_net(self,model_path):

        # graph definition
        _, graph = parser.parse_net(model_path,view=False)

        # get matrix and expected dimensions
        n_nodes         = len(matrix.get_node_list_matrix(graph))
        n_edges         = len(matrix.get_edge_list_matrix(graph))
        rates_matrix    = matrix.get_rates_matrix(graph)

        # check dimension of matrix
        self.assertEqual(rates_matrix.shape[0],n_edges)
        self.assertEqual(rates_matrix.shape[1],n_nodes)

        # check rank of matrix
        self.assertEqual(matrix_rank(rates_matrix),n_nodes-1)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RATES BALANCED MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

# @ddt.ddt
# class TestBalancedRatesMatrix(unittest.TestCase):

#     @ddt.data(
#         "examples/models/lenet.onnx",
#     )
#     def test_net(self,model_path):

#         # graph definition
#         graph, node_info = parser.parse_net(model_path,view=False)

#         # get matrix and expected dimensions
#         n_nodes                 = len(matrix.get_node_list_matrix(graph))
#         n_edges                 = len(matrix.get_edge_list_matrix(graph))
#         balanced_rates_matrix   = matrix.get_balanced_rates_matrix(graph,node_info)
#         # check dimension of matrix
#         self.assertEqual(balanced_rates_matrix.shape[0],n_edges)
#         self.assertEqual(balanced_rates_matrix.shape[1],n_nodes)

#         # check rank of matrix
#         self.assertEqual(matrix_rank(balanced_rates_matrix),n_nodes-1)

#         rates_matrix = matrix.get_rates_matrix(graph,node_info)
#         # check rate ratios
#         ## iterate over columns
#         for col_index in range(rates_matrix.shape[1]):
#             node = matrix.get_node_list_matrix(graph)[col_index]
#             if (node in node_info) and (node_info[node]['type'] == LAYER_TYPE.Convolution):
#                 ## go over every in out combination
#                 for row_index in range(rates_matrix.shape[0]):
#                     for i in range(rates_matrix.shape[0]):
#                         ratio            = np.nan_to_num(rates_matrix[row_index,col_index] / rates_matrix[i,col_index])
#                         ratio_balanced   = np.nan_to_num(balanced_rates_matrix[row_index,col_index] / balanced_rates_matrix[i,col_index])
#                         self.assertAlmostEqual(ratio,ratio_balanced)

#         # check the edges all have the same rate
#         for row_index in range(rates_matrix.shape[0]):
#             self.assertEqual(np.sum(balanced_rates_matrix[row_index,:]), 0.0)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
WORKLOAD MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""
@ddt.ddt
class TestWorkloadMatrix(unittest.TestCase):

    @ddt.data(*glob.glob("tests/models/*.onnx"))
    def test_net(self,model_path):

        # graph definition
        _, graph = parser.parse_net(model_path,view=False)

        # get matrix and expected dimensions
        n_nodes         = len(matrix.get_node_list_matrix(graph))
        n_edges         = len(matrix.get_edge_list_matrix(graph))
        workload_matrix = matrix.get_workload_matrix(graph)

        # check dimension of matrix
        self.assertEqual(workload_matrix.shape[0],n_edges)
        self.assertEqual(workload_matrix.shape[1],n_nodes)

        # check rank of matrix
        self.assertEqual(matrix_rank(workload_matrix),n_nodes-1)

        # check edges all have the same workload
        for i in range(workload_matrix.shape[0]):
            self.assertEqual(np.sum(workload_matrix[i,:]), 0,
                    f"Mismatching workload for edge: {matrix.get_edge_list_matrix(graph)[i]}")

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TOPOLOGY MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

@ddt.ddt
class TestTopologyMatrix(unittest.TestCase):

    @ddt.data(*glob.glob("tests/models/*.onnx"))
    def test_net(self,model_path):

        # graph definition
        _, graph = parser.parse_net(model_path,view=False)

        # get matrix and expected dimensions
        n_nodes         = len(matrix.get_node_list_matrix(graph))
        n_edges         = len(matrix.get_edge_list_matrix(graph))
        topology_matrix = matrix.get_topology_matrix(graph)

        # check dimension of matrix
        self.assertEqual(topology_matrix.shape[0],n_edges)
        self.assertEqual(topology_matrix.shape[1],n_nodes)

        # check rank of matrix
        self.assertEqual(matrix_rank(topology_matrix),n_nodes-1)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
INTERVAL MATRIX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

@ddt.ddt
class TestIntervalMatrix(unittest.TestCase):

    @ddt.data(*glob.glob("tests/models/*.onnx"))
    def test_net(self,model_path):

        # graph definition
        _, graph = parser.parse_net(model_path,view=False)

        # get matrix and expected dimensions
        n_nodes         = len(matrix.get_node_list_matrix(graph))
        n_edges         = len(matrix.get_edge_list_matrix(graph))
        interval_matrix = matrix.get_interval_matrix(graph)

        # check dimension of matrix
        self.assertEqual(interval_matrix.shape[0],n_edges)
        self.assertEqual(interval_matrix.shape[1],n_nodes)

        # check rank of matrix
        self.assertEqual(matrix_rank(interval_matrix),n_nodes-1)

        # check theres only one value
        #interval = np.unique(np.absolute(interval_matrix).astype(int)).tolist()
        #self.assertEqual(len(interval),2)

