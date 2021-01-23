import unittest
import fpgaconvnet_optimiser.tools.graphs as graphs
import fpgaconvnet_optimiser.tools.parser as parser
from numpy.linalg import matrix_rank
import scipy
import numpy as np

class TestGraphInverse(unittest.TestCase):

    def test_single_layer(self):
        # graph definition
        graph = {
            'layer' : []
        }

        # get the inverse
        graph_inv = graphs.get_graph_inv(graph)

        # assert node is there
        self.assertIn('layer', graph_inv)
        # assert connection
        self.assertEqual(graph_inv['layer'],[])

    def test_sequential(self):
        # graph definition
        graph = {
            'a' : ['b'],
            'b' : ['c'],
            'c' : []
        }

        # get the inverse
        graph_inv = graphs.get_graph_inv(graph)

        # assert node is there
        for node in graph:
            self.assertIn(node, graph_inv)
        # assert connection
        self.assertEqual(graph_inv['c'],['b'])
        self.assertEqual(graph_inv['b'],['a'])
        self.assertEqual(graph_inv['a'],[])

    def test_multipath(self):
        # graph definition
        graph = {
            'a' : ['b','c'],
            'b' : ['c'],
            'c' : ['d','e'],
            'd' : ['g'],
            'e' : ['f'],
            'f' : ['g'],
            'g' : []
        }

        # get the inverse
        graph_inv = graphs.get_graph_inv(graph)

        # assert node is there
        for node in graph:
            self.assertIn(node, graph_inv)
        # assert connection
        self.assertIn('f', graph_inv['g'])
        self.assertIn('d', graph_inv['g'])
        self.assertIn('e', graph_inv['f'])
        self.assertIn('c', graph_inv['e'])
        self.assertIn('c', graph_inv['d'])
        self.assertIn('b', graph_inv['c'])
        self.assertIn('a', graph_inv['c'])
        self.assertIn('a', graph_inv['b'])
        self.assertEqual(graph_inv['a'],[])

if __name__ == '__main__':
    unittest.main()
