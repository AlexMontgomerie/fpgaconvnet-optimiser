import unittest
import ddt
import tools.parser as parser
import tools.matrix as matrix
import transforms.coarse            as coarse
import transforms.fine              as fine
import transforms.weights_reloading as weights_reloading
import transforms.partition         as partition
import models.Network as Network
from tools.layer_enum import LAYER_TYPE

from numpy.linalg import matrix_rank
import scipy
import numpy as np
import pprint
import random
np.seterr(divide='ignore', invalid='ignore')

"""    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
COARSE TRANSFORM 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

@ddt.ddt
class TestCoarseTransform(unittest.TestCase):

    @ddt.data(
        #"test/sw/data/single_layer.prototxt",
        #"test/sw/data/sequential.prototxt",
        #"test/sw/data/multipath.prototxt",
        "data/models/lenet.prototxt",
        "data/models/mobilenet.prototxt",
        #"data/models/googlenet.prototxt",
        "data/models/googlenet_short.prototxt"
    )
    def test_random_coarse(self,model_path):

        # graph definition
        graph, node_info = parser.parse_net(model_path,view=False)
        # apply several times
        for _ in range(100):
            
            # apply random coarse transform
            coarse.apply_random_coarse(graph,node_info)
 
            # get streams matrix
            streams_matrix = matrix.get_streams_matrix(graph,node_info)
            
            # check all the layer's coarse folding is in feasible coarse folding
            for layer in node_info:
                self.assertIn(node_info[layer]['hw'].coarse_in , node_info[layer]['hw'].get_coarse_in_feasible() )
                self.assertIn(node_info[layer]['hw'].coarse_out, node_info[layer]['hw'].get_coarse_out_feasible())

            # check the stream connections match
           # get the nullspace
            null_space = scipy.linalg.null_space(streams_matrix)
            # check the dimensions of the null space
            self.assertEqual(null_space.shape[0],len(matrix.get_node_list_matrix(graph)))
            self.assertEqual(null_space.shape[1],1)
            # check that all elements are the same
            self.assertEqual(len(np.unique(np.around(null_space,decimals=4))),1)

"""    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PARTITION TRANSFORM 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

@ddt.ddt
class TestPartitionTransform(unittest.TestCase):

    @ddt.data(
        #"test/sw/data/single_layer.prototxt",
        #"test/sw/data/sequential.prototxt",
        #"test/sw/data/multipath.prototxt",
        "data/models/lenet.prototxt",
        "data/models/mobilenet.prototxt",
        #"data/models/googlenet.prototxt",
        "data/models/googlenet_short.prototxt"
    )
    def test_random_partition(self,model_path):

        # graph definition
        net = Network.Network('test',model_path)
        # apply several times
        for _ in range(100):
            
            # apply a random partition
            partition.apply_random_partition(net.partitions,random.randint(0,len(net.partitions)-1))

            for i in net.partitions:
                # check no empty partitions
                self.assertIsNotNone(i)
                # check there is no contact at the start
                self.assertNotEqual(net.node_info[i["input_node"]]['type'],LAYER_TYPE.Concat)
                # check there is no contact at the start
                self.assertNotEqual(net.node_info[i["output_node"]]['type'],LAYER_TYPE.Split)

    @ddt.data(
        "data/models/lenet.prototxt",
        "data/models/mobilenet.prototxt"
    )
    def test_complete_partition(self,model_path):

        # graph definition
        net = Network.Network('test',model_path)
            
        # apply complete partition
        partition.complete_partition(net.partitions,0)

        # check number partitions is number of layers
        self.assertEqual(len(net.partitions), len(net.node_info.keys()))


