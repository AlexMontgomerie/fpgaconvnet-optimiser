import unittest
import ddt
from optimiser.simulated_annealing import SimulatedAnnealing
import transforms.partition         as partition
import transforms.weights_reloading as weights_reloading
from tools.layer_enum import LAYER_TYPE

from numpy.linalg import matrix_rank
import scipy
import numpy as np
import json
np.seterr(divide='ignore', invalid='ignore')

"""    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
VALIDATE DESIGNS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

@ddt.ddt
class TestValidDesigns(unittest.TestCase):

    @ddt.data(
        "test/sw/data/single_layer.prototxt",
        "test/sw/data/sequential.prototxt",
        "test/sw/data/multipath.prototxt",
        "data/models/mobilenet.prototxt",
        "data/models/alexnet.prototxt",
        #"data/models/googlenet.prototxt",
        "data/models/googlenet_short.prototxt"
    )
    def test_validate_designs(self,model_path):

        # load network
        net = SimulatedAnnealing('test',model_path)
        net.DEBUG = False

        # get coefficients
        net.update_coefficients()

        # get platform
        with open('data/platforms/zc706.json','r') as f:
            platform = json.load(f)
        net.platform['constraints']['FF']   = platform['FF_max']
        net.platform['constraints']['DSP']  = platform['DSP_max']
        net.platform['constraints']['LUT']  = platform['LUT_max']
        net.platform['constraints']['BRAM'] = platform['BRAM_max']

        net.objective = 1
        batch_size    = 254 

        # partition completely
        partition.complete_partition(net.partitions,0)
        
        # complete weights reloading
        weights_reloading.apply_complete_weights_reloading(net.graph, net.node_info)

        # run optimiser
        net.run_optimiser()

        # check the correct number of streams
        for p in net.partitions:
            self.assertLessEqual(
                net.node_info[p['input_node']]['hw'].coarse_in + net.node_info[p['output_node']]['hw'].coarse_out,
                platform['ports'] )
            self.assertEqual(net.node_info[p['input_node']]['hw'].coarse_in, 1)
            self.assertEqual(net.node_info[p['output_node']]['hw'].coarse_out, 1)


        # check resource constraints

       
