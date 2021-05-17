import unittest
import ddt

from fpgaconvnet_optimiser.models.network import Network
from fpgaconvnet_optimiser.tools.layer_enum import LAYER_TYPE

from numpy.linalg import matrix_rank
import scipy
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

NETWORKS = [
    "examples/models/lenet.onnx",
    "examples/models/alexnet.onnx",
    "examples/models/vgg16.onnx",
    "examples/models/caffenet.onnx",
    "examples/models/caffenet.onnx",
]

PLATFORM = "examples/platforms/zedboard.json"

class TestNetworkTemplate():

    def run_test_validation(self,network):
        # run all validation checks
        network.check_ports()
        network.check_workload()
        network.check_streams()
        network.check_partitions()
        network.check_memory_bandwidth()

@ddt.ddt
class TestNetwork(TestNetworkTemplate, unittest.TestCase):

    @ddt.data(*NETWORKS)
    def test_network(self, network_path):
        # initialise network
        net = Network("test", network_path)
        # load platform
        net.update_platform(PLATFORM)
        # run all tests
        self.run_test_validation(net)
