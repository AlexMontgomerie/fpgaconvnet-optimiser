import unittest
import ddt
import json
from fpgaconvnet_optimiser.models.layers import *

class TestLayerTemplate():

    def run_test_dimensions(self, layer):
        # check input dimensions
        self.assertTrue(layer.rows_in(0) > 0) 
        self.assertTrue(layer.cols_in(0) > 0) 
        self.assertTrue(layer.channels_in(0) > 0) 
        # check output dimensions
        self.assertTrue(layer.rows_out(0) > 0) 
        self.assertTrue(layer.cols_out(0) > 0) 
        self.assertTrue(layer.channels_out(0) > 0) 
    
    def run_test_rates(self, layer):
        # check rate in
        self.assertTrue(layer.rate_in(0) >= 0.0) 
        self.assertTrue(layer.rate_in(0) <= 1.0) 
        # check rate out
        self.assertTrue(layer.rate_out(0) >= 0.0) 
        self.assertTrue(layer.rate_out(0) <= 1.0) 


@ddt.ddt
class TestPoolingLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/layers/pooling/config_0.json",
        "tests/configs/layers/pooling/config_1.json",
        "tests/configs/layers/pooling/config_2.json",
        "tests/configs/layers/pooling/config_3.json",
        "tests/configs/layers/pooling/config_4.json",
        "tests/configs/layers/pooling/config_5.json",
        "tests/configs/layers/pooling/config_6.json",
        "tests/configs/layers/pooling/config_7.json",
        "tests/configs/layers/pooling/config_8.json",
        "tests/configs/layers/pooling/config_9.json",
        "tests/configs/layers/pooling/config_10.json",
        "tests/configs/layers/pooling/config_11.json",
        "tests/configs/layers/pooling/config_12.json",
    )
    def test_layer_configurations(self, config_path):
        
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = PoolingLayer(
            config["rows"],
            config["cols"],
            config["channels"],
            config["coarse"],
            config["coarse"],
            k_size=config["kernel_size"],
            stride=config["stride"],
            pad=config["pad"],
        )
        
        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)

@ddt.ddt
class TestConvolutionLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/layers/convolution/config_0.json",
        "tests/configs/layers/convolution/config_1.json",
        "tests/configs/layers/convolution/config_2.json",
        "tests/configs/layers/convolution/config_3.json",
        "tests/configs/layers/convolution/config_4.json",
        "tests/configs/layers/convolution/config_7.json",
        "tests/configs/layers/convolution/config_8.json",
        "tests/configs/layers/convolution/config_9.json",
        "tests/configs/layers/convolution/config_10.json",
        "tests/configs/layers/convolution/config_11.json",
        "tests/configs/layers/convolution/config_12.json",
        "tests/configs/layers/convolution/config_13.json",
        "tests/configs/layers/convolution/config_14.json",
        "tests/configs/layers/convolution/config_15.json",
        "tests/configs/layers/convolution/config_16.json",
        "tests/configs/layers/convolution/config_17.json",
        "tests/configs/layers/convolution/config_18.json",
        "tests/configs/layers/convolution/config_19.json",
    )
    def test_layer_configurations(self, config_path):
        
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = ConvolutionLayer(
            config["filters"],
            config["rows"],
            config["cols"],
            config["channels"],
            config["coarse_in"],
            config["coarse_out"],
            k_size=config["kernel_size"],
            stride=config["stride"],
            groups=config["groups"],
            pad=config["pad"],
            fine=config["fine"],
        )
        
        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)

@ddt.ddt
class TestReLULayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/layers/relu/config_0.json",
        "tests/configs/layers/relu/config_1.json",
        "tests/configs/layers/relu/config_2.json",
        "tests/configs/layers/relu/config_3.json",
        "tests/configs/layers/relu/config_4.json",
        "tests/configs/layers/relu/config_5.json",
        "tests/configs/layers/relu/config_6.json",
        "tests/configs/layers/relu/config_7.json",
        "tests/configs/layers/relu/config_8.json",
        "tests/configs/layers/relu/config_9.json",
        "tests/configs/layers/relu/config_10.json",
    )
    def test_layer_configurations(self, config_path):
        
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = ReLULayer(
            config["rows"],
            config["cols"],
            config["channels"],
            config["coarse"],
            config["coarse"],
        )
        
        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)

@ddt.ddt
class TestInnerProductLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/layers/inner_product/config_0.json",
        "tests/configs/layers/inner_product/config_1.json",
        "tests/configs/layers/inner_product/config_2.json",
        "tests/configs/layers/inner_product/config_3.json",
        "tests/configs/layers/inner_product/config_4.json",
        "tests/configs/layers/inner_product/config_5.json",
        "tests/configs/layers/inner_product/config_6.json",
        "tests/configs/layers/inner_product/config_7.json",
        "tests/configs/layers/inner_product/config_8.json",
        "tests/configs/layers/inner_product/config_9.json",
    )
    def test_layer_configurations(self, config_path):
        
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = InnerProductLayer(
            config["filters"],
            config["rows"],
            config["cols"],
            config["channels"],
            config["coarse_in"],
            config["coarse_out"],
        )
        
        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)

@ddt.ddt
class TestSqueezeLayer(TestLayerTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/layers/squeeze/config_0.json",
        "tests/configs/layers/squeeze/config_1.json",
    )
    def test_layer_configurations(self, config_path):
        
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise layer
        layer = SqueezeLayer(
            config["rows"],
            config["cols"],
            config["channels"],
            config["coarse_in"],
            config["coarse_out"],
        )
        
        # run tests
        self.run_test_dimensions(layer)
        self.run_test_rates(layer)


