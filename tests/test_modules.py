import unittest
import ddt
import json
from fpgaconvnet_optimiser.models.modules import *


class TestModuleTemplate():
    
    def run_test_methods_exist(self, module):
        self.assertTrue(hasattr(module, "rows_in"))
        self.assertTrue(hasattr(module, "cols_in"))
        self.assertTrue(hasattr(module, "channels_in"))
        self.assertTrue(hasattr(module, "rows_out"))
        self.assertTrue(hasattr(module, "cols_out"))
        self.assertTrue(hasattr(module, "channels_out"))

    def run_test_dimensions(self, module):
        # check input dimensions
        self.assertGreater(module.rows_in(), 0) 
        self.assertGreater(module.cols_in(), 0) 
        self.assertGreater(module.channels_in(), 0) 
        # check output dimensions
        self.assertGreater(module.rows_out(), 0) 
        self.assertGreater(module.cols_out(), 0) 
        self.assertGreater(module.channels_out(), 0) 
    
    def run_test_rates(self, module):
        # check rate in
        self.assertGreaterEqual(module.rate_in(), 0.0) 
        self.assertLessEqual(module.rate_in(),1.0) 
        # check rate out
        self.assertGreaterEqual(module.rate_out(), 0.0) 
        self.assertLessEqual(module.rate_out(), 1.0) 

@ddt.ddt
class TestForkModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/modules/fork/config_0.json",
        "tests/configs/modules/fork/config_1.json",
        "tests/configs/modules/fork/config_2.json",
    )
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Fork([config["rows"],config["cols"],config["channels"]],
                config["kernel_size"],
                config["coarse"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
