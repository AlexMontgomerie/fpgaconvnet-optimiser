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

    def run_test_resources(self, module):
        
        rsc = module.rsc()
        self.assertGreaterEqual(rsc["LUT"], 0.0) 
        self.assertGreaterEqual(rsc["FF"], 0.0) 
        self.assertGreaterEqual(rsc["DSP"], 0.0) 
        self.assertGreaterEqual(rsc["BRAM"], 0.0) 


@ddt.ddt
class TestForkModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/modules/fork/config_0.json",
        "tests/configs/modules/fork/config_1.json",
        "tests/configs/modules/fork/config_2.json",
        "tests/configs/modules/fork/config_3.json",
        "tests/configs/modules/fork/config_4.json",
        "tests/configs/modules/fork/config_5.json",
        "tests/configs/modules/fork/config_6.json",
        "tests/configs/modules/fork/config_7.json",
        "tests/configs/modules/fork/config_8.json",
        "tests/configs/modules/fork/config_9.json",
    )
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Fork(config["rows"],config["cols"],config["channels"],
                config["kernel_size"],
                config["coarse"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)
        
@ddt.ddt
class TestAccumModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/modules/accum/config_0.json",
        "tests/configs/modules/accum/config_1.json",
        "tests/configs/modules/accum/config_2.json",
        "tests/configs/modules/accum/config_3.json",
        "tests/configs/modules/accum/config_4.json",
        "tests/configs/modules/accum/config_5.json",
    )
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Accum(config["rows"],config["cols"],config["channels"],
                config["filters"],config["groups"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)
        
        # additional checks
        self.assertGreater(module.filters,0)

@ddt.ddt
class TestConvModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/modules/conv/config_0.json",
        "tests/configs/modules/conv/config_1.json",
        "tests/configs/modules/conv/config_2.json",
        "tests/configs/modules/conv/config_3.json",
        "tests/configs/modules/conv/config_4.json",
        "tests/configs/modules/conv/config_5.json",
        "tests/configs/modules/conv/config_6.json",
        "tests/configs/modules/conv/config_7.json",
        "tests/configs/modules/conv/config_8.json",
        "tests/configs/modules/conv/config_9.json",
        "tests/configs/modules/conv/config_10.json",
    )
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Conv(config["rows"],config["cols"],config["channels"],
                config["filters"],config["fine"],config["kernel_size"],config["group"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestGlueModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/modules/glue/config_0.json",
        "tests/configs/modules/glue/config_1.json",
        "tests/configs/modules/glue/config_2.json",
        "tests/configs/modules/glue/config_3.json",
        "tests/configs/modules/glue/config_4.json",
        "tests/configs/modules/glue/config_5.json",
        "tests/configs/modules/glue/config_6.json",
        "tests/configs/modules/glue/config_7.json",
        "tests/configs/modules/glue/config_8.json",
        "tests/configs/modules/glue/config_9.json",
    )
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Glue(config["rows"],config["cols"],config["channels"],
                config["filters"],config["coarse_in"],config["coarse_out"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestSlidingWindowModule(TestModuleTemplate,unittest.TestCase):
    
    @ddt.data(
        "tests/configs/modules/sliding_window/config_0.json",
        "tests/configs/modules/sliding_window/config_1.json",
        "tests/configs/modules/sliding_window/config_2.json",
        "tests/configs/modules/sliding_window/config_3.json",
        "tests/configs/modules/sliding_window/config_4.json",
        "tests/configs/modules/sliding_window/config_5.json",
        "tests/configs/modules/sliding_window/config_6.json",
        "tests/configs/modules/sliding_window/config_7.json",
    )
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = SlidingWindow(config["rows"],config["cols"],config["channels"],
                config["kernel_size"],config["stride"],config["pad_top"],
                config["pad_right"],config["pad_bottom"],config["pad_left"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestPoolModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/modules/pool/config_0.json",
        "tests/configs/modules/pool/config_1.json",
        "tests/configs/modules/pool/config_2.json",
        "tests/configs/modules/pool/config_3.json",
        "tests/configs/modules/pool/config_4.json",
        "tests/configs/modules/pool/config_5.json",
        "tests/configs/modules/pool/config_6.json",
        "tests/configs/modules/pool/config_7.json",
        "tests/configs/modules/pool/config_8.json",
        "tests/configs/modules/pool/config_9.json",
        "tests/configs/modules/pool/config_10.json",
        "tests/configs/modules/pool/config_11.json",
    )
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Pool(config["rows"],config["cols"],config["channels"],
                config["kernel_size"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestSqueezeModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/modules/squeeze/config_0.json",
        "tests/configs/modules/squeeze/config_1.json",
    )
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = Squeeze(config["rows"],config["cols"],config["channels"],
                config["coarse_in"],config["coarse_out"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)

@ddt.ddt
class TestReLUModule(TestModuleTemplate,unittest.TestCase):

    @ddt.data(
        "tests/configs/modules/relu/config_0.json",
        "tests/configs/modules/relu/config_1.json",
        "tests/configs/modules/relu/config_2.json",
        "tests/configs/modules/relu/config_3.json",
        "tests/configs/modules/relu/config_4.json",
        "tests/configs/modules/relu/config_5.json",
        "tests/configs/modules/relu/config_6.json",
        "tests/configs/modules/relu/config_7.json",
        "tests/configs/modules/relu/config_8.json",
        "tests/configs/modules/relu/config_9.json",
    )
    def test_module_configurations(self, config_path):
        # open configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # initialise module
        module = ReLU(config["rows"],config["cols"],config["channels"])

        # run tests
        self.run_test_methods_exist(module)
        self.run_test_dimensions(module)
        self.run_test_rates(module)
        self.run_test_resources(module)


