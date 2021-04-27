import unittest
from fpgaconvnet_optimiser.models.layers import *

class TestLayerTemplate():

    def test_dimensions(self):
        # check input dimensions
        self.assertTrue(self.layer.rows_in() > 0) 
        self.assertTrue(self.layer.cols_in() > 0) 
        self.assertTrue(self.layer.channels_in() > 0) 
        # check output dimensions
        self.assertTrue(self.layer.rows_out() > 0) 
        self.assertTrue(self.layer.cols_out() > 0) 
        self.assertTrue(self.layer.channels_out() > 0) 
    
    def test_rates(self):
        # check rate in
        self.assertTrue(self.layer.rate_in(0) >= 0.0) 
        self.assertTrue(self.layer.rate_in(0) <= 1.0) 
        # check rate out
        self.assertTrue(self.layer.rate_out(0) >= 0.0) 
        self.assertTrue(self.layer.rate_out(0) <= 1.0) 

class TestPoolingLayer(TestLayerTemplate,unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # initialise layer
        self.layer = PoolingLayer(
            [10,10,10],
            k_size=2,
            stride=2,
            pad=0,
            coarse_in=1,
            coarse_out=1,
            fine=1,
        )

