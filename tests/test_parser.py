import unittest
import ddt
from tools.layer_enum import LAYER_TYPE
import os
os.environ['GLOG_minloglevel'] = '2' # suprress Caffe verbose prints
import caffe
import tools.parser as parser

class TestParserSingleLayer(unittest.TestCase):

    def setUp(self):
        self.graph, self.node_info = parser.parse_net("test/sw/data/single_layer.prototxt",view=False)

    def test_graph(self):
        
        # check graph
        self.assertEqual(self.graph['conv'],[])

    def test_layer_types(self):

        # check layer types
        self.assertEqual(self.node_info['conv']['type'],LAYER_TYPE.Convolution)

    def test_dimensions(self):

        # check layer dimensions
        self.assertEqual(self.node_info['conv']['hw'].rows, 28)
        self.assertEqual(self.node_info['conv']['hw'].cols, 28)
        self.assertEqual(self.node_info['conv']['hw'].channels, 1)

class TestParserSequentialNetwork(unittest.TestCase):

    def setUp(self):
        self.graph, self.node_info = parser.parse_net("test/sw/data/sequential.prototxt",view=False)

    def test_graph(self):
        
        # check graph
        self.assertEqual(self.graph['a'],['b'])
        self.assertEqual(self.graph['b'],['c'])
        self.assertEqual(self.graph['c'],[])

    def test_layer_types(self):

        # check layer types
        self.assertEqual(self.node_info['a']['type'],LAYER_TYPE.Convolution)
        self.assertEqual(self.node_info['b']['type'],LAYER_TYPE.Pooling)
        self.assertEqual(self.node_info['c']['type'],LAYER_TYPE.Convolution)

    def test_dimensions(self):

        # check layer dimensions
        self.assertEqual(self.node_info['a']['hw'].rows, 28)
        self.assertEqual(self.node_info['a']['hw'].cols, 28)
        self.assertEqual(self.node_info['a']['hw'].channels, 1)

        self.assertEqual(self.node_info['b']['hw'].rows, 24)
        self.assertEqual(self.node_info['b']['hw'].cols, 24)
        self.assertEqual(self.node_info['b']['hw'].channels, 20)

        self.assertEqual(self.node_info['c']['hw'].rows, 12)
        self.assertEqual(self.node_info['c']['hw'].cols, 12)
        self.assertEqual(self.node_info['c']['hw'].channels, 20)

class TestParserMultipathNetwork(unittest.TestCase):

    def setUp(self):
        self.graph, self.node_info = parser.parse_net("test/sw/data/multipath.prototxt",view=False)

    def test_graph(self):
       
        # check graph
        self.assertEqual(self.graph['a'],['a_split'])
        self.assertEqual(self.graph['a_split'],['b','c'])
        self.assertEqual(self.graph['b'],['c'])
        self.assertEqual(self.graph['c'],['c_split'])
        self.assertEqual(self.graph['c_split'],['d','e'])
        self.assertEqual(self.graph['d'],['g'])
        self.assertEqual(self.graph['e'],['f'])
        self.assertEqual(self.graph['f'],['g'])
        self.assertEqual(self.graph['g'],[])

    def test_layer_types(self):

        # check layer types
        self.assertEqual(self.node_info['a']['type'],LAYER_TYPE.Pooling)
        self.assertEqual(self.node_info['b']['type'],LAYER_TYPE.Convolution)
        self.assertEqual(self.node_info['c']['type'],LAYER_TYPE.Concat)
        self.assertEqual(self.node_info['d']['type'],LAYER_TYPE.Convolution)
        self.assertEqual(self.node_info['e']['type'],LAYER_TYPE.Convolution)
        self.assertEqual(self.node_info['f']['type'],LAYER_TYPE.ReLU)
        self.assertEqual(self.node_info['g']['type'],LAYER_TYPE.Concat)

    def test_dimensions(self):

        # check layer dimensions
        self.assertEqual(self.node_info['a']['hw'].rows, 56)
        self.assertEqual(self.node_info['a']['hw'].cols, 56)
        self.assertEqual(self.node_info['a']['hw'].channels, 32)

        self.assertEqual(self.node_info['b']['hw'].rows, 28)
        self.assertEqual(self.node_info['b']['hw'].cols, 28)
        self.assertEqual(self.node_info['b']['hw'].channels, 32)

        self.assertEqual(self.node_info['c']['hw'].rows, 28)
        self.assertEqual(self.node_info['c']['hw'].cols, 28)
        self.assertEqual(self.node_info['c']['hw'].channels[0], 32)

        self.assertEqual(self.node_info['c']['hw'].rows, 28)
        self.assertEqual(self.node_info['c']['hw'].cols, 28)
        self.assertEqual(self.node_info['c']['hw'].channels[1], 64)

        self.assertEqual(self.node_info['d']['hw'].rows, 28)
        self.assertEqual(self.node_info['d']['hw'].cols, 28)
        self.assertEqual(self.node_info['d']['hw'].channels, 96)

        self.assertEqual(self.node_info['e']['hw'].rows, 28)
        self.assertEqual(self.node_info['e']['hw'].cols, 28)
        self.assertEqual(self.node_info['e']['hw'].channels, 96)

        self.assertEqual(self.node_info['f']['hw'].rows, 26)
        self.assertEqual(self.node_info['f']['hw'].cols, 26)
        self.assertEqual(self.node_info['f']['hw'].channels, 64)

        self.assertEqual(self.node_info['g']['hw'].rows, 26)
        self.assertEqual(self.node_info['g']['hw'].cols, 26)
        self.assertEqual(self.node_info['g']['hw'].channels[0], 128)

        self.assertEqual(self.node_info['g']['hw'].rows, 26)
        self.assertEqual(self.node_info['g']['hw'].cols, 26)
        self.assertEqual(self.node_info['g']['hw'].channels[1], 64)

class TestParserLeNet(unittest.TestCase):

    def setUp(self):
        self.network_path = "data/models/lenet.prototxt"
        self.graph, self.node_info = parser.parse_net(self.network_path,view=False)

    def test_graph(self):
       
        # check graph
        self.assertEqual(self.graph['conv1'],['pool1'])
        self.assertEqual(self.graph['pool1'],['conv2'])
        self.assertEqual(self.graph['conv2'],['pool2'])
        self.assertEqual(self.graph['pool2'],['ip1'])
        self.assertEqual(self.graph['ip1'],['relu1'])
        self.assertEqual(self.graph['relu1'],['ip2'])
        self.assertEqual(self.graph['ip2'],['prob'])
        self.assertEqual(self.graph['prob'],[])

    def test_layer_types(self):

        # check layer types
        self.assertEqual(self.node_info['conv1']['type'],LAYER_TYPE.Convolution)
        self.assertEqual(self.node_info['pool1']['type'],LAYER_TYPE.Pooling)
        self.assertEqual(self.node_info['conv2']['type'],LAYER_TYPE.Convolution)
        self.assertEqual(self.node_info['pool2']['type'],LAYER_TYPE.Pooling)
        self.assertEqual(self.node_info['ip1']['type'],LAYER_TYPE.InnerProduct)
        self.assertEqual(self.node_info['relu1']['type'],LAYER_TYPE.ReLU)
        self.assertEqual(self.node_info['ip2']['type'],LAYER_TYPE.InnerProduct)
        self.assertEqual(self.node_info['prob']['type'],LAYER_TYPE.Softmax)

    def test_dimensions(self):

        # check layer dimensions
        caffe_net = caffe.Net(self.network_path,caffe.TEST)
        for node in self.node_info:
            if self.node_info[node]['type'] == (LAYER_TYPE.Convolution or LAYER_TYPE.Pooling or LAYER_TYPE.InnerProduct):
                shape = caffe_net.blobs[node].data[...][0].shape
                if len(shape) == 1:
                    shape = [shape[0],1,1]
                self.assertEqual(self.node_info[node]['hw'].rows_out(),     shape[1])
                self.assertEqual(self.node_info[node]['hw'].cols_out(),     shape[2])
                self.assertEqual(self.node_info[node]['hw'].channels_out(), shape[0])

class TestParserAlexNet(unittest.TestCase):

    def setUp(self):
        self.network_path = "data/models/alexnet.prototxt"
        self.graph, self.node_info = parser.parse_net(self.network_path,view=False)

    def test_graph(self):
       
        # check graph
        self.assertEqual(self.graph['conv1'],['relu1'])
        self.assertEqual(self.graph['relu1'],['pool1'])
        self.assertEqual(self.graph['pool1'],['norm1'])
        self.assertEqual(self.graph['norm1'],['conv2'])
        self.assertEqual(self.graph['conv2'],['relu2'])
        self.assertEqual(self.graph['relu2'],['pool2'])
        self.assertEqual(self.graph['pool2'],['norm2'])
        self.assertEqual(self.graph['norm2'],['conv3'])
        self.assertEqual(self.graph['conv3'],['relu3'])
        self.assertEqual(self.graph['relu3'],['conv4'])
        self.assertEqual(self.graph['conv4'],['relu4'])
        self.assertEqual(self.graph['relu4'],['conv5'])
        self.assertEqual(self.graph['conv5'],['relu5'])
        self.assertEqual(self.graph['relu5'],['pool5'])
        self.assertEqual(self.graph['pool5'],['fc6'])
        self.assertEqual(self.graph['fc6'],['relu6'])
        self.assertEqual(self.graph['relu6'],['fc7'])
        self.assertEqual(self.graph['fc7'],['relu7'])
        self.assertEqual(self.graph['relu7'],['fc8_sbt'])
        self.assertEqual(self.graph['fc8_sbt'],['prob'])
        self.assertEqual(self.graph['prob'],[])

    def test_layer_types(self):

        # check layer types
        self.assertEqual(self.node_info['conv1']['type'],LAYER_TYPE.Convolution)
        self.assertEqual(self.node_info['relu1']['type'],LAYER_TYPE.ReLU)
        self.assertEqual(self.node_info['pool1']['type'],LAYER_TYPE.Pooling)
        self.assertEqual(self.node_info['norm1']['type'],LAYER_TYPE.LRN)
        self.assertEqual(self.node_info['conv2']['type'],LAYER_TYPE.Convolution)
        self.assertEqual(self.node_info['relu2']['type'],LAYER_TYPE.ReLU)
        self.assertEqual(self.node_info['pool2']['type'],LAYER_TYPE.Pooling)
        self.assertEqual(self.node_info['norm2']['type'],LAYER_TYPE.LRN)
        self.assertEqual(self.node_info['conv3']['type'],LAYER_TYPE.Convolution)
        self.assertEqual(self.node_info['relu3']['type'],LAYER_TYPE.ReLU)
        self.assertEqual(self.node_info['conv4']['type'],LAYER_TYPE.Convolution)
        self.assertEqual(self.node_info['relu4']['type'],LAYER_TYPE.ReLU)
        self.assertEqual(self.node_info['conv5']['type'],LAYER_TYPE.Convolution)
        self.assertEqual(self.node_info['relu5']['type'],LAYER_TYPE.ReLU)
        self.assertEqual(self.node_info['pool5']['type'],LAYER_TYPE.Pooling)
        self.assertEqual(self.node_info['fc6']['type'],LAYER_TYPE.InnerProduct)
        self.assertEqual(self.node_info['relu6']['type'],LAYER_TYPE.ReLU)
        self.assertEqual(self.node_info['fc7']['type'],LAYER_TYPE.InnerProduct)
        self.assertEqual(self.node_info['relu7']['type'],LAYER_TYPE.ReLU)
        self.assertEqual(self.node_info['fc8_sbt']['type'],LAYER_TYPE.InnerProduct)
        self.assertEqual(self.node_info['prob']['type'],LAYER_TYPE.Softmax)

    def test_dimensions(self):

        # check layer dimensions
        caffe_net = caffe.Net(self.network_path,caffe.TEST)
        for node in self.node_info:
            if self.node_info[node]['type'] == (LAYER_TYPE.Convolution or LAYER_TYPE.Pooling or LAYER_TYPE.InnerProduct):
                shape = caffe_net.blobs[node].data[...][0].shape
                if len(shape) == 1:
                    shape = [shape[0],1,1]
                self.assertEqual(self.node_info[node]['hw'].rows_out(),     shape[1])
                self.assertEqual(self.node_info[node]['hw'].cols_out(),     shape[2])
                self.assertEqual(self.node_info[node]['hw'].channels_out(), shape[0])

class TestParserGoogleNet(unittest.TestCase):

    def setUp(self):
        self.network_path = "data/models/googlenet.prototxt"
        self.graph, self.node_info = parser.parse_net(self.network_path,view=False)

    def test_graph(self):
       
        # check graph
        pass

    def test_layer_types(self):

        # check layer types
        pass
    
    def test_dimensions(self):

        # check layer dimensions
        caffe_net = caffe.Net(self.network_path,caffe.TEST)
        for node in self.node_info:
            if self.node_info[node]['type'] == (LAYER_TYPE.Convolution or LAYER_TYPE.Pooling or LAYER_TYPE.InnerProduct):
                shape = caffe_net.blobs[node].data[...][0].shape
                if len(shape) == 1:
                    shape = [shape[0],1,1]
                self.assertEqual(self.node_info[node]['hw'].rows_out(),     shape[1])
                self.assertEqual(self.node_info[node]['hw'].cols_out(),     shape[2])
                self.assertEqual(self.node_info[node]['hw'].channels_out(), shape[0])

class TestParserResNet(unittest.TestCase):

    def setUp(self):
        self.network_path = "data/models/resnet.prototxt"
        self.graph, self.node_info = parser.parse_net(self.network_path,view=False)

    def test_graph(self):
       
        # check graph
        pass

    def test_layer_types(self):

        # check layer types
        pass
    
    def test_dimensions(self):

        # check layer dimensions
        caffe_net = caffe.Net(self.network_path,caffe.TEST)
        for node in self.node_info:
            if self.node_info[node]['type'] == (LAYER_TYPE.Convolution or LAYER_TYPE.Pooling or LAYER_TYPE.InnerProduct):
                shape = caffe_net.blobs[node].data[...][0].shape
                if len(shape) == 1:
                    shape = [shape[0],1,1]
                self.assertEqual(self.node_info[node]['hw'].rows_out(),     shape[1])
                self.assertEqual(self.node_info[node]['hw'].cols_out(),     shape[2])
                self.assertEqual(self.node_info[node]['hw'].channels_out(), shape[0])


