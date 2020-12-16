from models.modules.ReLU import ReLU
from models.layers.Layer import Layer

import numpy as np
import math
import tools.third_party.lmdb_io
import tools.third_party.prototxt
import tempfile
import caffe
import pydot

class ReLULayer(Layer):
    def __init__(
            self,
            dim,
            data_width  =16,
            coarse_in   =1,
            coarse_out  =1,
            sa          =0.5,
            sa_out      =0.5
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width)

        # init modules
        self.modules = {
            "relu" : ReLU(dim)
        }
        self.update()

        # switching activity
        self.sa     = sa
        self.sa_out = sa_out

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1):
        parameters.batch_size   = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in()
        parameters.cols_in      = self.cols_in()
        parameters.channels_in  = self.channels_in()
        parameters.rows_out     = self.rows_out()
        parameters.cols_out     = self.cols_out()
        parameters.channels_out = self.channels_out()
        parameters.coarse_in    = self.coarse_in
        parameters.coarse_out   = self.coarse_out
        parameters.coarse       = self.coarse_out

    """
    def layer_info(self):
        return {
            'type'          : 'RELU',
            'buffer_depth'  : self.buffer_depth,
            'rows'          : self.rows,
            'cols'          : self.cols,
            'channels'      : self.channels,
            'coarse'        : self.coarse_in,
            'coarse_in'     : self.coarse_in,
            'coarse_out'    : self.coarse_out,
            'size_in'       : int(self.rows*self.cols*self.channels),
            'size_out'      : int(self.rows_out()*self.cols_out()*self.channels_out()),
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }
    """

    ## UPDATE MODULES ##
    def update(self):
        self.modules['relu'].rows     = self.rows_in()
        self.modules['relu'].cols     = self.cols_in()
        self.modules['relu'].channels = int(self.channels/self.coarse_in)

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"relu",str(i)]), label="relu" ))

        return cluster, "_".join([name,"relu"]), "_".join([name,"relu"])

    def functional_model(self,data,batch_size=1):

        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"

        # create Caffe Layer
        net = caffe.NetSpec()

        lmdb_path = '/tmp/lmdb'
        lmdb = tools.third_party.lmdb_io.LMDB(lmdb_path)

        write_images = [(np.random.rand(self.rows, self.cols, self.channels)*255).astype(np.uint8)]*batch_size
        write_labels = [0]*batch_size
        lmdb.write(write_images,write_labels)

        # create inputs
        net.data, _ = caffe.layers.Data(
            batch_size = batch_size,
            backend = caffe.params.Data.LMDB,
            source = lmdb_path,
            transform_param = dict(scale = 1./255),
            ntop = 2)

        net.relu = caffe.layers.ReLU(net.data)
        #print(net.to_proto())

        train_prototxt = tempfile.NamedTemporaryFile(prefix='train_',suffix='.prototxt')
        test_prototxt  = tempfile.NamedTemporaryFile(prefix='test_',suffix='.prototxt')

        # write train file
        train_prototxt.write(str(net.to_proto()).encode('utf-8'))
        train_prototxt.seek(0)

        # convert to deploy prototxt
        tools.third_party.prototxt.train2deploy(train_prototxt.name,[batch_size,self.channels,self.rows,self.cols],test_prototxt.name)

        # Load network
        net = caffe.Net(test_prototxt.name,caffe.TEST)

        # Close temporary files
        train_prototxt.close()
        test_prototxt.close()

        # load network inputs
        net.blobs['data'].data[...][0] = np.moveaxis(data, -1, 0)

        # run network
        net.forward()

        return net.blobs['relu'].data[0]

if __name__ == "__main__":
    pass
