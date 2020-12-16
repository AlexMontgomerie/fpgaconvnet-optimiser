from models.modules.BatchNorm import BatchNorm
from models.layers.Layer import Layer

import numpy as np
import math
import tools.third_party.lmdb_io 
import tools.third_party.prototxt 
import tempfile
import caffe
import pydot

class BatchNormLayer(Layer):
    def __init__(
            self,
            dim,
            coarse_in   =1,
            coarse_out  =1,
            data_width  =16,
            sa          =0.5,
            sa_out      =0.5
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width)

        # init variables
        self.scale_layer = None

        # modules
        self.modules = {
            "batch_norm" : BatchNorm(dim)
        }
        self.update()

    ## LAYER INFO ##
    def layer_info(self):
        return {
            'type'          : 'BATCH NORM',
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

    ## UPDATE MODULES ##
    def update(self):
        # batch norm
        self.modules['batch_norm'].rows     = self.rows_in()
        self.modules['batch_norm'].cols     = self.cols_in()
        self.modules['batch_norm'].channels = int(self.channels/self.coarse_in)
   
    def functional_model(self,data,gamma,beta,batch_size=1):

        assert data.shape[0] == self.rows    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR (data): invalid channel dimension"

        assert gamma.shape[0] == self.channels , "ERROR (weights): invalid filter dimension"
        assert beta.shape[0]  == self.channels , "ERROR (weights): invalid filter dimension"

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
        net.batch_norm = caffe.layers.BatchNorm(net.data,use_global_stats=False)
        net.scale      = caffe.layers.Scale(net.batch_norm,bias_term=True)

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
        net.blobs['data'].data[...][0]  = np.moveaxis(data, -1, 0)
        net.params['scale'][0].data[...] = gamma 
        net.params['scale'][1].data[...] = beta

        # run network
        net.forward()

        return net.blobs['scale'].data[0], net.params['batch_norm'][0].data, net.params['batch_norm'][1].data
 

