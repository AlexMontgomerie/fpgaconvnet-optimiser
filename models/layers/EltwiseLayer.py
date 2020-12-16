from models.layers.Layer import Layer

import numpy as np
import math
import tools.third_party.lmdb_io  
import tools.third_party.prototxt
import tempfile
import caffe
import pydot

class EltwiseLayer(Layer):
    def __init__(
            self,
            dim,
            n_input,
            op_type     ='SUM',
            coarse_in   =1,
            coarse_out  =1,
            data_width  =16,
            sa          =0.5,
            sa_out      =0.5
        ):
        Layer.__init__(self,dim,coarse_in,coarse_out,data_width)

        # update flags
        self.flags['multi_input'] = True

        # init variables
        self.n_input    = n_input

    def layer_info(self):
        return {
            'buffer_depth'  : self.buffer_depth,
            'type'          : 'ELTWISE',
            'inputs'        : self.n_input,
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
        pass
    
    def functional_model(self,data,op_type='PROD',batch_size=1):

        assert len(data) == self.n_input , "ERROR: invalid number of inputs"
        for i in range(self.n_input):
            assert data[i].shape[0] == self.rows       , "ERROR: invalid row dimension"
            assert data[i].shape[1] == self.cols       , "ERROR: invalid column dimension"
            assert data[i].shape[2] == self.channels   , "ERROR: invalid channel dimension"

        # create Caffe Layer
        net = caffe.NetSpec()

        lmdb_path = '/tmp/lmdb'
        lmdb = tools.third_party.lmdb_io.LMDB(lmdb_path)

        # create inputs
        data_in = []
        dim = {}
        for i in range(self.n_input):
            write_images = [(np.random.rand(self.rows, self.cols, self.channels)*255).astype(np.uint8)]*batch_size
            write_labels = [0]*batch_size
            lmdb.write(write_images,write_labels)
            input_name = 'data_'+str(i)
            dim[input_name] = [batch_size,self.channels,self.rows,self.cols]
            tmp_data, _ = caffe.layers.Data(
                batch_size = batch_size,
                backend = caffe.params.Data.LMDB,
                source = lmdb_path,
                transform_param = dict(scale = 1./255),
                ntop = 2)
            setattr(net,input_name,tmp_data)
            data_in.append( getattr(net,input_name) )

        #net.eltwise = caffe.layers.Eltwise(*data_in, EltwiseOp=op_type)
        net.eltwise = caffe.layers.Eltwise(*data_in)

        print(net.to_proto())

        train_tmp = tempfile.NamedTemporaryFile(prefix='train_',suffix='.prototxt')
        test_tmp  = tempfile.NamedTemporaryFile(prefix='test_',suffix='.prototxt')

        # write train file
        train_tmp.write(str(net.to_proto()).encode('utf-8'))
        train_tmp.seek(0)

        # convert to deploy prototxt
        tools.third_party.prototxt.train2deploy_concat(train_tmp.name,dim,test_tmp.name)

        # Load network
        net = caffe.Net(test_tmp.name,caffe.TEST)

        # Close temporary files
        train_tmp.close()
        test_tmp.close()

        # load network inputs
        for i in range(self.n_input):
            input_name = 'data_'+str(i)
            net.blobs[input_name].data[...][0] = np.moveaxis(data[i],-1,0)
        # run network
        net.forward()

        return net.blobs['eltwise'].data[0]

