from models.layers.Layer import Layer

import numpy as np
import math
import tools.third_party.lmdb_io
import tools.third_party.prototxt
import tempfile
import caffe
import pydot

class SigmoidLayer(Layer):
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

