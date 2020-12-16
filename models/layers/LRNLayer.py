from models.layers.Layer import Layer

import numpy as np
import pydot

class LRNLayer(Layer):
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

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"lrn",str(i)]), label="lrn" ))

        return cluster, "_".join([name,"lrn"]), "_".join([name,"lrn"])
