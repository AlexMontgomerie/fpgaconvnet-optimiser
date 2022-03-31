"""
The split/fork/branch layer.
Takes one stream input and outputs several streams using the fork module.
"""

from typing import List

import pydot
import numpy as np
import os
import math

from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.layers import MultiPortLayer

class SplitLayer(MultiPortLayer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse: int = 1,
            ports_out: int = 1,
            data_width: int = 16
        ):
        """
        Parameters
        ----------
        rows: int
            row dimension of input featuremap
        cols: int
            column dimension of input featuremap
        channels: int
            channel dimension of input featuremap

        Attributes
        ----------
        buffer_depth: int, default: 0
            depth of incoming fifo buffers for each stream in.
        rows: list int
            row dimension of input featuremap
        cols: list int
            column dimension of input featuremap
        channels: list int
            channel dimension of input featuremap
        ports_in: int
            number of ports into the layer
        ports_out: int
            number of ports out of the layer
        coarse_in: list int
            number of parallel streams per port into the layer.
        coarse_out: NEED TO DEFINE
           TODO
        data_width: int
            bitwidth of featuremap pixels
        modules: dict
            dictionary of `module` instances that make
            up the layer. These modules are used for the
            resource and performance models of the layer.
        """

        # initialise parent class
        super().__init__(
                [rows],
                [cols],
                [channels],
                [coarse],
                [coarse],
                _rows_op     =   [rows]*ports_out,
                _cols_op     =   [cols]*ports_out,
                _channels_op =   [channels]*ports_out,
                ports_out   =   ports_out,
                data_width  =   data_width)

        # parameters
        self._coarse = coarse

        # init modules
        #One fork module, fork coarse_out corresponds to number of layer output ports
        self.modules["fork"] = Fork( self.rows_out(), self.cols_out(),
                self.channels_out(), 1, self.ports_out) #kernel default 1

        # update the modules
        self.update()

    """
    row out properties for multiple output ports
    """
    @property
    def coarse(self) -> int:
        return self._coarse

    @property
    def coarse_in(self) -> List[int]:
        return [self._coarse]

    @property
    def coarse_out(self) -> List[int]:
        return [self._coarse]*self.ports_out

    """
    split layer setters for multiple output ports
    """
    @coarse.setter
    def coarse(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = [val]
        self.coarse_out = [val]
        self.update()

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = [val]
        self._coarse_out = [val]
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = [val]
        self._coarse_out = [val]
        self.update()

    def layer_info(self,parameters,batch_size=1):
        MultiPortLayer.layer_info(self, parameters, batch_size)
        parameters.coarse = self.coarse
        parameters.ports_out = self.ports_out

    def update(self):
        # fork
        self.modules['fork'].rows     = self.rows_in()
        self.modules['fork'].cols     = self.cols_in()
        #FIXME treating coarse as consistent for now
        #self.modules['fork'].channels = self.channels_in()//self.coarse
        self.modules['fork'].channels = self.channels_in()
        self.modules['fork'].coarse   = self.ports_out

#=======
#    ## LAYER INFO ##
#    def layer_info(self,parameters,batch_size=1) :
#        parameters.batch_size = batch_size
#        parameters.buffer_depth = self.buffer_depth
#        parameters.rows_in      = self.rows_in(0)
#        parameters.cols_in      = self.cols_in(0)
#        parameters.channels_in  = self.channels_in(0)
#        parameters.rows_out     = self.rows_out(0)
#        parameters.cols_out     = self.cols_out(0)
#        parameters.channels_out = self.channels_out(0)
#        parameters.coarse_in    = self.coarse
#        parameters.coarse_out   = self.coarse
#        parameters.ports_out    = self.ports_out
#
#    ## UPDATE MODULES ##
#    def update(self):
#        # fork
#        self.modules['fork'].rows     = self.rows_out(0)
#        self.modules['fork'].cols     = self.cols_out(0)
#        self.modules['fork'].channels = int(self.channels[0]/self.coarse)
#        self.modules['fork'].coarse   = self.ports_out
#
#    def update_coarse_in(self, coarse_in, port_index=0):
#        self.coarse = coarse_in
#        self.coarse_in[0] = self.coarse
#        for i in range(self.ports_out):
#            self.coarse_out[0] = self.coarse
#
#    def update_coarse_out(self, coarse_out, port_index=0):
#        self.coarse = coarse_out
#        self.coarse_in[0] = self.coarse
#        for i in range(self.ports_out):
#            self.coarse_out[0] = self.coarse
#
#>>>>>>> b273d34... started split layer (#26)
#>>>>>>> c10e653... fixing MIMO layers to produce workload correctly

    def resource(self):

        # get module resources
        fork_rsc = self.modules['fork'].rsc()

        #Total
        return {
            "LUT"   :   fork_rsc['LUT']*self.coarse,
            "FF"    :   fork_rsc['FF']*self.coarse,
            "BRAM"  :   fork_rsc['BRAM']*self.coarse,
            "DSP"   :   fork_rsc['DSP']*self.coarse
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in):
            cluster.add_node(pydot.Node( "_".join([name,"split",str(i)]), label="split" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"fork",str(i)]) for i in range(self.coarse) ]
        nodes_out = [ "_".join([name,"fork",str(i)]) for i in range(self.ports_out*self.coarse) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self, data, batch_size=1):
        #default is port_index of 0
        assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"
        #assert data.shape[0] == self.rows_in(0)    , "ERROR (data): invalid row dimension"
        #assert data.shape[1] == self.cols_in(0)    , "ERROR (data): invalid column dimension"
        #assert data.shape[2] == self.channels_in(0), "ERROR (data): invalid channel dimension"


        out = np.ndarray((
            self.rows_in(0),
            self.cols_in(0),
            self.channels_in(0),
            self.ports_out),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[
              index[0],
              index[1],
              index[2]]

        return out

