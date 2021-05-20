"""
Buffering layer

Stores intermediate compute information such as results from Conv or Pool layers.
During DSE the required size will be calculated to store intermediate results at
branching layers. The position of the buffer layer will then be moved along a
given branch until the buffer size is feasible and the latency of the exit
condition is mitigated/matched. For effective pipelining I think.

Secondary function of the buffer is to "drop" a partial calculation.
Clear a FIFO - takes X number of cycles?
Drop signal will be control signal from the Exit Condition.

Future goal will be to have buffer as an offchip memory link.
In this case, the drop might not be used.

If "drop_mode" True then when True ctrl signal received, drop the data.
If "drop_mode" False then use inverted ctrl signal.
"""

import numpy as np
import math
import pydot
import torch

from fpgaconvnet_optimiser.models.modules import Buffer
#from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.layers import Layer

class BufferLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            coarse_in: int,
            coarse_out: int,
            ctrledge,
            drop_mode   =True,
            data_width  =16,
        ):
        # initialise parent class
        super().__init__([rows],[cols],[channels],[coarse_in],[coarse_out])

        #ctrledge links to exit condition layer
        self.ctrledge = ctrledge
        self.drop_mode = drop_mode

        #init modules
        self.modules = {
                "buffer" : Buffer(rows,cols,channels, ctrledge, data_width)
        }
        self.update()

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1):
        parameters.batch_size   = batch_size
        parameters.buffer_depth = self.buffer_depth
        parameters.rows_in      = self.rows_in(0)
        parameters.cols_in      = self.cols_in(0)
        parameters.channels_in  = self.channels_in0()
        parameters.rows_out     = self.rows_out(0)
        parameters.cols_out     = self.cols_out(0)
        parameters.channels_out = self.channels_out(0)
        parameters.coarse_in    = self.coarse_in
        parameters.coarse_out   = self.coarse_out

    ## UPDATE MODULES ##
    def update(self):
        self.modules['buffer'].rows     = self.rows_in(0)
        self.modules['buffer'].cols     = self.cols_in(0)
        self.modules['buffer'].channels = self.channels_in(0)
        #TODO work out if channels = int(self.channels/self.coarse_in)


    ### RATES ###
    def rates_graph(self):
        rates_graph = np.zeros( shape=(1,2) , dtype=float )
        #buffer
        rates_graph[0,0] = self.modules['buffer'].rate_in(0)
        rates_graph[0,1] = self.modules['buffer'].rate_out(0)
        return rates_graph

    def update_coarse_in(self, coarse_in):
        self.coarse_in  = coarse_in

    def update_coarse_out(self, coarse_out):
        self.coarse_out = coarse_out

    #def get_weights_reloading_feasible(self):

    def resource(self):

        buff_rsc    = self.modules['buffer'].rsc()

        # Total
        return {
            "LUT"  :  buff_rsc['LUT']*self.coarse_in,
            "FF"   :  buff_rsc['FF']*self.coarse_in,
            "BRAM" :  buff_rsc['BRAM']*self.coarse_in,
            "DSP" :   buff_rsc['DSP']*self.coarse_in,
        }

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in[0]):
            cluster.add_node(pydot.Node( "_".join([name,"buff",str(i)]), label="buff" ))

        # get nodes in and out
        nodes_in  = [ "_".join([name,"buff",str(i)]) for i in range(self.coarse_in[0]) ]
        nodes_out = nodes_in

        return cluster, nodes_in, nodes_out

    def functional_model(self, data, ctrl_drop):
        #Buffer is not an ONNX or pytorch op
        # check input dimensionality
        assert data.shape[0] == self.rows_in(0)    , "ERROR (data): invalid row dimension"
        assert data.shape[1] == self.cols_in(0)    , "ERROR (data): invalid column dimension"
        assert data.shape[2] == self.channels_in(0), "ERROR (data): invalid channel dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.channels),dtype=float)

        if self.drop_mode: #non-inverted
            if ctrl_drop:
                return out
            else:
                return data #pass through
        else: #inverted
            if not ctrl_drop:
                return out
            else:
                return data #pass through

