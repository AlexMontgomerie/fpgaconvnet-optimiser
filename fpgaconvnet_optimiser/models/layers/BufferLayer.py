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
            ctrledge: str,
            drop_mode: bool   =True,
            coarse: int = 1,
            data_width: int  =16,
        ):
        # initialise parent class
        super().__init__(rows, cols, channels, coarse, coarse,
                data_width=data_width)

        self._coarse = coarse
        #ctrledge links to exit condition layer
        self._ctrledge = ctrledge
        self._drop_mode = drop_mode

        #init modules
        self.modules = {
                "buffer" : Buffer(self.rows,self.cols,self.channels, self.ctrledge)
        }
        self.update()

    @property
    def ctrledge(self) -> str:
        return self._ctrledge

    @property
    def drop_mode(self) -> bool:
        return self._drop_mode

    @property
    def coarse(self) -> int:
        return self._coarse

    @property
    def coarse_in(self) -> int:
        return self._coarse

    @property
    def coarse_out(self) -> int:
        return self._coarse

    @coarse.setter
    def coarse(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse = val
        self._coarse_in = val
        self._coarse_out = val
        self.update()

    ## LAYER INFO ##
    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.coarse       = self.coarse
        parameters.ctrledge     = self.ctrledge
        parameters.drop_mode    = self.drop_mode


    def update(self):
        self.modules['buffer'].rows     = self.rows_in()
        self.modules['buffer'].cols     = self.cols_in()
        self.modules['buffer'].channels = int(self.channels/self.coarse_in)

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
        nodes_in  = [ "_".join([name,"buff",str(i)]) for i in range(self.coarse_in) ]
        nodes_out = [ "_".join([name,"buff",str(i)]) for i in range(self.coarse_out) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self, data, ctrl_drop, batch_size=1): #TODO implement batch size
        #Buffer is not an ONNX or pytorch op
        # check input dimensionality
        assert data.shape[0] == batch_size          , "ERROR: invalid mismatched batch"
        assert data.shape[1] == self.rows_in()     , "ERROR: invalid row dimension"
        assert data.shape[2] == self.cols_in()     , "ERROR: invalid column dimension"
        assert data.shape[3] == self.channels_in() , "ERROR: invalid channel dimension"

        data_out=[]
        for b, ctrl in zip(data, ctrl_drop):
            if self.drop_mode: #non-inverted
                if ctrl == 1.0:
                    continue
                else:
                    data_out.append(b) #pass through
            else: #inverted
                if not ctrl == 1.0:
                    continue
                else:
                    data_out.append(b) #pass through

        return np.asarray(data_out)
