import numpy as np
import math
import pydot
import torch
from typing import Union, List

from fpgaconvnet_optimiser.models.modules import Exponential
from fpgaconvnet_optimiser.models.modules import SoftMaxSum
from fpgaconvnet_optimiser.models.modules import Fork
from fpgaconvnet_optimiser.models.modules import ReduceMax
from fpgaconvnet_optimiser.models.modules import Compare
from fpgaconvnet_optimiser.models.layers import Layer

class SoftMaxCmpLayer(Layer):
    def __init__(
            self,
            rows: int,
            cols: int,
            channels: int,
            ctrledges: List[str], #expecting list
            threshold: float,
            coarse_in: int = 1,
            coarse_out: int = 1,
            cond_type: str   = 'top1',
            data_width: int  = 16
        ):
        super().__init__(rows,cols,channels,coarse_in,coarse_out,data_width=data_width)

        self._ctrledges = ctrledges
        self._cond_type = cond_type
        self._threshold = threshold
        self._coarse_in = coarse_in
        self._coarse_out = coarse_out

        self.ctrl_out_size = len(ctrledges)

        #update parameters TODO
        self.modules['exp'     ] = Exponential(self.rows_in(), self.cols_in(), self.channels_in())
        #kernel size=1 coarse=2
        self.modules['fork_in' ] = Fork(self.rows_in(), self.cols_in(), self.channels_in(), 1, 2)
        self.modules['sm_sum'  ] = SoftMaxSum(self.rows_in(), self.cols_in(), self.channels_in())
        self.modules['redmx'   ] = ReduceMax(self.rows_in(), self.cols_in(), self.channels_in())
        self.modules['cmp'     ] = Compare(self.rows_in(), self.cols_in(), self.channels_in(), self.threshold)
        #kernel size=1, coarse=num of ctrl signals
        self.modules['fork_out'] = Fork(self.rows_out(), self.cols_out(), self.channels_in(), 1, self.ctrl_out_size)

        self.update()

    def rows_out(self):
        return 1
    def cols_out(self):
        return 1
    def channels_out(self):
        return self.ctrl_out_size #TODO fix for coarse layer

    #properties to match updated layers
    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def ctrledges(self) -> List[str]:
        return self._ctrledges

    @property
    def cond_type(self) -> str:
        return self._cond_type

    @property
    def coarse_in(self) -> int:
        return self._coarse_in

    @property
    def coarse_out(self) -> int:
        return self._coarse_out

    @coarse_in.setter
    def coarse_in(self, val: int) -> None:
        self._coarse_in = val
        self.update()

    @coarse_out.setter
    def coarse_out(self, val: int) -> None:
        self._coarse_out = val
        self.update()

    def layer_info(self,parameters,batch_size=1):
        Layer.layer_info(self, parameters, batch_size)
        parameters.channels_out = self.channels_out()+1#NOTE implied connection to ID pipeline
        parameters.threshold    = self.threshold
        parameters.ctrl_out_size = self.ctrl_out_size+1 #NOTE implied connection to ID pipeline
        parameters.ctrledges.extend(self.ctrledges) #NOTE list to repeated

    def update(self):
        #TODO fix this when possible
        #FIXME coarse doesn't mean anything for this layer currently
        self.modules['exp'     ].rows     = self.rows_in()
        self.modules['exp'     ].cols     = self.cols_in()
        self.modules['exp'     ].channels = int(self.channels_in()/self.coarse_in)

        self.modules['fork_in' ].rows     = self.rows_in()
        self.modules['fork_in' ].cols     = self.cols_in()
        self.modules['fork_in' ].channels = int(self.channels_in()/self.coarse_in)

        self.modules['sm_sum'  ].rows     = self.rows_in()
        self.modules['sm_sum'  ].cols     = self.cols_in()
        self.modules['sm_sum'  ].channels = int(self.channels_in()/self.coarse_in)

        self.modules['redmx'   ].rows     = self.rows_in()
        self.modules['redmx'   ].cols     = self.cols_in()
        self.modules['redmx'   ].channels = int(self.channels_in()/self.coarse_in)

        self.modules['cmp'     ].rows     = self.rows_in()
        self.modules['cmp'     ].cols     = self.cols_in()
        self.modules['cmp'     ].channels = int(self.channels_in()/self.coarse_in)

        self.modules['fork_out'].rows     = self.rows_in()
        self.modules['fork_out'].cols     = self.cols_in()
        self.modules['fork_out'].channels = int(self.channels_in()/self.coarse_in)

    def resource(self): #TODO
        exp_rsc     = self.modules['exp'].rsc()
        fi_rsc      = self.modules['fork_in'].rsc()
        sm_sum_rsc  = self.modules['sm_sum'].rsc()
        redmx_rsc   = self.modules['redmx'].rsc()
        cmp_rsc     = self.modules['cmp'].rsc()
        fo_rsc      = self.modules['fork_out'].rsc()

        # Total
        return {
            "LUT"  :    exp_rsc['LUT']*self.coarse_in +
                        fi_rsc['LUT']*self.coarse_in +
                        sm_sum_rsc['LUT']*self.coarse_in +
                        redmx_rsc['LUT']*self.coarse_in +
                        cmp_rsc['LUT']*self.coarse_in +
                        fo_rsc['LUT']*self.coarse_in,
            "FF"   :    exp_rsc['FF']*self.coarse_in +
                        fi_rsc['FF']*self.coarse_in +
                        sm_sum_rsc['FF']*self.coarse_in +
                        redmx_rsc['FF']*self.coarse_in +
                        cmp_rsc['FF']*self.coarse_in +
                        fo_rsc['FF']*self.coarse_in,
            "BRAM" :    exp_rsc['BRAM']*self.coarse_in +
                        fi_rsc['BRAM']*self.coarse_in +
                        sm_sum_rsc['BRAM']*self.coarse_in +
                        redmx_rsc['BRAM']*self.coarse_in +
                        cmp_rsc['BRAM']*self.coarse_in +
                        fo_rsc['BRAM']*self.coarse_in,
            "DSP" :     exp_rsc['DSP']*self.coarse_in +
                        fi_rsc['DSP']*self.coarse_in +
                        sm_sum_rsc['DSP']*self.coarse_in +
                        redmx_rsc['DSP']*self.coarse_in +
                        cmp_rsc['DSP']*self.coarse_in +
                        fo_rsc['DSP']*self.coarse_in,
        }

    # NOTE hardcoded pipeline depth for 10 class confidence calc
    def pipeline_depth(self):
        # NOTE from normal version
        #return sum([ self.modules[module].pipeline_depth() for module in self.modules ])
        # TODO make classifier for more than 10 classes
        #return int((180/256)*self.batch_size) # 180 measured for batch size 256
        # FIXME LAYERS HAVE NO CONCEPT OF BATCH SIZE....
        return 180

    def visualise(self,name):
        cluster = pydot.Cluster(name,label=name)

        for i in range(self.coarse_in[0]): #TODO
            cluster.add_node(pydot.Node( "_".join([name,"exp",str(i)]), label="exp" ))
            cluster.add_node(pydot.Node( "_".join([name,"fork_i",str(i)]), label="fork_i" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"exp",str(i)]),
                                         "_".join([name,"fork_i",str(i)]) ))

            cluster.add_node(pydot.Node( "_".join([name,"sm_sum",str(i)]), label="sm_sum" ))
            cluster.add_node(pydot.Node( "_".join([name,"redmx",str(i)]), label="redmx" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"fork_i",str(i)]),
                                         "_".join([name,"sm_sum",str(i)]) ))
            cluster.add_edge(pydot.Edge( "_".join([name,"fork_i",str(i)]),
                                         "_".join([name,"redmx",str(i)]) ))

            cluster.add_node(pydot.Node( "_".join([name,"cmp",str(i)]), label="cmp" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"redmx",str(i)]),
                                         "_".join([name,"cmp",str(i)]) ))
            cluster.add_edge(pydot.Edge( "_".join([name,"sm_sum",str(i)]),
                                         "_".join([name,"cmp",str(i)]) ))

            cluster.add_node(pydot.Node( "_".join([name,"fork_o",str(i)]), label="fork_o" ))
            cluster.add_edge(pydot.Edge( "_".join([name,"cmp",str(i)]),
                                         "_".join([name,"fork_o",str(i)]) ))


        # get nodes in and out
        nodes_in  = [ "_".join([name,"exp",str(i)]) for i in range(self.coarse_in[0]) ]
        #nodes_out = [ "_".join([name,"cmp",str(i)]) for i in range(self.coarse_out[0]) ]
        nodes_out = [ "_".join([name,"fork_o",str(i)]) for i in range(len(self.ctrledges)) ]

        return cluster, nodes_in, nodes_out

    def functional_model(self, data, batch_size=1):
        # CONVERTING TO SINGLE PRECISION
        data = data.astype(np.float32)

        batched_flag=False
        print(data.shape)
        if len(data.shape) > 3:
            batched_flag=True
            assert data.shape[0] == batch_size          , "ERROR: invalid mismatched batch"
            assert data.shape[1] == self.rows_in()    , "ERROR (data): invalid row dimension"
            assert data.shape[2] == self.cols_in()    , "ERROR (data): invalid column dimension"
            assert data.shape[3] == self.channels_in(), "ERROR (data): invalid channel dimension"
        else:
            assert data.shape[0] == self.rows_in()    , "ERROR (data): invalid row dimension"
            assert data.shape[1] == self.cols_in()    , "ERROR (data): invalid column dimension"
            assert data.shape[2] == self.channels_in(), "ERROR (data): invalid channel dimension"

        softmax_layer = torch.nn.Softmax(dim=-1)
        out = np.zeros((batch_size, 3))
        for b in range(batch_size):
            pk = softmax_layer(torch.from_numpy(data[b])).detach()
            print(pk)
            #get max value
            top1 = torch.max(pk) #torch.from_numpy(data))
            print(top1)
            #True = early exit, drop buffered data
            if top1 > self.threshold:
                out[b][0] = 1
                out[b][1] = 1
                out[b][2] = 1
                #return 1.0
            else:
                out[b][0] = 0
                out[b][1] = 0
                out[b][2] = 0
                #return 0.0
        print(out.shape)
        return out
