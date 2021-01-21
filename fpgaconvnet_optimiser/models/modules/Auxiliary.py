from fpgaconvnet_optimiser.models.modules import Module
import numpy as np

class Split(Module):
    def __init__(
            self,
            dim,
            inputs,
            coarse,
            data_width=16
        ):
        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.inputs = inputs
        self.coarse = coarse

        

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self, data):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == int(self.channels/self.coarse) , "ERROR: invalid channel dimension"
        assert data.shape[3] == self.coarse  , "ERROR: invalid coarse dimension"

        out = np.ndarray((
            self.rows,
            self.cols,
            int(self.channels/self.coarse),
            self.inputs,
            self.coarse),dtype=float)

        for index,_ in np.ndenumerate(out):
            out[index] = data[
              index[0],
              index[1],
              index[2],
              index[4]]

        return out


