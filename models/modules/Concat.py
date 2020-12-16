from models.modules.Module import Module
import numpy as np

class Concat(Module):
    def __init__(
            self,
            dim,
            n_input,
            data_width=16
        ):
        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.n_input = n_input 

        #self.channels_in  = lambda : self.channels/self.coarse
        #self.channels_out = lambda : self.channels*self.n_input/self.coarse

    '''
    PERFORMANCE MODELS
    '''

    def get_latency(self):
        return 0

    '''
    USAGE MODELS
    '''

    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self, data):
        # check input dimensionality
        assert len(data) == self.n_input , "ERROR: invalid row dimension"
        for i in range(self.n_input):
            assert data[i].shape[0] == self.rows       , "ERROR: invalid column dimension"
            assert data[i].shape[1] == self.cols       , "ERROR: invalid column dimension"
            assert data[i].shape[2] == self.channels[i], "ERROR: invalid channel dimension"
        
        out = np.ndarray((
            self.rows,
            self.cols,
            sum(self.channels)),dtype=float)

        channel_offset = 0
        for i in range(self.n_input):
            for index,_ in np.ndenumerate(data[i]):
                out[index[0],index[1],channel_offset+index[2]] = data[i][index]
            channel_offset += self.channels[i]
        return out

if __name__ == "__main__":
    concat = Concat([[2,1],1,1],2)
    print(concat.channels)
    data = [
        np.zeros((
            1,
            1,
            2),dtype=float),
        np.zeros((
            1,
            1,
            1),dtype=float)]
    

    data[0][0,0,0] = 1
    data[0][0,0,1] = 1

    out = concat.functional_model(data)
    print(out.shape,out)
