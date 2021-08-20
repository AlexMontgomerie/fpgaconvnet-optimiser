"""
The convolution module computes the dot product
between the feature map windows and the coefficients
of the convolution module. This module has a tunable
degree of parallelism across the kernel dot product,
affecting the throughput and number of ports of the
on-chip weights storage.

.. figure:: ../../../figures/conv_diagram.png
"""
import joblib
from fpgaconvnet_optimiser.models.modules import Module
import numpy as np
import math
import os

class Conv(Module):
    """
    Conv hardware model class.
    """
    def __init__(
            self,
            dim,
            filters,
            fine,
            k_size,
            groups,
            data_width=16,
            acc_width=30,
            weight_width=30
        ):
        """
        Parameters
        ----------
        dim: list
            dimensions of the input featuremap. Should contain
            `channels`, `rows`, `cols` in that order.

        Attributes
        ----------
        filters: int
            output channel dimension of the featuremap.
        fine: int
            
        rows: int
            row dimension of input featuremap
        cols: int
            column dimension of input featuremap
        channels: int
            channel dimension of input featuremap
        data_width: int
            bitwidth of featuremap pixels (default is 16) 
        rsc_coef: list
            list of resource model coefficients. Corresponds
            to `LUT`, `BRAM`, `DSP` and `FF` resources in 
            that order.
        """

        # module name
        self.name = "conv"

        # init module
        Module.__init__(self,dim,data_width)

        # init variables
        self.filters = filters
        self.groups  = groups
        self.fine    = fine
        self.k_size  = k_size
        self.data_width=data_width
        self.acc_width=acc_width
        self.weight_width=weight_width
        RSC_TYPES=["LUT", "FF", "BRAM", "DSP"]
        self.rsc_buildmodel = {
        }
        self.rsc_coef = {
        }        
        #for rsc in RSC_TYPES:
            #self.rsc_buildmodel[rsc]=joblib.load("/home/wz2320/fpgaconvnet-optimiser/fpgaconvnet_optimiser/coefficients/conv_"+str(rsc))  
        #for rsc in RSC_TYPES:
              #self.rsc_coef[rsc]=np.load(os.path.join(os.path.dirname(__file__),"../../coefficients/conv_"+str(rsc).lower()+".npy"))
        self.rsc_buildmodel['LUT']=joblib.load("/home/wz2320/fpgaconvnet-optimiser/fpgaconvnet_optimiser/coefficients/conv_"+'LUT(randomforest)')
        self.rsc_buildmodel['FF']=joblib.load("/home/wz2320/fpgaconvnet-optimiser/fpgaconvnet_optimiser/coefficients/conv_"+'FF(randomforest)')
        self.rsc_buildmodel['BRAM']=joblib.load("/home/wz2320/fpgaconvnet-optimiser/fpgaconvnet_optimiser/coefficients/conv_"+'BRAM(randomforest)') 
        self.rsc_coef['DSP']=np.load(os.path.join(os.path.dirname(__file__),"../../coefficients/conv_"+str('DSP').lower()+".npy"))                         

    def dynamic_model(self, freq, rate, sa_in, sa_out):
        return [
            self.data_width*freq,
            self.data_width*sa_in*freq*rate*self.fine/float(self.channels),
            self.data_width*sa_in*freq*rate*self.fine,
            self.data_width*sa_out*freq*rate*self.fine,
            self.data_width*sa_out*freq*rate*self.fine*self.fine/float(self.k_size*self.k_size),
            self.data_width*sa_out*freq*rate*self.fine/float(self.k_size*self.k_size),
        ]

    def utilisation_model1(self): 
        if self.data_width<=4 or self.weight_width<=4:
              coefficient=0
        elif self.data_width<=8 and self.weight_width<=8:
              coefficient=0
        elif  (self.data_width<=28 and self.weight_width<=18) or (self.data_width<=18 and self.weight_width<=28):
              coefficient=1
        elif  (self.data_width>=30 and self.weight_width<=18 and self.weight_width>=6) or (self.data_width<=18 and self.weight_width>=6 and self.weight_width>=30):
              coefficient=2              
        elif  (self.data_width<=24 or self.weight_width<=24) and self.data_width+self.weight_width>=42:
              coefficient=2
        else:
              coefficient=4  
        a=self.k_size*self.channels*self.filters  
        b=self.channels*self.filters
        if self.fine==self.k_size:
          if a<=2048:
               coeff=1
          if a<=4096:
               coeff=2
          elif a<=8192:
                coeff=5  
          elif a<=16384:
                coeff=9
          else :
                coeff=18  
        elif self.fine==self.k_size*self.k_size:
          if b<=2048:
               coeff=1
          if b<=4096:
               coeff=2
          else :
               coeff=5  
        else:
            coeff=9                                       
                                                   
        return {
            "LUT"   : np.array([1,math.log(self.filters,2),math.log(self.cols*self.rows,2),math.log(self.channels,2)]),
            "FF"    : np.array([1,math.log(self.filters,2),math.log(self.cols*self.rows,2),math.log(self.channels,2)]),
            "DSP"   : np.array([coefficient*self.fine if self.fine>=self.k_size else 1]),
            "BRAM"  : np.array([coeff*self.fine if self.weight_width<=16 else 2*coeff*self.fine])
                }
    def utilisation_model(self):
        if self.data_width<=4 or self.weight_width<=4:
              coefficient=0
        elif self.data_width<=8 and self.weight_width<=8:
              coefficient=0
        elif  (self.data_width<=28 and self.weight_width<=18) or (self.data_width<=18 and self.weight_width<=28):
              coefficient=1
        elif  (self.data_width>=30 and self.weight_width<=18 and self.weight_width>=6) or (self.data_width<=18 and self.weight_width>=6 and self.weight_width>=30):
              coefficient=2              
        elif  (self.data_width<=24 or self.weight_width<=24) and self.data_width+self.weight_width>=42:
              coefficient=2
        else:
              coefficient=4        
        return {
            "LUT"   : np.array([self.cols,self.rows,self.channels,self.data_width,self.acc_width,self.weight_width,self.filters,self.fine,self.k_size]),
            "FF"    : np.array([self.cols,self.rows,self.channels,self.data_width,self.acc_width,self.weight_width,self.filters,self.fine,self.k_size]),
            "DSP"   : np.array([coefficient*self.fine if self.fine>=self.k_size else 1]),
            "BRAM"  : np.array([self.channels,self.weight_width,self.filters,self.fine,self.k_size])
        }                   

    def channels_out(self):
        return int(self.filters/float(self.groups))

    def rate_in(self):
        return self.fine*self.groups/float(self.k_size*self.k_size*self.filters)

    def rate_out(self):
        return self.fine/float(self.k_size*self.k_size)

    def pipeline_depth(self):
        return self.fine 

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'filters'   : self.filters,
            'groups'    : self.groups,
            'fine'      : self.fine,
            'kernel_size'   : self.k_size,
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def rsc1(self,coef=None): # TODO: improve DSP utilisation for different bitwidths
        if coef == None:
            coef = self.rsc_coef
        return {
          "LUT"  : int(np.dot(self.utilisation_model()["LUT"], coef["LUT"])),
          "BRAM" : int(np.dot(self.utilisation_model()["BRAM"], coef["BRAM"])),
          "DSP"  : int(np.dot(self.utilisation_model()["DSP"], coef["DSP"])),
          "FF"   : int(np.dot(self.utilisation_model()["FF"], coef["FF"])),
        }

    def rsc(self,buildmodel=None,coef=None):
        if buildmodel == None:          
              buildmodel = self.rsc_buildmodel     
        if coef == None:          
              coef = self.rsc_coef           
        return {
          "LUT"  : int(buildmodel["LUT"].predict(self.utilisation_model()["LUT"].reshape(1, -1))),
          "FF"   : int(buildmodel["FF"].predict(self.utilisation_model()["FF"].reshape(1, -1))),          
          "BRAM" : int(buildmodel["BRAM"].predict(self.utilisation_model()["BRAM"].reshape(1, -1))),
          "DSP"  : int(np.dot(self.utilisation_model()["DSP"], coef["DSP"]))
        }      
        
    def rsc2(self,buildmodel=None,scaler=None):
        if buildmodel == None:          
              buildmodel = self.rsc_buildmodel         
        return {
          "LUT"  : int(buildmodel["LUT"].predict(scaler["LUT"].transform(self.utilisation_model()["LUT"].reshape(1, -1)))),
          "FF"   : int(buildmodel["FF"].predict(scaler["FF"].transform(self.utilisation_model()["FF"].reshape(1, -1)))),          
          "BRAM" : int(buildmodel["BRAM"].predict(scaler["BRAM"].transform(self.utilisation_model()["BRAM"].reshape(1, -1)))),
          "DSP"  : int(buildmodel["DSP"].predict(scaler["DSP"].transform(self.utilisation_model()["DSP"].reshape(1, -1))))
        }          
    '''
    FUNCTIONAL MODEL
    '''

    def functional_model(self,data,weights):
        # check input dimensionality
        assert data.shape[0] == self.rows    , "ERROR: invalid row dimension"
        assert data.shape[1] == self.cols    , "ERROR: invalid column dimension"
        assert data.shape[2] == self.channels, "ERROR: invalid channel dimension"
        assert data.shape[3] == self.k_size  , "ERROR: invalid column dimension"
        assert data.shape[4] == self.k_size  , "ERROR: invalid column dimension"
        # check weight dimensionality
        assert weights.shape[0] == self.channels, "ERROR: invalid channel dimension"
        assert weights.shape[1] == int(self.filters/float(self.groups)) , "ERROR: invalid filter dimension"
        assert weights.shape[2] == self.k_size  , "ERROR: invalid column dimension"
        assert weights.shape[3] == self.k_size  , "ERROR: invalid column dimension"

        out = np.zeros((
            self.rows,
            self.cols,
            self.channels,
            int(self.filters/self.groups)
        ),dtype=float)

        for index,_ in np.ndenumerate(out):
            for k1 in range(self.k_size):
                for k2 in range(self.k_size):
                    out[index] += data[
                      index[0],index[1],index[2],k1,k2]*weights[
                      index[2],index[3],k1,k2]

        return out

