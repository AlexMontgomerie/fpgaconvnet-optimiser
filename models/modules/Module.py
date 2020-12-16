'''
Template for any module
'''

import numpy as np
import copy

class Module:
    def __init__(self,dim,data_width=16):
        # init variables
        self.rows       = dim[1]
        self.cols       = dim[2]
        self.channels   = dim[0]

        self.data_width = data_width

        # coefficients
        self.static_coef  = [ 0 ]
        self.dynamic_coef = [ 0 ]
        self.rsc_coef     = [ 0,0,0,0 ]

    def dynamic_model(self, freq, rate, sa_in, sa_out):
        return [0]

    def utilisation_model(self):
        return [0]

    def module_info(self):
        return {
            'type'      : self.__class__.__name__.upper(),
            'rows'      : self.rows_in(),
            'cols'      : self.cols_in(),
            'channels'  : self.channels_in(),
            'rows_out'      : self.rows_out(),
            'cols_out'      : self.cols_out(),
            'channels_out'  : self.channels_out()
        }

    def load_coef(self,static_coef_path,dynamic_coef_path,rsc_coef_path):
        self.static_coef  = np.load(static_coef_path) 
        self.dynamic_coef = np.load(dynamic_coef_path) 
        self.rsc_coef     = np.load(rsc_coef_path) 

    """    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    DIMENSIONS    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def rows_in(self):
        return self.rows

    def cols_in(self):
        return self.cols

    def channels_in(self):
        return self.channels

    def rows_out(self):
        return self.rows

    def cols_out(self):
        return self.cols

    def channels_out(self):
        return self.channels

    """    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RATES    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def rate_in(self):
        return 1.0

    def rate_out(self):
        return 1.0

    """    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    METRICS    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def get_latency(self):
        latency_in  = int((self.rows_in() *self.cols_in() *self.channels_in() )/self.rate_in() )
        latency_out = int((self.rows_out()*self.cols_out()*self.channels_out())/self.rate_out())
        return max(latency_in,latency_out)

    def pipeline_depth(self):
        return 0 

    def wait_depth(self):
        return 0

    """    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    USAGE 
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    
    def static_power(self):
        return np.dot(self.utilisation_model(), self.static_coef)

    def dynamic_power(self,freq,rate,sa,sa_out):
        return np.dot(
            self.dynamic_model(freq,rate,sa,sa_out), 
            self.dynamic_coef)

    def rsc(self):
        return {
          "LUT"  : int(np.dot(self.utilisation_model(), self.rsc_coef[0])),
          "BRAM" : int(np.dot(self.utilisation_model(), self.rsc_coef[1])),
          "DSP"  : int(np.dot(self.utilisation_model(), self.rsc_coef[2])),
          "FF"   : int(np.dot(self.utilisation_model(), self.rsc_coef[3])),
        }

