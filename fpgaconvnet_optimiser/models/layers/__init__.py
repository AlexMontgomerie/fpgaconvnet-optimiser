"""
Layers are comprised of modules. They have the same functionality of the equivalent layers of the CNN model.
"""

from .Layer             import Layer
from .MultiPortLayer    import MultiPortLayer

from .BatchNormLayer    import BatchNormLayer
from .InnerProductLayer import InnerProductLayer
from .PoolingLayer      import PoolingLayer
from .ReLULayer         import ReLULayer
from .ConvolutionLayer  import ConvolutionLayer
from .SqueezeLayer      import SqueezeLayer
from .SplitLayer        import SplitLayer
#from .LRNLayer          import LRNLayer
#from .SoftMaxLayer      import SoftMaxLayer
#EE Layers
from .SplitLayer            import SplitLayer
from .BufferLayer           import BufferLayer
from .ExitConditionLayer    import ExitConditionLayer
from .ExitMergeLayer        import ExitMergeLayer
from .SoftMaxCmpLayer       import SoftMaxCmpLayer
