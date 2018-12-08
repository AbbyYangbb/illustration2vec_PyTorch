# Add PyTorch model

from i2v.base_1 import Illustration2VecBase

caffe_available = False
chainer_available = False

""" Added PyTorch model """
pytorch_available = False

try:
    from i2v.pytorch_i2v import PytorchI2V, make_i2v_with_pytorch
    pytorch_available = True
except ImportError:
    pass    

try:
    from i2v.caffe_i2v import CaffeI2V, make_i2v_with_caffe
    caffe_available = True
except ImportError:
    pass

try:
    from i2v.chainer_i2v import ChainerI2V, make_i2v_with_chainer
    chainer_available = True
except ImportError:
    pass

"""
if not any([caffe_available, chainer_available]):
    raise ImportError('i2v requires caffe or chainer package')
"""    
if not any([pytorch_available, caffe_available, chainer_available]):
    raise ImportError('i2v requires pytorch, caffe or chainer package')    
