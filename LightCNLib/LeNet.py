import numpy as np
try:
    import cupy
    if cupy.cuda_is_available():
        np = cupy
except:
    pass

from .linear import LinearLayer
from .tanh import TanhLayer
from .convolution import ConvolutionLayer
from .flatten import FlattenLayer
from .pool import AvgPoolLayer
from .model import Model
from .loss import softmax
import math

class LeNet(Model):
    def __init__(self, ins, classes, channels):
        size_change = lambda x, k, s: (x - k)//s + 1
        dims = [[5, 1], [2, 2], [5, 1], [2, 2], [5, 1]]
        in_out = ins
        for d in dims:
            in_out = size_change(in_out, *d)
        in_out += 1
        in_out **= 2
        in_out *= 120
        self.layer_list = [ConvolutionLayer(channels, 6, 5, 1, 0, False), TanhLayer(), AvgPoolLayer(2, 2), 
                           ConvolutionLayer(6, 16, 5, 1, 0, False), TanhLayer(), AvgPoolLayer(2, 2),
                           ConvolutionLayer(16, 120, 4, 1, 0, False), TanhLayer(), FlattenLayer(),
                           LinearLayer(300000, 120), LinearLayer(120, 86), TanhLayer(), LinearLayer(86, 10)]
    
    def compute_logits(self, x : np.ndarray):
        out = x
        for layer in self.layer_list:  
            layer.do_grads = False
            out = layer.forward(out)
            layer.do_grads = True
        return softmax(out)
    
    def compute_preds(self, x : np.ndarray):
        out = x
        for layer in self.layer_list:  
            layer.do_grads = False
            out = layer.forward(out)
            layer.do_grads = True
        return np.argmax(softmax(out), axis=-1)