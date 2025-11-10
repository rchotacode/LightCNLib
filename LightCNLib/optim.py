import numpy
try:
    import cupy
except:
    pass
if cupy is not None and cupy.is_available():
    np = cupy
else:
    np = numpy

import math
from .loss import softmax
from .regularizer import Regularizer

class Optimizer:
    def __init__(self, loss, model, reg : Regularizer=None):
        self.f = [(lambda x : x * 1e-2,lambda x : x * 1e-2)]
        self.layers = []
        self.loss = loss
        self.reg = reg
        self.model = model
        
    def run_through(self, preds, target):
        if self.reg is not None:
            self.reg.set_batch_size(preds.shape[0])
        loss, upstream = self.loss(preds, target)
        logits = softmax(preds)
        self.model.backwards(upstream, self.reg)
        for i, l in enumerate(self.model.layer_list[::-1]):
            if l.weights is not None and self.reg is not None:
                loss += self.reg.loss(l.weights)
            l.update(self.f[i % len(self.f)])
        return loss
    
    def zero_grads(self):
        for l in self.model.layer_list:
            l.w_grad = None
            l.b_grad = None