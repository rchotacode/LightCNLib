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
from .layer import Layer

class ReLU_layer(Layer):
    def forward(self, x):
        self.last_in = x 
        return np.maximum(x, 0)
    def backwards(self, upstream):
        grad = np.zeros(self.last_in.shape)
        grad[self.last_in > 1] = 1
        return upstream * grad
