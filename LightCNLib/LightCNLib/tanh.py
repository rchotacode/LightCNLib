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

class TanhLayer(Layer):
    def forward(self, x):
        self.last_out = np.tanh(x)
        return self.last_out
    def backwards(self, upstream):
        return (1 - np.power(np.tanh(self.last_out), 2)) * upstream
