import numpy
try:
    import cupy
except:
    pass
if cupy is not None and cupy.is_available():
    np = cupy
else:
    np = numpy

from .regularizer import Regularizer

class Model:
    layer_list = []
    weights = None
    def forward(self, x : np.ndarray):
        out = x
        for layer in self.layer_list:
            out = layer.forward(out)

        return out

    def backwards(self, upstream : np.ndarray, reg : Regularizer=None):
        for layer in self.layer_list[::-1]:
            if reg is not None and layer.weights is not None:
                upstream += reg.backwards(upstream)
            upstream = layer.backwards(upstream)

