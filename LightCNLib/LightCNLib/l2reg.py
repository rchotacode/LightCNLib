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

class l2Reg(Regularizer):
    def __init__(self, l):
        self.batch_size = 1
        self.l = l
    def loss(self, w):
        return np.sum(w * w) * self.l / 2 / self.batch_size
    
    def set_batch_size(self, b):
        self.batch_size = b

    def backwards(self, w):
        return self.l  * w / self.batch_size
