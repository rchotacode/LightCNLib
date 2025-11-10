import numpy
try:
    import cupy
except:
    pass
if cupy is not None and cupy.is_available():
    np = cupy
else:
    np = numpy

class Regularizer:
    def __init__(self, l):
        self.l = l
    def loss(self, w : np.ndarray):
        pass
    def back(self, w : np.ndarray) -> np.ndarray:
        pass