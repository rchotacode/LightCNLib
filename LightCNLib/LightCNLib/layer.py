import numpy as np
try:
    import cupy
    if cupy.cuda_is_available():
        np = cupy
except:
    pass

class Layer:
    do_grad = True
    weights = None
    def forward(self, x : np.ndarray) -> np.ndarray:
        pass
    def backwards(self, upstream) -> np.ndarray:
        pass
    def update(self, f):
        pass