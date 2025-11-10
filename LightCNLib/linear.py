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

class LinearLayer(Layer):
    def __init__(self, in_size, out_size):
        bound = 1.0 / math.sqrt(in_size)
        self.weights = np.random.uniform(low=-bound, high=+bound, size=(in_size, out_size))
        self.biases = np.random.uniform(low=-bound, high=+bound, size=(out_size))
        self.grad_w = None
        self.grad_b = None
        self.last_in = None

    def forward(self, x : np.ndarray) -> np.ndarray:
        if self.do_grad:
            self.last_in = x
        return self.last_in @ self.weights + self.biases  # [in_size, a].T [in_size, out_size] = [a, out_size]

    def backwards(self, upstream) -> np.ndarray:
        if self.do_grad:
            self.grad_w = self.last_in.T @ upstream #[batch_size, in].T @ [batch_size, current_out]
            self.grad_b = upstream.sum(axis=0) #[current_out]
        return upstream @ self.weights.T #[batch_size, current_out] @ [in, current_out].T
    
    def update(self, f):
        if self.do_grad and self.grad_w is not None:
            self.weights -= f[0](self.grad_w)
            self.biases -= f[1](self.grad_b)
            self.grad_w = None
            self.grad_b = None
