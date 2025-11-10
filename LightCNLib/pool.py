import numpy
try:
    import cupy
except:
    pass
if cupy is not None and cupy.is_available():
    np = cupy
    from cupy.lib.stride_tricks import sliding_window_view
else:
    np = numpy
    from numpy.lib.stride_tricks import sliding_window_view



from .layer import Layer

class AvgPoolLayer(Layer):
    def __init__(self, k_size, stride=None):
        self.k_size = k_size
        self.stride = stride if stride is not None else k_size

    def forward(self, x : np.ndarray):
        slides = sliding_window_view(x, (self.k_size, self.k_size), axis=(2,3))
        slides = slides[:, :, ::self.stride, ::self.stride]
        slides = np.mean(slides, axis=(-1, -2))
        return slides
    
    def backwards(self , upstream):
        new_h = (upstream.shape[-2] - 1) * self.stride + 1
        new_w = (upstream.shape[-1] - 1)  * self.stride + 1
        pad = self.k_size - 1
        dout = np.zeros((*upstream.shape[:2], new_h, new_w))
        dout[:, :, ::self.stride, ::self.stride] = upstream
        if pad < 0:
            dout = dout[:, :, -pad:pad, -pad:pad]
        elif(pad > 0):
            dout = np.pad(dout, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
        slides = sliding_window_view(dout, (self.k_size, self.k_size), axis=(-2, -1))
        dout = np.mean(slides, axis=(-2, -1))
        return dout

