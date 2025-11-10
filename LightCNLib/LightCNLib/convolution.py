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

class ConvolutionLayer(Layer):
    def __init__(self, ins, outs, filter_shape, stride, padding, use_biases):
        bound   = np.sqrt(6.0 / ins * filter_shape**2)
        self.outs = outs
        self.ins = ins
        self.weights = np.random.uniform(-bound, bound,
                                         size=(outs, ins, filter_shape, filter_shape))
        if use_biases:
            self.biases  = np.zeros((outs,), dtype=self.weights.dtype)
        else:
            self.biases = None

        self.stride  = stride
        self.pad     = padding

    def forward(self, x: np.ndarray) -> np.ndarray:

        if self.pad != 0:
            x_padded = np.pad(
                x, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)), mode='constant'
            )
        else:
            x_padded = x
    
        i_h, i_w = self.weights.shape[2:]
        #get all heightxwidth sub matrixes of the image
        slides = sliding_window_view(x_padded, (i_h, i_w), axis=(2, 3))
        #remove all that are not within stride
        slides = slides[:, :, ::self.stride, ::self.stride, :, :]

        if self.do_grad:
            self.last_slides = slides

        #for each batch, y, and x
        #multiply each height by width sub matrix by the height by width kernels and sum them accross input channels
        #this leaves batch by output channels (the number of kernels) by y by x where each y/x is the coordinates of a kernel
        out = np.einsum('b i y x h w, o i h w -> b o y x', slides, self.weights)

        if self.biases is not None:
            out = out + np.asarray(self.biases)[None, :]

        return out


    def backwards(self, dout):

        k_size = self.weights.shape[3]

        if self.do_grad:
            if self.biases is not None:
                self.grad_b = dout.sum(axis=(0,2,3))   # → (outs,)
            self.grad_w = np.einsum('b o y x, b c y x h w -> o c h w', dout, self.last_slides)

        w_flip = np.flip(self.weights, axis=(2, 3))
        #dout: b o y x w_flip: o i h w 
        #y and x have to be equal to the input dimensions with no stride
        new_h = (dout.shape[2] - 1) * self.stride + 1
        new_w = (dout.shape[3] - 1) * self.stride + 1
        new_dout = np.zeros((dout.shape[0], dout.shape[1], new_h, new_w))
        new_dout[:, :, ::self.stride, ::self.stride] = dout[:, :, :, :]
        
        #padding time 
        new_pad = k_size - self.pad - 1 

        #if we have to crop
        if new_pad < 0:
            new_pad = -new_pad
            new_dout = new_dout[:, :, new_pad:-new_pad, new_pad:-new_pad]
        elif new_pad > 0:
            new_dout = np.pad(new_dout, ((0, 0), (0, 0), (new_pad, new_pad), (new_pad, new_pad)))
        
        
        slides = sliding_window_view(new_dout, (k_size, k_size), axis=(2, 3))
        dx = np.einsum('b o y x h w, o i h w -> b i y x', slides, w_flip)

        return dx

    def update(self, f):
        if self.do_grad and self.grad_w is not None:
            self.weights -= f[0](self.grad_w)
            if self.biases is not None:
                self.biases -= f[1](self.grad_b)
            self.grad_w = None
            self.grad_b = None