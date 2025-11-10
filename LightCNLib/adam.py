import numpy
try:
    import cupy
except:
    pass
if cupy is not None and cupy.is_available():
    np = cupy
else:
    np = numpy

def adam_generator(lr, betas):
    m, v, lr, (b_1, b_2) = 0, 0, lr, betas
    def adam(grad):
        nonlocal m, v, lr, b_1, b_2
        m,v = b_1 * m + (1 - b_1) * grad,   b_2 * v + (1 - b_2) * (grad ** 2)
        m_hat, v_hat = m / (1 - b_1),   v / (1 - b_2)
        ret = lr * m_hat / (np.sqrt(v_hat) + 10e-8)
        return ret
    return adam
