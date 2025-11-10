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

def softmax(preds):
    ex = np.exp(preds)
    sums = np.sum(ex, axis=-1, keepdims=True)
    ex /= sums 
    return ex

def softmax_loss(preds, targets):
    z = preds - np.max(preds, axis=-1, keepdims=True)
    grads = softmax(z)
    grads[np.arange(grads.shape[0]), targets] -= 1
    losses = -z[np.arange(z.shape[0]), targets] + np.log(np.sum(np.exp(z), axis=-1))
    return np.mean(losses), grads / grads.shape[0]
