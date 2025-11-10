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

from loss import softmax_loss
import torch 
from tensorflow.keras.datasets import mnist
from torch.utils.data import TensorDataset, DataLoader
from LeNet import LeNet
from optim import Optimizer
from adam import adam_generator


#init data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')[..., None] / 255.0  # -> (60000,28,28,1)
x_test  = x_test .astype('float32')[..., None] / 255.0


x_train_t = torch.from_numpy(x_train).permute(0,3,1,2)  # (60000,1,28,28)
y_train_t = torch.from_numpy(y_train).long()

train_ds = TensorDataset(x_train_t, y_train_t)
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)


lnet = LeNet(10, 1)
adams = []
for n in range(len(lnet.layer_list)):
    adams.append((adam_generator(1e-3, (0.9, 0.99)),adam_generator(1e-3, (0.9, 0.99))))

optim = Optimizer(softmax_loss, lnet)
optim.f = adams

for epoch in range(10):
    for data in train_loader:
        x_in, t = data
        out = lnet.forward(np.asarray(x_in))    
        optim.run_through(out, t)
        optim.zero_grads()




