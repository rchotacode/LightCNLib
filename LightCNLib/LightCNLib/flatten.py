from .layer import Layer

class FlattenLayer(Layer):
    def forward(self, x):
        self.last_shape = x.shape
        return x.reshape((x.shape[0], -1))
    
    def backwards(self, upstream):
        return upstream.reshape(self.last_shape)