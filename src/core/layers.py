import numpy as np

from core.functions import softmax, cee

class ReLU:
    def __init__(self) -> None:
        self.mask = None
        
    def forward(self, X):
        self.mask = (X <= 0)
        out = X.copy()
        out[self.mask] = 0
        
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dX = dout
        
        return dX

class Sigmoid:
    def __init__(self) -> None:
        self.out = None
        
    def forward(self, X):
        out = 1 / (1 + np.exp(-X))
        self.out = out
        
        return out

    def backward(self, dout):
        dX = dout * (1.0 - self.out) * self.out
        return dX

class Affine:
    def __init__(self, W, B):
        self.W = W
        self.B = B
        self.X = None
        self.dW = None
        self.dB = None
        
    def forward(self, X):
        self.X = X
        out = np.dot(X, self.W) + self.B
        
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.X.T, dout)
        self.dB = np.sum(dout, axis=0)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.Y = None
        self.T = None
        
    def forward(self, X, T):
        self.T = T
        self.y = softmax(X)
        self.loss = cee(self.Y, T)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.T.shape[0]
        dX = (self.Y - self.T) / batch_size
        
        return dX