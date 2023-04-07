import numpy as np

from core.functions import *


class ReLU:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, X):
        self.mask = X <= 0
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
        self.Y = softmax(X)
        self.loss = cee(self.Y, T)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.T.shape[0]
        dX = (self.Y - self.T) / batch_size

        return dX


class Swish:
    def __init__(self) -> None:
        self.out = None

    def forward(self, X):
        out = swish(X)
        self.X = X

        return out

    def backward(self, dout):
        dX = dout * (sigmoid(self.X) + swish(self.X) * (1 - sigmoid(self.X)))

        return dX


class Mish:
    """Mish
    Mish is a variation of Swish function
    """

    def __init__(self):
        self.X = None

    def forward(self, X):
        out = X * np.tanh(np.log(1 + np.exp(X)))
        self.X = X

        return out

    def backward(self, dout):
        X = self.X
        omega = (
            np.exp(3 * X) + 4 * np.exp(2 * X) + (6 + 4 * X) * np.exp(X) + 4 * (1 + X)
        )
        delta = 1 + pow((np.exp(X) + 1), 2)
        dX = dout * (np.exp(X) * omega / pow(delta, 2))

        return dX


class LeakyReLU:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, X):
        self.mask = X <= 0
        out = X.copy()
        out[self.mask] *= 0.01

        return out

    def backward(self, dout):
        dout[self.mask] = 0.01
        dX = dout

        return dX


class ELU:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, X):
        self.mask = X <= 0
        out = X.copy()


class GELU:
    """GELU
    GELU is a variation of ReLU function
    Stable Diffusion, BERT, GPT-3 uses GELU for its activation function
    """

    def __init__(self) -> None:
        self.X = None

    def forward(self, X):
        a = pow(X, 3)
        b = X + 0.044715 * a
        c = 1 + np.tanh(np.sqrt(2 / np.pi) * b)
        out = 0.5 * X * c
        self.X = X
        return out

    def backward(self, dout):
        return dout * (0.0592789 * (self.X**3) + 0.662852 * self.X + 0.5)
