from collections import OrderedDict
import numpy as np

from core.functions import *
from core.gradient import *
from core.layers import *

class BackPropagationNet:
    def __init__(
        self, input_size, hidden_size, output_size, weight_init_std=0.01, activation_layer=ReLU
    ) -> None:
        self.params = {}

        self.params["W1"] = weight_init_std / np.random.randn(input_size, hidden_size)
        self.params["B1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std / np.random.randn(hidden_size, output_size)
        self.params["B2"] = np.zeros(output_size)
        self.params["W3"] = weight_init_std / np.random.randn(hidden_size, output_size)
        self.params["B3"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["B1"])
        self.layers["Activation_layer1"] = activation_layer()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["B2"])
        self.layers["Activation_layer2"] = activation_layer()
        self.layers["Affine3"] = Affine(self.params["W3"], self.params["B3"])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, X):
        for layer in self.layers.values():
            X = layer.forward(X)

        return X

    def loss(self, X, T):
        Y = self.predict(X)
        return self.last_layer.forward(Y, T)

    def accuracy(self, X, T):
        Y = self.predict(X)
        Y = np.argmax(Y, axis=1)

        if T.ndim != 1:
            T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])

        return accuracy

    def numerical_gradient(self, X, T):
        loss_W = lambda W: self.loss(X, T)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["B1"] = numerical_gradient(loss_W, self.params["B1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["B2"] = numerical_gradient(loss_W, self.params["B2"])
        grads["W3"] = numerical_gradient(loss_W, self.params["W3"])
        grads["B3"] = numerical_gradient(loss_W, self.params["B3"])

        return grads

    def gradient(self, X, T):
        self.loss(X, T)

        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["B1"] = self.layers["Affine1"].dB
        grads["W2"] = self.layers["Affine2"].dW
        grads["B2"] = self.layers["Affine2"].dB
        grads["W3"] = self.layers["Affine3"].dW
        grads["B3"] = self.layers["Affine3"].dB

        return grads