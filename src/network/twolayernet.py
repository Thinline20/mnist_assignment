import numpy as np

from core.functions import *
from core.gradient import *

class TwoLayerNet:
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, activation_fn, output_classifier, weight_init_std=0.01
    ) -> None:
        """Initialize Two Layer Network

        Args:
            input_size (int): size of input data
            hidden_size (int): size of hidden layer
            output_size (int): size of output
            activation_fn (function): an activation function for the network
            output_classifier (function): a function that classify the output
            weight_init_std (float, optional): initial weight. Defaults to 0.01.
        """
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["B1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["B2"] = np.zeros(output_size)

        self.activation_fn = activation_fn
        self.output_classifier = output_classifier

    def predict(self, X):
        W1, W2 = self.params["W1"], self.params["W2"]
        B1, B2 = self.params["B1"], self.params["B2"]

        Z = self.activation_fn(np.dot(X, W1) + B1)
        Y = self.output_classifier(np.dot(Z, W2) + B2)

        return Y

    def loss(self, X, T):
        return cee(self.predict(X), T)

    def accuracy(self, X, T):
        Y = self.predict(X)
        Y = np.argmax(Y, axis=1)
        T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])

        return accuracy

    def numerical_gradient(self, X, T):
        loss = lambda _: self.loss(X, T)
        
        grads = {}
        grads["W1"] = numerical_gradient(loss, self.params["W1"])
        grads["B1"] = numerical_gradient(loss, self.params["B1"])
        grads["W1"] = numerical_gradient(loss, self.params["W1"])
        grads["B1"] = numerical_gradient(loss, self.params["B1"])

        return grads