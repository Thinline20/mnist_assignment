from collections import OrderedDict
import numpy as np

from core.functions import *
from core.gradient import *
from core.layers import *


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["B1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["B2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        B1, B2 = self.params["B1"], self.params["B2"]

        a1 = np.dot(x, W1) + B1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + B2
        y = softmax(a2)

        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return cee(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["B1"] = numerical_gradient(loss_W, self.params["B1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["B2"] = numerical_gradient(loss_W, self.params["B2"])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params["W1"], self.params["W2"]
        B1, B2 = self.params["B1"], self.params["B2"]
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + B1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + B2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads["W2"] = np.dot(z1.T, dy)
        grads["B2"] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads["W1"] = np.dot(x.T, dz1)
        grads["B1"] = np.sum(dz1, axis=0)

        return grads

class BackPropagationNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01) -> None:
        self.params = {}
        
        self.params["W1"] = weight_init_std / np.random.randn(input_size, hidden_size)
        self.params["B1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std / np.random.randn(hidden_size, output_size)
        self.params["B2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["B1"])
        self.layers["ReLU1"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["B2"])
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