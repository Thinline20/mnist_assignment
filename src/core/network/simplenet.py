import numpy as np

from core.functions import *

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, X):
        return np.dot(X, self.W)

    def loss(self, X, T):
        return cee(softmax(self.predict(X)), T)