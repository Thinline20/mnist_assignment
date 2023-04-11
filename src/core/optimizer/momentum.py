import numpy as np

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9) -> None:
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v == None:
            self.v = {}
            for k, v in params.items():
                self.v[k] = np.zeros_like(v)
                
        for k in params.keys():
            self.v[k] = self.momentum * self.v[k] + self.lr * grads[k]
            params[k] += self.v[k]