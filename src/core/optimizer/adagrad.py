import numpy as np

class AdaGrad:
    def __init__(self, lr=0.01) -> None:
        self.lr = lr
        self.h = None
        
    def update(self, params, grads, delta=1e-7):
        if self.h == None:
            self.h = {}
            for k, v in params.items():
                self.h[k] = np.zeros_like(v)
        
        for k in params.keys():
            self.h[k] += grads[k] * grads[k]
            params[k] -= self.lr * grads[k] / (np.sqrt(self.h[k]) + delta)