from tkinter import N
import numpy as np

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads, delta=1e-7):
        if self.m == None:
            self.m = {}
            self.v = {}
            for k, v in params.items():
                self.m[k] = np.zeros_like(v)
                self.v[k] = np.zeros_like(v)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        
        for k in params.keys():
            self.m[k] += (1.0 - self.beta1) * (grads[k] - self.m[k])
            self.v[k] += (1.0 - self.beta2) * (grads[k] ** 2 - self.v[k])
            
            params[k] -= (lr_t * self.m[k]) / (np.sqrt(self.v[k]) + delta)