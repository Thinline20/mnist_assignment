from enum import Enum
import numpy as np

class SGD:
    def __init__(self, lr=0.01) -> None:
        self.lr = lr
        
    def update(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]
            
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
            self.v[k] = self.momentum * self.v[k] - self.lr * grads[k]
            params[k] += self.v[k]

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

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads, epsilon=1e-7):
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
            
            params[k] -= (lr_t * self.m[k]) / (np.sqrt(self.v[k]) + epsilon)

class AdamWScheduler(Enum):
    Cosine_With_Restart = 1
    Cosine = 2

class AdamW:
    def __init__(self, batch_size, train_data_count, epoch, lr=0.01, decay_rate_norm=0.5, beta1=0.9, beta2=0.999, scheduler=AdamWScheduler.Cosine_With_Restart) -> None:
        self.lr = lr
        self.decay_rate = decay_rate_norm * np.sqrt(batch_size / train_data_count * epoch)
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        self.scheduler = scheduler

    def update(self, params, grads, epsilon=1e-7):
        if self.m == None:
            self.m = {}
            self.v = {}
            for k, v in params.items():
                self.m[k] = np.zeros_like(v)
                self.v[k] = np.zeros_like(v)

        self.iter += 1
        
        