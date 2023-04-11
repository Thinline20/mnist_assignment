class SGD:
    def __init__(self, lr=0.01) -> None:
        self.lr = lr
        
    def update(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]