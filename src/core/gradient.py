import numpy as np

def numerical_diff(f, x, delta_x=1e-4):
    return (f(x+delta_x) - f(x-delta_x)) / (2 * delta_x)

def _numerical_gradient_no_batch(f, x, delta_x):
    grad = np.zeros_like(x)
    
    for i in range(x.size):
        x[i] += delta_x
        fxh1 = f(x)
        
        x[i] -= 2 * delta_x
        fxh2 = f(x)
        
        grad[i] = (fxh1 - fxh2) / (2 * delta_x)
        x[i] += delta_x
        
    return grad

def numerical_gradient(f, X, delta_x=1e-4):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X, delta_x)
    
    grad = np.zeros_like(X)
    
    for i, x in enumerate(X):
        grad[i] = _numerical_gradient_no_batch(f, x, delta_x)
    
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    
    for _ in range(step_num):
        x -= lr * numerical_gradient(f, x)

    return x

def gradient_descent_with_history(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = [x.copy()]
    
    for _ in range(step_num):
        x -= lr * numerical_gradient(f, x)
        x_history.append(x.copy())

    return x, np.array(x_history)