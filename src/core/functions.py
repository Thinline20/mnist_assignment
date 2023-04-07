import numpy as np

# Activation Functions

def step(x):
    """step function

    Args:
        x (np.array): input data

    Returns:
        np.array: 1 if bigger than 0, 0 otherwise
    """
    return np.array(x > 0, dtype=np.int32)

def sigmoid(x):
    """sigmoid function

    Args:
        x (np.array): input data

    Returns:
        np.array: (1 / 1 + np.exp(-x))
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    """sigmoid gradient

    Args:
        x (np.array): input data

    Returns:
        np.array: (1.0 - sigmoid(x)) * sigmoid(x)
    """
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    """ReLU function

    Args:
        x (np.array): input data

    Returns:
        np.array: x if bigger than 0, 0 otherwise
    """
    return np.maximum(0, x) 

def relu_grad(x):
    """ReLU Gradient

    Args:
        x (np.array): input data
        
    Returns:
        np.array
    """
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad

def softmax(x):
    """softmax

    Args:
        x (np.array): input data

    Returns:
        np.array: (x - e^x) / (sum of x - e^x)
    """
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T
    
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def softplus(x):
    return np.log(1 + np.exp(x))

def swish(x):
    return x * sigmoid(x)

def mish(x):
    return x * tanh(softplus(x))

def gelu_approx(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x ** 3))))

# Loss Functions

def cee(y, t, delta=1e-7):
    """Cross Entropy error with batch processing capability

    Args:
        y (np.array): y
        t (np.array): t
        delta (float, optional): delta. Defaults to 1e-7.

    Returns:
        float: cross entropy
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
