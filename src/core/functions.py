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
    return np.array(1 / (1 + np.exp(-x)))

def relu(x):
    """ReLU function

    Args:
        x (np.array): input data

    Returns:
        np.array: x if bigger than 0, 0 otherwise
    """
    return np.array(np.maximum(0, x))

def softmax(x):
    """softmax

    Args:
        x (np.array): input data

    Returns:
        np.array: (x - e^x) / (sum of x - e^x)
    """
    t = np.exp(x - np.max(x))
    return np.array(t / np.sum(t))

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
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size