import numpy as np

def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["B1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["B2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["B3"] = np.array([0.1, 0.2])
    
    return network
    
def forward(X, network, activation_fn, output_classifier):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    B1, B2, B3 = network["B1"], network["B2"], network["B3"]
    
    a1 = np.dot(X, W1) + B1
    z1 = activation_fn(a1)
    a2 = np.dot(z1, W2) + B2
    z2 = activation_fn(a2)
    a3 = np.dot(z2, W3) + B3
    y = output_classifier(a3)
    
    return y

def predict(network, X, activation_fn, output_classifier):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    B1, B2, B3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(X, W1) + B1
    z1 = activation_fn(a1)
    a2 = np.dot(z1, W2) + B2
    z2 = activation_fn(a2)
    a3 = np.dot(z2, W3) + B3
    y = output_classifier(a3)

    return y