import urllib.request
import os
import gzip
import pickle
import numpy as np

files = {
    "train_img":"train-images-idx3-ubyte.gz",
    "train_label":"train-labels-idx1-ubyte.gz",
    "test_img":"t10k-images-idx3-ubyte.gz",
    "test_label":"t10k-labels-idx1-ubyte.gz"
}

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

def _convert_filename(filename):
    return f"{dataset_dir}/{filename}"

def _load(path, offset):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), np.uint8, offset=offset)

def _load_label(filename):
    return _load(_convert_filename(filename), 8)

def _load_img(filename):
    return _load(_convert_filename(filename), 16).reshape(-1, img_size)

def _convert_numpy():
    dataset = {}

    dataset["train_img"] = _load_img(files["train_img"])
    dataset["train_label"] = _load_label(files["train_label"])
    dataset["test_img"] = _load_img(files["test_img"])
    dataset["test_label"] = _load_label(files["test_label"])

    return dataset

def init_mnist():
    if os.path.isfile(save_file):
        pass
    
    with open(save_file, "wb") as f:
        pickle.dump(_convert_numpy(), f, -1)

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for i, row in enumerate(T):
        row[X[i]] = 1
        
    return T
    
def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """Load MNIST Dataset
    
    Parameters
    ----------
    normalize     : normalize pixel values of image to 0.0~1.0
    flatten       : flat input image to rank 1 array
    one_hot_label : change label to one-hot array
        (e.g. [0, 0, 1, 0, 0, 0])
        
    Returns
    -------
    (train_img, train_label), (test_image, test_label)
    """

    init_mnist()
    
    with open(save_file, "rb") as f:
        dataset = pickle.load(f)
        
    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset["train_label"] = _change_one_hot_label(dataset["train_label"])
        dataset["test_label"] = _change_one_hot_label(dataset["test_label"])

    if flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset["train_img"], dataset["train_label"]), (dataset["test_img"], dataset["test_label"])