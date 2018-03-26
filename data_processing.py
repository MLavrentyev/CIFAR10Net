import pickle
import numpy as np

def load_batch_data(path, meta=False):
    with open(path, 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
    
    if meta:
        return dict["label_names"]
    else: 
        return dict["data"], dict["labels"]

def reshape_image_data(data):
    num_entries = data.shape[0]
    return np.reshape(data, (num_entries, 3, 32, 32))
