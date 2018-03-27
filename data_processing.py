import pickle
import numpy as np


def load_all_data(path, meta=False):
    with open(path, 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
    if meta:
        return dict[b"label_names"]
    else:
        return dict[b"data"], dict[b"labels"]


def get_batch(data, batch_size, offset):
    offset = offset % data.shape[0]
    return data[offset : offset + batch_size]


def reshape_image_data(data):
    num_entries = data.shape[0]
    data = np.reshape(data, (num_entries, 3, 32, 32))
    data = np.swapaxes(data, 1, 3)

    return data


def reshape_labels(labels, num_classes):
    array = np.zeros((len(labels), num_classes), dtype=np.float32)

    for row, index in zip(array, labels):
        row[index] = 1.

    return array

#TODO: add data normalization
def normalize_image_data(data):
    data /= 128
    data -= 1
    return data