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


def reshape_image_data(data, to_grayscale=False):
    num_entries = data.shape[0]
    width = 32
    height = 32
    shaped_data = np.reshape(data, (num_entries, 3, height, width))

    if to_grayscale:
        shaped_data = 0.299 * shaped_data[:, 0]+ 0.587 * shaped_data[:, 1] + 0.114 * shaped_data[:, 2]
        shaped_data = np.reshape(shaped_data, (num_entries, 1, height, width))

    shaped_data = np.swapaxes(shaped_data, 1, 3)
    shaped_data = np.swapaxes(shaped_data, 1, 2)

    shaped_data = shaped_data.astype(dtype=np.uint8)

    return shaped_data


def reshape_labels(labels, num_classes):
    array = np.zeros((len(labels), num_classes), dtype=np.float32)

    for row, index in zip(array, labels):
        row[index] = 1.

    return array


def normalize_image_data(data):
    return (data / 255)

if __name__ == "__main__":
    all_data = load_all_data("data/cifar-10-batches-py/data")[0]
    data_1 = reshape_image_data(all_data)
    data_2 = reshape_image_data(all_data, to_grayscale=True)
    print(data_2[0].shape)
    print(data_2[0])
    print(data_1[0].shape)

    from PIL import Image

    gimg = Image.fromarray(data_2[7, :, :, 0], 'L')
    gimg.save("gray.png")

    img = Image.fromarray(data_1[7], 'RGB')
    g2_img = img.convert("L")
    g2_img.save("gray2.png")
    img.save("normal.png")