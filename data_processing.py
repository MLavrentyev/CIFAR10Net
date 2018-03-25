import pickle

def load_batch_data(path, meta=False):
    with open(path, 'rb') as file:
        dict = pickle.load(file, encoding='bytes')
    
    if meta:
        return dict["label_names"]
    else: 
        return dict["data"], dict["labels"]

