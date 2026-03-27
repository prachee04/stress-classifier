import pickle

def load_subject(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    chest = data['signal']['chest']
    labels = data['label']

    return chest, labels