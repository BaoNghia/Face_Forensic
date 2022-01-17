import os, glob
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def data_split(data, test_size):
    X = data["image"]
    y = data["class"]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    train_data = data.iloc[x_train.index, :]
    test_data = data.iloc[x_test.index, :]
    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)

def data2csv(path, data, split = True):
    from sklearn.utils.class_weight import compute_class_weight
    os.makedirs(path, exist_ok=True) 
    train, test = data_split(data, test_size = 0.15)
    train, valid = data_split(train, test_size = 0.1764)
    data.to_csv(os.path.join(path, "all_data.csv"), index=None)
    train.to_csv(os.path.join(path, "train.csv"), index=None)
    valid.to_csv(os.path.join(path, "valid.csv"), index=None)
    test.to_csv(os.path.join(path, "test.csv"), index=None)
    return train, valid, test

DATASET_PATHS = {
    'original': 'original_sequences/youtube',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}
DEEPFEAKES_DETECTION_DATASET = {
    'DeepFakeDetection_original': 'original_sequences/actors',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
}
COMPRESSION = ['c0', 'c23', 'c40']

if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input_path', '-i', type=str, default="data/FaceForensics++_frame")
    p.add_argument('--dataset', '-d', type=str,
                   choices=list(DATASET_PATHS.keys()) + ['all'],
                   default='all')
    args = p.parse_args()

    data = {"image": [], "class": []}
    label_dict = {"manipulated_sequences": 1, "original_sequences": 0} 
    if args.dataset == "all" :
        for label in os.listdir(args.input_path):
            for key, value in DATASET_PATHS.items():
                if label in value:
                    data_path = os.path.join(args.input_path, value, "**/*.png")
                    files = [f for f in glob.glob(data_path, recursive=True)]
                    data['image'].extend(files)
                    data['class'].extend([label_dict[label]]*len(files))
        all_data = pd.DataFrame(data)

    elif args.dataset == "deepfakedetection":
        for label in os.listdir(args.input_path):
            for key, value in DEEPFEAKES_DETECTION_DATASET.items():
                if label in value:
                    data_path = os.path.join(args.input_path, value, "**/*.png")
                    files = [f for f in glob.glob(data_path, recursive=True)]
                    data['image'].extend(files)
                    data['class'].extend([label_dict[label]]*len(files))
        all_data = pd.DataFrame(data)

    train, valid, test = data2csv(path = "./data/csv", data = all_data)
    _, train_10 = data_split(train, test_size=0.1)
    _, valid_10 = data_split(valid, test_size=0.1)
    _, test_10 = data_split(test, test_size=0.1)
    path = "./data/csv10"
    os.makedirs(path, exist_ok=True)
    train_10.to_csv(os.path.join(path, "train.csv"), index=None)
    valid_10.to_csv(os.path.join(path, "valid.csv"), index=None)
    test_10.to_csv(os.path.join(path, "test.csv"), index=None)