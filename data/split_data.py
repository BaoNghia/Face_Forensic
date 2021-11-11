import os, glob
import argparse
import pandas as pd

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
        df = pd.DataFrame(data)

    elif args.dataset == "deepfakedetection":
        for label in os.listdir(args.input_path):
            for key, value in DEEPFEAKES_DETECTION_DATASET.items():
                if label in value:
                    data_path = os.path.join(args.input_path, value, "**/*.png")
                    files = [f for f in glob.glob(data_path, recursive=True)]
                    data['image'].extend(files)
                    data['class'].extend([label_dict[label]]*len(files))
        df = pd.DataFrame(data)
