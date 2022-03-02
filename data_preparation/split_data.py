import os, glob
from pathlib import Path
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm


def calc_mse_loss(category_grouped_df_main, df, class_name, group_id):
    category_grouped_df = df.groupby(class_name).count()[[group_id]]/len(df)*100

    df_temp = category_grouped_df_main.join(category_grouped_df, on = class_name, how = 'left', lsuffix = '_main')
    df_temp.fillna(0, inplace=True)
    df_temp['diff'] = (df_temp['group_id_main'] - df_temp['group_id'])**2
    mse_loss = np.mean(df_temp['diff'])
    return mse_loss


def StratifiedGroupShuffleSplit(df_main, class_name: str, group_id: str, train_proportion=0.7, hparam_mse_wgt = 0.1):
    np.random.seed(1234)
    df_main = df_main.reindex(np.random.permutation(df_main.index)) # shuffle dataset
    # create empty train, val and test datasets
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()

    assert(0 <= hparam_mse_wgt <= 1)
    assert(0 <= train_proportion <= 1)
    val_test_proportion = (1-train_proportion)/2

    category_grouped_df_main = df_main.groupby(class_name).count()[[group_id]]/len(df_main)*100
    subject_grouped_df_main = df_main.groupby([group_id], sort=False, as_index=False)

    # init first 3 group 
    list_group = list(subject_grouped_df_main.groups)
    df_train = pd.concat([df_train, subject_grouped_df_main.get_group(list_group[0])], ignore_index=True)
    df_val = pd.concat([df_val, subject_grouped_df_main.get_group(list_group[1])], ignore_index=True)
    df_test = pd.concat([df_test, subject_grouped_df_main.get_group(list_group[2])], ignore_index=True)

    for id, group in enumerate(list_group[3:], start=3):
        df_group = subject_grouped_df_main.get_group(list_group[id])
        
        tmp_df_train = pd.concat([df_train, df_group], ignore_index=True)
        tmp_df_val = pd.concat([df_val, df_group], ignore_index=True)
        tmp_df_test = pd.concat([df_test, df_group], ignore_index=True)

        # loss for imbalance label
        mse_loss_diff_train = calc_mse_loss(category_grouped_df_main, df_train, class_name, group_id) \
            - calc_mse_loss(category_grouped_df_main, tmp_df_train, class_name, group_id)
        mse_loss_diff_val = calc_mse_loss(category_grouped_df_main, df_val, class_name, group_id) \
            - calc_mse_loss(category_grouped_df_main, tmp_df_val, class_name, group_id)
        mse_loss_diff_test = calc_mse_loss(category_grouped_df_main, df_test, class_name, group_id) \
            - calc_mse_loss(category_grouped_df_main, tmp_df_test, class_name, group_id)

        # loss for train_val_test proportion
        total_records = len(df_train) + len(df_val) + len(df_test)
        len_diff_train = (train_proportion - (len(df_train)/total_records))
        len_diff_val = (val_test_proportion - (len(df_val)/total_records))
        len_diff_test = (val_test_proportion - (len(df_test)/total_records))

        len_loss_diff_train = len_diff_train * abs(len_diff_train)
        len_loss_diff_val = len_diff_val * abs(len_diff_val)
        len_loss_diff_test = len_diff_test * abs(len_diff_test)

        loss_train = (hparam_mse_wgt * mse_loss_diff_train) + ((1-hparam_mse_wgt) * len_loss_diff_train)
        loss_val = (hparam_mse_wgt * mse_loss_diff_val) + ((1-hparam_mse_wgt) * len_loss_diff_val)
        loss_test = (hparam_mse_wgt * mse_loss_diff_test) + ((1-hparam_mse_wgt) * len_loss_diff_test)

        if (max(loss_train,loss_val,loss_test) == loss_train):
            df_train = tmp_df_train
        elif (max(loss_train,loss_val,loss_test) == loss_val):
            df_val = tmp_df_val
        else:
            df_test = tmp_df_test

        print(f"Group {id}/{len(list_group)}. Loss train: {loss_train} | Loss val: {loss_val} | Loss test: {loss_test}")

    return df_train, df_val, df_test


def get_part_of_data(data, size = 0.1):
    from sklearn.model_selection import train_test_split

    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=size, stratify=y, random_state=42)
    train_data = data.iloc[x_train.index, :]
    test_data = data.iloc[x_test.index, :]
    return train_data.reset_index(drop=True), test_data.reset_index(drop=True)
    

def sklearn_StratifiedGroupShuffleSplit(df_main):
    from sklearn.model_selection import StratifiedGroupKFold

    X = df_main.copy()
    y = df_main['class'].values
    groups = df_main['group_id'].values
    orig_ratio = y.mean()
    eps  = 0.1
    print("ORIGINAL POSITIVE RATIO:", orig_ratio)
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True)
    for fold, (train_idxs, test_idxs) in enumerate(cv.split(X, y, groups)):
        print("Fold :", fold)
        print("TRAIN POSITIVE RATIO: ", y[train_idxs].mean())
        print("TEST POSITIVE RATIO: ", y[test_idxs].mean())
        print("LEN TRAIN: ", len(y[train_idxs]))
        print("LEN TEST: ", len(y[test_idxs]))

    return


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

    data = {"image": [], "label": [], 'group_id': []}
    label_dict = {"manipulated_sequences": 1, "original_sequences": 0}
    label_dict = {
        'original': 0,
        'Deepfakes': 1,
        'Face2Face': 2,
        'FaceSwap': 3,
        'FaceShifter': 4,
        'NeuralTextures': 5
    }
    if args.dataset == "all" :
        for label in os.listdir(args.input_path):
            for key, value in DATASET_PATHS.items():
                if label in value:
                    data_path = os.path.join(args.input_path, value, "**/*.png")
                    files = [f for f in glob.glob(data_path, recursive=True)]
                    data['image'].extend(files)
                    data['label'].extend([label_dict[key]]*len(files))
                    meta_data = [f.split("/") for f in files]
                    data['group_id'].extend(f'{f[-5]}/{f[-2]}' for f in meta_data)
        all_data = pd.DataFrame(data)

    elif args.dataset == "deepfakedetection":
        for label in os.listdir(args.input_path):
            for key, value in DEEPFEAKES_DETECTION_DATASET.items():
                if label in value:
                    data_path = os.path.join(args.input_path, value, "**/*.png")
                    files = [f for f in glob.glob(data_path, recursive=True)]
                    data['image'].extend(files)
                    data['label'].extend([label_dict[key]]*len(files))
                    meta_data = [f.split("/") for f in files]
                    data['group_id'].extend(f'{f[-5]}/{f[-2]}' for f in meta_data)
    
    
    df_main = pd.DataFrame(data)
    main_data_path = Path('data/csv6labels/')

    # Split data
    class_name = 'label' # name of y(target) column
    group_id = 'group_id' # name of group column

    if not (main_data_path.exists() and os.listdir(main_data_path)):
        main_data_path.mkdir(parents=True, exist_ok=True)
        df_train, df_val, df_test = StratifiedGroupShuffleSplit(df_main, class_name = class_name, group_id = group_id)
        
        print("Train: ", df_train[class_name].value_counts().to_dict())
        print("Valid", df_val[class_name].value_counts().to_dict())
        print("Test", df_test[class_name].value_counts().to_dict())

        # Check not overlap group id
        train_group_set = df_train[group_id].unique().tolist()
        valid_group_set = df_val[group_id].unique().tolist()
        test_group_set = df_test[group_id].unique().tolist()

        if len(np.intersect1d(train_group_set, valid_group_set)) == 0 \
        and len(np.intersect1d(valid_group_set, test_group_set)) == 0 \
        and len(np.intersect1d(train_group_set, test_group_set)) == 0:
            df_main.to_csv(main_data_path / 'data.csv', index = False)
            df_train.to_csv(main_data_path / 'data_train.csv', index = False)
            df_val.to_csv(main_data_path / 'data_val.csv', index = False)
            df_test.to_csv(main_data_path / 'data_test.csv', index = False)
    else:
        df_train = pd.read_csv(main_data_path / 'data_train.csv')
        df_val = pd.read_csv(main_data_path / 'data_val.csv')
        df_test = pd.read_csv(main_data_path / 'data_test.csv')


    # ## get partial data
    # print("\nGet partial data")
    # time.sleep(1)
    # _, train_10 = get_part_of_data(df_train, size = 0.1)
    # _, valid_10 = get_part_of_data(df_val, size = 0.1)
    # _, test_10 = get_part_of_data(df_test, size = 0.2)

    # print(train_10[class_name].value_counts().to_dict())
    # print(valid_10[class_name].value_counts().to_dict())
    # print(test_10[class_name].value_counts().to_dict())

    
    # partial_path = Path('./data/csv10')
    # partial_path.mkdir(parents=True, exist_ok=True)
    # train_10.to_csv(partial_path / "data_train.csv", index = False)
    # valid_10.to_csv(partial_path / "data_val.csv", index = False)
    # test_10.to_csv(partial_path / "data_test.csv", index = False)

    print("Finished!!!")