"""
Generate a fully-prepared PhysioNet-2012 dataset and save into files for PyPOTS to use.
"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

import sys
import os
import pandas as pd
from mcar_augmentation import mcar_augmentation
from pypots.data.saving import save_dict_into_h5, pickle_dump
from pypots.utils.logging import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset_config import ARTIFICIALLY_MISSING_RATE, FOLD
import pickle
import warnings
warnings.filterwarnings("ignore")

saving_dir = "data/eicu"

if __name__ == "__main__":
    data_path='../healthcare_datasets/eicu/data_nan.pkl'
    label_path='../healthcare_datasets/eicu/label.pkl'

    # Loading the kfold dataset
    kfold_data = pickle.load(open(data_path, 'rb'))
    kfold_label = pickle.load(open(label_path, 'rb'))

    # Get dataset
    train_data = kfold_data[FOLD][0]
    train_label = kfold_label[FOLD][0]

    valid_data = kfold_data[FOLD][1]
    valid_label = kfold_label[FOLD][1]

    test_data = kfold_data[FOLD][2]
    test_label = kfold_label[FOLD][2]

    train_X = train_data.reshape(-1, train_data.shape[-1])
    val_X = valid_data.reshape(-1, valid_data.shape[-1])
    test_X = test_data.reshape(-1, test_data.shape[-1])

    # normalization
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    val_X = scaler.transform(val_X)
    test_X = scaler.transform(test_X)

    # reshape into time series samples
    train_X = train_X.reshape(train_data.shape[0], 48, -1)
    val_X = val_X.reshape(valid_data.shape[0], 48, -1)
    test_X = test_X.reshape(test_data.shape[0], 48, -1)

    train_set_dict = {
        "X": train_X,
        "y": train_label,
    }

    # mask values in the validation set as ground truth
    val_X_ori = val_X
    val_X = mcar_augmentation(val_X, ARTIFICIALLY_MISSING_RATE)
    val_set_dict = {
        "X": val_X,
        "X_ori": val_X_ori,
        "y": valid_label,
    }

    # mask values in the test set as ground truth
    test_X_ori = test_X
    test_X = mcar_augmentation(test_X, ARTIFICIALLY_MISSING_RATE)
    test_set_dict = {
        "X": test_X,
        "X_ori": test_X_ori,
        "y": test_label,
    }

    save_dict_into_h5(train_set_dict, saving_dir, "train.h5")
    save_dict_into_h5(val_set_dict, saving_dir, "val.h5")
    save_dict_into_h5(test_set_dict, saving_dir, "test.h5")
    pickle_dump(scaler, os.path.join(saving_dir, "scaler.pkl"))

    logger.info(f"Total sample number: {len(train_X) + len(val_X) + len(test_X)}")
    logger.info(f"Number of steps: {train_X.shape[1]}")
    logger.info(f"Number of features: {train_X.shape[2]}")

    logger.info("All done.")
