import os
import utils
import torch
import numpy as np
import pandas as pd
from data_loader import transforms, sampler
from sklearn.model_selection import train_test_split
from data_preparation.split_data import StratifiedGroupShuffleSplit

def get_label(dataset):
	return dataset.classes

def data_split(data, test_size):
	X = data["image"]
	y = data["label"]
	x_train, x_test, y_train, y_test = train_test_split(X, y, 
														test_size=test_size,
														stratify = y)
	return x_train, x_test, y_train, y_test

def get_dataset(cfg):
	collocation = None
	data = cfg["data"]["data_csv_name"]
	valid_data = cfg["data"]["validation_csv_name"]
	train_set = pd.read_csv(data)

	if (valid_data == ""):
		split_ratio = float(cfg["data"]["validation_ratio"])
		assert split_ratio > 0, 'validation_ratio should greater than 0'
		print("No validation set available, auto split the training into validation")
		print("Splitting dataset into train and valid....")
		train_set, valid_set, _ , _ = data_split(train_set, split_ratio)
		print("Done Splitting !!!")
	else:
		print(f"Creating validation set from file: {valid_data}")
		valid_set = pd.read_csv(valid_data)

	test_data = cfg["data"]["test_csv_name"]
	if test_data == "":
		test_data = valid_set.copy()
	else:
		test_set = pd.read_csv(test_data)
	
	# Get Custom Dataset inherit from torch.utils.data.Dataset
	dataset, module, _ = utils.general.get_attr_by_name(cfg["data"]["data.class"])
	# Create Dataset
	train_data = dataset(train_set, transform = transforms.train_transform)
	valid_data = dataset(valid_set, transform = transforms.val_transform)
	test_data = dataset(test_set, transform = transforms.val_transform)
	return train_data, valid_data, test_data


def get_dataloader(train_data, valid_data, test_data, batch_size = 8):
	kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
	# train_sampler = sampler.ImbalancedDatasetSampler(train_data)
	train_loader = torch.utils.data.DataLoader(
		train_data, sampler = None,
		batch_size=batch_size, **kwargs
	)
	
	valid_loader = torch.utils.data.DataLoader(
		valid_data, sampler = None,
		batch_size=batch_size, **kwargs
	)

	test_loader = torch.utils.data.DataLoader(
		test_data, sampler=None,
		batch_size=batch_size, **kwargs
	)
	return train_loader, valid_loader, test_loader