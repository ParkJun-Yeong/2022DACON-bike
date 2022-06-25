import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class BikeDataset(Dataset):
    def __init__(self, train=False, test=False):
        super(BikeDataset, self).__init__()
        if (train & test) | ((not train) & (not test)):
            raise NoOptionException

        self.train_flag = train
        self.test_flag = test

        self.train_path = './dataset/train.csv'
        self.test_path = './dataset/test.csv'

        self.train_df = pd.read_csv(self.train_path)
        self.data_columns = self.train_df.columns.difference(['rental', 'date'])
        self.train_dataset = self.train_df[self.data_columns].values.tolist()
        self.train_label = self.train_df['rental'].values.tolist()

        self.test_df = pd.read_csv(self.test_path)
        self.test_dataset = self.test_df[self.data_columns].values.tolist()

    def __len__(self):
        if self.train_flag:
            ret = len(self.train_dataset)
        elif self.test_flag:
            ret = len(self.test_dataset)
        return ret

    def __getitem__(self, idx):
        if self.train_flag:
            ret = (self.train_dataset[idx], self.train_label[idx])
            # ret = (torch.tensor(self.train_dataset[idx]), self.train_label[idx])
        elif self.test_flag:
            ret = self.test_dataset[idx]
        return ret


def collate_fn(data):
    x, label = zip(*data)

    return torch.tensor(x), label


class NoOptionException(Exception):
    def __init__(self):
        super(NoOptionException, self).__init__("Select Train or Test mode")