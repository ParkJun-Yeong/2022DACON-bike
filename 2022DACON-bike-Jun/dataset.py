import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning) # FutureWarning 제거
pd.set_option('mode.chained_assignment', None)  # SettingWithCopyWarning 경고 무시


class BikeDataset(Dataset):
    def __init__(self, train=False, test=False):
        super(BikeDataset, self).__init__()
        if (train & test) | ((not train) & (not test)):
            raise NoOptionException

        self.train_flag = train
        self.test_flag = test

        self.train_path = './dataset/train.csv'
        self.test_path = './dataset/test.csv'

        # Train
        df = pd.read_csv(self.train_path).drop(["date", "Unnamed: 0"], axis=1)
        data_columns = df.columns.difference(['rental'])

        self.train_df = df[data_columns]
        self.label_df = df['rental']

        # preprocess (나중에 preprocess.py로 옮기기)
        scaler = MinMaxScaler()
        self.train_df.iloc[:, :-1] = scaler.fit_transform(self.train_df.iloc[:, :-1])
        # self.train_df.iloc[:, :-1] = scaler.transform(self.train_df.iloc[:, :-1])

        self.train_dataset = self.train_df[data_columns].values.tolist()
        self.train_label = self.label_df.values.tolist()

        # Test
        self.test_df = pd.read_csv(self.test_path).drop(["date", "Unnamed: 0"], axis=1)
        self.test_df.iloc[:, :-1] = scaler.transform(self.test_df.iloc[:, :-1])
        self.test_dataset = df[data_columns].values.tolist()

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