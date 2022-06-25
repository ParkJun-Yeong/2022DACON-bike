import pandas as pd
import os


if __name__ == "__main__":
    base_path = './dataset'
    path = {'train': 'train.csv', 'test': 'test.csv'}

    raw = {}
    raw['train'] = pd.read_csv(os.path.join(base_path, path['train']))
    raw['test'] = pd.read_csv(os.path.join(base_path, path['test']))

