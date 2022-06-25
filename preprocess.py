import pandas as pd
import os


# 결측치 제거

if __name__ == "__main__":
    base_path = './dataset'
    path = {'train': 'train.csv', 'test': 'test.csv'}

    raw = {}
    raw['train'] = pd.read_csv(os.path.join(base_path, path['train']))
    raw['test'] = pd.read_csv(os.path.join(base_path, path['test']))

    # sunshine sum 결측치 처리

