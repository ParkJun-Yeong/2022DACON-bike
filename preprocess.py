import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

import warnings
pd.set_option('mode.chained_assignment', None)  # SettingWithCopyWarning 경고 무시
warnings.simplefilter(action='ignore', category=[FutureWarning, UserWarning, ]) # FutureWarning 제거

# preprocess 먼저 실행, 아래 메소드는 필요 시 개별로 사용
def get_train_test():
    train = pd.read_csv("./dataset/train.csv")
    test = pd.read_csv("./dataset/test.csv")
    test_label = pd.read_csv("./dataset/raw/test_label.csv")

    # 데이터 처리
    columns = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'PM10',
               'PM2.5', 'humidity', 'sunshine_sum', 'sunshine_rate', 'wind_mean',
               'wind_max', 'month', 'day']
    columns_year = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'PM10',
                    'PM2.5', 'humidity', 'sunshine_sum', 'sunshine_rate', 'wind_mean',
                    'wind_max', 'month', 'day', 'year']  # 'year' 추가

    # 전체 데이터 (이거 사용해도 됨)
    X = np.array(train[columns].values.tolist())  # year 없음
    X_year = np.array(train[columns_year].values.tolist())  # year 있음
    y = np.array(train['rental'].values.tolist())

    X_test = np.array(test[columns].values.tolist())  # year 없음
    X_test_year = np.array(test[columns_year].values.tolist())  # year 있음

    y_test = np.array(test_label['rental'].values.tolist())

    return X, X_year, y, X_test, X_test_year, y_test

# 결측치 제거

if __name__ == "__main__":
    # Load raw dataset
    r_train = pd.read_csv("./dataset/raw/train.csv")
    r_test = pd.read_csv("./dataset/raw/test.csv")

    # 시간에 따른 데이터 양상
    from matplotlib.dates import MonthLocator, DateFormatter

    # date 분리 in train
    train = r_train
    train['date'] = pd.to_datetime(train['date'])
    train['year'] = r_train['date'].dt.year
    train['month'] = r_train['date'].dt.month
    train['day'] = r_train['date'].dt.day
    train = train.drop(['date'], axis=1)

    # sunshine sum 결측치 처리 in train
    nan_idx_ss = train['sunshine_sum'][train['sunshine_sum'].isnull()].index
    print("sunshine sum nan indice: ", nan_idx_ss)

    tgt_ss = train['sunshine_sum'].dropna().values.reshape(-1, 1)
    x_sr = train['sunshine_rate'].drop(nan_idx_ss).values.reshape(-1, 1)

    # LR 학습
    lreg_ss = LinearRegression()
    lreg_ss.fit(x_sr, tgt_ss)

    pred_ss = lreg_ss.predict(train['sunshine_rate'].loc[nan_idx_ss].values.reshape(-1, 1))
    pred_ss = list(pred_ss)
    print(pred_ss)

    # PM10, PM2.5, sunshine_sum 결측치 채우기 in train

    train["PM10"] = train["PM10"].fillna(train["PM10"].mean())
    train["PM2.5"] = train["PM2.5"].fillna(train["PM2.5"].mean())

    for nan, pred in zip(nan_idx_ss, pred_ss):
        train["sunshine_sum"].iloc[nan] = pred

    # precipitation(강수량) 결측치 채우기 in train
    nan_idx_pp = train['precipitation'][train['precipitation'].isnull()].index
    print(nan_idx_pp)

    tgt_pp = train['precipitation'].dropna()
    columns = ['sunshine_rate', 'sunshine_sum', 'month']
    x_srss = train[columns].drop(nan_idx_pp)

    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(x_srss)
    x_scaled = scaler.transform(x_srss)

    # LR 학습
    lreg_pp = LinearRegression()
    lreg_pp.fit(x_scaled, tgt_pp)

    x_nanpp = scaler.transform(train[columns].iloc[nan_idx_pp])
    pred_pp = lreg_pp.predict(x_nanpp)
    pred_pp = list(pred_pp)

    for nan, pred in zip(nan_idx_pp, pred_pp):
        train["precipitation"].iloc[nan] = pred


    # date 분리 in test
    test = r_test
    test['date'] = pd.to_datetime(test['date'])
    test['year'] = r_test['date'].dt.year
    test['month'] = r_test['date'].dt.month
    test['day'] = r_test['date'].dt.day
    test = test.drop(['date'], axis=1)

    # sunshine sum 결측치 처리 in test
    nan_idx_ss = test['sunshine_sum'][test['sunshine_sum'].isnull()].index
    print("sunshine sum nan indice: ", nan_idx_ss)

    tgt_ss = test['sunshine_sum'].dropna().values.reshape(-1, 1)
    x_sr = test['sunshine_rate'].drop(nan_idx_ss).values.reshape(-1, 1)

    # LR 학습
    lreg_ss = LinearRegression()
    lreg_ss.fit(x_sr, tgt_ss)

    pred_ss = lreg_ss.predict(test['sunshine_rate'].loc[nan_idx_ss].values.reshape(-1, 1))
    pred_ss = list(pred_ss)
    print(pred_ss)

    # PM10, PM2.5, sunshine_sum 결측치 채우기 in test
    test["PM10"] = test["PM10"].fillna(test["PM10"].mean())
    test["PM2.5"] = test["PM2.5"].fillna(test["PM2.5"].mean())

    for nan, pred in zip(nan_idx_ss, pred_ss):
        test["sunshine_sum"].iloc[nan] = pred

    # precipitation(강수량) 결측치 채우기 in test
    nan_idx_pp = test['precipitation'][test['precipitation'].isnull()].index
    print(nan_idx_pp)

    tgt_pp = test['precipitation'].dropna()
    columns = ['sunshine_rate', 'sunshine_sum', 'month']
    x_srss = test[columns].drop(nan_idx_pp)

    # Scaling
    scaler = MinMaxScaler()
    scaler.fit(x_srss)
    x_scaled = scaler.transform(x_srss)

    # LR 학습
    lreg_pp = LinearRegression()
    lreg_pp.fit(x_scaled, tgt_pp)

    x_nanpp = scaler.transform(test[columns].iloc[nan_idx_pp])
    pred_pp = lreg_pp.predict(x_nanpp)
    pred_pp = list(pred_pp)

    for nan, pred in zip(nan_idx_pp, pred_pp):
        test["precipitation"].iloc[nan] = pred

    # train 결측치 전부 채움
    print("train null sum: \n", train.isnull().sum())
    # test 결측치 전부 채움
    print("train null sum: \n", train.isnull().sum())

    print("save processed dataset...")
    train.to_csv("./dataset/train.csv", sep=',', na_rep="NaN", index_label="index")
    test.to_csv("./dataset/test.csv", sep=',', na_rep="NaN", index_label="index")


    # sunshine sum 결측치 처리

