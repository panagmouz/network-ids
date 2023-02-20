import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler, NeighbourhoodCleaningRule, OneSidedSelection
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler


def basic(data, isMulticlass=False):
    y = data[['label', 'mode']]
    if isMulticlass:
        tmp = y['label'].copy()
        tmp = pd.get_dummies(tmp)
        y = y.drop(columns='label', axis=1)
        y = pd.concat([y, tmp], axis=1)

    x = data
    x = x.drop(columns='label', axis=1)

    x_train = x[x['mode'] == 'Train']
    x_test = x[x['mode'] == 'Test']
    x_train = x_train.drop(columns='mode', axis=1)
    x_test = x_test.drop(columns='mode', axis=1)

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = y[y['mode'] == 'Train']
    y_test = y[y['mode'] == 'Test']
    y_train = y_train.drop(columns='mode', axis=1)
    y_test = y_test.drop(columns='mode', axis=1)

    x_train = np.array(x_train, dtype="float32")
    x_test = np.array(x_test, dtype="float32")
    y_train = np.array(y_train.values.tolist(), dtype="float32")
    y_test = np.array(y_test.values.tolist(), dtype="float32")

    features, num_cls = x_train.shape[1], y_train.shape[1]

    return (x_train, x_test, y_train, y_test), (features, num_cls)


def random_sampling_SMOTE(data, args, isMulticlass=False):
    y = data[['label', 'mode']]
    if isMulticlass:
        tmp = y['label'].copy()
        tmp = pd.get_dummies(tmp)
        y = y.drop(columns='label', axis=1)
        y = pd.concat([y, tmp], axis=1)

    x = data
    x = x.drop(columns='label', axis=1)

    x_train = x[x['mode'] == 'Train']
    x_test = x[x['mode'] == 'Test']
    x_train = x_train.drop(columns='mode', axis=1)
    x_test = x_test.drop(columns='mode', axis=1)

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = y[y['mode'] == 'Train']
    y_test = y[y['mode'] == 'Test']
    y_train = y_train.drop(columns='mode', axis=1)
    y_test = y_test.drop(columns='mode', axis=1)

    x_train = np.array(x_train, dtype="float32")
    x_test = np.array(x_test, dtype="float32")
    y_train = np.array(y_train.values.tolist(), dtype="float32")
    y_test = np.array(y_test.values.tolist(), dtype="float32")

    sampling_strategy_down = None
    sampling_strategy_up = None
    if args.isMulticlass:
        if args.dataset == 'KDD':
            sampling_strategy_down = {0: 10000, 2: 10000}
            sampling_strategy_up = {3: 500}
        elif args.dataset == 'NSL_KDD':
            sampling_strategy_down = {0: 10000, 1: 10000, 2: 10000}
            sampling_strategy_up = {3: 500}
        elif args.dataset == 'UNSW_NB15':
            sampling_strategy_down = {0: 20000, 5: 20000, 6: 20000}
            sampling_strategy_up = {3: 2000, 8: 2000, 9: 1000}
        else:
            print("Not valid dataset..Exit")
            exit()

    sm = RandomUnderSampler(sampling_strategy=sampling_strategy_down, random_state=args.seed)
    x_train, y_train = sm.fit_resample(x_train, y_train)

    sm = SMOTE(sampling_strategy=sampling_strategy_up, random_state=args.seed)
    x_train, y_train = sm.fit_resample(x_train, y_train)

    features, num_cls = x_train.shape[1], y_train.shape[1]

    return (x_train, x_test, y_train, y_test), (features, num_cls)


def mySMOTE(data, args, isMulticlass=False):
    y = data[['label', 'mode']]
    if isMulticlass:
        tmp = y['label'].copy()
        tmp = pd.get_dummies(tmp)
        y = y.drop(columns='label', axis=1)
        y = pd.concat([y, tmp], axis=1)

    x = data
    x = x.drop(columns='label', axis=1)

    x_train = x[x['mode'] == 'Train']
    x_test = x[x['mode'] == 'Test']
    x_train = x_train.drop(columns='mode', axis=1)
    x_test = x_test.drop(columns='mode', axis=1)

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = y[y['mode'] == 'Train']
    y_test = y[y['mode'] == 'Test']
    y_train = y_train.drop(columns='mode', axis=1)
    y_test = y_test.drop(columns='mode', axis=1)

    x_train = np.array(x_train, dtype="float32")
    x_test = np.array(x_test, dtype="float32")
    y_train = np.array(y_train.values.tolist(), dtype="float32")
    y_test = np.array(y_test.values.tolist(), dtype="float32")

    sampling_strategy_up = None
    if args.isMulticlass:
        if args.dataset == 'KDD':
            sampling_strategy_up = {3: 500}
        elif args.dataset == 'NSL_KDD':
            sampling_strategy_up = {3: 500}
        elif args.dataset == 'UNSW_NB15':
            sampling_strategy_up = {3: 2000, 8: 2000, 9: 1000}
        else:
            print("Not valid dataset..Exit")
            exit()

    sm = SMOTE(sampling_strategy=sampling_strategy_up, random_state=args.seed)
    x_train, y_train = sm.fit_resample(x_train, y_train)

    features, num_cls = x_train.shape[1], y_train.shape[1]

    return (x_train, x_test, y_train, y_test), (features, num_cls)
