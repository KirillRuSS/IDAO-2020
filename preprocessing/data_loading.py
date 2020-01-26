import os.path
import numpy as np
import pandas as pd
from datetime import datetime, date, time

import preprocessing.extra_points_removing as epr

import config as c


def load_train_dataframe():
    df = None

    if os.path.isfile(c.DATASET_DIR + 'cor_' + c.TRAIN_PATH):
        df = pd.read_csv(c.DATASET_DIR + 'cor_' + c.TRAIN_PATH)
    else:
        df = pd.read_csv(c.TRAIN_PATH)
        epr.remove_excess_points(df)
        df.to_csv(c.DATASET_DIR + 'cor_' + c.TRAIN_PATH, index=False, sep=',')

    return df


def load_test_dataframe():
    df = None

    if os.path.isfile(c.DATASET_DIR + 'cor_' + c.TEST_PATH):
        df = pd.read_csv(c.DATASET_DIR + 'cor_' + c.TEST_PATH)
    else:
        df = pd.read_csv(c.TEST_PATH)
        epr.remove_excess_points(df)
        df.to_csv(c.DATASET_DIR + 'cor_' + c.TEST_PATH, index=False, sep=',')

    return df
