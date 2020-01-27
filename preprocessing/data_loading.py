import os.path
import numpy as np
import pandas as pd
from datetime import datetime, date, time

import preprocessing.extra_points_removing as epr

import config as c


def load_train_dataframe():
    if os.path.isfile(c.DATASET_DIR + 'cor_' + c.TRAIN_CSV):
        df = pd.read_csv(c.DATASET_DIR + 'cor_' + c.TRAIN_CSV)
    else:
        df = pd.read_csv(c.DATASET_DIR + c.TRAIN_CSV)
        df = epr.remove_excess_points(df)
        df.to_csv(c.DATASET_DIR + 'cor_' + c.TRAIN_CSV, index=False, sep=',')

    return df


def load_test_dataframe():
    if os.path.isfile(c.DATASET_DIR + 'cor_' + c.TEST_CSV):
        df = pd.read_csv(c.DATASET_DIR + 'cor_' + c.TEST_CSV)
    else:
        df = pd.read_csv(c.DATASET_DIR + c.TEST_CSV)
        df = epr.remove_excess_points(df)
        df.to_csv(c.DATASET_DIR + 'cor_' + c.TEST_CSV, index=False, sep=',')

    return df