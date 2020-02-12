import numpy as np
import pandas as pd
import config as c
from datetime import datetime

from LinearRegressionModel import load_models
from utils.kepler_utils import kepler_numba

import time

t0 = time.time()
print(time.time() - t0)

models = load_models('models')

df_test = pd.read_csv("test.csv")
df_test.epoch = pd.to_datetime(df_test.epoch)
df_test['total_seconds'] = (df_test['epoch'] - datetime(2014,1,1)).dt.total_seconds()
print(time.time() - t0)

predictions = []

for sat_id in df_test['sat_id'].unique():
    d = df_test[df_test['sat_id'] == sat_id]

    predictions.append(models[sat_id].predict_test_df(d, -10))
print(time.time() - t0)
predictions = np.concatenate(predictions, axis=0)

df_test[['x', 'y', 'z', 'Vx', 'Vy', 'Vz']] = pd.DataFrame(predictions, columns=['x', 'y', 'z', 'Vx', 'Vy', 'Vz'])
df_test[["id", "x", "y", "z", "Vx", "Vy", "Vz"]].to_csv("submission.csv", index=False)
print(time.time() - t0)