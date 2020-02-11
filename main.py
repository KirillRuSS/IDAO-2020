import numpy as np
import pandas as pd
from datetime import datetime

#from LinearRegressionModel import load_models
from utils.kepler_utils import kepler_numba

import time

t0 = time.time()
#print(time.time() - t0)

#models = load_models('models')
#if len(models) != 5:
#    a = 0 / 0

df_test = pd.read_csv("test.csv")
df_test.epoch = pd.to_datetime(df_test.epoch)
df_test['total_seconds'] = (df_test['epoch'] - datetime(2014,1,1)).dt.total_seconds()


df = pd.read_csv("df.csv")
quick_predictions = []

for sat_id in df_test['sat_id'].unique():
    sat = df[df['sat_id'] == sat_id]

    r1 = np.array(sat.iloc[-1][['x', 'y', 'z']]).astype('float64')
    v1 = np.array(sat.iloc[-1][['Vx', 'Vy', 'Vz']]).astype('float64')
    total_seconds = sat.iloc[-1]['total_seconds']

    sat = df_test[df_test['sat_id'] == sat_id]

    quick_predictions.append(np.array(list(sat['total_seconds'].map(
        lambda t: kepler_numba(r1, v1, t - total_seconds, numiter=300, rtol=1e-9)))))
quick_predictions = pd.DataFrame(np.concatenate(quick_predictions, axis=0), columns=['x', 'y', 'z', 'Vx', 'Vy', 'Vz'])

df_test[['x', 'y', 'z', 'Vx', 'Vy', 'Vz']] = quick_predictions
df_test[["id", "x", "y", "z", "Vx", "Vy", "Vz"]].to_csv("submission.csv", index=False)
