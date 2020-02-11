import pickle
import config as c
import numpy as np
import pandas as pd
from numpy import sqrt
from sklearn.linear_model import LinearRegression

import utils.metrics as metrics
from utils.kepler_utils import get_next_rv_nu, get_mean_anomaly, kepler_numba, rv_from_elements


class LinearRegressionModel:
    def __init__(self):
        self.delta_time_list = None
        self.positions_self = None
        self.nu_list = None
        self.reg = None
        self.delta_time_reg = None
        self.is_dynamic_time = True

        self.test_score = None
        self.p = None
        self.p_time = None

    def get_x(self, data):
        pd.options.mode.chained_assignment = None
        data.loc[:, 'total_seconds_square'] = data['total_seconds'] ** 1.1
        return np.array(data[['total_seconds', 'total_seconds_square']])

    def fit(self, data):
        self.delta_time_reg = None

        x = self.get_x(data)
        y = data[['a', 'ecc', 'inc', 'raan', 'argp']]

        self.reg = LinearRegression().fit(x, y)

        p = np.array(data[['a', 'ecc', 'inc', 'raan', 'argp', 'nu']].iloc[0])
        positions = self.step_by_step_predict(data, p, 0)

        # Ошибка в определении времени
        self.delta_time_list = []
        for a, e, t1, t0 in zip(data['a'], data['ecc'], data['nu'], self.nu_list):
            mean_anomaly = get_mean_anomaly(a, e, t0, t1)
            if mean_anomaly < -4:
                mean_anomaly += np.pi * 2
            elif mean_anomaly > 4:
                mean_anomaly -= np.pi * 2

            delta_t = mean_anomaly / sqrt(c.mu / a ** 3)
            self.delta_time_list.append(delta_t)

        self.delta_time_reg = LinearRegression().fit(x, np.array(self.delta_time_list))

        return positions

    def predict(self, row):
        p = self.reg.predict(row)
        b = np.zeros((len(p), 6))
        b[:, :-1] = p

        return b


    def predict_time(self, row):
        if self.delta_time_reg is not None:
            return self.delta_time_reg.predict(row)
        else:
            return np.zeros(len(row))

    def test(self, data):
        border = len(data) // 2
        d = data.iloc[:border]
        self.fit(d)

        original_smape = metrics.smape(np.array(d[c.sim_columns]),
                                       np.array(d[c.real_columns]))

        p_time = d['total_seconds'].iloc[-1]
        p = np.array(d[['a', 'ecc', 'inc', 'raan', 'argp', 'nu']])[-1]
        d = data.iloc[border:]

        positions = self.step_by_step_predict(d, p, p_time)
        self.positions_self = positions

        model_smape = metrics.smape(positions, d[c.real_columns].to_numpy())

        self.test_score = (original_smape - model_smape) * len(data)
        return (1 - original_smape) * 100, (1 - model_smape) * 100, original_smape / model_smape

    def step_by_step_predict(self, d, p, p_time):
        positions = []
        self.nu_list = []

        _predict_time = self.predict_time(np.array([[p_time, p_time ** 1.1]]))[0]

        X = self.get_x(d)
        predict_time_array = self.predict_time(X)
        predict_p_array = self.predict(X)

        for i, x in enumerate(X):
            if self.is_dynamic_time:
                rv, nu = get_next_rv_nu(p, x[0] - p_time + (predict_time_array[i] - _predict_time))
            else:
                rv, nu = get_next_rv_nu(p, x[0] - p_time)

            _predict_time = predict_time_array[i]
            positions.append(rv)
            p_time = x[0]
            nu += p[-1]

            self.nu_list.append(nu)

            if p[1] < 0.9:
                p = predict_p_array[i]
            p[-1] = nu
        positions = np.array(positions)

        return positions

    def get_quick_predictions(self, d, p, p_time):
        r1, v1 = rv_from_elements(p[0], p[1], p[2], p[3], p[4], p[5])

        quick_prediction = np.array(list(d['total_seconds'].map(
            lambda t: kepler_numba(r1, v1, t - p_time, numiter=100, rtol=1e-9))))
        return quick_prediction

    def set_p_and_p_time(self, p, p_time):
        self.p = p
        self.p_time = p_time

    def predict_test_df(self, test_df, minimum_allowable_test_score):
        if self.test_score >= minimum_allowable_test_score:
            return self.step_by_step_predict(test_df, self.p, self.p_time)
        else:
            return self.get_quick_predictions(test_df, self.p, self.p_time)


def save_models(model: [], name: str):
    with open('pickle_data/' + name + '.pkl', 'wb') as f:
        pickle.dump(model, f)


def load_models(name: str) -> []:
    with open('pickle_data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
