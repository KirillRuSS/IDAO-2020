import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from utils.math_utils import mean_without_k_outlies


def get_sat_list(df: pd.DataFrame):
    sat_list = []
    for sat_id in range(600):
        sat = {}
        sat['sat_id'] = sat_id
        sat['d'] = df[df['sat_id'] == sat_id]
        sat_list.append(sat)

    # Решаем проблему с непрерывностью argp, raan и inc
    for sat_id in range(600):
        sat = sat_list[sat_id]
        d = sat['d']

        if max(d['argp']) > 6 * min(d['argp']) < 0.3:
            d.loc[:, 'argp'] = d['argp'] + (d['argp'] < np.mean(d['argp'])) * np.pi * 2

        if max(d['raan']) > 6 * min(d['raan']) < 0.3:
            d.loc[:, 'raan'] = d['raan'] + (d['raan'] < np.mean(d['raan'])) * np.pi * 2

        if max(d['inc']) > 6 * min(d['inc']) < 0.3:
            d.loc[:, 'inc'] = d['inc'] + (d['inc'] < np.mean(d['inc'])) * np.pi * 2

        sat['d'] = d

    # Усредняем параметры орбиты

    averaged_columns = ['a', 'ecc', 'inc', 'raan', 'argp']
    for sat_id in range(600):
        sat = sat_list[sat_id]
        d = sat['d']

        main_part = d[averaged_columns][:len(d) // 24 * 24].to_numpy().reshape((-1, 24, 5))
        residue = d[averaged_columns][-24:].to_numpy()

        averaged_values = np.apply_along_axis(mean_without_k_outlies, 1, main_part)
        averaged_values = np.append(averaged_values,
                                    np.apply_along_axis(mean_without_k_outlies, 0, residue).reshape((1, 5)), axis=0)

        x = np.linspace(0, 1, len(averaged_values))

        a = interp1d(x, averaged_values[:, 0], kind='cubic')
        a = a(np.linspace(0, 1, len(d)))

        ecc = interp1d(x, averaged_values[:, 1], kind='cubic')
        ecc = ecc(np.linspace(0, 1, len(d)))

        inc = interp1d(x, averaged_values[:, 2], kind='cubic')
        inc = inc(np.linspace(0, 1, len(d)))

        raan = interp1d(x, averaged_values[:, 3], kind='cubic')
        raan = raan(np.linspace(0, 1, len(d)))

        argp = interp1d(x, averaged_values[:, 4], kind='cubic')
        argp = argp(np.linspace(0, 1, len(d)))

        d.loc[:, ['a']] = a
        d.loc[:, ['ecc']] = ecc
        d.loc[:, ['inc']] = inc
        d.loc[:, ['raan']] = raan
        d.loc[:, ['argp']] = argp
        d.loc[:, ['nu']] = d['nu']# - d['argp']

        sat['d'] = d
    return sat_list

