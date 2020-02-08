import numpy as np
import pandas as pd
from datetime import datetime, date, time


def remove_excess_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Скрипт преднозначен для удаления лишних точек, которые присутствуют в исходных данных, и появились вероятно в результате некорректной склейки данных
    :param df: исходная таблица точек
    :return: таблица точек без лишних точек и смещенными симулированными параметрами
    """
    df['epoch'] = df.epoch.map(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f'))
    df['total_seconds'] = df.epoch.map(lambda x: (x - datetime(2014, 1, 1)).total_seconds())
    df['delta_time'] = 0

    drop_list = []

    for sat_id in df['sat_id'].unique():
        sat = df[df['sat_id'] == sat_id]
        sat['delta_time'] = sat['total_seconds'].diff()

        # Дописываем в первую ячейку дельту времени из второй
        sat.loc[list(sat.index)[0], 'delta_time'] = sat.loc[list(sat.index)[1], 'delta_time']

        sat_array = sat[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']].to_numpy()
        new_sat_array = sat[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']].to_numpy()

        j = 0
        for i, dt in enumerate(sat['delta_time']):
            if dt < 1:
                j += 1
            new_sat_array[i] = sat_array[i - j]

        sat[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']] = new_sat_array

        drop_list += list(sat[sat['delta_time'] < 1].index)

        df[df['sat_id'] == sat_id] = sat

    return df.drop(drop_list)


def remove_outlies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Скрипт преднозначен для удаления выбросов в данных
    :param df: исходная таблица точек
    :return: таблица точек без выбросов
    """
    mean_ecc = df['ecc'].rolling(window=24, center=True).sum() / 24
    ecc_deviation = abs(mean_ecc - df['ecc'])

    k = 2
    outlies = ecc_deviation > df['ecc'].rolling(window=24, center=True).std() * 3
    outlies &= (ecc_deviation.shift(24) / ecc_deviation > k) | (ecc_deviation.shift(24) / ecc_deviation < 1 / k)
    outlies &= df['sat_id'].astype('float64') == (df['sat_id'].rolling(window=24, center=True).sum() / 24)

    for index in df[outlies].index:
        df.at[index, ['a', 'ecc', 'inc', 'raan', 'argp']] = (df.iloc[index - 1][['a', 'ecc', 'inc', 'raan', 'argp']] +
                                                             df.iloc[index + 1][['a', 'ecc', 'inc', 'raan', 'argp']]) / 2

    return df


def arg_decomposition_into_sin_cos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Разложение аргумента на sin и cos
    :param df: исходная таблица
    :return: таблица c sin и cos аргумента
    """
    df['sin_argp'] = np.sin(df['argp'])
    df['cos_argp'] = np.cos(df['argp'])

    return df
