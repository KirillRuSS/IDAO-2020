import pandas as pd
from datetime import datetime, date, time


def remove_excess_points(df: pd.datetime) -> pd.datetime:
    """
    Скрипт преднозначен для удаления лишних точек, которые присутствуют в исходных данных, и появились вероятно в результате некорректной склейки данных
    :param df: исходная таблица точек
    :return: таблица точек без лишних точек и смещенными симулированными параметрами
    """
    df['total_seconds'] = df.epoch.map(lambda x: (datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f') - datetime(2014, 1, 1)).total_seconds())
    df['delta_time'] = 0

    drop_list = []

    for sat_id in df['sat_id'].unique():
        sat = df[df['sat_id'] == sat_id]
        sat['delta_time'] = sat['total_seconds'].diff()

        sat_array = sat[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']].to_numpy()
        new_sat_array = sat[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']].to_numpy()

        j = 0
        for i, dt in enumerate(sat['delta_time']):
            if dt < 1:
                j += 1
            new_sat_array[i] = sat_array[i - j]

        sat[['x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']] = new_sat_array

        drop_list.append(list(sat.index)[0])
        drop_list += list(sat[sat['delta_time'] < 1].index)

        df[df['sat_id'] == sat_id] = sat

    return df.drop(drop_list)
