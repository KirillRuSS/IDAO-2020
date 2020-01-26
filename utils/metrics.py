import numpy as np


def smape(satellite_predicted_values: np.array, satellite_true_values: np.array) -> float:
    """
    Симметричная средняя абсолютная процентная ошибка
    :param satellite_predicted_values: предсказанное значение
    :param satellite_true_values: истинное значение
    :return: величина ошибки
    """
    return float(np.mean(np.abs((satellite_predicted_values - satellite_true_values)
                                / (np.abs(satellite_predicted_values) + np.abs(satellite_true_values)))))


def score(satellite_predicted_values: np.array, satellite_true_values: np.array) -> float:
    """
    Скор на лидерборде
    :param satellite_predicted_values: предсказанное значение
    :param satellite_true_values: истинное значение
    :return: скор
    """
    return 100 * (1 - smape(satellite_predicted_values, satellite_true_values))
