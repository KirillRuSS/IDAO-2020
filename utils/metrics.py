import numpy as np
import keras.backend as K


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


def smape_loss():
    """
    Функция, для передачи метрики керасу
    :return: функция потерь
    """
    def loss(satellite_predicted_values, satellite_true_values):
        return K.mean(K.abs((satellite_predicted_values - satellite_true_values)
                            / (K.abs(satellite_predicted_values) + K.abs(satellite_true_values))))

    return loss
