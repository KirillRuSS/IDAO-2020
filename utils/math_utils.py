import numba
import numpy as np
from numpy.linalg import norm
from numpy import (array, dot, arccos, clip)


def normalize(a, k):
    a = a / a.mean() - 1
    a *= k
    a = a - a.min()
    return a


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def mean_without_k_outlies(a, k=2):
    a = np.delete(a, np.argmax(abs(a - np.mean(a))))
    return np.mean(a)


def get_angle(r0: np.array, r1: np.array):
    return np.arccos(clip(dot(r0, r1) / norm(r0) / norm(r1), -1, 1))
