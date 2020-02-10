import numba
import numpy as np
from utils.stumpff import c2, c3


def kepler_numba(r0, v0, tof, k=398600.44180000003, numiter=35, rtol=1e-10):
    """Propagates Keplerian orbit.

    Parameters
    ----------
    k : float
        Гравитационная постоянная главного аттрактора (km^3 / s^2).
    r0 : array
        Initial position (km).
    v0 : array
        Initial velocity (km).
    tof : float
        Время полета (s).
    numiter : int, optional
        Максимальное количество итераций, по умолчанию 35.
    rtol : float, optional
        Максимально допустимая относительная ошибка, по умолчанию 1e-10.

    Raises
    ------
    RuntimeError
        If the algorithm didn't converge.

    Notes
    -----
    Этот алгоритм основан на реализации Vallado и выполняет базовую итерацию Ньютона
    для уравнения Кеплера, написанного с использованием универсальных переменных.
    Battin утверждает, что его алгоритм использует тот же объем памяти,
    но работает на 40-85% быстрее.
    """
    if tof == 0.0:
        tof = 1e-10

    # Compute Lagrange coefficients
    try:
        f, g, fdot, gdot = _kepler(k, r0, v0, tof, numiter, rtol)
    except RuntimeError:
        raise RuntimeError("Convergence could not be achieved under "
                           "%d iterations" % numiter)

    assert np.abs(f * gdot - fdot * g - 1) < 1e-5  # Fixed tolerance

    # Return position and velocity vectors
    r = f * r0 + g * v0
    v = fdot * r0 + gdot * v0


    return np.concatenate((r, v), axis=0)


@numba.njit('f8(f8[:], f8[:])')
def dot(u, v):
    dp = 0.0
    for ii in range(u.shape[0]):
        dp += u[ii] * v[ii]
    return dp


@numba.njit
def _kepler(k, r0, v0, tof, numiter, rtol):
    # Cache some results
    dot_r0v0 = dot(r0, v0)
    norm_r0 = dot(r0, r0) ** .5
    sqrt_mu = k ** .5
    alpha = -dot(v0, v0) / k + 2 / norm_r0


    # First guess
    if alpha > 0:
        # Elliptic orbit
        xi_new = sqrt_mu * tof * alpha
    elif alpha < 0:
        # Hyperbolic orbit
        xi_new = (np.sign(tof) * (-1 / alpha) ** .5 *
                  np.log((-2 * k * alpha * tof) / (dot_r0v0 + np.sign(tof) * np.sqrt(-k / alpha) * (1 - norm_r0 * alpha))))

    # Newton-Raphson iteration on the Kepler equation
    count = 0
    while count < numiter:
        xi = xi_new
        psi = xi * xi * alpha
        c2_psi = c2(psi)
        c3_psi = c3(psi)
        norm_r = xi * xi * c2_psi + dot_r0v0 / sqrt_mu * xi * (1 - psi * c3_psi) + norm_r0 * (1 - psi * c2_psi)
        xi_new = xi + (sqrt_mu * tof - xi * xi * xi * c3_psi - dot_r0v0 / sqrt_mu * xi * xi * c2_psi -
                       norm_r0 * xi * (1 - psi * c3_psi)) / norm_r
        if abs((xi_new - xi) / xi_new) < rtol:
            break
        else:
            count += 1
    else:
        raise RuntimeError

    # Compute Lagrange coefficients
    f = 1 - xi ** 2 / norm_r0 * c2_psi
    g = tof - xi ** 3 / sqrt_mu * c3_psi

    gdot = 1 - xi ** 2 / norm_r * c2_psi
    fdot = sqrt_mu / (norm_r * norm_r0) * xi * (psi * c3_psi - 1)


    return f, g, fdot, gdot
