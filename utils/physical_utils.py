import numpy as np
import pandas as pd
from datetime import timedelta, datetime

from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit


def get_orbital_elements_from_vectors(x, y, z, Vx, Vy, Vz, total_seconds):
    r = [x, y, z] * u.km
    v = [Vx, Vy, Vz] * u.km / u.s
    dt = datetime(2014, 1, 1) + timedelta(seconds=total_seconds)
    ss = Orbit.from_vectors(Earth, r, v, epoch=Time(dt))
    return ss


def get_orbit_from_orbital_elements(orbital_elements: np.array, seconds_from_01_01: float) -> Orbit:
    return Orbit.from_classical(Earth, orbital_elements[0] * u.km, orbital_elements[1] * u.one, orbital_elements[2] * u.rad, \
                                orbital_elements[3] * u.rad, orbital_elements[4] * u.rad, orbital_elements[5] * u.rad,\
                                epoch=Time(datetime(2014, 1, 1) + timedelta(seconds=seconds_from_01_01)))


def get_vectors_from_orbit(ss: Orbit, delta_seconds: float) -> np.array:
    ss = ss.propagate(delta_seconds * u.s)
    return np.concatenate([ss.r.to_value(), ss.v.to_value()])


def add_orbit_elements_to_df(df: pd.DataFrame) -> pd.DataFrame:

    df['orbit'] = df.apply(lambda r: get_orbital_elements_from_vectors(r.x, r.y, r.z, r.Vx, r.Vy, r.Vz, r.total_seconds), axis=1)

    lambdafunc = lambda r: pd.Series([r['orbit'].a.to_value(),
                                      r['orbit'].ecc.to_value(),
                                      r['orbit'].inc.to_value(),
                                      r['orbit'].raan.to_value(),
                                      r['orbit'].argp.to_value(),
                                      r['orbit'].nu.to_value()])

    df[['a', 'ecc', 'inc', 'raan', 'argp', 'nu']] = df.apply(lambdafunc, axis=1)

    return df
