from typing import Union, Tuple
import numpy as np
import pandas as pd

from const import FREE_SPACE_PERMITTIVITY


# Decibels conversion
# -------------------

def to_db(
    x: Union[float, np.ndarray, pd.Series],
    output_dbm: bool = False
) -> Union[float, np.ndarray]:
    dbm_coef = 1000 if output_dbm else 1
    return 10 * np.log10(x * dbm_coef)


def from_db(
    x_db: Union[float, np.ndarray, pd.Series],
    input_dbm: bool = False
) -> Union[float, np.ndarray]:
    dbm_coef = 1000 if input_dbm else 1
    return np.power(10, x_db / 10) / dbm_coef


# Antenna arrays
# --------------

def get_lower_bound_angular_resolution_ula(
    n_array_elements: int,
    spacing_array_elements: float  # in multiples of the carrier wavelength
):
    """
    Angular resolution of a uniform linear array (ULA) for any direction theta:
    |cos(theta) - cos(theta + d_theta)| = lambda / array_length
    Under high enough resolutions (i.e. large enough array_length):
    |cos(theta) - cos(theta + d_theta)| \approx |d_theta * sin(theta)|
    The lower bound is obtained by taking |sin(theta)| = 1, yielding |d_theta| = lambda / array_length.
    """
    return 1 / (n_array_elements * spacing_array_elements)


# Materials
# ---------

def get_complex_permittivity(
    real_permittivity: Union[float, np.ndarray, pd.Series],
    conductivity: Union[float, np.ndarray, pd.Series],
    frequency: float
) -> Union[complex, np.ndarray]:
    _real_permittivity = np.asarray(real_permittivity, dtype=np.float32)
    _conductivity = np.asarray(conductivity, dtype=np.float32)
    complex_part = -_conductivity / (
            FREE_SPACE_PERMITTIVITY * 2 * np.pi * frequency
    )
    return _real_permittivity + (1j * complex_part)


def get_material_reflection_coefficients(
    real_permittivity: float,
    conductivity: float,
    frequency: float,
    angle_of_incidence: Union[float, np.ndarray]
) -> Tuple[
    Union[complex, np.ndarray],  # Complex reflection coefficient for polarizations parallel to the incidence plane
    Union[complex, np.ndarray]  # Complex reflection coefficient for polarizations perpendicular to the incidence plane
]:
    """
    Get material complex reflection coefficient for s-polarized and p-polarized fields along
    the given angles of incidence (incident media is assumed to be vacuum-like).
    """
    complex_permittivity = get_complex_permittivity(
        real_permittivity=real_permittivity,
        conductivity=conductivity,
        frequency=frequency
    )
    cos_angle = np.complex64(np.cos(angle_of_incidence))
    sin2_angle = np.complex64(
        np.power(np.sin(angle_of_incidence), 2)
    )
    sqrt_term = np.sqrt(complex_permittivity - sin2_angle)
    r_perpendicular = (cos_angle - sqrt_term) / (cos_angle + sqrt_term)
    r_parallel = (
        (complex_permittivity * cos_angle) - sqrt_term
    ) / (
        (complex_permittivity * cos_angle) + sqrt_term
    )

    return r_parallel, r_perpendicular
