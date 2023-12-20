import numpy as np
import tensorflow as tf
from sionna.channel import subcarrier_frequencies
from sionna.rt import Scene, Paths

from src.utils.sionna_utils import get_mask_paths
from src.utils.tensor_utils import cast_to_pure_imag, cast_to_complex
from src.projections.time_lattice import get_paths_delays, get_evenly_spaced_time_lattice


def _get_time_projections_complex_coefficients(
    num_subcarriers: int,
    subcarrier_spacing: float,
    time_lattice: tf.Tensor,  # Shape [N_RX, N_TX, N_POINTS]
) -> tf.Tensor:  # Shape [N_RX, N_TX, N_SUBCARRIERS, N_POINTS]
    f_subcarriers_baseband = subcarrier_frequencies(  # Shape [N_SUBCARRIERS]
        num_subcarriers=num_subcarriers,
        subcarrier_spacing=subcarrier_spacing
    )

    # Format shapes
    # To shape [N_RX=1, N_TX=1, N_SUBCARRIERS, N_POINTS=1]
    f_subcarriers_baseband = f_subcarriers_baseband[tf.newaxis, tf.newaxis, :, tf.newaxis]
    time_lattice = time_lattice[:, :, tf.newaxis, :]  # To shape [N_RX, N_TX, N_SUBCARRIERS=1, N_POINTS]

    # Get projection phase vector; shape [N_RX, N_TX, N_SUBCARRIERS, N_POINTS]
    projection_phase = (
        -2 * np.pi *
        f_subcarriers_baseband *
        time_lattice
    )

    return tf.exp(cast_to_pure_imag(projection_phase))


def get_paths_delays_time_projections(
    paths: Paths,
    num_subcarriers: int,
    subcarrier_spacing: float,
) -> tf.Tensor:  # Shape [N_RX, N_TX, N_SUBCARRIERS, N_POINTS=N_PATHS]
    """
    Get the complex coefficients for the time projection of the CFR of each (Rx, Tx) pair.
    For each (Rx, Tx) pairs, the time projection points are taken as the times of arrival of each path.
    Entries that do not correspond to any path (0 padding in the CIR tensor) are set to 0.
    Note: values must be conjugated before being multiplied with the CFR.
    """
    # Get time points (shape [N_RX, N_TX, N_POINTS])
    time_lattice = get_paths_delays(paths=paths)

    # Get projection coefficients ; shape [N_RX, N_TX, N_SUBCARRIERS, N_POINTS]
    projections_coefficients = _get_time_projections_complex_coefficients(
        num_subcarriers=num_subcarriers,
        subcarrier_spacing=subcarrier_spacing,
        time_lattice=time_lattice
    )
    mask_paths = get_mask_paths(paths)  # Shape [N_RX, N_TX, N_PATH]
    mask_paths = cast_to_complex(tf.cast(mask_paths, tf.float32))
    # To shape [N_RX, N_TX, N_SUBCARIERS=1, N_POINTS=N_PATHS]
    mask_paths = mask_paths[:, :, tf.newaxis, :]

    return projections_coefficients * mask_paths


def get_evenly_spaced_time_projections(
    paths: Paths,
    num_subcarriers: int,
    subcarrier_spacing: float,
    force_n_points: int = None  # Force the number of points (ignores bandwidth)
) -> tf.Tensor:  # Shape [N_RX, N_TX, N_SUBCARRIERS, N_POINTS]
    """
    Get the complex coefficients for the time projection of the CFR of each (Rx, Tx) pair.
    For each (Rx, Tx) pairs, the time projection points are between the first and the last time of arrival, where the
    number of points is imposed by the time resolution of the system (i.e., its bandwidth).
    Note: values must be conjugated before being multiplied with the CFR.
    """
    # Get time lattice
    bandwidth = num_subcarriers * subcarrier_spacing
    time_lattice = get_evenly_spaced_time_lattice(  # Shape [N_RX, N_TX, N_POINTS]
        paths=paths,
        bandwidth=bandwidth,
        force_n_points=force_n_points
    )

    # Get projection coefficients
    return _get_time_projections_complex_coefficients(
        num_subcarriers=num_subcarriers,
        subcarrier_spacing=subcarrier_spacing,
        time_lattice=time_lattice
    )
