from typing import Tuple

import numpy as np
import tensorflow as tf
from sionna.rt import Paths

from src.utils.sionna_utils import get_mask_paths, set_path_delays_normalization
from src.utils.tensor_utils import reduce_masked_max, reduce_masked_min


def get_paths_delays(
    paths: Paths,
    normalize_delays: bool = False
) -> tf.Tensor:  # Shape [N_RX, N_TX, N_POINTS=N_PATHS]
    set_path_delays_normalization(paths=paths, normalize_delays=normalize_delays)

    _, delays_raw = paths.cir()
    delays = delays_raw[0]  # Shape [N_RX, N_TX, N_PATHS]

    # Check shape corresponds to a synthetic array (single time of arrival per path)
    # If not synthetic arrays, delays.shape=[N_RX, N_ARR_RX, N_TX, N_ARR_TX, N_PATHS]
    if len(delays.shape) > 3:
        raise ValueError("Projection functions are only defined for synthetic arrays")

    return delays


def get_evenly_spaced_time_lattice(
    paths: Paths,
    bandwidth: float,  # in [Hz]
    normalize_delays: bool = False,
    force_n_points: int = None
) -> tf.Tensor:  # Shape [N_RX, N_TX, N_POINTS]
    """
    Get equidistant points on the time axis between the first and last time of arrival for each (Rx, Tx) pair.
    The number of points is computed from the time resolution of the system (1 / bandwidth) for the longest delay spread
    among the given (Rx, Tx) pairs (max_spread / time_res).
    If <force_n_points> is set, the number of computed points is <force_n_points> and the time resolution of the
    system is bypassed.
    """
    delays = get_paths_delays(paths=paths, normalize_delays=normalize_delays)

    mask_paths = get_mask_paths(paths)
    max_delay = reduce_masked_max(
        tensor=delays,
        mask=mask_paths,
        axis=-1
    )
    min_delay = reduce_masked_min(
        tensor=delays,
        mask=mask_paths,
        axis=-1
    )

    if force_n_points is not None:
        n_points = force_n_points
    else:
        max_delay_spread = tf.reduce_max(max_delay - min_delay)
        n_points = np.ceil(max_delay_spread.numpy() * bandwidth)

    return tf.linspace(
        min_delay,
        max_delay,
        n_points,
        axis=-1
    )
