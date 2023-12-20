from typing import List
import numpy as np
import tensorflow as tf
from sionna.rt import Paths, Scene

from src.utils.tensor_utils import cast_to_pure_imag
from src.utils.sionna_utils import get_subcarrier_frequencies, set_path_delays_normalization


def compute_cfr_per_mpc_from_paths(
    scene: Scene,
    paths: Paths,
    num_subcarriers: int,
    subcarrier_spacing: float,
    carrier_modulation: bool,
    normalization_constant: float = 1.0,
    normalize_delays: bool = False
) -> tf.Tensor:  # Shape [N_RX, N_TX, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    """
    Use paths generated within the scene to compute the channel frequency response for each individual multipath
    component
    """
    # Generate channel impulse response (CIR) using Sionna
    set_path_delays_normalization(paths=paths, normalize_delays=normalize_delays)
    (
        amplitudes_raw,  # ([BATCH=1, N_RX, N_ARR_RX, N_TX, N_ARR_TX, N_PATH, N_TIME_STEPS=1]
        delays_raw  # [BATCH=1, N_RX, N_TX, N_PATH] or [BATCH=1, N_RX, N_ARR_RX, N_TX, N_ARR_TX, N_PATH]
    ) = paths.cir(diffraction=False, scattering=False)

    # Keep only useful dims and re-order
    # <delays> and <amplitudes> shape: [N_RX, N_TX, N_ARR_RX, N_ARR_TX, N_PATH]
    amplitudes = tf.transpose(amplitudes_raw, [0, 6, 1, 3, 2, 4, 5])[0, 0]
    if len(delays_raw.shape) == 4:  # format [BATCH=1, N_RX, N_TX, N_PATH]
        delays = delays_raw[0, :, :, tf.newaxis, tf.newaxis, :]
    elif len(delays_raw.shape) == 6:  # format [BATCH=1, N_RX, N_ARR_RX, N_TX, N_ARR_TX, N_PATH]
        delays = tf.transpose(delays_raw, [0, 1, 3, 2, 4, 5])[0]
    else:
        raise ValueError(f"Unknown delays tensor dimension format : '{delays_raw.shape}'...")

    # Generate array of subcarrier frequencies (with or without carrier modulation)
    subcarrier_frequencies = get_subcarrier_frequencies(
        scene=scene,
        n_subcarriers=num_subcarriers,
        subcarrier_spacing=subcarrier_spacing,
        carrier_modulation=carrier_modulation
    )

    # Compute channel frequency responses (CFR)
    freq_delay = (  # [N_SUBCARRIERS, N_RX, N_TX, N_ARR_RX, N_ARR_TX, N_PATH]
        -2 * np.pi *
        subcarrier_frequencies[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis] *
        delays
    )
    phase_shift_propagation = tf.exp(cast_to_pure_imag(freq_delay))

    normalized_cfr_per_mpc = (  # [N_SUBCARRIERS, N_RX, N_TX, N_ARR_RX, N_ARR_TX, N_PATH]
        phase_shift_propagation *
        amplitudes
    ) / normalization_constant

    # [N_RX, N_TX, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    return tf.transpose(normalized_cfr_per_mpc, [1, 2, 0, 3, 4, 5])
