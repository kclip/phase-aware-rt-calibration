import numpy as np
import tensorflow as tf
from sionna.channel import subcarrier_frequencies
from sionna.rt import Paths

from src.utils.tensor_utils import cast_to_pure_imag
from src.utils.sionna_utils import set_path_delays_normalization


def compute_frequency_response_from_paths(
    paths: Paths,
    freq_axis_baseband: tf.Tensor,  # Frequencies f at which the response is computed, must be in baseband form (f - f_carrier) ; shape [N_POINTS]
    normalization_constant: float = 1.0,
    normalize_delays: bool = False
) -> tf.Tensor:  # Shape [N_RX, N_TX, N_POINTS, N_ARR_RX, N_ARR_TX, N_PATH]
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

    # Compute channel frequency responses (CFR)
    freq_delay = (  # [N_SUBCARRIERS, N_RX, N_TX, N_ARR_RX, N_ARR_TX, N_PATH]
        -2 * np.pi *
        freq_axis_baseband[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis] *
        delays
    )
    phase_shift_propagation = tf.exp(cast_to_pure_imag(freq_delay))

    normalized_cfr_per_mpc = (  # [N_SUBCARRIERS, N_RX, N_TX, N_ARR_RX, N_ARR_TX, N_PATH]
        phase_shift_propagation *
        amplitudes
    ) / normalization_constant

    # [N_RX, N_TX, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    return tf.transpose(normalized_cfr_per_mpc, [1, 2, 0, 3, 4, 5])


def compute_cfr_per_mpc_from_paths(
    paths: Paths,
    num_subcarriers: int,
    subcarrier_spacing: float,
    normalization_constant: float = 1.0,
    normalize_delays: bool = False
) -> tf.Tensor:  # Shape [N_RX, N_TX, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    """
    Use paths generated within the scene to compute the channel frequency response for each individual multipath
    component
    """
    # Generate array of baseband subcarrier frequencies
    # Note: paths.cir() returns the baseband amplitudes "a_i * e^{- j 2 \pi f_carrier \tau_i}" ; there is no need to add
    # "f_carrier" to <f_subcarriers_baseband> when computing the CFR
    f_subcarriers_baseband = subcarrier_frequencies(
        num_subcarriers=num_subcarriers,
        subcarrier_spacing=subcarrier_spacing
    )

    # Compute frequency response at subcarriers
    return compute_frequency_response_from_paths(
        paths=paths,
        freq_axis_baseband=f_subcarriers_baseband,
        normalization_constant=normalization_constant,
        normalize_delays=normalize_delays
    )


