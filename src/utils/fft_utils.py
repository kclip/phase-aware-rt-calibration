from typing import Tuple

import numpy as np
import tensorflow as tf

from src.utils.tensor_utils import cast_to_pure_imag


# Time domain
# -----------

def _get_time_axis(
    freq_axis: tf.Tensor  # Shape [N_POINTS]
) -> tf.Tensor:
    n_points = freq_axis.shape[0]
    bandwidth = freq_axis[-1] - freq_axis[0]
    time_step = 1 / bandwidth
    return tf.linspace(
        start=tf.constant(0.0, dtype=tf.float32),
        stop=(tf.cast(n_points, tf.float32) - 1.0) * time_step,
        num=n_points
    )


def get_time_impulse_response(
    freq_axis: tf.Tensor,  # evenly spaced frequency points from lowest to highest ; [Hz] ; shape [N_POINTS]
    freq_response: tf.Tensor,  # shape [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX, N_ARR_TX] or [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX, N_ARR_TX, N_PATHS]
    central_frequency: float = None  # carrier frequency ; in Hz ; if None, freq_response is treated as the baseband response
) -> Tuple[
    tf.Tensor,  # time axis ; shape [N_POINTS]
    tf.Tensor,  # time impulse response ; shape [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX, N_ARR_TX] or [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX, N_ARR_TX, N_PATHS]
]:
    time_axis = _get_time_axis(freq_axis)
    time_response = np.fft.ifft(
        np.fft.ifftshift(freq_response.numpy(),  axes=(1,)),
        axis=1
    )
    time_response = tf.constant(time_response, dtype=tf.complex64)

    if central_frequency is not None:
        carrier_phasor = tf.exp(cast_to_pure_imag(2 * np.pi * central_frequency * time_axis))
        carrier_phasor = carrier_phasor[tf.newaxis, :, tf.newaxis, tf.newaxis]
        if len(time_response.shape) == 5:  # If tensor has paths dimension
            carrier_phasor = carrier_phasor[..., tf.newaxis]
        time_response = time_response * carrier_phasor

    return time_axis, time_response


def get_power_profile(
    freq_axis: tf.Tensor,  # Shape [N_POINTS]
    freq_response: tf.Tensor,  # Shape [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX, N_ARR_TX] or [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX, N_ARR_TX, N_PATHS]
) -> Tuple[
    tf.Tensor,  # time axis ; shape [N_POINTS]
    tf.Tensor  # Shape [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX, N_ARR_TX] or [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX, N_ARR_TX, N_PATHS]
]:
    time_axis, time_response = get_time_impulse_response(
        freq_axis=freq_axis,
        freq_response=freq_response
    )

    return time_axis, tf.pow(tf.abs(time_response), 2)


# Frequency domain
# ----------------

def filter_freq_response(
    freq_axis: tf.Tensor,  # Shape [N_POINTS]
    freq_response: tf.Tensor,  # Shape [..., N_POINTS, ...]
    axis: int,
    central_frequency: float,
    bandwidth: float
) -> Tuple[
    tf.Tensor,  # Axis of selected frequencies ; shape [N_POINTS_SELECTED]
    tf.Tensor,  # Frequency responses for selected frequencies ; shape [N_RX, N_POINTS_SELECTED]
]:
    indices = tf.squeeze(
        tf.where(
            (freq_axis > (central_frequency - (bandwidth / 2))) &
            (freq_axis < (central_frequency + (bandwidth / 2)))
        )
    )
    return (
        tf.gather(params=freq_axis, indices=indices),
        tf.gather(params=freq_response, indices=indices, axis=axis)
    )
