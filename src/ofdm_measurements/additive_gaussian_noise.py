from typing import Tuple
import numpy as np
import tensorflow as tf

from src.utils.tensor_utils import sample_complex_standard_normal_tensor


def get_additive_gaussian_noise(
    measurement_without_additive_noise: tf.Tensor,  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]
    mean_channel_power: float,  # Power of the measured channel on average
    measurement_snr: float,  # Define additive noise power from SNR
    normalization_constant: float  # Normalization constant of the CFR (avoid numerical errors due to small values)
) -> Tuple[
    float,  # Normalized measurement noise std
    tf.Tensor  # Noise sample ; shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]
]:
    # TODO: handle power and SNR per (rx, tx) position (note: current experiments only use a single position)

    # Compute Gaussian noise std such that SNR = mean_channel_power / std^2
    measurement_noise_std = np.sqrt(mean_channel_power / np.float32(measurement_snr))
    normalized_measurement_noise_std = float(measurement_noise_std) / normalization_constant

    # Sample additive Gaussian noise
    additive_gaussian_noise = sample_complex_standard_normal_tensor(
        shape=measurement_without_additive_noise.shape,
        std=normalized_measurement_noise_std,
    )

    return normalized_measurement_noise_std, additive_gaussian_noise
