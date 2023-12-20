from typing import Tuple
import tensorflow as tf

from src.utils.tensor_utils import sample_complex_standard_normal_tensor


def get_additive_gaussian_noise(
    measurement_without_additive_noise: tf.Tensor,  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]
    mean_channel_power: tf.Tensor,  # Power of the measured channel per Rx-Tx pair ; shape [N_RX_TX_PAIRS]
    measurement_snr: float,  # Define additive noise power from SNR
    normalization_constant: float  # Normalization constant of the CFR (avoid numerical errors due to small values)
) -> Tuple[
    tf.Tensor,  # Normalized measurement noise std ; shape [N_RX_TX_PAIRS]
    tf.Tensor  # Noise sample ; shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]
]:
    n_measurements = measurement_without_additive_noise.shape[0]
    n_carriers = measurement_without_additive_noise.shape[2]
    n_arr_rx = measurement_without_additive_noise.shape[3]
    n_arr_tx = measurement_without_additive_noise.shape[4]

    # Compute Gaussian noise std such that SNR = mean_channel_power / std^2
    measurement_noise_std = tf.sqrt(mean_channel_power / tf.constant(measurement_snr, dtype=tf.float32))
    normalized_measurement_noise_std = measurement_noise_std / normalization_constant

    # Sample additive Gaussian noise
    normalized_measurement_noise_std_tiled = tf.tile(  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]
        normalized_measurement_noise_std[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis],
        [n_measurements, 1, n_carriers, n_arr_rx, n_arr_tx]
    )
    additive_gaussian_noise = sample_complex_standard_normal_tensor(
        shape=measurement_without_additive_noise.shape,
        std=normalized_measurement_noise_std_tiled,
    )

    return normalized_measurement_noise_std, additive_gaussian_noise
