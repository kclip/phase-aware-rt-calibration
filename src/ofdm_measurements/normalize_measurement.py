import tensorflow as tf

from src.utils.channel_power import compute_channel_power_uniform_phase_error


def get_measurement_mean_power(
    cfr_measurements_per_mpc: tf.Tensor,  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
) -> tf.Tensor:  # Shape [N_RX_TX_PAIRS]
    # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]
    channel_power_per_channel_coefficient = compute_channel_power_uniform_phase_error(
        cfr_per_mpc=cfr_measurements_per_mpc
    )
    # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS]
    mean_channel_power_per_measurement = tf.reduce_mean(channel_power_per_channel_coefficient, axis=[-1, -2, -3])
    # Shape [N_RX_TX_PAIRS]
    mean_channel_power = tf.reduce_mean(mean_channel_power_per_measurement, axis=0)

    return mean_channel_power
