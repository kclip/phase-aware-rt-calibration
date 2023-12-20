import tensorflow as tf
from src.ofdm_measurements.main import sum_paths_of_cfr_measurements


def least_squares_loss(
    channel_measurements: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]
    cfr_train: tf.Tensor,  # [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
) -> tf.float32:
    """
    Train assuming ray-tracing computes the perfect phases for each path (minimise MSE between simulation and
    measurements)
    """
    # Compute predicted measurement by summing the paths contributions of the RT generated CFR
    channel_measurements_train = sum_paths_of_cfr_measurements(
        cfr_measurements_per_mpc=cfr_train[tf.newaxis, :, :, :, :, :]
    )

    # Match measurement data shape
    n_measurements = channel_measurements.shape[0]
    channel_measurements_train_tiled = tf.tile(
        channel_measurements_train,
        [n_measurements, 1, 1, 1, 1]
    )

    # Compute normalized MSE
    return (
        tf.reduce_mean(tf.abs(channel_measurements - channel_measurements_train_tiled) ** 2) /
        tf.reduce_mean(tf.abs(channel_measurements) ** 2)
    )
