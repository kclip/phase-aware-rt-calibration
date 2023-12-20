import tensorflow as tf

from src.utils.tensor_utils import dot


def _compute_projected_power(
    measurements_or_cfr: tf.Tensor,  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATHS]
    time_projections: tf.Tensor,  # Shape [N_RX_TX_PAIRS, N_SUBCARRIERS, N_POINTS_TIME]
    rx_angle_projections: tf.Tensor,  # Shape [N_RX_TX_PAIRS, N_ARR_RX, N_POINTS_ANGLE]
    tx_angle_projections: tf.Tensor  # Shape [N_RX_TX_PAIRS, N_ARR_RX, N_POINTS_ANGLE]
) -> tf.Tensor:  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_POINTS_TIME, N_POINTS_ANGLE, N_POINTS_ANGLE]
    # Time projection
    projected_power = dot(  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_ARR_RX, N_ARR_TX, N_PATHS, N_POINTS_TIME]
        x=measurements_or_cfr[..., tf.newaxis],
        y=tf.math.conj(time_projections[tf.newaxis, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]),
        axis=2
    ) / time_projections.shape[2]  # Normalization of dot product

    # Rx array projection
    projected_power = dot(  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_ARR_TX, N_PATHS, N_POINTS_TIME, N_POINTS_ANGLE]
        x=projected_power[..., tf.newaxis],
        y=tf.math.conj(rx_angle_projections[tf.newaxis, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]),
        axis=2
    ) / rx_angle_projections.shape[2]  # Normalization of dot product

    # Tx array projection
    # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS, N_POINTS_TIME, N_POINTS_ANGLE, N_POINTS_ANGLE]
    projected_power = dot(
        x=projected_power[..., tf.newaxis],
        y=tf.math.conj(tx_angle_projections[tf.newaxis, :, :, tf.newaxis, tf.newaxis, tf.newaxis, :]),
        axis=2
    ) / tx_angle_projections.shape[2]  # Normalization of dot product

    # Get power
    projected_power = tf.pow(
        tf.abs(projected_power),
        2
    )

    # Sum power over path axis
    return tf.reduce_sum(projected_power, axis=2)


def compute_projected_power_cfr(
    cfr_per_mpc: tf.Tensor,  # Shape [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATHS]
    time_projections: tf.Tensor,  # Shape [N_RX_TX_PAIRS, N_SUBCARRIERS, N_POINTS_TIME]
    rx_angle_projections: tf.Tensor,  # Shape [N_RX_TX_PAIRS, N_ARR_RX, N_POINTS_ANGLE]
    tx_angle_projections: tf.Tensor  # Shape [N_RX_TX_PAIRS, N_ARR_RX, N_POINTS_ANGLE]
) -> tf.Tensor:  # Shape [N_MEASUREMENTS=1, N_RX_TX_PAIRS, N_POINTS_TIME, N_POINTS_ANGLE, N_POINTS_ANGLE]
    """
    Compute the predicted channel power from the CFR for the specified time/angle projections, under the assumption of
    uniform phase errors.
    """
    return _compute_projected_power(
        measurements_or_cfr=cfr_per_mpc[tf.newaxis, ...],
        time_projections=time_projections,
        rx_angle_projections=rx_angle_projections,
        tx_angle_projections=tx_angle_projections
    )


def compute_projected_power_measurements(
    channel_measurements: tf.Tensor,  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX]
    time_projections: tf.Tensor,  # Shape [N_RX_TX_PAIRS, N_SUBCARRIERS, N_POINTS_TIME]
    rx_angle_projections: tf.Tensor,  # Shape [N_RX_TX_PAIRS, N_ARR_RX, N_POINTS_ANGLE]
    tx_angle_projections: tf.Tensor  # Shape [N_RX_TX_PAIRS, N_ARR_RX, N_POINTS_ANGLE]
) -> tf.Tensor:  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_POINTS_TIME, N_POINTS_ANGLE, N_POINTS_ANGLE]
    """
    Compute the measured channel power for the specified time/angle projections.
    """
    return _compute_projected_power(
        measurements_or_cfr=channel_measurements[..., tf.newaxis],
        time_projections=time_projections,
        rx_angle_projections=rx_angle_projections,
        tx_angle_projections=tx_angle_projections
    )
