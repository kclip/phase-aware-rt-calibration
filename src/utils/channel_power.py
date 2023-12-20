import tensorflow as tf

from src.utils.tensor_utils import cast_to_pure_imag, compute_bessel_ratio


def compute_channel_power_uniform_phase_error(
    # Shape [*DIMS, N_PATHS] (usually [N_RX, N_TX, N_CARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    # or [N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX, N_PATH])
    cfr_per_mpc: tf.Tensor
) -> tf.Tensor:  # Shape DIMS (usually [N_RX, N_TX, N_CARRIERS, N_ARR_RX, N_ARR_TX] or [N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX])
    """
    Channel power for uniformly distributed phase errors (sum of the power of each path).
    Power is computed analytically for each set of (Rx, Tx) pair, subcarriers and antenna pairs.
    """
    return tf.reduce_sum(
        tf.pow(tf.abs(cfr_per_mpc), 2),
        axis=-1
    )


def compute_channel_power_rt_simulated_phases(
    # Shape [*DIMS, N_PATHS] (usually [N_RX, N_TX, N_CARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    # or [N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX, N_PATH])
    cfr_per_mpc: tf.Tensor
) -> tf.Tensor:  # Shape DIMS (usually [N_RX, N_TX, N_CARRIERS, N_ARR_RX, N_ARR_TX] or [N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX])
    """
    Channel power with phases predicted from the ray tracer (power of the sum of all paths based on the
    RT-simulated phases).
    Power is computed for each set of (Rx, Tx) pair, subcarriers and antenna pairs.
    """
    return tf.pow(
        tf.abs(tf.reduce_sum(cfr_per_mpc, axis=-1)),
        2
    )


def weight_rt_predicted_and_uniform_phase_power_maps_by_von_mises_concentration(
    power_uniform_phases: tf.Tensor,  # Shape [dim1, ..., dimN]
    power_rt_simulated_phases: tf.Tensor,  # Shape [dim1, ..., dimN]
    von_mises_concentration: float,  # greater than 0
) -> tf.Tensor:
    bessel_ratio_squared = tf.pow(
        compute_bessel_ratio(tf.constant(von_mises_concentration, dtype=tf.float32)),
        2
    )
    power_von_mises_weighted = (
        ((1 - bessel_ratio_squared) * power_uniform_phases) +
        (bessel_ratio_squared * power_rt_simulated_phases)
    )

    return power_von_mises_weighted


def compute_channel_power_von_mises_phase_error(
    # Shape [N_RX, N_TX, N_CARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    # or [N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    cfr_per_mpc: tf.Tensor,
    von_mises_prior_concentration: float,  # greater than 0
    von_mises_prior_mean: float = None  # in [-pi, pi[
) -> tf.Tensor:  # Shape [N_RX, N_TX, N_CARRIERS, N_ARR_RX, N_ARR_TX] or [N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]
    """
    Mean channel power for von Mises distributed phase errors.
    Power is computed analytically for each set of (Rx, Tx) pair, subcarriers and antenna pairs.
    """
    # Phase shift from von Mises mean
    if von_mises_prior_mean is not None:
        phase_shift_coefficient = tf.exp(cast_to_pure_imag(
            tf.constant(von_mises_prior_mean, dtype=tf.float32)
        ))
    else:
        phase_shift_coefficient = tf.constant(1.0, dtype=tf.complex64)
    cfr_per_mpc_mean_phase_centered = phase_shift_coefficient * cfr_per_mpc

    # Channel powers for uniform and RT-simulated phases
    power_uniform_phases = compute_channel_power_uniform_phase_error(cfr_per_mpc_mean_phase_centered)
    power_rt_simulated_phases = compute_channel_power_rt_simulated_phases(cfr_per_mpc_mean_phase_centered)

    # Powers weighting depending on von Mises concentration
    return weight_rt_predicted_and_uniform_phase_power_maps_by_von_mises_concentration(
        power_uniform_phases=power_uniform_phases,
        power_rt_simulated_phases=power_rt_simulated_phases,
        von_mises_concentration=von_mises_prior_concentration
    )
