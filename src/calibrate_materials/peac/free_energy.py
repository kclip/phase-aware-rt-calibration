from typing import Union
import numpy as np
import tensorflow as tf

from src.utils.tensor_utils import cast_to_pure_imag, cast_to_complex, squared_norm, compute_log_bessel_i0, \
    compute_bessel_ratio


def _tile_von_mises_scalar_param(
    value: Union[tf.Tensor, tf.Variable],  # Shape []
    n_measurements: int,
    n_rx_tx_pairs: int,
    n_paths: int
) -> tf.Tensor:  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    return tf.tile(
        value[tf.newaxis, tf.newaxis, tf.newaxis],
        (n_measurements, n_rx_tx_pairs, n_paths)
    )


def _cross_entropy_von_mises(
    von_mises_posterior_mean_params: tf.Variable,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    bessel_ratio_posterior: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    von_mises_prior_mean_params: tf.Tensor = None,  # [N_RX_TX_PAIRS, N_PATHS]
    von_mises_prior_concentration_params: tf.Tensor = None  # [N_RX_TX_PAIRS, N_PATHS]
) -> tf.Tensor:
    n_paths = von_mises_posterior_mean_params.shape[2]
    if (von_mises_prior_mean_params is None) or (von_mises_prior_concentration_params is None):
        return n_paths * tf.math.log(2 * np.pi)
    else:
        cross_entropies_per_component = (
            tf.math.log(2 * np.pi) +
            compute_log_bessel_i0(von_mises_prior_concentration_params) - (
                von_mises_prior_concentration_params *
                bessel_ratio_posterior *
                tf.math.cos(von_mises_posterior_mean_params - von_mises_prior_mean_params)
            )
        )
        return tf.reduce_sum(cross_entropies_per_component, axis=2)


def _entropy_von_mises(
    von_mises_concentration_params: tf.Variable,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    bessel_ratio: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
) -> tf.Tensor:
    entropies_per_component = (
        tf.math.log(2 * np.pi) +
        compute_log_bessel_i0(von_mises_concentration_params) - (
            von_mises_concentration_params * bessel_ratio
        )
    )
    return tf.reduce_sum(entropies_per_component, axis=2)


def _expected_nll_exp_partition_unscaled(
    cfr_flat: tf.Tensor,  # [N_RX_TX_PAIRS, N_SUBCARRIERS*N_ARR_RX*N_ARR_TX, N_PATHS]
    measurements_flat: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS*N_ARR_RX*N_ARR_TX]
    von_mises_mean_params: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    bessel_ratio: tf.Tensor  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
) -> tf.Tensor:  # [N_MEASUREMENTS, N_RX_TX_PAIRS]
    """
    Expectation of the negative log likelihood exponential-partition given the latent variables, i.e.,
    E_q[-log(expPartition)] for likelihood P(measurement | latent) = a*Exp((1/std) * expPartition).
    The returned value does not comprise the noise std scaling 1/std.
    """
    # Order 1 and 2 circular moments of posterior ([N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS])
    circular_mean = cast_to_complex(bessel_ratio) * tf.math.exp(cast_to_pure_imag(von_mises_mean_params))
    circular_variance = 1 - tf.math.pow(bessel_ratio, 2)
    # Note: the circular variance is taken with respect to its definition in classical statistics, which differs from
    # its definition in directional statistics

    # Intermediary variables
    cfr_moment_1 = tf.squeeze(  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS*N_ARR_RX*N_ARR_TX]
        (
            cfr_flat[tf.newaxis, :, :, :] @  # [N_MEASUREMENTS=1, N_RX_TX_PAIRS, N_SUBCARRIERS*N_ARR_RX*N_ARR_TX, N_PATHS]
            circular_mean[:, :, :, tf.newaxis]  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS, 1]
        ),
        axis=3
    )
    squared_cfr = squared_norm(cfr_flat, axis=1)  # [N_RX_TX_PAIRS, N_PATHS]
    squared_cfr_var = (  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
        squared_cfr[tf.newaxis, :, :] *  # [N_MEASUREMENTS=1, N_RX_TX_PAIRS, N_PATHS]
        circular_variance  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    )

    # Compute NLL (without log-partition)
    expected_nll = (  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS]
        squared_norm(cfr_moment_1 - measurements_flat, axis=-1) +
        tf.reduce_sum(squared_cfr_var, axis=2)
    )
    return expected_nll


def expected_nll_normalized(
    cfr_train: tf.Tensor,  # [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATHS]
    channel_measurements: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX]
    von_mises_mean_params: tf.Variable,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    von_mises_concentration_params: tf.Variable,  # Shape [] if amortized, else [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    von_mises_amortize_concentration: bool
) -> tf.Tensor:  # [N_MEASUREMENTS, N_RX_TX_PAIRS]
    """Expected negative log-likelihood exponential-partition normalized by the measured channel power"""
    # Init
    n_measurements = channel_measurements.shape[0]
    n_pairs = channel_measurements.shape[1]
    n_paths = cfr_train.shape[-1]
    measurements_flat = tf.reshape(channel_measurements, [n_measurements, n_pairs, -1])
    cfr_flat = tf.reshape(cfr_train, [n_pairs, -1, n_paths])

    # Shape von Mises posterior concentrations tensor
    if von_mises_amortize_concentration:
        von_mises_tiled_concentration_params = _tile_von_mises_scalar_param(
            value=von_mises_concentration_params,
            n_measurements=n_measurements,
            n_rx_tx_pairs=n_pairs,
            n_paths=n_paths
        )
    else:
        von_mises_tiled_concentration_params = von_mises_concentration_params

    # Compute expected NLL (without log-partition)
    bessel_ratio = compute_bessel_ratio(von_mises_tiled_concentration_params)
    expected_nll = _expected_nll_exp_partition_unscaled(
        cfr_flat=cfr_flat,
        measurements_flat=measurements_flat,
        von_mises_mean_params=von_mises_mean_params,
        bessel_ratio=bessel_ratio
    )

    # Normalize
    return expected_nll / squared_norm(measurements_flat, axis=-1)


def free_energy_von_mises(
    cfr_train: tf.Tensor,  # [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATHS]
    channel_measurements: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX]
    von_mises_mean_params: tf.Variable,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    von_mises_concentration_params: tf.Variable,  # Shape [] if amortized, else [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    von_mises_amortize_concentration: bool,
    measurement_noise_std: tf.Tensor,  # [N_RX_TX_PAIRS]
    # Latent prior/global distribution
    von_mises_global_mean: tf.Tensor,  # Shape []
    von_mises_global_concentration_param: tf.Variable,  # Shape []
    normalize_by_snr: bool = False,
    normalize_by_arrays_and_subcarriers_size: bool = True
) -> tf.Tensor:  # [N_MEASUREMENTS, N_RX_TX_PAIRS]
    n_measurements = channel_measurements.shape[0]
    n_pairs = channel_measurements.shape[1]
    n_paths = cfr_train.shape[-1]
    measurements_flat = tf.reshape(channel_measurements, [n_measurements, n_pairs, -1])
    cfr_flat = tf.reshape(cfr_train, [n_pairs, -1, n_paths])
    arrays_and_subcarriers_size = measurements_flat.shape[2]  # N_SUBCARRIERS * N_ARR_RX * N_ARR_TX

    # Init
    # ----

    # Shape von Mises tensors
    # Posterior concentration
    if von_mises_amortize_concentration:
        von_mises_tiled_concentration_params = _tile_von_mises_scalar_param(
            value=von_mises_concentration_params,
            n_measurements=n_measurements,
            n_rx_tx_pairs=n_pairs,
            n_paths=n_paths
        )
    else:
        von_mises_tiled_concentration_params = von_mises_concentration_params
    # Prior/Global mean and concentration
    von_mises_tiled_global_mean = _tile_von_mises_scalar_param(
        value=von_mises_global_mean,
        n_measurements=n_measurements,
        n_rx_tx_pairs=n_pairs,
        n_paths=n_paths
    )
    von_mises_tiled_global_concentration_param = _tile_von_mises_scalar_param(
        value=von_mises_global_concentration_param,
        n_measurements=n_measurements,
        n_rx_tx_pairs=n_pairs,
        n_paths=n_paths
    )

    # Useful values
    bessel_ratio = compute_bessel_ratio(von_mises_tiled_concentration_params)

    # Cross-entropy of variational posterior with prior/global distribution
    # ---------------------------------------------------------------------
    vi_cross_entropy_posterior_prior = _cross_entropy_von_mises(
        von_mises_posterior_mean_params=von_mises_mean_params,
        bessel_ratio_posterior=bessel_ratio,
        von_mises_prior_mean_params=von_mises_tiled_global_mean,
        von_mises_prior_concentration_params=von_mises_tiled_global_concentration_param
    )

    # Entropy of variational posterior
    # --------------------------------
    vi_posterior_entropy = _entropy_von_mises(
        von_mises_concentration_params=von_mises_tiled_concentration_params,
        bessel_ratio=bessel_ratio
    )

    # Expectation of negative log-likelihood, E_q [-log(P(measurement | latent))]
    # ---------------------------------------------------------------------------
    # shape [N_RX_TX_PAIRS]
    meas_noise_power = tf.pow(measurement_noise_std, 2)
    inv_meas_noise_power = 1 / meas_noise_power
    expected_nll_log_partition = arrays_and_subcarriers_size * tf.math.log(meas_noise_power * np.pi)
    # shape [N_MEASUREMENTS, N_RX_TX_PAIRS]
    expected_nll_exp_partition = inv_meas_noise_power * _expected_nll_exp_partition_unscaled(
        cfr_flat=cfr_flat,
        measurements_flat=measurements_flat,
        von_mises_mean_params=von_mises_mean_params,
        bessel_ratio=bessel_ratio
    )
    expected_nll = expected_nll_log_partition[tf.newaxis, :] + expected_nll_exp_partition

    # Free Energy
    # -----------
    free_energy = (
        vi_cross_entropy_posterior_prior -
        vi_posterior_entropy +
        expected_nll
    )

    if normalize_by_snr:
        free_energy = meas_noise_power * free_energy
    if normalize_by_arrays_and_subcarriers_size:
        free_energy = free_energy / arrays_and_subcarriers_size
    return free_energy





