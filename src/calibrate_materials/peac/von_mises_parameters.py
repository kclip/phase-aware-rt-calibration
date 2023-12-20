import tensorflow as tf

from src.utils.tensor_utils import hermitian, cast_to_complex, cast_to_pure_imag, squared_norm, count_non_null, \
    compute_bessel_ratio, compute_bessel_ratio_inverse


def compute_von_mises_posterior_mean(
    cfr_per_mpc: tf.Tensor,  # [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    channel_measurements: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX]
    von_mises_global_concentration: tf.float32,
    measurement_noise_std: float
) -> tf.Tensor:  # Shape [N_RX_TX_PAIRS, N_PATH]
    # Init
    # ----
    n_measurements = channel_measurements.shape[0]
    n_pairs = channel_measurements.shape[1]
    n_paths = cfr_per_mpc.shape[-1]
    measurements_flat = tf.reshape(channel_measurements, [n_measurements, n_pairs, -1])
    cfr_flat = tf.reshape(cfr_per_mpc, [n_pairs, -1, n_paths])

    # Useful intermediary term
    # ------------------------
    cfr_herm = hermitian(cfr_flat)

    # (CFR^H @ CFR)^-1 term
    # ---------------------
    inv_pairs = []
    # CFRs are padded with zeros for non-existing paths; padding must be removed before inverting
    for n_pair in range(n_pairs):
        n_paths_pair = count_non_null(cfr_flat[n_pair], axis=-1)
        cfr_inv_pair = tf.linalg.inv(cfr_herm[n_pair, :n_paths_pair, :] @ cfr_flat[n_pair, :, :n_paths_pair])
        n_pad = n_paths - n_paths_pair
        cfr_inv_pair = tf.pad(cfr_inv_pair, [[0, n_pad], [0, n_pad]], "CONSTANT")
        inv_pairs.append(cfr_inv_pair)
    cfr_inv = tf.stack(inv_pairs, axis=0)

    # CFR @ Measurements term
    # -----------------------
    cfr_meas = cfr_herm @ measurements_flat[:, :, :, tf.newaxis]

    # KL(Posterior VM || Prior VM) term
    # ---------------------------------
    kl_term = tf.constant(((measurement_noise_std ** 2) * von_mises_global_concentration) / 2, dtype=tf.float32)
    kl_term = cast_to_complex(kl_term)
    kl_term = tf.tile(
        kl_term[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis],
        [n_measurements, n_pairs, n_paths, 1]
    )

    # Posterior mean
    # --------------
    posterior_circular_mean = cfr_inv @ (kl_term + cfr_meas)
    vm_mean = tf.math.angle(posterior_circular_mean)

    return tf.squeeze(vm_mean, axis=3)


def compute_von_mises_posterior_concentration(
    cfr_per_mpc: tf.Tensor,  # [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    n_measurements: int,
    measurement_noise_std: float,
    amortize_concentration: bool
):
    # Get total path power over additive noise power
    # ----------------------------------------------
    squared_mpc_amplitude_per_subcarrier_antenna = tf.pow(tf.abs(cfr_per_mpc), 2)
    # Sum path energy for all subcarrier and antenna-pairs combinations
    squared_mpc_amplitude = tf.reduce_sum(  # Shape [N_RX_TX_PAIRS, N_PATH]
        squared_mpc_amplitude_per_subcarrier_antenna,
        axis=[1, 2, 3]
    )
    total_path_power_over_noise_power = (
         squared_mpc_amplitude / tf.constant((measurement_noise_std ** 2), dtype=tf.float32)
    )

    # Compute posterior concentration each Rx-Tx pair
    # -----------------------------------------------
    # If total_path_power_over_noise_power > 1 -> compute lower bound solution to the fixed-point equation
    # Otherwise -> set posterior concentration to 0
    clipped_path_snr = tf.clip_by_value(
        total_path_power_over_noise_power,
        clip_value_min=1,
        clip_value_max=total_path_power_over_noise_power.dtype.max
    )
    vm_concentration_per_rx_tx_pair = 2 * tf.sqrt(clipped_path_snr - 1) * tf.sqrt(clipped_path_snr)

    # Tile concentration over measurements (same for every measurement in the same (Rx, Tx) position)
    # ------------------------------------------------------------------------------------------------
    vm_concentration = tf.tile(
        vm_concentration_per_rx_tx_pair[tf.newaxis, :, :],
        [n_measurements, 1, 1]
    )

    if amortize_concentration:
        avg_vm_concentration = tf.constant(0.0, dtype=vm_concentration.dtype)
        n_entries = tf.constant(0.0, dtype=vm_concentration.dtype)
        n_pairs = cfr_per_mpc.shape[0]
        n_paths = cfr_per_mpc.shape[-1]
        cfr_flat = tf.reshape(cfr_per_mpc, [n_pairs, -1, n_paths])  # [N_RX_TX_PAIRS, N_ARR_CARRIERS, N_PATH]
        for n_pair in range(cfr_per_mpc.shape[0]):
            n_paths_pair = count_non_null(cfr_flat[n_pair], axis=-1)
            vm_concentration_pair = vm_concentration[:, n_pair, :n_paths_pair]
            avg_vm_concentration += tf.reduce_sum(vm_concentration_pair)
            n_entries += tf.cast(tf.reduce_prod(vm_concentration_pair.shape), vm_concentration.dtype)
        return avg_vm_concentration / n_entries
    else:
        return vm_concentration


def compute_von_mises_global_concentration(
    von_mises_posterior_mean: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATH]
    von_mises_posterior_concentration: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATH] or tf.float32
):
    bessel_ratio = tf.reduce_mean(
        compute_bessel_ratio(von_mises_posterior_concentration) *
        tf.math.cos(von_mises_posterior_mean)
    )
    vm_global_concentration = compute_bessel_ratio_inverse(bessel_ratio)
    if vm_global_concentration is not None:
        return vm_global_concentration
    else:  # No solution, minimal value of KL divergence is at 0.0
        return tf.constant(0.0, dtype=tf.float32)
