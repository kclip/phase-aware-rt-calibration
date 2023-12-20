from typing import List
import tensorflow as tf

from src.utils.tensor_utils import hermitian, cast_to_complex, cartesian_product, count_non_null, \
    compute_bessel_ratio, compute_bessel_ratio_inverse


def get_valid_paths_indices_per_rx_tx_pair(
    cfr_per_mpc: tf.Tensor,  # [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
) -> List[tf.Tensor]:  # List of tf.Tensor[N_PATH_PAIR]
    """
    Since each (Rx, Tx) pair might have a different number of possible paths, the obtained CFR for a given pair usually
    contains entries for non-existing paths that are set to 0.0 as a placeholder.
    This function extracts a list of valid paths indices (last CFR axis) per (Rx, Tx) pair by searching for path-slices
    with at least one non-zero element across all the L=N_SUBCARRIERS*N_ARR_RX*N_ARR_TX entries of the pair's CFR.
    """
    n_pairs = cfr_per_mpc.shape[0]
    n_paths = cfr_per_mpc.shape[-1]
    cfr_flat = tf.reshape(cfr_per_mpc, [n_pairs, -1, n_paths])  # Shape [N_RX_TX_PAIRS, L, N_PATH]
    valid_paths_indices = []
    for n_pair in range(n_pairs):
        valid_paths_indices_pair = tf.reshape(  # Shape [N_PATH]
            tf.where(
                tf.reduce_any(cfr_flat[n_pair] != 0, axis=0)
            ),
            [-1]
        )
        valid_paths_indices.append(valid_paths_indices_pair)
    return valid_paths_indices


def compute_von_mises_posterior_mean(
    valid_paths_indices: List[tf.Tensor],
    cfr_per_mpc: tf.Tensor,  # [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    channel_measurements: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX]
    von_mises_global_concentration: tf.float32,
    measurement_noise_std: tf.Tensor  # [N_RX_TX_PAIRS]
) -> tf.Tensor:  # Shape [N_RX_TX_PAIRS, N_PATH]
    # Init
    # ----
    n_measurements = channel_measurements.shape[0]
    n_pairs = channel_measurements.shape[1]
    n_paths = cfr_per_mpc.shape[-1]
    measurements_flat = tf.reshape(  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, L]
        channel_measurements,
        [n_measurements, n_pairs, -1]
    )
    cfr_flat = tf.reshape(cfr_per_mpc, [n_pairs, -1, n_paths])  # Shape [N_RX_TX_PAIRS, L, N_PATH]

    # (CFR^H @ CFR)^-1 term
    # ---------------------
    inv_pairs = []
    # Each (Rx, Tx) pair in the CFR matrix has a different number of valid paths, inversion must be performed for each
    # pair independently by accounting only for valid paths
    for idx_pair, valid_paths_indices_pair in enumerate(valid_paths_indices):
        # Gather valid paths
        cfr_flat_pair = tf.gather(  # Shape [L, N_PATH_PAIR]
            params=cfr_flat[idx_pair],
            indices=valid_paths_indices_pair,
            axis=-1
        )
        # Compute inverse
        cfr_herm_pair = hermitian(cfr_flat_pair)  # Shape [N_PATH_PAIR, L]
        cfr_inv_pair = tf.linalg.inv(cfr_herm_pair @ cfr_flat_pair)  # Shape [N_PATH_PAIR, N_PATH_PAIR]
        # Restore to original number of paths (placeholder=0.0 at invalid path indices)
        cfr_inv_pair = tf.scatter_nd(
            indices=cartesian_product(valid_paths_indices_pair, valid_paths_indices_pair),
            updates=tf.reshape(cfr_inv_pair, -1),
            shape=[n_paths, n_paths]
        )
        inv_pairs.append(cfr_inv_pair)
    cfr_inv = tf.stack(inv_pairs, axis=0)   # Shape [N_RX_TX_PAIRS, N_PATH, N_PATH]

    # CFR @ Measurements term
    # -----------------------
    cfr_herm = hermitian(cfr_flat)  # Shape [N_RX_TX_PAIRS, N_PATH, L]
    cfr_meas = cfr_herm @ measurements_flat[:, :, :, tf.newaxis]

    # KL(Posterior VM || Prior VM) term
    # ---------------------------------
    kl_term = ((tf.pow(measurement_noise_std, 2)) * von_mises_global_concentration) / 2
    kl_term = cast_to_complex(kl_term)
    kl_term = tf.tile(
        kl_term[tf.newaxis, :, tf.newaxis, tf.newaxis],
        [n_measurements, 1, n_paths, 1]
    )

    # Posterior mean
    # --------------
    posterior_circular_mean = cfr_inv @ (kl_term + cfr_meas)
    vm_mean = tf.math.angle(posterior_circular_mean)

    return tf.squeeze(vm_mean, axis=3)


def compute_von_mises_posterior_concentration(
    valid_paths_indices: List[tf.Tensor],
    cfr_per_mpc: tf.Tensor,  # [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    n_measurements: int,
    measurement_noise_std: tf.Tensor,  # [N_RX_TX_PAIRS]
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
         squared_mpc_amplitude / tf.pow(measurement_noise_std[:, tf.newaxis], 2)
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
        sum_vm_concentration = tf.constant(0.0, dtype=vm_concentration.dtype)
        n_entries = tf.constant(0.0, dtype=vm_concentration.dtype)
        for n_pair, valid_paths_indices_pair in enumerate(valid_paths_indices):
            # Gather valid paths
            vm_concentration_pair = tf.gather(  # Shape [N_MEASUREMENTS, N_PATH_PAIR]
                params=vm_concentration[:, n_pair, :],
                indices=valid_paths_indices_pair,
                axis=-1
            )
            sum_vm_concentration += tf.reduce_sum(vm_concentration_pair)
            n_entries += tf.cast(tf.reduce_prod(vm_concentration_pair.shape), vm_concentration.dtype)
        return sum_vm_concentration / n_entries
    else:
        return vm_concentration


def compute_von_mises_global_concentration(
    valid_paths_indices: List[tf.Tensor],
    von_mises_posterior_mean: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATH]
    von_mises_posterior_concentration: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATH] or tf.float32
):
    sum_bessel_ratio = tf.constant(0.0, dtype=von_mises_posterior_concentration.dtype)
    n_entries = tf.constant(0.0, dtype=von_mises_posterior_concentration.dtype)
    for n_pair, valid_paths_indices_pair in enumerate(valid_paths_indices):
        # Gather posterior parameters for valid paths
        vm_posterior_mean_pair = tf.gather(  # Shape [N_MEASUREMENTS, N_PATH_PAIR]
            params=von_mises_posterior_mean[:, n_pair, :],
            indices=valid_paths_indices_pair,
            axis=-1
        )
        vm_posterior_concentration_pair = tf.gather(  # Shape [N_MEASUREMENTS, N_PATH_PAIR]
            params=von_mises_posterior_concentration[:, n_pair, :],
            indices=valid_paths_indices_pair,
            axis=-1
        )
        # Compute entries and add to total sum
        bessel_ratio_pair = compute_bessel_ratio(vm_posterior_concentration_pair) * tf.math.cos(vm_posterior_mean_pair)
        sum_bessel_ratio += tf.reduce_sum(bessel_ratio_pair)
        n_entries += tf.cast(tf.reduce_prod(bessel_ratio_pair.shape), von_mises_posterior_concentration.dtype)
    bessel_ratio = sum_bessel_ratio / n_entries
    vm_global_concentration = compute_bessel_ratio_inverse(bessel_ratio)
    if vm_global_concentration is not None:
        return vm_global_concentration
    else:  # No solution, minimal value of free-energy is at 0.0
        return tf.constant(0.0, dtype=tf.float32)
