from itertools import product
import tensorflow as tf
import matplotlib.pyplot as plt
from sionna.rt import Scene, Paths

from src.utils.tensor_utils import dot
from src.utils.sionna_utils import get_mask_paths, set_path_delays_normalization, select_rx_tx_pairs
from src.data_classes import MeasurementPhaseNoiseType
from src.ofdm_measurements.channel_frequency_response import compute_cfr_per_mpc_from_paths
from src.ofdm_measurements.normalize_measurement import get_measurement_mean_power
from src.ofdm_measurements.additive_gaussian_noise import get_additive_gaussian_noise
from src.ofdm_measurements.simulate_phase_noise import simulate_phase_noise_per_mpc
from src.ofdm_measurements.main import sum_paths_of_cfr_measurements
from src.projections.time_lattice import get_paths_delays, get_evenly_spaced_time_lattice
from src.projections.time_projection import get_paths_delays_time_projections, get_evenly_spaced_time_projections


def _simulate_phase_noise_measurements(
    selected_cfr,
    n_measurements,
    component_noise_type,
    measurement_snr,
    von_mises_prior_concentration,
):
    phase_noise_per_mpc = simulate_phase_noise_per_mpc(
        n_measurements=n_measurements,
        n_rx_tx_pairs=selected_cfr.shape[0],
        n_paths=selected_cfr.shape[-1],
        component_noise_type=component_noise_type,
        von_mises_prior_mean=0.0,
        von_mises_prior_concentration=von_mises_prior_concentration,
        seed=None
    )
    cfr_measurements_per_mpc = tf.tile(
        selected_cfr[tf.newaxis, :, :, :, :, :],
        [n_measurements, 1, 1, 1, 1, 1]
    )
    measurement_without_additive_noise = sum_paths_of_cfr_measurements(
        cfr_measurements_per_mpc,
        phase_noise_per_mpc=phase_noise_per_mpc
    )
    mean_channel_power = get_measurement_mean_power(cfr_measurements_per_mpc=cfr_measurements_per_mpc)

    _, additive_measurement_noise = get_additive_gaussian_noise(
        measurement_without_additive_noise=measurement_without_additive_noise,
        mean_channel_power=mean_channel_power,
        measurement_snr=measurement_snr,
        normalization_constant=1.0
    )

    measurement = measurement_without_additive_noise + additive_measurement_noise

    return measurement


def compute_time_projections(
    scene: Scene,
    paths: Paths,
    num_subcarriers: int,
    subcarrier_spacing: float,
    paths_projections: bool = False,
    force_n_points: int = None,
    idx_path: int = None,
    carrier_modulation: bool = False,
    n_measurements: int = 1,
    component_noise_type: str = MeasurementPhaseNoiseType.VON_MISES_PHASE,
    measurement_snr: float = 100,  # 20 dB
    von_mises_prior_concentration=100.0  # Set to 0 for uniform phase error
) -> tf.Tensor:  # Shape [N_RX, N_TX, N_POINTS]
    # Compute selected CFRs (shape [N_RX*N_TX, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATHS])
    # -------------------------------------------------------------------------------------
    cfr_per_mpc = compute_cfr_per_mpc_from_paths(
        scene=scene,
        paths=paths,
        num_subcarriers=num_subcarriers,
        subcarrier_spacing=subcarrier_spacing,
        carrier_modulation=carrier_modulation
    )
    n_rx = cfr_per_mpc.shape[0]
    n_tx = cfr_per_mpc.shape[1]
    selected_rx_tx_pairs = list(product(range(n_rx), range(n_tx)))
    selected_cfr = select_rx_tx_pairs(
        rx_tx_indexed_tensor=cfr_per_mpc,
        rx_tx_indexes=selected_rx_tx_pairs
    )

    # Compute measurements (shape [N_MEASURMENTS, N_RX*N_TX, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX])
    # ------------------------------------------------------------------------------------------
    if idx_path is None:  # Create measurement
        measurements = _simulate_phase_noise_measurements(
            selected_cfr=selected_cfr,
            n_measurements=n_measurements,
            component_noise_type=component_noise_type,
            measurement_snr=measurement_snr,
            von_mises_prior_concentration=von_mises_prior_concentration
        )
    else:
        measurements = selected_cfr[..., idx_path]
        measurements = measurements[tf.newaxis, ...]

    # Compute projections (shape [N_RX, N_TX, N_SUBCARRIERS, N_POINTS])
    # -----------------------------------------------------------------
    if paths_projections:
        time_proj = get_paths_delays_time_projections(
            scene=scene,
            paths=paths,
            num_subcarriers=num_subcarriers,
            subcarrier_spacing=subcarrier_spacing,
            carrier_modulation=carrier_modulation
        )
    else:
        time_proj = get_evenly_spaced_time_projections(
            scene=scene,
            paths=paths,
            num_subcarriers=num_subcarriers,
            subcarrier_spacing=subcarrier_spacing,
            carrier_modulation=carrier_modulation,
            force_n_points=force_n_points
        )
    n_points = time_proj.shape[-1]

    # Project measurements
    # --------------------
    time_proj_selected = select_rx_tx_pairs(
        rx_tx_indexed_tensor=time_proj,
        rx_tx_indexes=selected_rx_tx_pairs
    )
    # To shape [N_MEASUREMENTS=1, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX=1, N_ARR_TX=1, N_POINTS]
    time_proj_selected = time_proj_selected[tf.newaxis, :, :, tf.newaxis, tf.newaxis, :]

    # To shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_POINTS=1]
    measurements_tiled = tf.tile(
        measurements[..., tf.newaxis],
        [1, 1, 1, 1, 1, 1]
    )

    projected_measurements = dot(  # Shape [N_MEASURMENTS, N_RX*N_TX, N_ARR_RX, N_ARR_TX, N_POINTS]
        measurements_tiled,
        tf.math.conj(time_proj_selected),
        axis=2  # subcarriers axis
    ) / num_subcarriers  # Normalize
    projected_abs_measurements = tf.math.abs(projected_measurements)
    # Average over measurements and rx/tx arrays
    mean_abs_proj = tf.reduce_mean(  # Shape [N_RX*N_TX, N_POINTS]
        projected_abs_measurements,
        axis=[0, 2, 3]
    )

    return tf.reshape(mean_abs_proj, [n_rx, n_tx, n_points])


def plot_cir_vs_delay(
    paths: Paths,
    idx_rx: int,
    idx_tx: int,
    idx_path: int = None,
    normalize_delays: bool = False
):
    set_path_delays_normalization(paths=paths, normalize_delays=normalize_delays)
    amplitudes_all_devices, delays_all_devices = paths.cir()
    mask_all_devices = get_mask_paths(paths)
    mask = mask_all_devices[idx_rx, idx_tx, :]
    n_paths = tf.reduce_sum(tf.cast(mask, tf.int64))

    amplitudes = amplitudes_all_devices[0, idx_rx, :, idx_tx, :, :n_paths, 0]
    delays = delays_all_devices[0, idx_rx, idx_tx, :]
    amplitudes_abs = tf.reduce_mean(
        tf.abs(amplitudes),
        axis=[0, 1]
    )

    fig, ax = plt.subplots()
    ax.set_title("Channel impulse response")
    if idx_path is None:
        ax.stem(delays, amplitudes_abs)
    else:
        ax.stem(delays[idx_path], amplitudes_abs[idx_path])
    ax.set_xlabel(r"Delay [s]")
    ax.set_ylabel(r"Path amplitude")

    return ax


def plot_time_projection(
    paths: Paths,
    projected_measurements: tf.Tensor,  # Shape [N_RX, N_TX, N_POINTS]
    idx_rx: int,
    idx_tx: int,
    idx_path: int = None,
    paths_projections: bool = False
):
    n_points = projected_measurements.shape[2]
    if paths_projections:
        time_axes = get_paths_delays(paths=paths)
    else:
        time_axes = get_evenly_spaced_time_lattice(
            paths=paths,
            bandwidth=None,
            force_n_points=n_points
        )
    time_axis = time_axes[idx_rx, idx_tx]
    idx_sorted_time_axis = tf.argsort(time_axis)
    time_axis = tf.gather(time_axis, idx_sorted_time_axis)
    proj = tf.gather(projected_measurements[idx_rx, idx_tx], idx_sorted_time_axis)
    print(proj)

    # Plot
    ax = plot_cir_vs_delay(
        paths=paths,
        idx_rx=idx_rx,
        idx_tx=idx_tx,
        idx_path=idx_path
    )
    ax.plot(time_axis, proj)
