from typing import Union
import numpy as np
import tensorflow as tf
from sionna.rt import Scene, Paths

from src.data_classes import MeasurementType, MeasurementNoiseType, MeasurementDataMaxwell, MeasurementDataPhaseNoise, \
    MeasurementDataPositionNoise
from src.utils.sionna_utils import select_rx_tx_pairs, check_all_rx_tx_pairs_have_at_least_one_path
from src.ofdm_measurements import _MeasurementMetadataBase
from src.ofdm_measurements.channel_frequency_response import compute_cfr_per_mpc_from_paths
from src.ofdm_measurements.compute_paths import compute_paths_from_traces, compute_paths_traces, compute_paths
from src.ofdm_measurements.normalize_measurement import get_measurement_mean_power
from src.ofdm_measurements.additive_gaussian_noise import get_additive_gaussian_noise
from src.ofdm_measurements.simulate_phase_noise import simulate_phase_noise_per_mpc
from src.ofdm_measurements.simulate_position_noise import sample_position_noise, simulate_position_noise_cfr
from src.ofdm_measurements.maxwell_simulation_toy_example import get_mean_channel_power_maxwell_simulation, \
    cfr_and_freq_response_maxwell_simulation_toy_example


# Paths from metadata
# -------------------

def compute_paths_traces_from_metadata(
    scene: Scene,
    measurement_metadata: _MeasurementMetadataBase,
    check_scene: bool = False,
):
    return compute_paths_traces(
        scene=scene,
        max_depth=measurement_metadata.max_depth_path,
        num_samples=measurement_metadata.num_samples_path,
        check_scene=check_scene
    )


def compute_paths_from_metadata(
    scene: Scene,
    measurement_metadata: _MeasurementMetadataBase,
    check_paths: bool = False,
    check_scene: bool = False
) -> Paths:
    paths = compute_paths(
        scene=scene,
        max_depth=measurement_metadata.max_depth_path,
        num_samples=measurement_metadata.num_samples_path,
        check_scene=check_scene
    )

    # Optional checks (mainly used at the beginning of calibration and during measurements)
    if check_paths:
        check_all_rx_tx_pairs_have_at_least_one_path(
            paths=paths,
            rx_tx_indexes=measurement_metadata.rx_tx_indexes
        )

    return paths


# Channel Frequency Response
# --------------------------

def compute_selected_cfr_from_paths_traces(
    scene: Scene,
    traced_paths,
    measurement_metadata: _MeasurementMetadataBase,
    normalization_constant: float,  # Normalization constant (obtained from measurement during normalization)
    check_paths: bool = False
) -> tf.Tensor:  # Selected CFR [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    # Compute paths
    paths = compute_paths_from_traces(scene=scene, traced_paths=traced_paths)

    # Optional checks (mainly used at the beginning of calibration and during measurements)
    if check_paths:
        check_all_rx_tx_pairs_have_at_least_one_path(
            paths=paths,
            rx_tx_indexes=measurement_metadata.rx_tx_indexes
        )

    # Get CFRs
    cfr_per_mpc = compute_cfr_per_mpc_from_paths(
        paths=paths,
        num_subcarriers=measurement_metadata.num_subcarriers,
        subcarrier_spacing=measurement_metadata.subcarrier_spacing,
        normalization_constant=normalization_constant
    )
    return select_rx_tx_pairs(
        rx_tx_indexed_tensor=cfr_per_mpc,
        rx_tx_indexes=measurement_metadata.rx_tx_indexes
    )


def compute_selected_cfr_from_scratch(
    scene: Scene,
    measurement_metadata: _MeasurementMetadataBase,
    normalization_constant: float,  # Normalization constant (obtained from measurement during normalization)
    check_scene: bool = False,
    check_paths: bool = False
) -> tf.Tensor:  # Selected CFR [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    # Compute paths
    traced_paths = compute_paths_traces_from_metadata(
        scene=scene,
        measurement_metadata=measurement_metadata,
        check_scene=check_scene
    )
    return compute_selected_cfr_from_paths_traces(
        scene=scene,
        traced_paths=traced_paths,
        measurement_metadata=measurement_metadata,
        normalization_constant=normalization_constant,
        check_paths=check_paths
    )


# Measurements / Channel observations
# -----------------------------------

def sum_paths_of_cfr_measurements(
    cfr_measurements_per_mpc: tf.Tensor,  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    phase_noise_per_mpc: tf.Tensor = None
) -> tf.Tensor:  # CFR measurements ; shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX]
    _, _,  n_carriers, n_arr_rx, n_arr_tx, _ = cfr_measurements_per_mpc.shape
    phase_noise_tiled = tf.tile(
        phase_noise_per_mpc[:, :, tf.newaxis, tf.newaxis, tf.newaxis, :],
        [1, 1, n_carriers, n_arr_rx, n_arr_tx, 1]
    ) if phase_noise_per_mpc is not None else tf.constant(1.0, dtype=cfr_measurements_per_mpc.dtype)

    return tf.reduce_sum(
        cfr_measurements_per_mpc * phase_noise_tiled,
        axis=5
    )


def simulate_measurements(
    scene: Scene,
    measurement_metadata: _MeasurementMetadataBase
) -> Union[MeasurementDataMaxwell, MeasurementDataPhaseNoise, MeasurementDataPositionNoise]:
    # Init
    n_rx_tx_pairs = len(measurement_metadata.rx_tx_indexes)
    freq_axis_maxwell = None
    freq_response_maxwell = None

    if measurement_metadata.__measurement_type__ == MeasurementType.RAY_TRACING:
        # Predicted CFR ; shape [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
        predicted_cfr_per_mpc = compute_selected_cfr_from_scratch(
            scene=scene,
            measurement_metadata=measurement_metadata,
            normalization_constant=1.0,  # No normalization yet
            check_scene=True,
            check_paths=True  # Normalization of measurements is not robust to (Rx, Tx) pairs with no paths
        )

        # Simulate mismatch between RT prediction and actual ground-truth conditions.
        # The obtained CFR measurements mainly differ from the RT-predicted CFR in terms of phase at each path
        # Position noise
        if measurement_metadata.__noise_type__ == MeasurementNoiseType.POSITION:
            # Simulate small errors (of order of the carrier's wavelength) between the positions of the devices in the
            # simulation and during ground-truth measurements
            position_noise_rx, position_noise_tx = sample_position_noise(
                carrier_wavelength=scene.wavelength,
                n_measurements=measurement_metadata.n_measurements_per_channel,
                n_rx=len(scene.receivers),
                n_tx=len(scene.transmitters),
                displacement_amplitude=measurement_metadata.displacement_amplitude,
                displace_rx=measurement_metadata.displace_rx,
                displace_tx=measurement_metadata.displace_tx
            )
            cfr_measurements_per_mpc = simulate_position_noise_cfr(
                scene=scene,
                n_measurements=measurement_metadata.n_measurements_per_channel,
                max_depth_path=measurement_metadata.max_depth_path,
                num_samples_path=measurement_metadata.num_samples_path,
                num_subcarriers=measurement_metadata.num_subcarriers,
                subcarrier_spacing=measurement_metadata.subcarrier_spacing,
                rx_tx_indexes=measurement_metadata.rx_tx_indexes,
                position_noise_rx=position_noise_rx,
                position_noise_tx=position_noise_tx
            )
        else:
            # All measured CFRs are equivalent to the RT-simulated CFR (before adding phase noise)
            cfr_measurements_per_mpc = tf.tile(
                predicted_cfr_per_mpc[tf.newaxis, :, :, :, :, :],
                [measurement_metadata.n_measurements_per_channel, 1, 1, 1, 1, 1]
            )
            position_noise_rx = None
            position_noise_tx = None

        # Phase noise (i.i.d.)
        if measurement_metadata.__noise_type__ == MeasurementNoiseType.PHASE:
            # Simulate errors between the RT-predicted phases of each path and their ground-truth phase by adding a phase
            # noise term per path and per measurement
            n_paths = cfr_measurements_per_mpc.shape[-1]
            optional_kwargs_phase_noise = {
                kw_function: getattr(measurement_metadata, kw_metadata)
                for kw_metadata, kw_function in [
                    ("von_mises_prior_mean", "von_mises_prior_mean"),
                    ("von_mises_prior_concentration", "von_mises_prior_concentration"),
                    ("manual_phases", "manual_phases")
                ]
                if hasattr(measurement_metadata, kw_metadata)
            }
            phase_noise_per_mpc = simulate_phase_noise_per_mpc(
                n_measurements=measurement_metadata.n_measurements_per_channel,
                n_rx_tx_pairs=n_rx_tx_pairs,
                n_paths=n_paths,
                component_noise_type=measurement_metadata.__component_noise_type__,
                seed=measurement_metadata.seed,
                **optional_kwargs_phase_noise
            )
        else:
            phase_noise_per_mpc = None

        # Sum paths contributions (from paths frequency responses to channel frequency response)
        measurement_without_additive_noise = sum_paths_of_cfr_measurements(
            cfr_measurements_per_mpc=cfr_measurements_per_mpc,
            phase_noise_per_mpc=phase_noise_per_mpc
        )
        # Get average power of the measured channel (used to normalize the CFR and/or define the power of additive noise)
        mean_channel_power = get_measurement_mean_power(cfr_measurements_per_mpc=cfr_measurements_per_mpc)
    elif measurement_metadata.__measurement_type__ == MeasurementType.MAXWELL_SIMULATION:
        (
            measurement_without_additive_noise,
            freq_axis_maxwell,
            freq_response_maxwell
        ) = cfr_and_freq_response_maxwell_simulation_toy_example(
            measurement_metadata=measurement_metadata,
            carrier_frequency=scene.frequency
        )
        mean_channel_power = get_mean_channel_power_maxwell_simulation(
            freq_axis=freq_axis_maxwell,
            freq_response=freq_response_maxwell
        )
    else:
        raise ValueError(f"Unknown measurement type '{measurement_metadata.__measurement_type__}'")

    # Normalization (mean channel power = 1): helps with floating point errors due to small numbers during optimization
    normalization_constant = (
        float(tf.reduce_mean(tf.sqrt(mean_channel_power)).numpy())
        if measurement_metadata.normalize else
        1.0
    )
    measurement_without_additive_noise = (
        measurement_without_additive_noise /
        tf.constant(normalization_constant, dtype=measurement_without_additive_noise.dtype)
    )

    # Simulate additive Gaussian noise during measurements
    if measurement_metadata.with_additive_noise:
        normalized_measurement_noise_std, additive_measurement_noise = get_additive_gaussian_noise(
            measurement_without_additive_noise=measurement_without_additive_noise,
            mean_channel_power=mean_channel_power,
            measurement_snr=measurement_metadata.measurement_snr,
            normalization_constant=normalization_constant,
        )
    else:
        # Note: if no noise is added to the data (if the SNR is 'infinite' or if the available data is already noisy),
        # the <normalized_measurement_noise_std> variable reflects the SNR assumed by the calibration procedure
        normalized_measurement_noise_std = (
            tf.constant(1 / np.sqrt(measurement_metadata.measurement_snr), dtype=tf.float32) *
            tf.ones(n_rx_tx_pairs, dtype=tf.float32)
        )
        additive_measurement_noise = tf.zeros(measurement_without_additive_noise.shape, dtype=tf.complex64)
    measurement = measurement_without_additive_noise + additive_measurement_noise

    # Return measurement data
    if measurement_metadata.__measurement_type__ == MeasurementType.RAY_TRACING:
        if measurement_metadata.__noise_type__ == MeasurementNoiseType.PHASE:
            return MeasurementDataPhaseNoise(
                measurement=measurement,
                normalization_constant=normalization_constant,
                normalized_measurement_noise_std=normalized_measurement_noise_std,
                measurement_noise=additive_measurement_noise,
                cfr_per_mpc=predicted_cfr_per_mpc,
                components_noise=phase_noise_per_mpc
            )
        elif measurement_metadata.__noise_type__ == MeasurementNoiseType.POSITION:
            return MeasurementDataPositionNoise(
                measurement=measurement,
                normalization_constant=normalization_constant,
                normalized_measurement_noise_std=normalized_measurement_noise_std,
                measurement_noise=additive_measurement_noise,
                position_noise_rx=position_noise_rx,
                position_noise_tx=position_noise_tx
            )
        else:
            raise ValueError(f"Unknown measurement noise type '{measurement_metadata.__noise_type__}'")
    elif measurement_metadata.__measurement_type__ == MeasurementType.MAXWELL_SIMULATION:
        return MeasurementDataMaxwell(
            measurement=measurement,
            normalization_constant=normalization_constant,
            normalized_measurement_noise_std=normalized_measurement_noise_std,
            measurement_noise=additive_measurement_noise,
            freq_axis_maxwell=freq_axis_maxwell,
            freq_response_maxwell=freq_response_maxwell
        )
    else:
        raise ValueError(f"Unknown measurement type '{measurement_metadata.__measurement_type__}'")