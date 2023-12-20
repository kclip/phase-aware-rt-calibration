import os
from itertools import product
from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from sionna.rt import Scene

from settings import LOGS_FOLDER
from src.data_classes import MeasurementDataMaxwell, MaterialsMapping, VonMisesCalibrationInfo
from src.utils.save_utils import SafeOpen
from src.utils.tensor_utils import cast_to_pure_imag, compute_bessel_ratio, cast_to_complex
from src.utils.fft_utils import get_power_profile, get_time_impulse_response, filter_freq_response
from src.scenarios import get_scenario
from src.scenarios.toy_example_maxwell.main import N_RX_MAXWELL_TOY_EXAMPLE
from src.ofdm_measurements import MeasurementMaxwellSimulationMetadata
from src.ofdm_measurements.main import compute_paths_from_metadata, select_rx_tx_pairs
from src.ofdm_measurements.maxwell_simulation_toy_example import cfr_and_freq_response_maxwell_simulation_toy_example
from src.ofdm_measurements.channel_frequency_response import compute_frequency_response_from_paths
from study.calibration.experiment_config import get_scenario_metadata, get_measurement_metadata, get_rx_tx_indexes
from study.calibration.experiment_protocol import load_experiment_protocol
from study.calibration.individual_runs.run_measurements import get_measurements_run_name
from study.calibration.individual_runs.run_calibration import get_calibration_run_name
from study.calibration.utils import PROTOCOL_POWER_PROFILES_DATA_FILENAME, CalibrationType, power_profiles_plots_folder


_CARRIER_FREQUENCY = 6e9
_BANDWIDTH = 1e9  # Gathered bandwidth in freq domain and bandwidth used for IFFT (time domain) ; in [Hz]

# Frequency and time windows to store
_SAVE_BANDWIDTH = 1e9  # 1 GHz
_SAVE_MAX_TOA = 1e-6  # 1 ms

_RUN_INFO_COLS = [
    "measurement_snr",
    "measurement_additive_noise",
    "bandwidth",
    "n_channel_measurements",
]


def _expand_info_cols(data, n_rows):
    return {
        k: [v] * n_rows
        for k, v in data.items()
        if k in _RUN_INFO_COLS
    }


def _compute_paths_power_and_freq_response(
    scene: Scene,
    measurement_metadata: MeasurementMaxwellSimulationMetadata,
    freq_axis: tf.Tensor,  # Frequencies at which the response is computed ; shape [N_POINTS]
) -> Tuple[
    tf.Tensor,  # Times of arrival [N_RX_TX_PAIRS, N_PATH]
    tf.Tensor,  # Power at each path [N_RX_TX_PAIRS, N_PATH, N_ARR_RX, N_ARR_TX]
    tf.Tensor  # Frequency response per path on given freq axis [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX, N_ARR_TX, N_PATH]
]:
    # Get paths
    paths = compute_paths_from_metadata(
        scene=scene,
        measurement_metadata=measurement_metadata,
        check_paths=True,
        check_scene=True
    )

    # Get frequency response at each path
    freq_response = compute_frequency_response_from_paths(
        paths=paths,
        freq_axis_baseband=freq_axis - scene.frequency,
        normalization_constant=1.0
    )
    freq_response = select_rx_tx_pairs(
        rx_tx_indexed_tensor=freq_response,
        rx_tx_indexes=measurement_metadata.rx_tx_indexes
    )

    # Get time of arrival and power of each path
    paths_coefs, toa_axes = paths.cir()
    if len(toa_axes.shape) != 4:
        raise ValueError(
            "ToA axis is only supported for synthetic antennas (tau.shape = [BATCH=1, N_RX, N_TX, N_PATH])"
        )
    paths_coefs = tf.transpose(paths_coefs, [0, 6, 1, 3, 5, 2, 4])[0, 0]
    paths_power = tf.pow(tf.abs(paths_coefs), 2)
    paths_power = select_rx_tx_pairs(
        rx_tx_indexed_tensor=paths_power,
        rx_tx_indexes=measurement_metadata.rx_tx_indexes
    )
    toa_axes = select_rx_tx_pairs(
        rx_tx_indexed_tensor=toa_axes[0],
        rx_tx_indexes=measurement_metadata.rx_tx_indexes
    )

    return toa_axes, paths_power, freq_response


def _average_antennas_normalize_and_mask(
    power_profile: np.ndarray,  # shape [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX, N_ARR_TX]
    normalization_values: np.ndarray,   # shape [N_RX_TX_PAIRS]
    mask: np.ndarray = None  # Shape [N_POINTS]
) -> np.ndarray:  # shape [N_RX_TX_PAIRS, N_POINTS_SELECTED]
    if mask is not None:
        power_profile = power_profile[:, mask, :, :]
    return np.mean(
        np.reshape(
            power_profile,
            [power_profile.shape[0], power_profile.shape[1], -1]
        ),
        axis=2
    ) / normalization_values[:, np.newaxis]


def _get_power_profiles_from_path_responses(
    freq_axis: tf.Tensor,  # Shape [N_POINTS]
    freq_response_per_path: tf.Tensor,  # Shape [N_RX_TX_PAIR, N_POINTS, N_ARR_RX, N_ARR_TX, N_PATH]
    phase_shift_per_path: tf.Tensor = None,  # Shape [N_RX_TX_PAIR, N_PATHS]
    bessel_ratio_per_path: tf.Tensor = None  # If None, deterministic path phases ; shape [N_RX_TX_PAIR, N_PATH]
) -> Tuple[
     tf.Tensor,  # Frequency power profile ; shape [N_RX_TX_PAIR, N_POINTS, N_ARR_RX, N_ARR_TX]
     tf.Tensor  # Time power profile ; shape [N_RX_TX_PAIR, N_POINTS, N_ARR_RX, N_ARR_TX]
]:
    # Apply phase correction if specified
    if phase_shift_per_path is not None:
        phase_shift_phasors = tf.exp(cast_to_pure_imag(tf.constant(phase_shift_per_path, dtype=tf.float32)))
        freq_response_per_path = freq_response_per_path * phase_shift_phasors[:, tf.newaxis, tf.newaxis, tf.newaxis, :]

    # Squared bessel ratio per path
    if bessel_ratio_per_path is None:
        bessel_ratio_per_path = tf.ones(
            [freq_response_per_path.shape[0], freq_response_per_path.shape[-1]],
            dtype=tf.float32
        )
    bessel_ratio = bessel_ratio_per_path[:, tf.newaxis, tf.newaxis, tf.newaxis, :]

    # Get time responses
    _, time_response_per_path = get_time_impulse_response(
        freq_axis=freq_axis,
        freq_response=freq_response_per_path
    )

    # Fraction of power profile due to stochastic phases ((1-squared bessel ratio)-weighted sum the power of each path)
    bessel_ratio_stochastic_factor = 1 - tf.pow(bessel_ratio, 2)
    freq_power_profile_stochastic_phases = tf.reduce_sum(
        tf.pow(tf.abs(freq_response_per_path), 2) * bessel_ratio_stochastic_factor,
        axis=-1
    )
    time_power_profile_stochastic_phases = tf.reduce_sum(
        tf.pow(tf.abs(time_response_per_path), 2) * bessel_ratio_stochastic_factor,
        axis=-1
    )

    # Fraction of power profile with deterministic phases (power of the (bessel ratio)-weighted sum of the paths)
    bessel_ratio_deterministic_factor = cast_to_complex(bessel_ratio)
    freq_power_profile_predicted_phases = tf.pow(
        tf.abs(
            tf.reduce_sum(freq_response_per_path * bessel_ratio_deterministic_factor, axis=-1)
        ),
        2
    )
    time_power_profile_predicted_phases = tf.pow(
        tf.abs(
            tf.reduce_sum(time_response_per_path * bessel_ratio_deterministic_factor, axis=-1)
        ),
        2
    )

    # Sum power profile contributions
    freq_power_profile = freq_power_profile_stochastic_phases + freq_power_profile_predicted_phases
    time_power_profile = time_power_profile_stochastic_phases + time_power_profile_predicted_phases

    return freq_power_profile, time_power_profile


def run_gather_power_profiles_maxwell_simulation(
    protocol_name: str,
    n_runs_to_load: int,
):
    data = []

    # Loop over experiments
    for run_parameters in load_experiment_protocol(protocol_name):
        data_run = {
            attr_name: getattr(run_parameters, attr_name)
            for attr_name in _RUN_INFO_COLS
        }

        print("=============================")
        print(f"Extracting power profiles from run: \n{run_parameters}")
        print("=============================")

        # Get measurement metadata and include all receivers
        measurement_metadata = get_measurement_metadata(
            measurement_snr=run_parameters.measurement_snr,
            measurement_additive_noise=run_parameters.measurement_additive_noise,
            measurement_perfect_phase=run_parameters.measurement_perfect_phase,
            measurement_von_mises_mean=run_parameters.measurement_von_mises_mean,
            measurement_von_mises_concentration=run_parameters.measurement_von_mises_concentration,
            bandwidth=run_parameters.bandwidth,
            position_noise_amplitude=run_parameters.position_noise_amplitude
        )
        measurement_metadata.rx_tx_indexes = get_rx_tx_indexes(n_rx=N_RX_MAXWELL_TOY_EXAMPLE)

        # Get measurement frequency response
        _, freq_axis, freq_response_maxwell = cfr_and_freq_response_maxwell_simulation_toy_example(
            measurement_metadata=measurement_metadata,
            carrier_frequency=_CARRIER_FREQUENCY
        )
        freq_axis, freq_response_maxwell = filter_freq_response(
            freq_axis=freq_axis,
            freq_response=freq_response_maxwell,
            axis=1,
            central_frequency=_CARRIER_FREQUENCY,
            bandwidth=_BANDWIDTH
        )

        # Measurement power profiles
        freq_power_profile_measurement = tf.pow(tf.abs(freq_response_maxwell), 2)
        time_axis, time_power_profile_measurement = get_power_profile(
            freq_axis=freq_axis,
            freq_response=freq_response_maxwell
        )
        normalization_values = np.sum(  # Shape [N_RX_TX_PAIR]
            time_power_profile_measurement.numpy().mean(axis=2).mean(axis=2),
            axis=1
        )

        # Get masks for datapoints to save per domain
        central_freq = tf.reduce_mean(freq_axis)
        freq_save_mask = (
                (freq_axis.numpy() > central_freq.numpy() - (_SAVE_BANDWIDTH / 2)) &
                (freq_axis.numpy() < central_freq.numpy() + (_SAVE_BANDWIDTH / 2))
        )
        time_save_mask = (time_axis.numpy() < _SAVE_MAX_TOA)

        # Store measurement power profiles
        data_run["freq_axis"] = freq_axis.numpy()[freq_save_mask]
        data_run["time_axis"] = time_axis.numpy()[time_save_mask]
        data_run["measurement"] = {
            "freq_power_profile": _average_antennas_normalize_and_mask(
                freq_power_profile_measurement.numpy(),
                normalization_values=normalization_values,
                mask=freq_save_mask
            ),
            "time_power_profile": _average_antennas_normalize_and_mask(
                time_power_profile_measurement.numpy(),
                normalization_values=normalization_values,
                mask=time_save_mask
            ),
        }

        # Get RT power profile with DT model and ground_truth parameters
        scenario_metadata = get_scenario_metadata(ground_truth_geometry=False)
        scene = get_scenario(scenario_metadata)
        toa_axes, paths_power, freq_response_per_path = _compute_paths_power_and_freq_response(
            scene=scene,
            measurement_metadata=measurement_metadata,
            freq_axis=freq_axis,
        )
        freq_power_profile_gt, time_power_profile_gt = _get_power_profiles_from_path_responses(
            freq_axis=freq_axis,
            freq_response_per_path=freq_response_per_path
        )
        data_run["toa_axes"] = toa_axes.numpy()
        data_run["ground_truth_materials"] = {
            "freq_power_profile": _average_antennas_normalize_and_mask(
                freq_power_profile_gt.numpy(),
                normalization_values=normalization_values,
                mask=freq_save_mask
            ),
            "time_power_profile": _average_antennas_normalize_and_mask(
                time_power_profile_gt.numpy(),
                normalization_values=normalization_values,
                mask=time_save_mask
            ),
            "paths_power": _average_antennas_normalize_and_mask(
                paths_power.numpy(),
                normalization_values=normalization_values
            )
        }

        # Get power profiles for calibrated materials
        data_run["calibration"] = dict()
        for calibration_type in run_parameters.run_calibration_types:

            data_calibration = {
                "freq_power_profile_raw": [],
                "time_power_profile_raw": [],
                "freq_power_profile_assumption": [],
                "time_power_profile_assumption": [],
                "paths_power": []
            }

            for n_run in range(n_runs_to_load):
                # Init scene with learned materials
                calibration_run_name = get_calibration_run_name(
                    calibration_type=calibration_type,
                    measurement_snr=run_parameters.measurement_snr,
                    measurement_additive_noise=run_parameters.measurement_additive_noise,
                    measurement_perfect_phase=run_parameters.measurement_perfect_phase,
                    measurement_von_mises_mean=run_parameters.measurement_von_mises_mean,
                    measurement_von_mises_concentration=run_parameters.measurement_von_mises_concentration,
                    bandwidth=run_parameters.bandwidth,
                    position_noise_amplitude=run_parameters.position_noise_amplitude,
                    n_channel_measurements=run_parameters.n_channel_measurements,
                    n_run=n_run
                )
                calibration_dir = os.path.join(LOGS_FOLDER, calibration_run_name)
                materials_mapping = MaterialsMapping.load(calibration_dir)
                scenario_metadata = get_scenario_metadata(ground_truth_geometry=False)
                scene = get_scenario(scenario_metadata, materials_mapping=materials_mapping)

                # Compute calibrated frequency responses per path
                toa_axes, paths_power, freq_response_per_path = _compute_paths_power_and_freq_response(
                    scene=scene,
                    measurement_metadata=measurement_metadata,
                    freq_axis=freq_axis,
                )
                if not tf.reduce_all(toa_axes == data_run["toa_axes"]):
                    raise ValueError("Multiple different ToA axes...")

                # Get power profiles without the phase assumption of the calibration model
                freq_power_profile_raw, time_power_profile_raw = _get_power_profiles_from_path_responses(
                    freq_axis=freq_axis,
                    freq_response_per_path=freq_response_per_path
                )

                # Get power profiles with the phase assumption of the calibration model
                phase_shift_per_path = None
                bessel_ratio_per_path = None
                n_rx_tx_indexes = len(measurement_metadata.rx_tx_indexes)
                n_paths = freq_response_per_path.shape[-1]
                if calibration_type in (CalibrationType.PEAC, CalibrationType.PEAC_FIXED_PRIOR):
                    # Mean phase correction and concentration at calibration positions
                    von_mises_info = VonMisesCalibrationInfo.load(calibration_dir)
                    vm_cal_mean = tf.constant(von_mises_info.von_mises_means, dtype=tf.float32)
                    vm_cal_mean = tf.reduce_mean(vm_cal_mean, axis=0)
                    vm_cal_concentration = tf.reduce_mean(
                        tf.constant(von_mises_info.von_mises_concentrations, dtype=tf.float32),
                        axis=0
                    )
                    # Prior phase correction and concentration at new locations
                    n_new_rx_tx_pairs = n_rx_tx_indexes - vm_cal_concentration.shape[0]
                    prior_params_shape = [n_new_rx_tx_pairs, vm_cal_concentration.shape[1]]
                    vm_prior_mean = tf.zeros(prior_params_shape, dtype=tf.float32)
                    vm_prior_concentration = (
                        von_mises_info.von_mises_global_concentration *
                        tf.ones(prior_params_shape, dtype=tf.float32)
                    )
                    # Concat
                    phase_shift_per_path = tf.concat([vm_cal_mean, vm_prior_mean], axis=0)
                    vm_concentration = tf.concat([vm_cal_concentration, vm_prior_concentration], axis=0)
                    bessel_ratio_per_path = compute_bessel_ratio(vm_concentration)
                elif calibration_type in (CalibrationType.UPEC, CalibrationType.UPEC_PATHS_PROJ):
                    bessel_ratio_per_path = tf.zeros([n_rx_tx_indexes, n_paths], dtype=tf.float32)
                elif calibration_type == CalibrationType.PEOC:
                    bessel_ratio_per_path = tf.ones([n_rx_tx_indexes, n_paths], dtype=tf.float32)
                else:
                    raise ValueError(f"Unknown calibration type '{calibration_type}'")
                freq_power_profile_assumption, time_power_profile_assumption = _get_power_profiles_from_path_responses(
                    freq_axis=freq_axis,
                    freq_response_per_path=freq_response_per_path,
                    phase_shift_per_path=phase_shift_per_path,
                    bessel_ratio_per_path=bessel_ratio_per_path
                )

                # Store data
                data_calibration["paths_power"].append(
                    _average_antennas_normalize_and_mask(
                        paths_power.numpy(),
                        normalization_values=normalization_values
                    )
                )
                data_calibration["freq_power_profile_raw"].append(
                    _average_antennas_normalize_and_mask(
                        freq_power_profile_raw.numpy(),
                        normalization_values=normalization_values,
                        mask=freq_save_mask
                    )
                )
                data_calibration["time_power_profile_raw"].append(
                    _average_antennas_normalize_and_mask(
                        time_power_profile_raw.numpy(),
                        normalization_values=normalization_values,
                        mask=time_save_mask
                    )
                )
                data_calibration["freq_power_profile_assumption"].append(
                    _average_antennas_normalize_and_mask(
                        freq_power_profile_assumption.numpy(),
                        normalization_values=normalization_values,
                        mask=freq_save_mask
                    )
                )
                data_calibration["time_power_profile_assumption"].append(
                    _average_antennas_normalize_and_mask(
                        time_power_profile_assumption.numpy(),
                        normalization_values=normalization_values,
                        mask=time_save_mask
                    )
                )

            # Mean performance across runs
            data_calibration_median = dict()
            # data_calibration_low_quartile = dict()
            # data_calibration_high_quartile = dict()
            for profile_name, power_profiles in data_calibration.items():
                data_calibration_median[profile_name] = np.mean(
                    np.stack(power_profiles, axis=0),
                    axis=0
                )
                # data_calibration_low_quartile[profile_name] = np.quantile(
                #     np.stack(power_profiles, axis=0),
                #     q=0.25,
                #     axis=0
                # )
                # data_calibration_high_quartile[profile_name] = np.quantile(
                #     np.stack(power_profiles, axis=0),
                #     q=0.75,
                #     axis=0
                # )

            # Store data for belonging to the same calibration scheme
            data_run["calibration"][calibration_type] = {
                "median": data_calibration_median,
                # "low_quartile": data_calibration_low_quartile,
                # "high_quartile": data_calibration_high_quartile,
            }

        # Store data with same run parameters
        data.append(data_run)

    # Convert data to dataframe
    df_list = []

    for data_run in data:
        df_run_list = []
        calibration_type_default = None
        assumption_default = None

        # Frequency and time profiles
        for domain, value_type in product(
            ("FREQ", "TIME"),
            ("MEASUREMENT", "GROUND_TRUTH_MATERIALS", "CALIBRATION")
        ):
            is_freq_domain = (domain == "FREQ")
            axis = data_run["freq_axis"] if is_freq_domain else data_run["time_axis"]
            n_rows = len(axis)
            if value_type == "MEASUREMENT":
                power_profile_name = "freq_power_profile" if is_freq_domain else "time_power_profile"
                for rx_tx_pair, power_profile in enumerate(data_run["measurement"][power_profile_name]):
                    df_run_list.append(pd.DataFrame.from_dict({
                        **_expand_info_cols(data_run, n_rows),
                        "value_type": [value_type] * n_rows,
                        "domain": [domain] * n_rows,
                        "rx_tx_pair": [rx_tx_pair] * n_rows,
                        "calibration_type": [calibration_type_default] * n_rows,
                        "assumption": [assumption_default] * n_rows,
                        "axis": axis,
                        "value": power_profile
                    }))
            elif value_type == "GROUND_TRUTH_MATERIALS":
                power_profile_name = "freq_power_profile" if is_freq_domain else "time_power_profile"
                for rx_tx_pair, power_profile in enumerate(data_run["ground_truth_materials"][power_profile_name]):
                    df_run_list.append(pd.DataFrame.from_dict({
                        **_expand_info_cols(data_run, n_rows),
                        "value_type": [value_type] * n_rows,
                        "domain": [domain] * n_rows,
                        "rx_tx_pair": [rx_tx_pair] * n_rows,
                        "calibration_type": [calibration_type_default] * n_rows,
                        "assumption": [assumption_default] * n_rows,
                        "axis": axis,
                        "value": power_profile
                    }))
            elif value_type == "CALIBRATION":
                for calibration_type, data_calibration in data_run["calibration"].items():
                    for assumption in (True, False):
                        power_profile_name = "freq_power_profile" if is_freq_domain else "time_power_profile"
                        power_profile_name = (
                            f"{power_profile_name}_assumption" if assumption else f"{power_profile_name}_raw"
                        )
                        for rx_tx_pair, power_profile in enumerate(data_calibration["median"][power_profile_name]):
                            df_run_list.append(pd.DataFrame.from_dict({
                                **_expand_info_cols(data_run, n_rows),
                                "value_type": [value_type] * n_rows,
                                "domain": [domain] * n_rows,
                                "rx_tx_pair": [rx_tx_pair] * n_rows,
                                "calibration_type": [calibration_type] * n_rows,
                                "assumption": [assumption] * n_rows,
                                "axis": axis,
                                "value": power_profile
                            }))
            else:
                raise ValueError

        # Power per path
        domain = "PATHS"
        toa_axes = data_run["toa_axes"]
        for rx_tx_pair, power_profile in enumerate(data_run["ground_truth_materials"]["paths_power"]):
            mask = (power_profile != 0.0)
            axis = toa_axes[rx_tx_pair, mask]
            n_paths = len(axis)
            df_run_list.append(pd.DataFrame.from_dict({
                **_expand_info_cols(data_run, n_paths),
                "value_type": ["GROUND_TRUTH_MATERIALS"] * n_paths,
                "domain": [domain] * n_paths,
                "rx_tx_pair": [rx_tx_pair] * n_paths,
                "calibration_type": [calibration_type_default] * n_paths,
                "assumption": [assumption_default] * n_paths,
                "axis": axis,
                "value": power_profile[mask]
            }))
        for calibration_type, data_calibration in data_run["calibration"].items():
            for rx_tx_pair, power_profile in enumerate(data_calibration["median"]["paths_power"]):
                mask = (power_profile != 0.0)
                axis = toa_axes[rx_tx_pair, mask]
                n_paths = len(axis)
                df_run_list.append(pd.DataFrame.from_dict({
                    **_expand_info_cols(data_run, n_paths),
                    "value_type": ["CALIBRATION"] * n_paths,
                    "domain": [domain] * n_paths,
                    "rx_tx_pair": [rx_tx_pair] * n_paths,
                    "calibration_type": [calibration_type] * n_paths,
                    "assumption": [assumption_default] * n_paths,
                    "axis": axis,
                    "value": power_profile[mask]
                }))

        # Append data
        df_list += df_run_list

    # Stack dataframes
    df = pd.concat(df_list, ignore_index=True)

    # Store dataframe
    with SafeOpen(
        power_profiles_plots_folder(protocol_name=protocol_name, n_loaded_runs=n_runs_to_load),
        PROTOCOL_POWER_PROFILES_DATA_FILENAME,
        "wb"
    ) as file:
        df.to_parquet(path=file, engine="fastparquet", compression='gzip', index=False)

