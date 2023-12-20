import os
import pandas as pd
import numpy as np
import tensorflow as tf

from settings import LOGS_FOLDER
from src.utils.save_utils import SafeOpen
from src.utils.python_utils import rad2deg
from src.utils.telecom_utils import to_db
from src.utils.plot_utils import estimation_error
from src.data_classes import CoverageMapPowerData
from src.scenarios import get_scenario
from src.coverage_map.channel_power_map import InterferenceType, get_channel_power_map
from src.mappings.von_mises_std_concentration_mapping.mapping_funcs import map_von_mises_concentration_to_std
from study.calibration.utils import power_map_data_subfolder, PROTOCOL_POWER_MAP_DATA_FILENAME, \
    RunType, protocol_plots_folder, coverage_map_filename
from study.calibration.experiment_config import PLOT_QUANTILE_VALUE, PLOT_QUANTILE_METHOD, SUBCARRIER_SPACING, \
    NUM_SUBCARRIERS, get_scenario_metadata, get_num_subcarriers_from_bandwidth
from study.calibration.experiment_protocol import load_experiment_protocol
from study.calibration.individual_runs.run_coverage_map import get_coverage_map_run_name


def run_gather_power_map_data(
    ground_truth_coverage_map_run_name: str,
    protocol_name: str,
    array_config: str,
    n_runs_to_load: int,
):
    # Init
    # ----
    save_dir = os.path.join(
        protocol_plots_folder(protocol_name=protocol_name, n_loaded_runs=n_runs_to_load),
        power_map_data_subfolder(array_config=array_config)
    )
    # Load scene
    # Note: all coverage maps (included the ones estimated at the DT) are displayed in the ground-truth scene
    scenario_metadata = get_scenario_metadata(ground_truth_geometry=True)
    scene = get_scenario(scenario_metadata)
    # Load ground-truth data
    ground_truth_power_map_data = CoverageMapPowerData.load(
        os.path.join(LOGS_FOLDER, ground_truth_coverage_map_run_name)
    )
    # Dataframe data
    data = []

    kwargs_power_maps = dict(idx_tx=0)
    # Load and compute coverage map for each parameter set in protocol
    # ----------------------------------------------------------------
    for run_parameters in load_experiment_protocol(protocol_name):
        print("=============================")
        print(f"Loading runs from parameters: \n{run_parameters}")

        # Load ground-truth coverage map
        # ------------------------------
        ground_truth_interference_type = InterferenceType.UNIFORM_PHASE
        ground_truth_power_map = get_channel_power_map(
            scene=scene,
            coverage_map_power_data=ground_truth_power_map_data,
            interference_type=ground_truth_interference_type,
            **kwargs_power_maps
        )
        # We only analyze cells with power coverage in the ground-truth setting
        mask_ground_truth_power_map = tf.logical_and(
            ground_truth_power_map.as_tensor() != 0,
            tf.logical_not(tf.math.is_nan(ground_truth_power_map.as_tensor()))
        )

        # Load calibrated coverage maps
        # -----------------------------
        all_calibration_info = []
        for n_run in range(n_runs_to_load):
            for calibration_type in run_parameters.run_calibration_types:
                # Load coverage map
                coverage_map_data_run_name = get_coverage_map_run_name(
                    array_config=array_config,
                    load_materials_run_type=RunType.CALIBRATION,
                    measurement_snr=run_parameters.measurement_snr,
                    measurement_additive_noise=run_parameters.measurement_additive_noise,
                    measurement_perfect_phase=run_parameters.measurement_perfect_phase,
                    measurement_von_mises_mean=run_parameters.measurement_von_mises_mean,
                    measurement_von_mises_concentration=run_parameters.measurement_von_mises_concentration,
                    calibration_type=calibration_type,
                    bandwidth=run_parameters.bandwidth,
                    position_noise_amplitude=run_parameters.position_noise_amplitude,
                    n_channel_measurements=run_parameters.n_channel_measurements,
                    n_run=n_run,
                )
                calibrated_power_map_data = CoverageMapPowerData.load(
                    os.path.join(LOGS_FOLDER, coverage_map_data_run_name)
                )

                # Get predicted channel power map at the DT after calibration
                calibration_interference_type = InterferenceType.UNIFORM_PHASE
                calibrated_power_map = get_channel_power_map(
                    scene=scene,
                    coverage_map_power_data=calibrated_power_map_data,
                    interference_type=calibration_interference_type,
                    **kwargs_power_maps
                )

                # Store run data
                all_calibration_info.append(
                    (n_run, calibrated_power_map, calibration_type)
                )

        print(f"Loaded data from all {n_runs_to_load} runs !")

        # Compare calibrated coverage maps to ground-truth
        # ------------------------------------------------
        estimation_error_tensors_all_calibration_types = {
            calibration_type: []
            for calibration_type in run_parameters.run_calibration_types
        }
        num_subcarriers = get_num_subcarriers_from_bandwidth(
            bandwidth=run_parameters.bandwidth,
            default_num_subcarriers=NUM_SUBCARRIERS,
            subcarrier_spacing=SUBCARRIER_SPACING
        )
        for (
            n_run,
            calibrated_power_map,
            calibration_type
        ) in all_calibration_info:
            # Estimation error power map
            # ==========================
            calibrated_power_tensor = calibrated_power_map.as_tensor()
            calibrated_power_tensor = tf.where(  # Fill NaN cells in estimated power with 0 (i.e. no power coverage)
                tf.math.is_nan(calibrated_power_tensor),
                tf.constant(0, dtype=tf.float32),
                calibrated_power_tensor
            )
            estimation_error_tensor = estimation_error(  # Compute estimation error
                estimated_value=calibrated_power_tensor,
                ground_truth_value=ground_truth_power_map.as_tensor()
            )
            estimation_error_tensor = tf.where(  # We only compute the error on cells covered by the ground-truth map
                mask_ground_truth_power_map,
                estimation_error_tensor,
                tf.constant(0, dtype=tf.float32)
            )
            # Store estimation error power map
            estimation_error_tensors_all_calibration_types[calibration_type].append(estimation_error_tensor)

            # Get power map mean error data (average error on cells with ground-truth coverage)
            # =================================================================================
            error_array = estimation_error_tensor[mask_ground_truth_power_map].numpy()  # Note: this should not contain any NaN value
            error_mean = error_array.mean()
            # Get error quantiles
            error_q1 = np.quantile(error_array, PLOT_QUANTILE_VALUE, method=PLOT_QUANTILE_METHOD)
            error_q2 = np.quantile(error_array, 1 - PLOT_QUANTILE_VALUE, method=PLOT_QUANTILE_METHOD)
            if error_q1 < error_q2:
                error_low_q, error_high_q = error_q1, error_q2
            else:
                error_low_q, error_high_q = error_q2, error_q1

            data.append(dict(
                # Run params
                measurement_snr=run_parameters.measurement_snr,
                measurement_additive_noise=run_parameters.measurement_additive_noise,
                measurement_perfect_phase=run_parameters.measurement_perfect_phase,
                measurement_von_mises_mean=run_parameters.measurement_von_mises_mean,
                measurement_von_mises_concentration=run_parameters.measurement_von_mises_concentration,
                subcarrier_spacing=SUBCARRIER_SPACING,
                num_subcarriers=num_subcarriers,
                bandwidth=SUBCARRIER_SPACING * num_subcarriers,
                calibration_type=calibration_type,
                # Loaded and computed data
                n_run=n_run,
                error_mean=error_mean,
                error_low_q=error_low_q,
                error_high_q=error_high_q
            ))

        # Save mean-error coverage maps
        # -----------------------------
        for (
            calibration_type,
            estimation_error_tensors_all_runs
        ) in estimation_error_tensors_all_calibration_types.items():
            if len(estimation_error_tensors_all_runs) > 0:
                # Average errors across runs
                mean_estimation_error_tensor = tf.stack(estimation_error_tensors_all_runs, axis=0)
                mean_estimation_error_tensor = tf.reduce_mean(mean_estimation_error_tensor, axis=0)

                # Save CM data
                filename = coverage_map_filename(calibration_type=calibration_type, run_parameters=run_parameters)
                with SafeOpen(save_dir, filename, "wb") as file:
                    np.save(file, mean_estimation_error_tensor.numpy())

        print("Mean error across runs coverage maps saved !")

    # Save coverage map data
    # ----------------------
    df = pd.DataFrame(data)

    # Add von Mises phase-error std
    std_col = "measurement_von_mises_std_deg"
    df[std_col] = 0.0
    mask_von_mises_calibration = (~df["measurement_perfect_phase"])
    df.loc[mask_von_mises_calibration, std_col] = map_von_mises_concentration_to_std(
        concentrations=df.loc[mask_von_mises_calibration, "measurement_von_mises_concentration"],
        round_std=3,
        print_errors=True
    )
    df[std_col] = rad2deg(df[std_col])

    # SNR in dB
    df[f"measurement_snr_db"] = to_db(df["measurement_snr"])

    with SafeOpen(save_dir, PROTOCOL_POWER_MAP_DATA_FILENAME, "w") as file:
        df.to_csv(file, index=False)
