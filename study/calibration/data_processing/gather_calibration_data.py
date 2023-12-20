import os
import tensorflow as tf
import pandas as pd

from settings import LOGS_FOLDER
from src.data_classes import MaterialsCalibrationInfo, VonMisesCalibrationInfo, CalibrationChannelsPower
from src.utils.python_utils import rad2deg
from src.utils.save_utils import SafeOpen
from src.utils.telecom_utils import to_db
from src.mappings.von_mises_std_concentration_mapping.mapping_funcs import map_von_mises_concentration_to_std
from study.calibration.experiment_config import SUBCARRIER_SPACING, NUM_SUBCARRIERS, get_num_subcarriers_from_bandwidth
from study.calibration.experiment_protocol import load_experiment_protocol
from study.calibration.individual_runs.run_measurements import get_measurements_run_name
from study.calibration.individual_runs.run_calibration import get_calibration_run_name
from study.calibration.utils import PROTOCOL_CALIBRATION_DATA_FILENAME, CalibrationType, calibration_plots_folder


def run_gather_calibration_data(
    protocol_name: str,
    n_runs_to_load: int
):
    # Init
    df_list = []

    # Get calibration data
    # --------------------
    for run_parameters in load_experiment_protocol(protocol_name):
        print("=============================")
        print(f"Loading runs from parameters: \n{run_parameters}")

        for n_run in range(n_runs_to_load):
            for calibration_type in run_parameters.run_calibration_types:
                # Measurement run name
                measurements_run_name = get_measurements_run_name(
                    measurement_snr=run_parameters.measurement_snr,
                    measurement_additive_noise=run_parameters.measurement_additive_noise,
                    measurement_perfect_phase=run_parameters.measurement_perfect_phase,
                    measurement_von_mises_mean=run_parameters.measurement_von_mises_mean,
                    measurement_von_mises_concentration=run_parameters.measurement_von_mises_concentration,
                    bandwidth=run_parameters.bandwidth,
                    position_noise_amplitude=run_parameters.position_noise_amplitude,
                    n_run=n_run
                )
                measurement_dir = os.path.join(LOGS_FOLDER, measurements_run_name)
                # Load calibration data
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
                calibration_info = MaterialsCalibrationInfo.load(calibration_dir)
                von_mises_info = None
                if calibration_type in (CalibrationType.PEAC, CalibrationType.PEAC_FIXED_PRIOR):
                    von_mises_info = VonMisesCalibrationInfo.load(calibration_dir)

                # Get ground-truth power at calibration locations
                ground_truth_power = CalibrationChannelsPower.load(measurement_dir).uniform_phases_power
                mean_ground_truth_power = tf.reduce_mean(ground_truth_power)
                # Get predicted power at calibration locations
                calibrated_power = CalibrationChannelsPower.load(calibration_dir).uniform_phases_power                
                mean_calibrated_power = tf.reduce_mean(calibrated_power)
                mean_estimation_error_power = tf.reduce_mean(
                    tf.abs(ground_truth_power - calibrated_power) / ground_truth_power
                )

                # Get info
                num_subcarriers = get_num_subcarriers_from_bandwidth(
                    bandwidth=run_parameters.bandwidth,
                    default_num_subcarriers=NUM_SUBCARRIERS,
                    subcarrier_spacing=SUBCARRIER_SPACING
                )
                _calibration_selected_info = {
                    # Metadata
                    "n_run": n_run,
                    "calibration_type": calibration_type,
                    "measurement_snr": run_parameters.measurement_snr,
                    "measurement_additive_noise": run_parameters.measurement_additive_noise,
                    "measurement_perfect_phase": run_parameters.measurement_perfect_phase,
                    "measurement_von_mises_mean": run_parameters.measurement_von_mises_mean,
                    "measurement_von_mises_concentration": run_parameters.measurement_von_mises_concentration,
                    "subcarrier_spacing": SUBCARRIER_SPACING,
                    "num_subcarriers": num_subcarriers,
                    "bandwidth": SUBCARRIER_SPACING * num_subcarriers,
                    "position_noise_amplitude": run_parameters.position_noise_amplitude,
                    "n_channel_measurements": run_parameters.n_channel_measurements,
                    # Calibration
                    "ground_truth_conductivity": calibration_info.ground_truth_materials_conductivity[0],
                    "ground_truth_permittivity": calibration_info.ground_truth_materials_permittivity[0],
                    "conductivity": calibration_info.calibrated_materials_conductivity[0],
                    "permittivity": calibration_info.calibrated_materials_permittivity[0],
                    "calibrated_von_mises_concentration": None,
                    # Power
                    "mean_ground_truth_power": float(mean_ground_truth_power.numpy()),
                    "mean_calibrated_power": float(mean_calibrated_power.numpy()),
                    "mean_estimation_error_power": float(mean_estimation_error_power.numpy()),
                }
                if von_mises_info is not None:
                    _calibration_selected_info[
                        "calibrated_von_mises_concentration"
                    ] = von_mises_info.von_mises_global_concentration

                # Store calibration run info
                df_list.append(pd.DataFrame([_calibration_selected_info]))

            print(f"Loaded run {n_run} / {n_runs_to_load}")

    # Concat data of all runs into single dataframe
    df = pd.concat(df_list, ignore_index=True)

    # Process data
    # ------------
    # Map measurement and calibration von Mises concentration to standard deviation in degrees
    for concentration_col, perfect_phase_mask, std_col in [
        (
                "measurement_von_mises_concentration",
                df["measurement_perfect_phase"],
                "measurement_von_mises_std"
        ),
        (
                "calibrated_von_mises_concentration",
                df["calibrated_von_mises_concentration"].isnull(),
                "calibrated_von_mises_std"
        ),
    ]:
        # Perfect phase rows
        df[std_col] = 0.0
        # Rows with phase noise
        df.loc[~perfect_phase_mask, std_col] = map_von_mises_concentration_to_std(
            concentrations=df.loc[~perfect_phase_mask, concentration_col],
            round_std=3,
            print_errors=True
        )
        # To degrees
        df[std_col] = rad2deg(df[std_col])

    # SNR in dB
    df["measurement_snr_db"] = to_db(df["measurement_snr"])
    
    # Bandwidth in MHz
    df["bandwidth_mhz"] = df["bandwidth"] / 1e6

    # Store dataframe
    with SafeOpen(
        calibration_plots_folder(protocol_name=protocol_name, n_loaded_runs=n_runs_to_load),
        PROTOCOL_CALIBRATION_DATA_FILENAME,
        "w"
    ) as file:
        df.to_csv(file, index=False)
