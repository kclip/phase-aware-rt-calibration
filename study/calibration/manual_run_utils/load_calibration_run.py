import os
from typing import Tuple

from settings import LOGS_FOLDER
from src.data_classes import _MeasurementDataBase, MaterialsCalibrationInfo, VonMisesCalibrationInfo, \
    load_measurement_data
from study.calibration.individual_runs.run_calibration import get_calibration_run_name
from study.calibration.individual_runs.run_measurements import get_measurements_run_name
from study.calibration.utils import CalibrationType


def load_calibration_run(
    measurement_snr: float,
    calibration_type: str,  # CalibrationType
    measurement_additive_noise: bool = True,
    measurement_perfect_phase: bool = False,
    measurement_von_mises_mean: float = None,
    measurement_von_mises_concentration: float = None,
    bandwidth: int = None,
    position_noise_amplitude: float = None,
    n_channel_measurements: int = None,
    n_run: int = None
) -> Tuple[_MeasurementDataBase, MaterialsCalibrationInfo, VonMisesCalibrationInfo]:
    # Load measurement data
    meas_run_name = get_measurements_run_name(
        measurement_snr=measurement_snr,
        measurement_additive_noise=measurement_additive_noise,
        measurement_perfect_phase=measurement_perfect_phase,
        measurement_von_mises_mean=measurement_von_mises_mean,
        measurement_von_mises_concentration=measurement_von_mises_concentration,
        bandwidth=bandwidth,
        position_noise_amplitude=position_noise_amplitude,
        n_run=n_run
    )
    meas_data = load_measurement_data(os.path.join(LOGS_FOLDER, meas_run_name))

    # Load calibration info and von mises parameters info
    cal_run_name = get_calibration_run_name(
        measurement_snr=measurement_snr,
        calibration_type=calibration_type,
        measurement_additive_noise=measurement_additive_noise,
        measurement_perfect_phase=measurement_perfect_phase,
        measurement_von_mises_mean=measurement_von_mises_mean,
        measurement_von_mises_concentration=measurement_von_mises_concentration,
        bandwidth=bandwidth,
        position_noise_amplitude=position_noise_amplitude,
        n_channel_measurements=n_channel_measurements,
        n_run=n_run
    )
    cal_dir = os.path.join(LOGS_FOLDER, cal_run_name)
    cal_info = MaterialsCalibrationInfo.load(cal_dir)
    if calibration_type in (CalibrationType.PEAC, CalibrationType.PEAC_FIXED_PRIOR):
        vm_info = VonMisesCalibrationInfo.load(cal_dir)
    else:
        vm_info = None

    return meas_data, cal_info, vm_info
