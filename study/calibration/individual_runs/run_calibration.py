import os
import argparse
import timeit

from settings import LOGS_FOLDER
from metadata import Metadata
from src.data_classes import MaterialsMapping, load_measurement_data
from src.scenarios import get_scenario
from src.utils.sionna_utils import setup_training_materials
from src.calibrate_materials.peac.calibrate import phase_error_aware_calibration
from src.calibrate_materials.peoc.calibrate import phase_error_oblivious_calibration
from src.calibrate_materials.upec.calibrate import uniform_phase_error_calibration
from study.calibration.utils import CalibrationType, format_parameter_folder
from study.calibration.experiment_config import EXPERIMENT_NAME, PRINT_FREQ, PEOC_PRINT_FREQ, \
    UPEC_PRINT_FREQ, PRINT_VM_PARAMS, NUM_SUBCARRIERS, get_scenario_metadata, get_measurement_metadata, \
    get_peac_calibration_metadata, get_peoc_calibration_metadata, \
    get_peac_fixed_prior_calibration_metadata, get_upec_calibration_metadata, \
    get_upec_paths_proj_calibration_metadata
from study.calibration.individual_runs.run_measurements import get_measurements_run_name


def get_calibration_run_name(
    measurement_snr: float,
    calibration_type: str,
    measurement_additive_noise: bool,
    measurement_perfect_phase: bool,
    measurement_von_mises_mean: float,
    measurement_von_mises_concentration: float,
    bandwidth: float,
    position_noise_amplitude: float,
    n_channel_measurements: int,
    n_run: int,
):
    calibration_type_folder_name = CalibrationType.get_folder_name(calibration_type=calibration_type)
    run_name = f"{EXPERIMENT_NAME}_{calibration_type_folder_name}_run_{n_run}"
    # Measurement Noise
    if (not measurement_additive_noise):
        run_name += "_no_additive_noise"
    else:
        run_name += f"_snr_{format_parameter_folder(measurement_snr)}"
    # Phase Noise
    label_params = []
    if measurement_perfect_phase:
        run_name = f"{run_name}_data_perfect_phase"
    else:
        label_params += [
            ("vm_data_mean", measurement_von_mises_mean),
            ("vm_data_concentration", measurement_von_mises_concentration)
        ]
    # Other params
    if bandwidth is not None:  # Legacy folder names do not include bandwidth
        label_params.append(("bandwidth", bandwidth))
    if position_noise_amplitude is not None:
        label_params.append(("pos_noise_amp", position_noise_amplitude))
    if n_channel_measurements is not None:
        label_params.append(("n_measurements_cal", n_channel_measurements))
    
    # Append params to name
    for label, param in label_params:
        if param is not None:
            str_param = str(float(param)).split(".")
            run_name += f"_{label}_{str_param[0]}_{str_param[1]}"
    
    return run_name


def run_calibration(
    measurement_snr: float,
    calibration_type: str,
    measurement_additive_noise: bool,
    measurement_perfect_phase: bool,
    measurement_von_mises_mean: float,
    measurement_von_mises_concentration: float,
    bandwidth: float,
    position_noise_amplitude: float,
    n_channel_measurements: int,
    n_run: int
):
    # Load scene (load geometric model, which might differ from the ground-truth geometry)
    scenario_metadata = get_scenario_metadata(ground_truth_geometry=False)
    scene = get_scenario(scenario_metadata)

    # Load measurements
    measurements_run_name = get_measurements_run_name(
        measurement_snr=measurement_snr,
        measurement_additive_noise=measurement_additive_noise,
        measurement_perfect_phase=measurement_perfect_phase,
        measurement_von_mises_mean=measurement_von_mises_mean,
        measurement_von_mises_concentration=measurement_von_mises_concentration,
        bandwidth=bandwidth,
        position_noise_amplitude=position_noise_amplitude,
        n_run=n_run
    )
    measurement_data = load_measurement_data(os.path.join(LOGS_FOLDER, measurements_run_name))
    if n_channel_measurements is not None:
        channel_measurements = measurement_data.measurement[:n_channel_measurements]
    else:
        channel_measurements = measurement_data.measurement

    # Set materials for training
    _ = setup_training_materials(scene)

    # Calibrate materials
    measurement_metadata = get_measurement_metadata(
        measurement_snr=measurement_snr,
        measurement_additive_noise=measurement_additive_noise,
        measurement_perfect_phase=measurement_perfect_phase,
        measurement_von_mises_mean=measurement_von_mises_mean,
        measurement_von_mises_concentration=measurement_von_mises_concentration,
        bandwidth=bandwidth,
        position_noise_amplitude=position_noise_amplitude
    )

    von_mises_calibration_info = None

    if calibration_type == CalibrationType.PEOC:
        calibration_metadata = get_peoc_calibration_metadata(
            normalization_constant=measurement_data.normalization_constant
        )
        materials_calibration_info = phase_error_oblivious_calibration(
            channel_measurements=channel_measurements,
            scene=scene,
            measurement_metadata=measurement_metadata,
            calibration_metadata=calibration_metadata,
            print_freq=PEOC_PRINT_FREQ
        )
    elif calibration_type in (CalibrationType.PEAC, CalibrationType.PEAC_FIXED_PRIOR):
        if calibration_type == CalibrationType.PEAC:
            calibration_metadata = get_peac_calibration_metadata(
                normalization_constant=measurement_data.normalization_constant,
                normalized_measurement_noise_std=measurement_data.normalized_measurement_noise_std.numpy().tolist()
            )
        else:  # calibration_type == CalibrationType.PEAC_FIXED_PRIOR
            calibration_metadata = get_peac_fixed_prior_calibration_metadata(
                normalization_constant=measurement_data.normalization_constant,
                normalized_measurement_noise_std=measurement_data.normalized_measurement_noise_std.numpy().tolist()
            )
        materials_calibration_info, von_mises_calibration_info = phase_error_aware_calibration(
            channel_measurements=channel_measurements,
            scene=scene,
            measurement_metadata=measurement_metadata,
            calibration_metadata=calibration_metadata,
            print_von_mises_params=PRINT_VM_PARAMS,
            print_freq=PRINT_FREQ
        )
    elif calibration_type in (CalibrationType.UPEC, CalibrationType.UPEC_PATHS_PROJ):
        if calibration_type == CalibrationType.UPEC:
            calibration_metadata = get_upec_calibration_metadata(
                normalization_constant=measurement_data.normalization_constant
            )
        else:  # calibration_type == CalibrationType.UPEC_PATHS_PROJ
            calibration_metadata = get_upec_paths_proj_calibration_metadata(
                normalization_constant=measurement_data.normalization_constant
            )
        materials_calibration_info = uniform_phase_error_calibration(
            channel_measurements=channel_measurements,
            scene=scene,
            measurement_metadata=measurement_metadata,
            calibration_metadata=calibration_metadata,
            print_freq=UPEC_PRINT_FREQ
        )
    else:
        raise ValueError(f"Unknown calibration type '{calibration_type}'")

    # Get mapped materials
    materials_mapping = MaterialsMapping.from_scene(scene)

    # Save data
    metadata = Metadata(
        measurement_metadata=measurement_metadata,
        calibrate_materials_metadata=calibration_metadata,
        scenario_metadata=scenario_metadata
    )
    calibration_run_name = get_calibration_run_name(
        measurement_snr=measurement_snr,
        measurement_additive_noise=measurement_additive_noise,
        measurement_perfect_phase=measurement_perfect_phase,
        measurement_von_mises_mean=measurement_von_mises_mean,
        measurement_von_mises_concentration=measurement_von_mises_concentration,
        calibration_type=calibration_type,
        bandwidth=bandwidth,
        position_noise_amplitude=position_noise_amplitude,
        n_channel_measurements=n_channel_measurements,
        n_run=n_run
    )
    save_run_dir = os.path.join(LOGS_FOLDER, calibration_run_name)
    dataclasses_to_store = [
        metadata,
        materials_calibration_info,
        materials_mapping
    ]
    if von_mises_calibration_info is not None:
        dataclasses_to_store.append(von_mises_calibration_info)
    for storable_dataclass in dataclasses_to_store:
        storable_dataclass.store(save_run_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Run material calibration from previously recorded measurements for the selected calibration type"
    )
    parser.add_argument("--meas-snr", dest="measurement_snr", type=float, help="SNR of measurements")
    parser.add_argument("--calibration-type", dest="calibration_type", type=str,
                        help=f"Select calibration type to run among : {'|'.join(CalibrationType.all_types())}")
    parser.add_argument("--meas-no-additive-noise", dest="measurement_no_additive_noise", default=False, action="store_true",
                        help="If set, do not add Gaussian noise to measurements")
    parser.add_argument("--meas-perfect-phase", dest="measurement_perfect_phase", default=False, action="store_true",
                        help="If True, do not add phase noise to the ray-traced path components")
    parser.add_argument("--meas-vm-mean", dest="measurement_von_mises_mean", type=float, default=0.0,
                        help="Mean of von Mises-distributed phase noise during measurements")
    parser.add_argument("--meas-vm-concentration", dest="measurement_von_mises_concentration", type=float, default=0.0,
                        help="Concentration of von Mises-distributed phase noise during measurements")
    parser.add_argument("--bandwidth", dest="bandwidth", type=float, default=None,
                        help=f"Bandwidth of the system (defines the number of subcarriers). "
                             f"If not specified, the number of subcarriers defaults to {NUM_SUBCARRIERS}")
    parser.add_argument("--position-noise-amplitude", dest="position_noise_amplitude", type=float, default=None,
                        help=f"Position noise amplitude (in multiples of the carrier wavelength); used in position"
                             f"noise measurements only.")
    parser.add_argument("--n-channel-measurements", dest="n_channel_measurements", type=int, default=None,
                        help=f"Number of channel measurements used for calibration "
                             f"(if None, use all available measurements).")
    parser.add_argument("--n-run", dest="n_run", type=int, help="Select run number")
    args = parser.parse_args()

    start = timeit.default_timer()

    run_calibration(
        measurement_snr=args.measurement_snr,
        calibration_type=args.calibration_type,
        measurement_additive_noise=(not args.measurement_no_additive_noise),
        measurement_perfect_phase=args.measurement_perfect_phase,
        measurement_von_mises_mean=args.measurement_von_mises_mean,
        measurement_von_mises_concentration=args.measurement_von_mises_concentration,
        bandwidth=args.bandwidth,
        position_noise_amplitude=args.position_noise_amplitude,
        n_channel_measurements=args.n_channel_measurements,
        n_run=args.n_run
    )

    end = timeit.default_timer()
    duration = end - start
    print(f"Calibration executed in {duration // 60} min and {duration % 60} s")
