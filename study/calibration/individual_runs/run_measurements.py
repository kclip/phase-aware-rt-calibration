import os
import argparse

from settings import LOGS_FOLDER
from metadata import Metadata
from src.data_classes import MaterialsMapping
from src.scenarios import get_scenario
from src.ofdm_measurements.main import simulate_measurements
from study.calibration.utils import format_parameter_folder
from study.calibration.experiment_config import EXPERIMENT_NAME, NUM_SUBCARRIERS, get_scenario_metadata, \
    get_measurement_metadata


def get_measurements_run_name(
    measurement_snr: float,
    measurement_additive_noise: bool,
    measurement_perfect_phase: bool,
    measurement_von_mises_mean: float,
    measurement_von_mises_concentration: float,
    bandwidth: float,
    position_noise_amplitude: float,
    n_run: int = 0
):
    run_name = f"{EXPERIMENT_NAME}_mesurement_run_{n_run}"
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
    
    # Append params to name
    for label, param in label_params:
        if param is not None:
            str_param = str(float(param)).split(".")
            run_name += f"_{label}_{str_param[0]}_{str_param[1]}"
    
    return run_name


def run_measurements(
    measurement_snr: float,
    measurement_additive_noise: bool,
    measurement_perfect_phase: bool,
    measurement_von_mises_mean: float,
    measurement_von_mises_concentration: float,
    bandwidth: float,
    position_noise_amplitude: float,
    n_run: int,
):
    # Compute measurements
    scenario_metadata = get_scenario_metadata(ground_truth_geometry=True)
    scene = get_scenario(scenario_metadata)
    measurement_metadata = get_measurement_metadata(
        measurement_snr=measurement_snr,
        measurement_additive_noise=measurement_additive_noise,
        measurement_perfect_phase=measurement_perfect_phase,
        measurement_von_mises_mean=measurement_von_mises_mean,
        measurement_von_mises_concentration=measurement_von_mises_concentration,
        bandwidth=bandwidth,
        position_noise_amplitude=position_noise_amplitude
    )
    measurement_data = simulate_measurements(
        scene=scene,
        measurement_metadata=measurement_metadata
    )

    # Get mapped materials
    materials_mapping = MaterialsMapping.from_scene(scene)

    # Store data and metadata
    metadata = Metadata(
        measurement_metadata=measurement_metadata,
        scenario_metadata=scenario_metadata
    )
    run_name = get_measurements_run_name(
        measurement_snr=measurement_snr,
        measurement_additive_noise=measurement_additive_noise,
        measurement_perfect_phase=measurement_perfect_phase,
        measurement_von_mises_mean=measurement_von_mises_mean,
        measurement_von_mises_concentration=measurement_von_mises_concentration,
        bandwidth=bandwidth,
        position_noise_amplitude=position_noise_amplitude,
        n_run=n_run
    )
    save_run_dir = os.path.join(LOGS_FOLDER, run_name)
    for storable_dataclass in [metadata, measurement_data, materials_mapping]:
        storable_dataclass.store(save_run_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Simulate channel measurements using ray-traced path components with von Mises-distributed phase noise or "
        "Rx position errors, and complex Gaussian measurement noise"
    )
    parser.add_argument("--meas-snr", dest="measurement_snr", type=float, help="SNR of measurements")
    parser.add_argument("--meas-no-additive-noise", dest="measurement_no_additive_noise", default=False, action="store_true",
                        help="If set, do not add Gaussian noise to measurements")
    parser.add_argument("--meas-perfect-phase", dest="measurement_perfect_phase", default=False, action="store_true",
                        help="If True, do not add phase noise to the ray-traced path components")
    parser.add_argument("--meas-vm-mean", dest="measurement_von_mises_mean", type=float, default=0.0,
                        help="Mean of von Mises-distributed phase noise")
    parser.add_argument("--meas-vm-concentration", dest="measurement_von_mises_concentration", type=float, default=0.0,
                        help="Concentration of von Mises-distributed phase noise")
    parser.add_argument("--bandwidth", dest="bandwidth", type=float, default=None,
                        help=f"Bandwidth of the system (defines the number of subcarriers). "
                             f"If not specified, the number of subcarriers defaults to {NUM_SUBCARRIERS}")
    parser.add_argument("--position-noise-amplitude", dest="position_noise_amplitude", type=float, default=None,
                        help=f"Position noise amplitude (in multiples of the carrier wavelength); used in position"
                             f"noise measurements only.")
    parser.add_argument("--n-run", dest="n_run", type=int, help="Select run number")
    args = parser.parse_args()

    run_measurements(
        measurement_snr=args.measurement_snr,
        measurement_additive_noise=(not args.measurement_no_additive_noise),
        measurement_perfect_phase=args.measurement_perfect_phase,
        measurement_von_mises_mean=args.measurement_von_mises_mean,
        measurement_von_mises_concentration=args.measurement_von_mises_concentration,
        bandwidth=args.bandwidth,
        position_noise_amplitude=args.position_noise_amplitude,
        n_run=args.n_run
    )
