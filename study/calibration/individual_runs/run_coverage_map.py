import os
import argparse

from settings import LOGS_FOLDER
from metadata import Metadata
from src.data_classes import MaterialsMapping
from src.scenarios import get_scenario
from src.coverage_map.compute_coverage_map_data import compute_coverage_map_data
from study.calibration.utils import RunType, CalibrationType
from study.calibration.experiment_config import EXPERIMENT_NAME, NUM_SUBCARRIERS, coverage_map_metadata, \
    get_scenario_metadata
from study.calibration.individual_runs.run_measurements import get_measurements_run_name
from study.calibration.individual_runs.run_calibration import get_calibration_run_name


def _load_run_name(
    # Specify which run materials should be loaded from
    load_materials_run_type: str = None,  # if None, use default materials in scenario
    measurement_snr: float = None,
    measurement_additive_noise: bool = True,
    measurement_perfect_phase: bool = False,
    measurement_von_mises_mean: float = None,
    measurement_von_mises_concentration: float = None,
    bandwidth: float = None,
    position_noise_amplitude: float = None,
    n_channel_measurements: int = None,
    calibration_type: str = None,
    n_run: int = 0
):
    if load_materials_run_type is None:
        return None
    elif load_materials_run_type == RunType.MEASUREMENT:
        return get_measurements_run_name(
            measurement_snr=measurement_snr,
            measurement_additive_noise=measurement_additive_noise,
            measurement_perfect_phase=measurement_perfect_phase,
            measurement_von_mises_mean=measurement_von_mises_mean,
            measurement_von_mises_concentration=measurement_von_mises_concentration,
            bandwidth=bandwidth,
            position_noise_amplitude=position_noise_amplitude,
            n_run=n_run
        )
    elif load_materials_run_type == RunType.CALIBRATION:
        return get_calibration_run_name(
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
    else:
        raise ValueError(f"Unknown run type '{load_materials_run_type}'")


def get_coverage_map_run_name(
    array_config: str,
    load_materials_run_type: str = None,
    measurement_snr: float = None,
    measurement_additive_noise: bool = True,
    measurement_perfect_phase: bool = False,
    measurement_von_mises_mean: float = None,
    measurement_von_mises_concentration: float = None,
    bandwidth: float = None,
    position_noise_amplitude: float = None,
    n_channel_measurements: int = None,
    calibration_type: str = None,
    n_run: int = 0
) -> str:
    run_name = f"{EXPERIMENT_NAME}_coverage_map_{array_config}"
    load_run_name = _load_run_name(
        load_materials_run_type=load_materials_run_type,
        measurement_snr=measurement_snr,
        measurement_additive_noise=measurement_additive_noise,
        measurement_perfect_phase=measurement_perfect_phase,
        measurement_von_mises_mean=measurement_von_mises_mean,
        measurement_von_mises_concentration=measurement_von_mises_concentration,
        bandwidth=bandwidth,
        position_noise_amplitude=position_noise_amplitude,
        n_channel_measurements=n_channel_measurements,
        calibration_type=calibration_type,
        n_run=n_run
    )
    if load_run_name is None:
        return f"{run_name}_default_scenario_materials_run_{n_run}"
    else:
        return f"{run_name}_from_{load_run_name[len(EXPERIMENT_NAME)+1:]}"


def run_coverage_map(
    # Load previous run parameters
    load_materials_run_type: str = None,
    measurement_snr: float = None,
    measurement_additive_noise: bool = True,
    measurement_perfect_phase: bool = False,
    measurement_von_mises_mean: float = None,
    measurement_von_mises_concentration: float = None,
    bandwidth: float = None,
    position_noise_amplitude: float = None,
    n_channel_measurements: int = None,
    calibration_type: str = None,
    array_config: str = "mimo",
    ignore_cfr_data: bool = False,
    n_run: int = 0,
):
    # Calibration type works only when loading data from calibration runs
    if load_materials_run_type != RunType.CALIBRATION:
        calibration_type = None

    # Get materials mapping from previous experiment
    load_run_name = _load_run_name(
        load_materials_run_type=load_materials_run_type,
        measurement_snr=measurement_snr,
        measurement_additive_noise=measurement_additive_noise,
        measurement_perfect_phase=measurement_perfect_phase,
        measurement_von_mises_mean=measurement_von_mises_mean,
        measurement_von_mises_concentration=measurement_von_mises_concentration,
        bandwidth=bandwidth,
        position_noise_amplitude=position_noise_amplitude,
        n_channel_measurements=n_channel_measurements,
        calibration_type=calibration_type,
        n_run=n_run
    )
    if load_run_name is None:
        materials_mapping = None
    else:
        load_run_dir = os.path.join(LOGS_FOLDER, load_run_name)
        materials_mapping = MaterialsMapping.load(load_run_dir)

    # Load scene with material mapping
    selected_scenario_metadata = get_scenario_metadata(ground_truth_geometry=True, array_config_name=array_config)
    scene = get_scenario(selected_scenario_metadata, materials_mapping=materials_mapping)

    # Compute coverage map
    coverage_map_cfr_data, coverage_map_power_data = compute_coverage_map_data(
        scene=scene,
        cm_metadata=coverage_map_metadata,
        ignore_cfr_data=ignore_cfr_data
    )

    # Metadata
    metadata = Metadata(
        coverage_map_metadata=coverage_map_metadata,
        scenario_metadata=selected_scenario_metadata
    )

    # Store run
    coverage_map_run_name = get_coverage_map_run_name(
        array_config=array_config,
        load_materials_run_type=load_materials_run_type,
        measurement_snr=measurement_snr,
        measurement_additive_noise=measurement_additive_noise,
        measurement_perfect_phase=measurement_perfect_phase,
        measurement_von_mises_mean=measurement_von_mises_mean,
        measurement_von_mises_concentration=measurement_von_mises_concentration,
        bandwidth=bandwidth,
        position_noise_amplitude=position_noise_amplitude,
        n_channel_measurements=n_channel_measurements,
        calibration_type=calibration_type,
        n_run=n_run
    )
    save_run_dir = os.path.join(LOGS_FOLDER, coverage_map_run_name)

    dataclasses_to_store = [metadata, coverage_map_power_data]
    if not ignore_cfr_data:
        dataclasses_to_store.append(coverage_map_cfr_data)
    if materials_mapping is not None:
        dataclasses_to_store.append(materials_mapping)
    for storable_dataclass in dataclasses_to_store:
        storable_dataclass.store(save_run_dir)


if __name__ == '__main__':
    _loadable_run_types = ' | '.join(["None", RunType.MEASUREMENT, RunType.CALIBRATION])
    parser = argparse.ArgumentParser(
        "Compute channel frequency response map using default material parameters or a material parameters mapping "
        "from a previous run"
    )
    parser.add_argument("--load-materials-run-type", dest="load_materials_run_type", type=str, default=None,
                        help=f"Loaded run: run type to load; possible values: {_loadable_run_types}. Leave empty to "
                             f"load the default materials parameters of the scene")
    parser.add_argument("--meas-snr", dest="measurement_snr", type=float, help="Loaded run: SNR of measurements")
    parser.add_argument("--meas-no-additive-noise", dest="measurement_no_additive_noise", default=False, action="store_true",
                        help="Loaded run: if set, loaded measurements do not have Gaussian noise")
    parser.add_argument("--meas-perfect-phase", dest="measurement_perfect_phase", default=False, action="store_true",
                        help="Loaded run: phase noise or perfect phase flag")
    parser.add_argument("--meas-vm-mean", dest="measurement_von_mises_mean", type=float, default=0.0,
                        help="Loaded run: mean of von Mises-distributed phase noise during measurements")
    parser.add_argument("--meas-vm-concentration", dest="measurement_von_mises_concentration", type=float, default=0.0,
                        help="Loaded run: concentration of von Mises-distributed phase noise during measurements")
    parser.add_argument("--calibration-type", dest="calibration_type", type=str, default=None,
                        help=f"If selected run type is 'calibration', selects from which calibration scheme the "
                             f"calibrated parameters should be extracted. "
                             f"Select among: {'|'.join(CalibrationType.all_types())}")
    parser.add_argument("--bandwidth", dest="bandwidth", type=float, default=None,
                        help=f"Bandwidth of the system (defines the number of subcarriers). "
                             f"If not specified, the number of subcarriers defaults to {NUM_SUBCARRIERS}")
    parser.add_argument("--position-noise-amplitude", dest="position_noise_amplitude", type=float, default=None,
                        help=f"Position noise amplitude (in multiples of the carrier wavelength); used in position"
                             f"noise measurements only.")
    parser.add_argument("--n-channel-measurements", dest="n_channel_measurements", type=int, default=None,
                        help=f"Number of channel measurements used for calibration "
                             f"(if None, use all available measurements).")
    parser.add_argument("--array-config", dest="array_config", type=str, default=None,
                        help=f"Config of simulated arrays. Choose among: 'mimo' | 'miso' | 'siso'")
    parser.add_argument("--ignore-cfr-data", dest="ignore_cfr_data", default=False, action="store_true",
                        help="If set, do not store CFR data (can save disk memory when using large arrays)")
    parser.add_argument("--n-run", dest="n_run", type=int, help="Loaded run: select run number")
    args = parser.parse_args()

    _calibration_type = None if args.calibration_type in (None, "") else args.calibration_type

    run_coverage_map(
        load_materials_run_type=args.load_materials_run_type,
        measurement_snr=args.measurement_snr,
        measurement_additive_noise=(not args.measurement_no_additive_noise),
        measurement_perfect_phase=args.measurement_perfect_phase,
        measurement_von_mises_mean=args.measurement_von_mises_mean,
        measurement_von_mises_concentration=args.measurement_von_mises_concentration,
        bandwidth=args.bandwidth,
        position_noise_amplitude=args.position_noise_amplitude,
        n_channel_measurements=args.n_channel_measurements,
        calibration_type=_calibration_type,
        array_config=args.array_config,
        ignore_cfr_data=args.ignore_cfr_data,
        n_run=args.n_run
    )
