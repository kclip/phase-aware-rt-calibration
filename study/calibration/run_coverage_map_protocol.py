import argparse
import timeit

from study.calibration.utils import RunType, CalibrationType
from study.calibration.experiment_protocol import load_experiment_protocol, get_all_protocol_names
from study.calibration.individual_runs.run_coverage_map import run_coverage_map


def run_coverage_map_protocol(
    n_run: int,
    protocol_name: str,
    # Force runs that are not specified in the given protocol
    force_calibration_type: str = None,
    array_config: str = "mimo",
    ignore_cfr_data: bool = False  # If True, do not store coverage map CFR
):
    protocol_parameters = load_experiment_protocol(protocol_name)
    for run_parameters in protocol_parameters:
        print(run_parameters)
        start_timer = timeit.default_timer()

        # Select which calibration types to run
        calibration_types_to_run = (
            [force_calibration_type]  # Run manually specified calibration type
            if force_calibration_type is not None else
            run_parameters.run_calibration_types  # Run calibration types specified in the protocol
        )

        # Run
        for calibration_type in calibration_types_to_run:
            run_coverage_map(
                load_materials_run_type=RunType.CALIBRATION,
                measurement_snr=run_parameters.measurement_snr,
                measurement_additive_noise=run_parameters.measurement_additive_noise,
                measurement_perfect_phase=run_parameters.measurement_perfect_phase,
                measurement_von_mises_mean=run_parameters.measurement_von_mises_mean,
                measurement_von_mises_concentration=run_parameters.measurement_von_mises_concentration,
                bandwidth=run_parameters.bandwidth,
                position_noise_amplitude=run_parameters.position_noise_amplitude,
                n_channel_measurements=run_parameters.n_channel_measurements,
                calibration_type=calibration_type,
                array_config=array_config,
                ignore_cfr_data=ignore_cfr_data,
                n_run=n_run
            )

        # End run
        end_timer = timeit.default_timer()
        duration_run = end_timer - start_timer
        print(f"Run executed in {duration_run // 60} min and {duration_run % 60:.1f} s")


if __name__ == '__main__':
    protocol_names = get_all_protocol_names()

    parser = argparse.ArgumentParser(
        "Run a series of CFR maps for the different SNRs and phase-error concentrations specified in the protocol"
    )
    parser.add_argument("--n-run", dest="n_run", type=int, help="Select run number")
    parser.add_argument("--protocol-name", dest="protocol_name", type=str,
                        help=f"Select protocol to run among: {'|'.join(protocol_names)}")
    parser.add_argument("--force-calibration-type", dest="force_calibration_type",  type=str, default=None,
                        help=f"Select calibration type to run among: {'|'.join(CalibrationType.all_types())}")
    parser.add_argument("--array-config", dest="array_config", type=str, default=None,
                        help=f"Config of simulated arrays. Choose among: 'mimo' | 'miso' | 'siso'")
    parser.add_argument("--ignore-cfr-data", dest="ignore_cfr_data", default=False, action="store_true",
                        help="If set, do not store CFR data (can save disk memory when using large arrays)")
    args = parser.parse_args()

    _force_calibration_type = None if args.force_calibration_type in (None, "") else args.force_calibration_type

    run_coverage_map_protocol(
        n_run=args.n_run,
        protocol_name=args.protocol_name,
        force_calibration_type=_force_calibration_type,
        array_config=args.array_config,
        ignore_cfr_data=args.ignore_cfr_data
    )
