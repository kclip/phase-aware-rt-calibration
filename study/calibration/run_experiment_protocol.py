import argparse
import timeit

from study.calibration.utils import CalibrationType, RunType
from study.calibration.experiment_protocol import load_experiment_protocol, get_all_protocol_names
from study.calibration.individual_runs.run_measurements import run_measurements
from study.calibration.individual_runs.run_calibration import run_calibration
from study.calibration.individual_runs.run_power_estimates import compute_channel_power_at_calibration_locations


def run_experiments_protocol(
    n_run: int,
    protocol_name: str,
    skip_measurement: bool = False,
    skip_channel_power: bool = False,
    # Force runs that are not specified in the given protocol
    force_calibration_type: str = None
):
    protocol_parameters = load_experiment_protocol(protocol_name)

    # Run experiments
    for run_parameters in protocol_parameters:
        print(run_parameters)
        start_timer = timeit.default_timer()

        # Run measurements
        if not skip_measurement:
            run_measurements(
                measurement_snr=run_parameters.measurement_snr,
                measurement_additive_noise=run_parameters.measurement_additive_noise,
                measurement_perfect_phase=run_parameters.measurement_perfect_phase,
                measurement_von_mises_mean=run_parameters.measurement_von_mises_mean,
                measurement_von_mises_concentration=run_parameters.measurement_von_mises_concentration,
                bandwidth=run_parameters.bandwidth,
                position_noise_amplitude=run_parameters.position_noise_amplitude,
                n_run=n_run
            )
            if not skip_channel_power:
                compute_channel_power_at_calibration_locations(
                    run_type=RunType.MEASUREMENT,
                    measurement_snr=run_parameters.measurement_snr,
                    measurement_additive_noise=run_parameters.measurement_additive_noise,
                    measurement_perfect_phase=run_parameters.measurement_perfect_phase,
                    measurement_von_mises_mean=run_parameters.measurement_von_mises_mean,
                    measurement_von_mises_concentration=run_parameters.measurement_von_mises_concentration,
                    bandwidth=run_parameters.bandwidth,
                    position_noise_amplitude=run_parameters.position_noise_amplitude,
                    n_channel_measurements=run_parameters.n_channel_measurements,
                    n_run=n_run,
                    calibration_type=None
                )

        calibration_types_to_run = (
            [force_calibration_type]  # Run manually specified calibration type
            if force_calibration_type is not None else
            run_parameters.run_calibration_types  # Run calibration types specified in the protocol
        )
        for calibration_type in calibration_types_to_run:
            run_calibration(
                measurement_snr=run_parameters.measurement_snr,
                calibration_type=calibration_type,
                measurement_additive_noise=run_parameters.measurement_additive_noise,
                measurement_perfect_phase=run_parameters.measurement_perfect_phase,
                measurement_von_mises_mean=run_parameters.measurement_von_mises_mean,
                measurement_von_mises_concentration=run_parameters.measurement_von_mises_concentration,
                bandwidth=run_parameters.bandwidth,
                position_noise_amplitude=run_parameters.position_noise_amplitude,
                n_channel_measurements=run_parameters.n_channel_measurements,
                n_run=n_run
            )
            if not skip_channel_power:
                compute_channel_power_at_calibration_locations(
                    run_type=RunType.CALIBRATION,
                    measurement_snr=run_parameters.measurement_snr,
                    measurement_additive_noise=run_parameters.measurement_additive_noise,
                    measurement_perfect_phase=run_parameters.measurement_perfect_phase,
                    measurement_von_mises_mean=run_parameters.measurement_von_mises_mean,
                    measurement_von_mises_concentration=run_parameters.measurement_von_mises_concentration,
                    bandwidth=run_parameters.bandwidth,
                    position_noise_amplitude=run_parameters.position_noise_amplitude,
                    n_channel_measurements=run_parameters.n_channel_measurements,
                    n_run=n_run,
                    calibration_type=calibration_type
                )

        # End run
        end_timer = timeit.default_timer()
        duration_run = end_timer - start_timer
        print(f"Run executed in {duration_run // 60} min and {duration_run % 60:.1f} s")


if __name__ == '__main__':
    protocol_names = get_all_protocol_names()

    parser = argparse.ArgumentParser(
        "Run series of measurement and calibration experiments for the different SNRs and phase-error concentrations "
        "specified in the protocol"
    )
    parser.add_argument("--n-run", dest="n_run", type=int, help="Select run number")
    parser.add_argument("--protocol-name", dest="protocol_name", type=str,
                        help=f"Select protocol to run among: {'|'.join(protocol_names)}")
    parser.add_argument("--skip-measurement", dest="skip_measurement", default=False, action="store_true",
                        help="If set, skip measurement runs")
    parser.add_argument("--skip-channel-power", dest="skip_channel_power", default=False, action="store_true",
                        help="If set, skip computation of channel powers at calibration locations runs")
    parser.add_argument("--force-calibration-type", dest="force_calibration_type",  type=str, default=None,
                        help=f"Select calibration type to run among: {'|'.join(CalibrationType.all_types())}")
    args = parser.parse_args()

    _force_calibration_type = None if args.force_calibration_type in (None, "") else args.force_calibration_type

    run_experiments_protocol(
        n_run=args.n_run,
        protocol_name=args.protocol_name,
        skip_measurement=args.skip_measurement,
        skip_channel_power=args.skip_channel_power,
        force_calibration_type=_force_calibration_type
    )
