import argparse

from study.calibration.experiment_protocol import get_all_protocol_names
from study.calibration.plots.calibration_errorbar_plots import run_calibration_vs_std_plots, \
    run_calibration_error_vs_std_plots, run_calibration_vs_snr_plots, run_calibration_error_vs_snr_plots, \
    run_calibration_vs_bandwidth_plots, run_calibration_error_vs_bandwidth_plots, \
    run_calibration_vs_position_noise_plots, run_calibration_error_vs_position_noise_plots, \
    run_calibration_error_vs_n_measurements_plots
from study.calibration.plots.plot_power_profiles import plot_power_profiles_protocol
from study.calibration.data_processing.gather_calibration_data import run_gather_calibration_data
from study.calibration.data_processing.gather_power_profiles import run_gather_power_profiles_maxwell_simulation


_DEFAULT_PLOT_TYPES_TO_FUNC = {
    "calibration_vs_std": run_calibration_vs_std_plots,
    "calibration_error_vs_std": run_calibration_error_vs_std_plots,
    "calibration_vs_snr": run_calibration_vs_snr_plots,
    "calibration_error_vs_snr": run_calibration_error_vs_snr_plots,
    "calibration_vs_bandwidth": run_calibration_vs_bandwidth_plots,
    "calibration_error_vs_bandwidth": run_calibration_error_vs_bandwidth_plots,
    "calibration_vs_position_noise": run_calibration_vs_position_noise_plots,
    "calibration_error_vs_position_noise": run_calibration_error_vs_position_noise_plots,
    "calibration_error_vs_n_measurements": run_calibration_error_vs_n_measurements_plots,
}
_PLOT_TYPES_TO_FUNC = {
    **_DEFAULT_PLOT_TYPES_TO_FUNC,
    "power_profiles": plot_power_profiles_protocol,
}

if __name__ == '__main__':
    protocol_names = get_all_protocol_names()

    parser = argparse.ArgumentParser("Extract data and compute plots for the experiment protocol")
    parser.add_argument("--n-runs", dest="n_runs", type=int, help="Number of runs to load for plots")
    parser.add_argument("--protocol-name", dest="protocol_name", type=str,
                        help=f"Select protocol to run among: {'|'.join(protocol_names)}")
    parser.add_argument("--plot-type", dest="plot_type", default=None, type=str,
                        help=f"If set, select which plot type to execute among: "
                             f"{' | '.join(_DEFAULT_PLOT_TYPES_TO_FUNC.keys())}")
    parser.add_argument("--skip-gather-data-calibration", dest="skip_gather_calibration_data",
                        default=False, action="store_true",
                        help="If set, do not compute the calibration data extraction step")
    parser.add_argument("--gather-data-maxwell", dest="gather_data_maxwell", default=False,
                        action="store_true",
                        help="If set, extract power profiles data (works only for Maxwell simulation measurements)")
    args = parser.parse_args()

    # Gather data
    if not args.skip_gather_calibration_data:
        run_gather_calibration_data(
            protocol_name=args.protocol_name,
            n_runs_to_load=args.n_runs
        )
    if args.gather_data_maxwell:
        run_gather_power_profiles_maxwell_simulation(
            protocol_name=args.protocol_name,
            n_runs_to_load=args.n_runs
        )

    # Plots
    if args.plot_type is None:
        for plot_func in _DEFAULT_PLOT_TYPES_TO_FUNC.values():
            plot_func(protocol_name=args.protocol_name, n_loaded_runs=args.n_runs)
    else:
        if args.plot_type not in _PLOT_TYPES_TO_FUNC.keys():
            raise ValueError(f"Unknown plot type '{args.plot_type}'")
        plot_func = _PLOT_TYPES_TO_FUNC[args.plot_type]
        plot_func(protocol_name=args.protocol_name, n_loaded_runs=args.n_runs)
