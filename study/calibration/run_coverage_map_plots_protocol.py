import argparse

from study.calibration.experiment_config import PLOT_CM_VMIN, PLOT_CM_VMAX, PLOT_CM_NUM_SAMPLES, \
    PLOT_CM_SHOW_DEVICES
from study.calibration.experiment_protocol import get_all_protocol_names
from study.calibration.data_processing.gather_power_map_data import run_gather_power_map_data
from study.calibration.plots.coverage_map_errorbar_plots import run_coverage_map_error_vs_std_plots, \
    run_coverage_map_error_vs_snr_plots
from study.calibration.plots.coverage_map_plots import run_coverage_map_plots
from study.calibration.plots.scenario_render_plot import run_scenario_render_plot


_PLOT_TYPES_TO_FUNC_AND_KWARGS = {
    "coverage_map_error_vs_std": (run_coverage_map_error_vs_std_plots, dict()),
    "coverage_map_error_vs_snr": (run_coverage_map_error_vs_snr_plots, dict()),
    "coverage_map": (
        run_coverage_map_plots,
        dict(
            cm_show_devices=PLOT_CM_SHOW_DEVICES,
            cm_show_tx_only=True,
            cm_vmin=PLOT_CM_VMIN,
            cm_vmax=PLOT_CM_VMAX,
            cm_num_samples=PLOT_CM_NUM_SAMPLES,
        )
    ),
    "scenario_render": (
        run_scenario_render_plot,
        dict(
            show_devices=True,
            show_paths=True,
            num_samples=PLOT_CM_NUM_SAMPLES,
            save_plot=True
        )
    )
}

if __name__ == '__main__':
    protocol_names = get_all_protocol_names()

    parser = argparse.ArgumentParser("Extract data and compute plots for the experiment protocol")
    parser.add_argument("--n-runs", dest="n_runs", type=int, help="Number of runs to load for plots")
    parser.add_argument("--protocol-name", dest="protocol_name", type=str,
                        help=f"Select protocol to run among: {'|'.join(protocol_names)}")
    parser.add_argument("--ground-truth-coverage-map-run-name", dest="ground_truth_coverage_map_run_name", type=str,
                        default=None,
                        help="Name of coverage map run with ground-truth material parameters (not needed if "
                             "'--skip-gather-data' is set)")
    parser.add_argument("--skip-gather-data", dest="skip_gather_data", default=False, action="store_true",
                        help="If set, do not compute the data extraction step")
    parser.add_argument("--array-config", dest="array_config", type=str, default=None,
                        help=f"Config of simulated arrays. Choose among: 'mimo' | 'miso' | 'siso'")
    parser.add_argument("--plot-type", dest="plot_type", default=None, type=str,
                        help=f"If set, select which plot type to execute among: "
                             f"{' | '.join(_PLOT_TYPES_TO_FUNC_AND_KWARGS.keys())}")
    args = parser.parse_args()

    # Gather data
    if not args.skip_gather_data:
        run_gather_power_map_data(
            ground_truth_coverage_map_run_name=args.ground_truth_coverage_map_run_name,
            protocol_name=args.protocol_name,
            array_config=args.array_config,
            n_runs_to_load=args.n_runs
        )

    # Plots
    if args.plot_type is None:
        for plot_func, plot_kwargs in _PLOT_TYPES_TO_FUNC_AND_KWARGS.values():
            plot_func(
                protocol_name=args.protocol_name,
                array_config=args.array_config,
                n_loaded_runs=args.n_runs, **plot_kwargs
            )
    else:
        if args.plot_type not in _PLOT_TYPES_TO_FUNC_AND_KWARGS.keys():
            raise ValueError(f"Unknown plot type '{args.plot_type}'")
        plot_func, plot_kwargs = _PLOT_TYPES_TO_FUNC_AND_KWARGS[args.plot_type]
        plot_func(
            protocol_name=args.protocol_name,
            array_config=args.array_config,
            n_loaded_runs=args.n_runs, **plot_kwargs
        )
