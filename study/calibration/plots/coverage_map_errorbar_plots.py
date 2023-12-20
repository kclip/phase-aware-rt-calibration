import os
from typing import List
import pandas as pd

from src.utils.save_utils import SafeOpen
from src.plot.plot_errorbars import plot_errorbars
from study.calibration.experiment_config import PLOT_QUANTILE_VALUE, PLOT_QUANTILE_METHOD
from study.calibration.experiment_protocol import get_protocol_by_name
from study.calibration.utils import protocol_plots_folder, power_map_data_subfolder, \
    power_map_plots_subfolder, PROTOCOL_POWER_MAP_DATA_FILENAME, CalibrationType
from study.calibration.plots.plot_legend import plot_legend


def _coverage_map_errorbar_plot(
    protocol_name: str,
    array_config: str,
    n_loaded_runs: int,
    x_column: str,
    x_label: str,
    plot_columns: List[str] = None,
    plot_name_prefix: str = None
):
    protocol_plots_dir = protocol_plots_folder(protocol_name=protocol_name, n_loaded_runs=n_loaded_runs)

    # Load data
    with SafeOpen(
        os.path.join(protocol_plots_dir, power_map_data_subfolder(array_config=array_config)),
        PROTOCOL_POWER_MAP_DATA_FILENAME,
        "r"
    ) as file:
        df_data = pd.read_csv(file)

    # Get calibration types (i.e., line_values) and their plot kwargs (i.e., line_plot_kwargs)
    protocol = get_protocol_by_name(protocol_name=protocol_name)
    line_values = protocol.run_calibration_types
    line_plot_kwargs = [
        CalibrationType.get_plot_kwargs(calibration_type)
        for calibration_type in line_values
    ]

    # Save plots directory
    save_plots_dir = os.path.join(
        protocol_plots_dir,
        power_map_plots_subfolder(array_config=array_config)
    )

    for annotate in [True, False]:
        plot_name_prefix_annot = f"{plot_name_prefix}_annotated" if annotate else plot_name_prefix
        plot_errorbars(
            df=df_data,
            x_column=x_column,
            x_label=x_label,
            y_columns=["error_mean"],
            y_labels=["Power Map Mean Error [dB]"],
            ground_truth_y_columns=None,
            quantile_value=PLOT_QUANTILE_VALUE,
            quantile_method=PLOT_QUANTILE_METHOD,
            line_column="calibration_type",
            line_values=line_values,
            line_plot_kwargs=line_plot_kwargs,
            plot_columns=plot_columns,
            x_log_scale=False,
            y_log_scale=False,
            y_db_scale=[True],
            plot_name_prefix=plot_name_prefix_annot,
            save_plots=True,
            save_plots_dir=save_plots_dir,
            annotate=annotate
        )
    # Add separate plot with only legends
    plot_legend(
        protocol_name=protocol_name,
        save_dir=save_plots_dir,
        estimation_error_legend=True
    )


def run_coverage_map_error_vs_std_plots(protocol_name: str, array_config: str, n_loaded_runs: int):
    _coverage_map_errorbar_plot(
        protocol_name=protocol_name,
        array_config=array_config,
        n_loaded_runs=n_loaded_runs,
        x_column="measurement_von_mises_std_deg",
        x_label="Phase Error Standard Deviation [deg]",
        plot_columns=["measurement_snr_db", "bandwidth"],
        plot_name_prefix="coverage_map_error_vs_std"
    )


def run_coverage_map_error_vs_snr_plots(protocol_name: str, array_config: str, n_loaded_runs: int):
    _coverage_map_errorbar_plot(
        protocol_name=protocol_name,
        array_config=array_config,
        n_loaded_runs=n_loaded_runs,
        x_column="measurement_snr_db",
        x_label="Signal-to-Noise Ratio [dB]",
        plot_columns=["measurement_von_mises_std_deg", "bandwidth"],
        plot_name_prefix="coverage_map_error_vs_snr"
    )
