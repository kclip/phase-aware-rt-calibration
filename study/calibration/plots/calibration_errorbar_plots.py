from typing import List
import pandas as pd

from src.utils.save_utils import SafeOpen
from src.utils.plot_utils import estimation_error
from src.plot.plot_errorbars import plot_errorbars
from study.calibration.experiment_config import PLOT_QUANTILE_VALUE, PLOT_QUANTILE_METHOD
from study.calibration.experiment_protocol import get_protocol_by_name
from study.calibration.utils import PROTOCOL_CALIBRATION_DATA_FILENAME, CalibrationType, calibration_plots_folder
from study.calibration.plots.plot_legend import plot_legend


_PROTOCOL_TO_ANNOTATION_KWARGS = {
    "snr_paper": dict(
        annotation_text_x_pos=0.55,
        annotation_arrow_x_start_pos=0.6,
        annotation_arrow_x_delta=0.15,
    ),
    "prior_concentration_paper": dict(
        annotation_text_x_pos=0.6,
        annotation_arrow_x_start_pos=0.7,
        annotation_arrow_x_delta=0.1,
    ),
    "bandwidth_paper": dict(
        annotation_text_x_pos=0.55,
        annotation_arrow_x_start_pos=0.7,
        annotation_arrow_x_delta=0.1,
    ),
    "pos_noise_paper": dict(
        annotation_text_x_pos=0.45,
        annotation_arrow_x_start_pos=0.35,
        annotation_arrow_x_delta=-0.15,
    ),
}


def _calibration_errorbar_plot(
    protocol_name: str,
    n_loaded_runs: int,
    x_column: str,
    x_label: str,
    y_columns: List[str],
    y_labels: List[str],
    ground_truth_y_columns: List[str] = None,
    plot_columns: List[str] = None,  # One plot per value in the z_column
    plot_estimation_error: bool = False,
    x_log_scale: bool = False,
    y_log_scale: bool = False,
    y_columns_in_db: List[str] = None,
    plot_name_prefix: str = None
):
    # Check input
    if plot_estimation_error and (len(y_columns) > 0) and (ground_truth_y_columns is None):
        raise ValueError("Ground-truth column must be defined for error plots")

    # Load data
    with SafeOpen(
        calibration_plots_folder(protocol_name=protocol_name, n_loaded_runs=n_loaded_runs),
        PROTOCOL_CALIBRATION_DATA_FILENAME,
        "r"
    ) as file:
        df_data = pd.read_csv(file)

    # Compute error
    if plot_estimation_error:
        for y_col, ground_truth_y_col in zip(y_columns, ground_truth_y_columns):
            df_data[y_col] = estimation_error(
                estimated_value=df_data[y_col],
                ground_truth_value=df_data[ground_truth_y_col]
            )

    # Compute dB scale indicator array
    y_db_scale = [False for _ in range(len(y_columns))]
    if y_columns_in_db is not None:
        for y_col in y_columns_in_db:
            if y_col in y_columns:
                idx = y_columns.index(y_col)
                y_db_scale[idx] = True

    # Get calibration types (i.e., line_values) and their plot kwargs (i.e., line_plot_kwargs)
    protocol = get_protocol_by_name(protocol_name=protocol_name)
    line_values = protocol.run_calibration_types
    line_plot_kwargs = [
        CalibrationType.get_plot_kwargs(calibration_type)
        for calibration_type in line_values
    ]

    # Get protocol specific options
    protocol_kwargs = _PROTOCOL_TO_ANNOTATION_KWARGS.get(protocol_name, dict())

    for annotate in [True, False]:
        # Save plots folder
        save_plots_dir = calibration_plots_folder(
            protocol_name=protocol_name,
            n_loaded_runs=n_loaded_runs,
            annotate=annotate
        )
        _ = plot_errorbars(
            df=df_data,
            x_column=x_column,
            x_label=x_label,
            y_columns=y_columns,
            y_labels=y_labels,
            x_log_scale=x_log_scale,
            y_log_scale=y_log_scale,
            y_db_scale=y_db_scale,
            ground_truth_y_columns=None if plot_estimation_error else ground_truth_y_columns,
            quantile_value=PLOT_QUANTILE_VALUE,
            quantile_method=PLOT_QUANTILE_METHOD,
            line_column="calibration_type",
            line_values=line_values,
            line_plot_kwargs=line_plot_kwargs,
            plot_columns=plot_columns,
            plot_name_prefix=plot_name_prefix,
            save_plots=True,
            save_plots_dir=save_plots_dir,
            annotate=annotate,
            **protocol_kwargs
        )

    # Add separate plot with only legends
    save_legend_dir = calibration_plots_folder(
        protocol_name=protocol_name,
        n_loaded_runs=n_loaded_runs,
        annotate=False
    )
    plot_legend(
        protocol_name=protocol_name,
        save_dir=save_legend_dir,
        estimation_error_legend=plot_estimation_error
    )


def run_calibration_vs_std_plots(protocol_name: str, n_loaded_runs: int):
    y_columns = ["conductivity", "permittivity", "mean_calibrated_power"]
    ground_truth_y_columns = ["ground_truth_conductivity", "ground_truth_permittivity", "mean_ground_truth_power"]
    y_labels = ["Conductivity", "Permittivity", "Channel Power [dB]"]
    y_columns_in_db = ["mean_calibrated_power"]
    kwargs = dict(
        protocol_name=protocol_name,
        n_loaded_runs=n_loaded_runs,
        x_column="measurement_von_mises_std",
        x_label="Phase Error Standard Deviation [deg]",
        plot_columns=["measurement_snr", "bandwidth"],
        y_log_scale=False,
        y_columns_in_db=y_columns_in_db,
        plot_estimation_error=False,
        plot_name_prefix="calibration_vs_std"
    )
    # Single plot
    _calibration_errorbar_plot(
        y_columns=y_columns,
        y_labels=y_labels,
        ground_truth_y_columns=ground_truth_y_columns,
        **kwargs
    )
    # Individual plots
    for y_column, ground_truth_y_column, y_label in zip(y_columns, ground_truth_y_columns, y_labels):
        _calibration_errorbar_plot(
            y_columns=[y_column],
            y_labels=[y_label],
            ground_truth_y_columns=[ground_truth_y_column],
            **kwargs
        )


def run_calibration_error_vs_std_plots(protocol_name: str, n_loaded_runs: int):
    y_columns = ["conductivity", "permittivity", "mean_calibrated_power"]
    ground_truth_y_columns = ["ground_truth_conductivity", "ground_truth_permittivity", "mean_ground_truth_power"]
    y_labels = ["Conductivity Error", "Permittivity Error", "Power Error [dB]"]
    y_columns_in_db = ["mean_calibrated_power"]
    kwargs = dict(
        protocol_name=protocol_name,
        n_loaded_runs=n_loaded_runs,
        x_column="measurement_von_mises_std",
        x_label="Phase Error Standard Deviation [deg]",
        plot_columns=["measurement_snr", "bandwidth"],
        y_log_scale=True,
        y_columns_in_db=y_columns_in_db,
        plot_estimation_error=True,
        plot_name_prefix="calibration_error_vs_std"
    )
    # Single plot
    _calibration_errorbar_plot(
        y_columns=y_columns,
        y_labels=y_labels,
        ground_truth_y_columns=ground_truth_y_columns,
        **kwargs
    )
    # Individual plots
    for y_column, ground_truth_y_column, y_label in zip(y_columns, ground_truth_y_columns, y_labels):
        _calibration_errorbar_plot(
            y_columns=[y_column],
            y_labels=[y_label],
            ground_truth_y_columns=[ground_truth_y_column],
            **kwargs
        )


def run_calibration_vs_snr_plots(protocol_name: str, n_loaded_runs: int):
    y_columns = ["conductivity", "permittivity", "mean_calibrated_power"]
    ground_truth_y_columns = ["ground_truth_conductivity", "ground_truth_permittivity", "mean_ground_truth_power"]
    y_labels = ["Conductivity", "Permittivity", "Channel Power [dB]"]
    y_columns_in_db = ["mean_calibrated_power"]
    kwargs = dict(
        protocol_name=protocol_name,
        n_loaded_runs=n_loaded_runs,
        x_column="measurement_snr_db",
        x_label="Signal-to-Noise Ratio [dB]",
        plot_columns=["measurement_von_mises_std", "bandwidth"],
        y_log_scale=False,
        y_columns_in_db=y_columns_in_db,
        plot_estimation_error=False,
        plot_name_prefix="calibration_vs_snr"
    )
    # Single plot
    _calibration_errorbar_plot(
        y_columns=y_columns,
        y_labels=y_labels,
        ground_truth_y_columns=ground_truth_y_columns,
        **kwargs
    )
    # Individual plots
    for y_column, ground_truth_y_column, y_label in zip(y_columns, ground_truth_y_columns, y_labels):
        _calibration_errorbar_plot(
            y_columns=[y_column],
            y_labels=[y_label],
            ground_truth_y_columns=[ground_truth_y_column],
            **kwargs
        )


def run_calibration_error_vs_snr_plots(protocol_name: str, n_loaded_runs: int):
    y_columns = ["conductivity", "permittivity", "mean_calibrated_power"]
    ground_truth_y_columns = ["ground_truth_conductivity", "ground_truth_permittivity", "mean_ground_truth_power"]
    y_labels = ["Conductivity Error", "Permittivity Error", "Power Error [dB]"]
    y_columns_in_db = ["mean_calibrated_power"]
    kwargs = dict(
        protocol_name=protocol_name,
        n_loaded_runs=n_loaded_runs,
        x_column="measurement_snr_db",
        x_label="Signal-to-Noise Ratio [dB]",
        plot_columns=["measurement_von_mises_std", "bandwidth"],
        y_log_scale=True,
        y_columns_in_db=y_columns_in_db,
        plot_estimation_error=True,
        plot_name_prefix="calibration_error_vs_snr"
    )
    # Single plot
    _calibration_errorbar_plot(
        y_columns=y_columns,
        y_labels=y_labels,
        ground_truth_y_columns=ground_truth_y_columns,
        **kwargs
    )
    # Individual plots
    for y_column, ground_truth_y_column, y_label in zip(y_columns, ground_truth_y_columns, y_labels):
        _calibration_errorbar_plot(
            y_columns=[y_column],
            y_labels=[y_label],
            ground_truth_y_columns=[ground_truth_y_column],
            **kwargs
        )


def run_calibration_vs_bandwidth_plots(protocol_name: str, n_loaded_runs: int):
    y_columns = ["conductivity", "permittivity", "mean_calibrated_power"]
    ground_truth_y_columns = ["ground_truth_conductivity", "ground_truth_permittivity", "mean_ground_truth_power"]
    y_labels = ["Conductivity", "Permittivity", "Channel Power [dB]"]
    y_columns_in_db = ["mean_calibrated_power"]
    kwargs = dict(
        protocol_name=protocol_name,
        n_loaded_runs=n_loaded_runs,
        x_column="bandwidth_mhz",
        x_label="Bandwidth [MHz]",
        plot_columns=["measurement_snr_db", "measurement_von_mises_std"],
        x_log_scale=True,
        y_log_scale=False,
        y_columns_in_db=y_columns_in_db,
        plot_estimation_error=False,
        plot_name_prefix="calibration_vs_bandwidth"
    )
    # Single plot
    _calibration_errorbar_plot(
        y_columns=y_columns,
        y_labels=y_labels,
        ground_truth_y_columns=ground_truth_y_columns,
        **kwargs
    )
    # Individual plots
    # With error computation
    for y_column, ground_truth_y_column, y_label in zip(y_columns, ground_truth_y_columns, y_labels):
        _calibration_errorbar_plot(
            y_columns=[y_column],
            y_labels=[y_label],
            ground_truth_y_columns=[ground_truth_y_column],
            **kwargs
        )


def run_calibration_error_vs_bandwidth_plots(protocol_name: str, n_loaded_runs: int):
    y_columns = ["conductivity", "permittivity", "mean_calibrated_power"]
    ground_truth_y_columns = ["ground_truth_conductivity", "ground_truth_permittivity", "mean_ground_truth_power"]
    y_labels = ["Conductivity Error", "Permittivity Error", "Power Error [dB]"]
    y_columns_in_db = ["mean_calibrated_power"]
    kwargs = dict(
        protocol_name=protocol_name,
        n_loaded_runs=n_loaded_runs,
        x_column="bandwidth_mhz",
        x_label="Bandwidth [MHz]",
        plot_columns=["measurement_snr_db", "measurement_von_mises_std"],
        x_log_scale=True,
        y_log_scale=True,
        y_columns_in_db=y_columns_in_db,
        plot_estimation_error=True,
        plot_name_prefix="calibration_error_vs_bandwidth"
    )
    # Single plot
    _calibration_errorbar_plot(
        y_columns=y_columns,
        y_labels=y_labels,
        ground_truth_y_columns=ground_truth_y_columns,
        **kwargs
    )
    # Individual plots
    # With error computation
    for y_column, ground_truth_y_column, y_label in zip(y_columns, ground_truth_y_columns, y_labels):
        _calibration_errorbar_plot(
            y_columns=[y_column],
            y_labels=[y_label],
            ground_truth_y_columns=[ground_truth_y_column],
            **kwargs
        )


def run_calibration_vs_position_noise_plots(protocol_name: str, n_loaded_runs: int):
    y_columns = ["conductivity", "permittivity", "mean_calibrated_power"]
    ground_truth_y_columns = ["ground_truth_conductivity", "ground_truth_permittivity", "mean_ground_truth_power"]
    y_labels = ["Conductivity", "Permittivity", "Channel Power [dB]"]
    y_columns_in_db = ["mean_calibrated_power"]
    kwargs = dict(
        protocol_name=protocol_name,
        n_loaded_runs=n_loaded_runs,
        x_column="position_noise_amplitude",
        x_label=r"Displacement amplitude [$\times \lambda^c$]",
        plot_columns=["bandwidth", "measurement_snr_db", "measurement_von_mises_std"],
        x_log_scale=False,
        y_log_scale=False,
        y_columns_in_db=y_columns_in_db,
        plot_estimation_error=False,
        plot_name_prefix="calibration_vs_pos_noise"
    )
    # Single plot
    _calibration_errorbar_plot(
        y_columns=y_columns,
        y_labels=y_labels,
        ground_truth_y_columns=ground_truth_y_columns,
        **kwargs
    )
    # Individual plots
    # With error computation
    for y_column, ground_truth_y_column, y_label in zip(y_columns, ground_truth_y_columns, y_labels):
        _calibration_errorbar_plot(
            y_columns=[y_column],
            y_labels=[y_label],
            ground_truth_y_columns=[ground_truth_y_column],
            **kwargs
        )


def run_calibration_error_vs_position_noise_plots(protocol_name: str, n_loaded_runs: int):
    y_columns = ["conductivity", "permittivity", "mean_calibrated_power"]
    ground_truth_y_columns = ["ground_truth_conductivity", "ground_truth_permittivity", "mean_ground_truth_power"]
    y_labels = ["Conductivity Error", "Permittivity Error", "Power Error [dB]"]
    y_columns_in_db = ["mean_calibrated_power"]
    kwargs = dict(
        protocol_name=protocol_name,
        n_loaded_runs=n_loaded_runs,
        x_column="position_noise_amplitude",
        x_label=r"Displacement amplitude [$\times \lambda^c$]",
        plot_columns=["bandwidth", "measurement_snr_db", "measurement_von_mises_std"],
        x_log_scale=False,
        y_log_scale=True,
        y_columns_in_db=y_columns_in_db,
        plot_estimation_error=True,
        plot_name_prefix="calibration_error_vs_pos_noise"
    )
    # Single plot
    _calibration_errorbar_plot(
        y_columns=y_columns,
        y_labels=y_labels,
        ground_truth_y_columns=ground_truth_y_columns,
        **kwargs
    )
    # Individual plots
    # With error computation
    for y_column, ground_truth_y_column, y_label in zip(y_columns, ground_truth_y_columns, y_labels):
        _calibration_errorbar_plot(
            y_columns=[y_column],
            y_labels=[y_label],
            ground_truth_y_columns=[ground_truth_y_column],
            **kwargs
        )


def run_calibration_error_vs_n_measurements_plots(protocol_name: str, n_loaded_runs: int):
    y_columns = ["conductivity", "permittivity", "mean_calibrated_power"]
    ground_truth_y_columns = ["ground_truth_conductivity", "ground_truth_permittivity", "mean_ground_truth_power"]
    y_labels = ["Conductivity Error", "Permittivity Error", "Power Error [dB]"]
    y_columns_in_db = ["mean_calibrated_power"]
    kwargs = dict(
        protocol_name=protocol_name,
        n_loaded_runs=n_loaded_runs,
        x_column="n_channel_measurements",
        x_label="Number of channel measurements",
        plot_columns=["bandwidth", "measurement_snr_db", "measurement_von_mises_std"],
        x_log_scale=False,
        y_log_scale=True,
        y_columns_in_db=y_columns_in_db,
        plot_estimation_error=True,
        plot_name_prefix="calibration_error_vs_n_measurements"
    )
    # Single plot
    _calibration_errorbar_plot(
        y_columns=y_columns,
        y_labels=y_labels,
        ground_truth_y_columns=ground_truth_y_columns,
        **kwargs
    )
    # Individual plots
    # With error computation
    for y_column, ground_truth_y_column, y_label in zip(y_columns, ground_truth_y_columns, y_labels):
        _calibration_errorbar_plot(
            y_columns=[y_column],
            y_labels=[y_label],
            ground_truth_y_columns=[ground_truth_y_column],
            **kwargs
        )
