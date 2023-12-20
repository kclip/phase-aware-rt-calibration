import os
from itertools import product
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from src.utils.save_utils import SafeOpen
from src.utils.plot_utils import set_rc_params
from study.calibration.experiment_protocol import RunParameters, load_experiment_protocol
from study.calibration.utils import PROTOCOL_POWER_PROFILES_DATA_FILENAME, power_profiles_plots_folder, CalibrationType
from study.calibration.plots.plot_legend import plot_legend

# Manual plot params (set to None to deactivate)
_MANUAL_X_LIM_TIME = [14, 35]  # in [ns]
_MANUAL_Y_LIM_TIME = [-0.02, 0.64]
_MANUAL_X_LIM_FREQ = [6 - 0.25, 6 + 0.25]  # in GHz
_NB_MARKERS_FREQ = 30
_NB_MARKERS_TIME = 15


_RUN_INFO_COLS = [
    "measurement_snr",
    "measurement_additive_noise",
    "bandwidth",
    "n_channel_measurements",
]


def _get_x_lim_time(
    time_axis: np.ndarray,
    power_profile_measurement: np.ndarray,
    axis_multiplier: float = 1.0,
    percentage_energy: float = 0.995
):
    if _MANUAL_X_LIM_TIME is not None:
        return _MANUAL_X_LIM_TIME
    mask = (power_profile_measurement.cumsum() / power_profile_measurement.sum()) < percentage_energy
    return [0, axis_multiplier * np.amax(time_axis[mask])]


def _get_x_lim_freq(
    freq_axis: np.ndarray,
    bandwidth: float,
    axis_multiplier: float = 1.0
):
    if _MANUAL_X_LIM_FREQ is not None:
        return _MANUAL_X_LIM_FREQ
    return [
        axis_multiplier * (freq_axis.mean() - (bandwidth / 2)),
        axis_multiplier * (freq_axis.mean() + (bandwidth / 2))
    ]


def _plot_power_profile(
    df: pd.DataFrame,
    run_parameters: RunParameters,
    rx_tx_pair: int,
    domain: str,
    calibration_assumption: bool,
    calibration_types: List[str],
    save_plots_dir: str
):
    # Select data corresponding to run parameters
    mask_run = pd.Series(True, index=df.index)
    for run_attr in _RUN_INFO_COLS:
        mask_run = mask_run & (
            df[run_attr] == getattr(run_parameters, run_attr)
            if getattr(run_parameters, run_attr) is not None else
            df[run_attr].isna()
        )
    df_run = df[mask_run]

    # Get masks
    mask_measurement = df_run["value_type"] == "MEASUREMENT"
    mask_calibration = df_run["value_type"] == "CALIBRATION"
    mask_rx_tx_pair = df_run["rx_tx_pair"] == rx_tx_pair
    if domain == "PATHS":
        mask_domain_measurement = df_run["domain"] == "TIME"
        mask_domain_calibration = df_run["domain"] == "PATHS"
        mask_assumption = df_run["assumption"].isna()
    else:
        mask_domain_measurement = mask_domain_calibration = df_run["domain"] == domain
        mask_assumption = df_run["assumption"] == calibration_assumption

    # Ground-truth
    mask_measurment_data = mask_measurement & mask_domain_measurement & mask_rx_tx_pair
    axis_measurement = df_run[mask_measurment_data]["axis"]
    power_profile_measurement = df_run[mask_measurment_data]["value"]

    # Get Axes
    y_lim = None
    if domain == "FREQ":
        x_axis_multiplier = 1e-9
        x_label = "Frequency [GHz]"
        x_lim = _get_x_lim_freq(
            freq_axis=axis_measurement.values,
            bandwidth=run_parameters.bandwidth,
            axis_multiplier=x_axis_multiplier
        )
        figsize = (10, 5)
        nbmarkers = _NB_MARKERS_FREQ
    elif domain in ("TIME", "PATHS"):
        x_axis_multiplier = 1e9
        x_label = "Time of Arrival [ns]"
        x_lim = _get_x_lim_time(
            time_axis=axis_measurement.values,
            power_profile_measurement=power_profile_measurement.values,
            axis_multiplier=x_axis_multiplier
        )
        if (domain == "TIME") and (_MANUAL_Y_LIM_TIME is not None):
            y_lim = _MANUAL_Y_LIM_TIME
        figsize = (5, 5)
        nbmarkers = _NB_MARKERS_TIME
    else:
        raise ValueError
    y_label = "Normalized Power"

    # Calibrated profiles
    power_profiles_calibration = []  # (calibration_type, values)
    for cal_type in calibration_types:
        cal_type_mask = df_run["calibration_type"] == cal_type
        mask_calibration_data = (
            mask_calibration &
            mask_domain_calibration &
            mask_rx_tx_pair &
            cal_type_mask &
            mask_assumption
        )
        axis = df_run[mask_calibration_data]["axis"]
        values = df_run[mask_calibration_data]["value"]
        power_profiles_calibration.append((cal_type, axis, values))

    # Plot
    set_rc_params()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    axis_step = x_axis_multiplier * (axis_measurement.values[1] - axis_measurement.values[0])
    n_points = abs(x_lim[0] - x_lim[1]) / axis_step
    plot_kwargs = dict(
        markevery=round(n_points / nbmarkers)
    )

    ax.plot(
        x_axis_multiplier * axis_measurement,
        power_profile_measurement,
        linestyle="--",
        color="black",
        label="Ground-Truth",
        linewidth=3,
        **plot_kwargs
    )
    for cal_type, axis, values in power_profiles_calibration:
        if domain == "PATHS":  # Discrete
            cal_plot_kwargs = CalibrationType.get_stemplot_kwargs(calibration_type=cal_type)
            ax.stem(
                x_axis_multiplier * axis,
                values,
                **cal_plot_kwargs
            )
        else:  # Continuous
            cal_plot_kwargs = CalibrationType.get_plot_kwargs(calibration_type=cal_type)
            ax.plot(
                x_axis_multiplier * axis,
                values,
                **cal_plot_kwargs,
                **plot_kwargs
            )

    # Save plot
    plot_name = (
        f"{domain.lower()}_power_profile" +
        f"_rx_tx_pair_{rx_tx_pair}" +
        ("_with_assumption" if calibration_assumption else "_no_assumption") +
        f"_snr_{run_parameters.measurement_snr}" +
        f"_bandwidth_{round(run_parameters.bandwidth / 1e6)}_mhz"
    )
    with SafeOpen(save_plots_dir, f"{plot_name}.png", "wb") as file:
        fig.savefig(file, dpi=300, bbox_inches="tight")


def plot_power_profiles_protocol(protocol_name: str, n_loaded_runs: int):
    # Load data
    with SafeOpen(
            power_profiles_plots_folder(protocol_name=protocol_name, n_loaded_runs=n_loaded_runs),
            PROTOCOL_POWER_PROFILES_DATA_FILENAME,
            "rb"
    ) as file:
        df = pd.read_parquet(file, engine="fastparquet")

    n_rx_tx_pair = len([v for v in df["rx_tx_pair"].unique() if v is not None])
    save_plots_dir = os.path.join(
        power_profiles_plots_folder(protocol_name=protocol_name, n_loaded_runs=n_loaded_runs),
        "plots"
    )

    # Plot profiles
    for run_parameters in load_experiment_protocol(protocol_name):
        # for rx_tx_pair, domain, assumption in product(range(n_rx_tx_pair), ("TIME", "FREQ"), (False, True)):
        for rx_tx_pair, domain in product(range(n_rx_tx_pair), ("PATHS", "TIME", "FREQ")):
            if domain == "PATHS":
                _plot_power_profile(
                    df=df,
                    run_parameters=run_parameters,
                    rx_tx_pair=rx_tx_pair,
                    domain=domain,
                    calibration_assumption=None,
                    calibration_types=run_parameters.run_calibration_types,
                    save_plots_dir=save_plots_dir
                )
            else:
                for assumption in (False, True):
                    _plot_power_profile(
                        df=df,
                        run_parameters=run_parameters,
                        rx_tx_pair=rx_tx_pair,
                        domain=domain,
                        calibration_assumption=assumption,
                        calibration_types=run_parameters.run_calibration_types,
                        save_plots_dir=save_plots_dir
                    )
    # Separate plot for legend
    plot_legend(
        protocol_name=protocol_name,
        save_dir=save_plots_dir,
        estimation_error_legend=False
    )
