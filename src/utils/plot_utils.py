from typing import List, Union, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils.telecom_utils import to_db


# Plot params
# -----------

_FONT_SIZE_LARGE = 26
_FONT_SIZE_MEDIUM = 22


def set_rc_params(scale: float = 1):
    plt.rc('text', usetex=True)
    plt.rc('font', size=scale*_FONT_SIZE_LARGE, family="Times New Roman")
    plt.rc('axes', titlesize=scale*_FONT_SIZE_LARGE)
    plt.rc('axes', labelsize=scale*_FONT_SIZE_LARGE)
    plt.rc('xtick', labelsize=scale*_FONT_SIZE_MEDIUM)
    plt.rc('ytick', labelsize=scale*_FONT_SIZE_MEDIUM)
    plt.rc('legend', fontsize=scale*_FONT_SIZE_MEDIUM)
    plt.rc('errorbar', capsize=scale*8)
    plt.rc('lines', linewidth=scale*2, markeredgewidth=scale*2)


# Plot functions
# --------------

def add_annotation(
    ax: plt.Axes,
    label: str,
    pos_arrow_data: Tuple[float, float],
    pos_text_axes: Tuple[float, float],
    arrow_kwargs: dict = None,
    text_kwargs: dict = None
):
    if arrow_kwargs is None:
        arrow_kwargs = dict()
    if text_kwargs is None:
        text_kwargs = dict()

    # Position of the arrow with respect to the text
    arrow_on_right_side = (pos_arrow_data[0] > pos_text_axes[0])
    _epsilon_y_axis = 0.05
    _pos_arrow_axes_coords = ax.transAxes.inverted().transform(  # display coords -> axes coords
        ax.transData.transform(pos_arrow_data)  # data coords -> display coords                                         
    )
    arrow_above_text = (_pos_arrow_axes_coords[1] > (pos_text_axes[1] + _epsilon_y_axis))
    arrow_below_text = (_pos_arrow_axes_coords[1] < (pos_text_axes[1] - _epsilon_y_axis))

    # Parameters dependent on relative arrow position
    angleB = None
    if arrow_on_right_side:
        angleA = 180
        ha_text = "right"
        if arrow_above_text:
            angleB = -100
        if arrow_below_text:
            angleB = 100
    else:
        angleA = 0
        ha_text = "left"
        if arrow_above_text:
            angleB = -80
        if arrow_below_text:
            angleB = 80
    if angleB is not None:
        connectionstyle = f"angle,angleA={angleA},angleB={angleB},rad=0.0"
    else:
        connectionstyle = None
    
    default_arrow_kwargs = dict(
        arrowstyle="->",
        color="black",
        connectionstyle=connectionstyle,
    )
    default_text_kwargs = dict(
        ha=ha_text,
        va="center",
        size=_FONT_SIZE_MEDIUM
    )
    ax.annotate(
        text=label,
        xy=pos_arrow_data, xycoords=ax.transData,
        xytext=pos_text_axes, textcoords=ax.transAxes,  # "axes fraction"
        arrowprops={
            **default_arrow_kwargs,
            **arrow_kwargs,
        },
        **{
            **default_text_kwargs,
            **text_kwargs,
        }
    )


def plot_ax_errorbar(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    y_err_low: np.ndarray,
    y_err_high: np.ndarray,
    plot_kwargs: dict = None
):
    if plot_kwargs is None:
        plot_kwargs = dict()
    y_err_low_diff = y - y_err_low
    y_err_high_diff = y_err_high - y
    ax.errorbar(
        x=x,
        y=y,
        yerr=np.stack([y_err_low_diff, y_err_high_diff], axis=0),
        **plot_kwargs
    )


def plot_fill_between(
    ax: plt.Axes,
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    y_err_low: Union[list, np.ndarray],
    y_err_high: Union[list, np.ndarray],
    plot_kwargs: dict = None
):
    if plot_kwargs is None:
        plot_kwargs = dict()

    # Plot
    ax.plot(x, y, **plot_kwargs)

    # Remove unecessary plot kwargs for shaded area
    kwargs_to_remove = ["label", "marker"]
    cleaned_plot_kwargs = {
        k: v
        for k, v in plot_kwargs.items()
        if k not in kwargs_to_remove
    }

    # Shaded area
    fill_plot_kwargs = {
        **cleaned_plot_kwargs,
        "alpha": 0.3
    }
    ax.fill_between(x, y_err_low, y_err_high, **fill_plot_kwargs)
    err_limits_kwargs = {
        **cleaned_plot_kwargs,
        "alpha": 0.5,
        "linewidth": 1
    }
    for y_err in [y_err_low, y_err_high]:
        ax.plot(x, y_err, **err_limits_kwargs)


# Metrics
# -------

def get_mean_and_quantiles(
    df: pd.DataFrame,
    x_columns: List[str],
    y_columns: List[str],
    quantile: float,
    quantile_method: str = "linear",
    y_db_scale: Union[bool, List[bool]] = False
) -> pd.DataFrame:
    if not isinstance(y_db_scale, list):
        y_db_scale = [y_db_scale] * len(y_columns)

    x_columns = [x_columns] if isinstance(x_columns, str) else x_columns
    y_columns = [y_columns] if isinstance(y_columns, str) else y_columns
    lower_quantile = quantile if quantile < 0.5 else (1 - quantile)
    higher_quantile = quantile if quantile >= 0.5 else (1 - quantile)

    df_groubpy = df[[*x_columns, *y_columns]].groupby(x_columns, as_index=False)
    df_mean = df_groubpy.mean().rename(columns={col: f"{col}_mean" for col in y_columns})
    df_median = df_groubpy.quantile(
        0.5, interpolation=quantile_method
    ).rename(columns={col: f"{col}_median" for col in y_columns})
    df_lower_quantile = df_groubpy.quantile(
        lower_quantile, interpolation=quantile_method
    ).rename(columns={col: f"{col}_low_q" for col in y_columns})
    df_higher_quantile = df_groubpy.quantile(
        higher_quantile, interpolation=quantile_method
    ).rename(columns={col: f"{col}_high_q" for col in y_columns})

    df_out = df_mean.merge(
        df_median, on=x_columns, how="left"
    ).merge(
        df_lower_quantile, on=x_columns, how="left"
    ).merge(
        df_higher_quantile, on=x_columns, how="left"
    )

    # To dB
    for y_col, y_col_in_db in zip(y_columns, y_db_scale):
        if y_col_in_db:
            for col in [f"{y_col}_mean", f"{y_col}_median", f"{y_col}_low_q", f"{y_col}_high_q"]:
                df_out[col] = to_db(df_out[col])

    return df_out


def estimation_error(
    estimated_value: Union[tf.Tensor, pd.Series, np.ndarray, float],
    ground_truth_value: Union[tf.Tensor, pd.Series, np.ndarray, float]
) -> Union[tf.Tensor, pd.Series, np.ndarray, np.float32]:
    if isinstance(estimated_value, tf.Tensor):
        return tf.divide(
            tf.abs(estimated_value - ground_truth_value),
            tf.abs(ground_truth_value)
        )
    elif isinstance(estimated_value, pd.Series):
        return (estimated_value - ground_truth_value).abs() / ground_truth_value.abs()
    elif isinstance(estimated_value, (np.ndarray, float)):
        return np.abs(estimated_value - ground_truth_value) / np.abs(ground_truth_value)
    else:
        raise ValueError(f"Type '{type(estimated_value)}' not handled by estimation error func")
