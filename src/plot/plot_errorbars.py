from typing import List, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from src.utils.plot_utils import set_rc_params, get_mean_and_quantiles, add_annotation, plot_fill_between
from src.utils.telecom_utils import to_db
from src.utils.save_utils import SafeOpen


def _annotate_aux(
    ax: plt.Axes,
    y_min: float,
    x_pos_text: float,
    labels_pos_arrow: List[Tuple[str, Tuple[float, float]]],
    log_scale: bool = False
):
    n_annotations = len(labels_pos_arrow)
    y_delta = -0.1  # y-axis diff from one annotation to another (as a multiple of the initial y-axis range)

    # Extend plot ylims to make room for annotation
    y_lim = ax.get_ylim()
    if log_scale:
        y_range = np.log10(y_lim[1] / y_lim[0])
        # y_bottom = y_min / (y_range * (n_annotations * 30) * np.abs(y_delta))
        y_bottom = y_min / (10 ** (y_range * (n_annotations + 3) * np.abs(y_delta)))
    else:
        y_range = y_lim[1] - y_lim[0]
        y_bottom = y_min - (y_range * (n_annotations + 3) * np.abs(y_delta))
    ax.set_ylim(
        bottom=min(y_bottom, y_lim[0]),
        top=y_lim[1]
    )

    y_pos = 0.0 - (n_annotations * y_delta)

    for label, pos_arrow in labels_pos_arrow:
        add_annotation(
            ax=ax,
            label=label,
            pos_arrow_data=pos_arrow,
            pos_text_axes=(x_pos_text, y_pos),
            arrow_kwargs=None,
            text_kwargs=None
        )
        y_pos += y_delta


def plot_errorbars(
    df: pd.DataFrame,
    # X-axis
    x_column: str,
    x_label: str,
    # Y-axis
    y_columns: List[str],  # One subplot per element in <y_columns>
    y_labels: List[str],
    ground_truth_y_columns: List[str] = None,  # Ground-truth y-value (dashed line)
    # Quantiles
    quantile_value: float = None,
    quantile_method: str = None,
    # Lines: One line per value in the <line_column> column
    # The data of each line is given as: df[<line_column>] == line_val for line_val in <line_values>
    line_column: str = None,  # If None, plot only one line
    line_values: list = None,
    line_plot_kwargs: List[dict] = None,
    # Plots: One plot per value in the <plot_columns> columns
    plot_columns: List[str] = None,  # If None, only one plot
    # Scales
    x_log_scale: bool = False,  # Plot x-axis in log scale
    y_log_scale: Union[bool, List[bool]] = False,  # Plot y-axes in log scale
    y_db_scale: Union[bool, List[bool]] = False,  # Plot y-axes in decibels
    # Annotation positions (in multiple of X-axis or Y-axis range)
    annotate: bool = True,
    annotation_text_x_pos: float = 0.55,  # X-axis position of annotation text
    annotation_arrow_x_start_pos: float = 0.6,  # X-axis starting point of arrows
    annotation_arrow_x_delta: float = 0.1,  # X-axis delta between two consecutive annotation arrows
    # Share axis
    sharex: bool = True,
    # Save plots
    plot_name_prefix: str = None,
    save_plots: bool = False,
    save_plots_dir: str = None
) -> List[
    Tuple[
        plt.Figure,  # Plot
        plt.Axes,
        Any  # Value of plot in <plot_column>
    ]
]:
    # Y-axes scales
    if not isinstance(y_log_scale, list):
        y_log_scale = [y_log_scale] * len(y_columns)
    if not isinstance(y_db_scale, list):
        y_db_scale = [y_db_scale] * len(y_columns)
    for i in range(len(y_columns)):
        y_log_scale[i] = (y_log_scale[i] and (not y_db_scale[i]))  # decibels are not plotted in log-scale

    # Columns on which to group by different runs
    group_by_columns = [x_column]
    if line_column is not None:
        group_by_columns.append(line_column)
    if plot_columns is not None:
        group_by_columns += plot_columns

    # Get median values and quantiles of predicted values
    df_mean_and_q = get_mean_and_quantiles(
        df=df,
        x_columns=group_by_columns,
        y_columns=y_columns,
        quantile=quantile_value,
        quantile_method=quantile_method,
        y_db_scale=y_db_scale
    )

    # Take mean of ground-truth values
    if ground_truth_y_columns is not None:
        df_ground_truth = df[
            [*group_by_columns, *ground_truth_y_columns]
        ].groupby(
            group_by_columns, as_index=False
        ).mean()
        df_mean_and_q = df_mean_and_q.merge(
            df_ground_truth,
            on=group_by_columns,
            how="left"
        )
    df_mean_and_q = df_mean_and_q.sort_values(x_column)

    # Separate dataframe per value in <plot_column>
    if plot_columns is not None:
        plot_value_and_df = []
        unique_plot_values = df_mean_and_q[plot_columns].drop_duplicates()
        for _, filter_values in unique_plot_values.iterrows():
            filter_val_list = filter_values.to_list()
            mask = pd.Series(
                [True for _ in range(len(df_mean_and_q))],
                index=df_mean_and_q.index
            )
            for col, val in zip(plot_columns, filter_val_list):
                mask = mask & (df_mean_and_q[col] == val)
            plot_value_and_df.append(
                (filter_val_list, df_mean_and_q[mask])
            )
    else:
        plot_value_and_df = [(None, df_mean_and_q)]

    # Plots
    all_plots = []
    set_rc_params()

    # One plot per value in "plot_column"
    for plot_values, plot_df in plot_value_and_df:
        # Separate plot_df per line
        if line_column is not None:
            if line_plot_kwargs is not None:
                lines_df_and_plot_kwargs = [
                    (plot_df[plot_df[line_column] == line_val], line_kwargs)
                    for line_val, line_kwargs in zip(line_values, line_plot_kwargs)
                ]
            else:
                lines_df_and_plot_kwargs = [
                    (plot_df[plot_df[line_column] == line_val], dict())
                    for line_val in line_values
                ]
        else:
            lines_df_and_plot_kwargs = [(plot_df, dict())]

        # Setup figure
        n_subplots = len(y_columns)
        fig, axs = plt.subplots(n_subplots, squeeze=False, sharex=sharex, sharey=False)
        axs = axs.reshape(-1)
        fig_width = 10
        fig_height = max(5.5, 5 * n_subplots)
        fig.set_size_inches((fig_width, fig_height))
        if sharex:
            axs[-1].set_xlabel(x_label)
        for ax, y_label, ax_y_log_scale in zip(axs, y_labels, y_log_scale):
            ax.set_ylabel(y_label)
            if not sharex:
                ax.set_xlabel(x_label)
            if sharex:
                ax.xaxis.set_tick_params(which='both', labelbottom=True)
            if x_log_scale:
                ax.set_xscale("log")
                ax.xaxis.set_major_formatter(ScalarFormatter())
            if ax_y_log_scale:
                ax.set_yscale("log")
        _plot_kwargs = dict()

        tiled_ground_truth_y_columns = (
            ground_truth_y_columns
            if ground_truth_y_columns is not None else
            [None] * n_subplots
        )

        # One subplot per entry in "y_columns"
        for ax, y_col, ground_truth_y_col, log_scale, db_scale in zip(
                axs,
                y_columns,
                tiled_ground_truth_y_columns,
                y_log_scale,
                y_db_scale
        ):
            # Plot ground-truth
            # =================
            if ground_truth_y_col is not None:
                df_gt = lines_df_and_plot_kwargs[0][0]  # select any df
                y_gt = df_gt[ground_truth_y_col].tolist()
                if db_scale:
                    y_gt = to_db(y_gt)
                ax.plot(
                    df_gt[x_column].tolist(),
                    y_gt,
                    linestyle="--",
                    color="black",
                    label="Ground-Truth",
                    **_plot_kwargs
                )

            # Plot calibration
            # ================
            # Store labels and pos of curves for annotation
            x_pos_arrow = annotation_arrow_x_start_pos
            labels_pos_arrow = []

            y_min = np.inf

            # One line per value in "line_column"
            for line_df, line_kwargs in lines_df_and_plot_kwargs:
                if not line_df.empty:
                    # Plot
                    current_plot_kwargs = {
                        **_plot_kwargs,
                        **line_kwargs
                    }
                    x = line_df[x_column].to_numpy()
                    y = line_df[f"{y_col}_median"].to_numpy()
                    plot_fill_between(
                        ax=ax,
                        x=x,
                        y=y,
                        y_err_low=line_df[f"{y_col}_low_q"].to_numpy(),
                        y_err_high=line_df[f"{y_col}_high_q"].to_numpy(),
                        plot_kwargs=current_plot_kwargs
                    )
                    # Annotation
                    if "label" in current_plot_kwargs:
                        # Position arrow
                        label = current_plot_kwargs["label"]
                        idx_pos_arrow = min(
                            int(round(x_pos_arrow * len(x))),
                            len(x) - 1
                        )
                        pos_arrow = (x[idx_pos_arrow], y[idx_pos_arrow])
                        labels_pos_arrow.append((label, pos_arrow))
                        # Minimal Y-value at text X-pos
                        idx_pos_text = min(
                            int(round(annotation_text_x_pos * len(x))),
                            len(x) - 1
                        )
                        y_min = min(y_min, y[idx_pos_text])
                        # Update arrow pos
                        x_pos_arrow += annotation_arrow_x_delta

            # Annotate curves
            if annotate:
                _annotate_aux(
                    ax=ax,
                    x_pos_text=annotation_text_x_pos,
                    y_min=y_min,
                    labels_pos_arrow=labels_pos_arrow,
                    log_scale=log_scale
                )

        # Append plot to output
        all_plots.append((fig, axs, plot_values))

        # Save plot
        plot_values_name_suffix = ""
        if plot_values is not None:
            plot_values_name_suffix = "_" + "_".join([
                f"{col}_{str(val)}"
                for col, val in zip(plot_columns, plot_values)
            ])
        plot_name = (
            (plot_name_prefix if plot_name_prefix is not None else "errorbar") +
            f"_{'_'.join(y_columns)}" +
            plot_values_name_suffix
        )
        if save_plots:
            with SafeOpen(save_plots_dir, f"{plot_name}.png", "wb") as file:
                fig.tight_layout()
                fig.savefig(file, dpi=300, bbox_inches="tight")

    return all_plots
