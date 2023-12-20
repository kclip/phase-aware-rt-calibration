import os
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plot_utils import set_rc_params
from src.utils.save_utils import SafeOpen
from study.calibration.utils import CalibrationType
from study.calibration.experiment_protocol import get_protocol_by_name


def plot_legend(
    protocol_name: str,
    save_dir: str,
    estimation_error_legend: bool = False,
    fig_width: float = 10,
):
    set_rc_params()

    protocol = get_protocol_by_name(protocol_name=protocol_name)
    line_plot_kwargs = [
        CalibrationType.get_plot_kwargs(calibration_type)
        for calibration_type in protocol.run_calibration_types
    ]
    if not estimation_error_legend:
        line_plot_kwargs = [dict(
            linestyle="--",
            color="black",
            label="Ground-Truth",
        )] + line_plot_kwargs

    # Plot empty line
    plot_line = lambda ax_plt, kwargs: ax_plt.plot([], [], **kwargs)[0]

    for single_col in [True, False]:
        filename = (
            "legend_calibration_error" if estimation_error_legend else "legend_calibration"
        ) + (
            "_single_column" if single_col else "_double_column"
        ) + ".png"
        filepath = os.path.join(save_dir, filename)

        if not os.path.exists(filepath):
            if single_col:
                n_cols = 1
            else:
                n_cols = 3 if estimation_error_legend else 2
            n_rows = int(np.ceil(len(line_plot_kwargs) / n_cols))
            width = fig_width if single_col else 2*fig_width
            height = 2 * n_rows

            fig, ax = plt.subplots()
            fig.set_size_inches((width, height))
            handles = [plot_line(ax, line_kwargs) for line_kwargs in line_plot_kwargs]
            legend = ax.legend(
                handles=handles,
                bbox_to_anchor=(0, 1.02, 1, 0.2),
                loc="lower left",
                mode="expand",
                ncol=n_cols,
                frameon=False
            )
            fig.canvas.draw()
            bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            
            with SafeOpen(save_dir, filename, "wb") as file:
                fig.savefig(file, dpi=300, bbox_inches=bbox)
            
