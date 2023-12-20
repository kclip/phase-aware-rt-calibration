import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import scipy.stats as st

from src.utils.plot_utils import set_rc_params
from src.utils.tensor_utils import count_non_null


def plot_von_mises_global_calibration(
    track_von_mises_global_concentration: tf.Tensor,  # [N_STEPS]
    ground_truth_von_mises_global_concentration: float
):
    n_steps = track_von_mises_global_concentration.shape[0]

    # Setup matplotlib params
    set_rc_params()

    # Plot
    plot_kwargs = {
        "color": "g",
        "lw": 3
    }
    fig, ax = plt.subplots()
    fig.set_size_inches((20, 6))
    x = range(1, n_steps + 1)
    ax.plot(x, track_von_mises_global_concentration, **plot_kwargs, label="Von Mises Global Concentration")
    if ground_truth_von_mises_global_concentration is not None:
        ax.plot(
            [1, n_steps], [ground_truth_von_mises_global_concentration]*2,
            linestyle="--", **plot_kwargs, label="Ground-truth Global Concentration"
        )
    ax.set_xlabel("Calibration step")
    ax.set_ylabel("Concentration")
    ax.legend()


def plot_von_mises_components_calibration(
    components_noise: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    track_von_mises_mean_params: tf.Tensor,  # [N_STEPS, N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    track_von_mises_concentration_params: tf.Tensor,  # [N_STEPS, N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS] or [N_STEPS]
    n_measurement: int = 0,
    n_rx_tx_pair: int = 0,
    center_angles_range_around_zero: bool = False,
    angles_ylim: list = None,
    n_paths: int = None
):
    # Get data
    n_paths = (
        n_paths
        if n_paths is not None else
        count_non_null(components_noise[n_measurement, n_rx_tx_pair, :], axis=0)
    )
    n_steps = track_von_mises_mean_params.shape[0]

    track_mean = track_von_mises_mean_params[:, n_measurement, n_rx_tx_pair, :]
    true_angles = tf.math.angle(components_noise[n_measurement, n_rx_tx_pair, :])
    shift_plot_range = np.pi if center_angles_range_around_zero else 0.0
    track_mean = tf.math.mod(track_mean + shift_plot_range, 2 * np.pi) - shift_plot_range
    true_angles = tf.math.mod(true_angles + shift_plot_range, 2 * np.pi) - shift_plot_range
    if len(track_von_mises_concentration_params.shape) == 4:
        track_concentration = track_von_mises_concentration_params[:, n_measurement, n_rx_tx_pair, :]
    else:
        track_concentration = tf.tile(track_von_mises_concentration_params[:, tf.newaxis], (1, n_paths))

    # Setup matplotlib params
    set_rc_params()

    # Plot
    fig, axs = plt.subplots(n_paths, 2, squeeze=False)
    fig.set_size_inches((20, 6 * n_paths))
    for ax in axs[:, 0]:
        ax.set_ylim(
            [0 - shift_plot_range, 2 * np.pi - shift_plot_range]
            if angles_ylim is None else
            angles_ylim
        )
        ax.set_xlabel("Calibration step")
        ax.set_ylabel("Angle")
    for ax in axs[:, 1]:
        ax.set_xlabel("Calibration step")
        ax.set_ylabel("Concentration")

    plot_kwargs = {
        "color": "g",
        "lw": 3
    }
    x = range(1, n_steps + 1)
    for i in range(n_paths):
        # Plot mean angle
        axs[i, 0].plot(x, track_mean[:, i], **plot_kwargs, label="Von Mises Mean")
        axs[i, 0].plot(
            [1, n_steps], [true_angles[i], true_angles[i]],
            linestyle="--", **plot_kwargs, label="True Angle"
        )
        # Plot concentration
        axs[i, 1].plot(x, track_concentration[:, i], **plot_kwargs, label="Von Mises Concentration")

    # Display legend
    for ax in axs.reshape(-1):
        ax.legend()


def plot_von_mises(
    von_mises_mean_params: np.ndarray,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    von_mises_concentration_params: np.ndarray,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS] or []
    von_mises_global_mean: float = 0.0,
    von_mises_global_concentration: float = None,
    components_noise: tf.Tensor = None,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    n_measurement: int = 0,
    n_rx_tx_pair: int = 0,
    center_angles_range_around_zero: bool = True,
    n_points_plot: int = 1000
):
    # Get data
    n_paths = count_non_null(components_noise[n_measurement, n_rx_tx_pair, :], axis=0)
    mean_params = von_mises_mean_params[n_measurement, n_rx_tx_pair, :]
    if len(von_mises_concentration_params.shape) == 3:
        concentration_params = von_mises_concentration_params[n_measurement, n_rx_tx_pair, :]
    else:
        concentration_params = np.repeat(von_mises_concentration_params, n_paths)
    shift_plot_range = np.pi if center_angles_range_around_zero else 0.0
    mean_params = np.mod(mean_params + shift_plot_range, 2 * np.pi) - shift_plot_range
    true_angles = None
    if components_noise is not None:
        true_angles = np.angle(components_noise[n_measurement, n_rx_tx_pair, :])
        true_angles = np.mod(true_angles + shift_plot_range, 2 * np.pi) - shift_plot_range

    # Setup matplotlib params
    set_rc_params()

    # Plot
    fig, ax = plt.subplots()
    fig.set_size_inches((20, 10))
    plot_kwargs = {
        "lw": 3
    }
    x = np.linspace(0 - shift_plot_range, 2 * np.pi - shift_plot_range, n_points_plot)
    von_mises_pdfs = np.zeros((n_paths, n_points_plot))
    for i in range(n_paths):
        rv = st.vonmises(concentration_params[i], loc=mean_params[i])
        von_mises_pdfs[i, :] = rv.pdf(x)
    max_pdf = np.max(von_mises_pdfs)
    if von_mises_global_concentration is not None:
        rv = st.vonmises(von_mises_global_concentration, loc=von_mises_global_mean)
        global_pdf = rv.pdf(x)
        ax.plot(x, global_pdf, color="black", linestyle="--", **plot_kwargs)
        max_pdf = max(max_pdf, np.max(global_pdf))
    for i in range(n_paths):
        c = plt.cm.tab10.colors[i]
        ax.plot(x, von_mises_pdfs[i, :], color=c, **plot_kwargs)
        if true_angles is not None:
            ax.vlines(true_angles[i], 0, max_pdf, color=c, linestyle="--", **plot_kwargs)

    return ax
