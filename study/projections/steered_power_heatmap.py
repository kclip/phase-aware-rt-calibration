import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sionna.rt import Scene, Paths

from src.utils.tensor_utils import dot, squared_norm
from src.projections.angle_lattice import get_angle_linspace
from src.projections.relative_position_array_elements import get_relative_position_array_elements
from src.projections.angle_projection import get_steering_vectors


def compute_steered_power_heatmap(
    scene: Scene,
    paths: Paths,
    n_points: int
):
    # Init
    # ----
    min_el, max_el = 0, np.pi
    min_az, max_az = -np.pi, np.pi

    # Get position array elements
    # ---------------------------
    rx_arr_pos, tx_arr_pos = get_relative_position_array_elements(scene)
    n_rx = rx_arr_pos.shape[0]
    n_tx = tx_arr_pos.shape[0]

    # Get projection angles
    # ---------------------
    elevation, azimuth = get_angle_linspace(
        n_points=n_points,
        min_elevation=min_el,
        max_elevation=max_el,
        min_azimuth=min_az,
        max_azimuth=max_az
    )
    grid_el, grid_az = tf.meshgrid(elevation, azimuth, indexing="ij")
    projection_angles = tf.stack([
        tf.reshape(grid_el, [-1]),
        tf.reshape(grid_az, [-1])
    ], axis=-1)
    projection_angles = projection_angles[tf.newaxis, tf.newaxis, :, :]  # Shape [N_RX=1, N_TX=1, N_POINTS, 2]
    n_projections = projection_angles.shape[2]

    # Compute steering vectors
    # ------------------------
    steering_rx, steering_tx = get_steering_vectors(
        rx_array_elements_positions=rx_arr_pos,
        tx_array_elements_positions=tx_arr_pos,
        rx_aoa_steering_angles=projection_angles,
        tx_aod_steering_angles=projection_angles,
        frequency=scene.frequency
    )

    # Get path amplitudes
    # -------------------
    # Shape [batch_size=1, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps=1]
    amplitudes, _ = paths.cir()
    # To shape [N_RX, N_ARR_RX, N_TX, N_ARR_TX, N_PATH, N_POINTS=1]
    amplitudes = amplitudes[0, :, :, :, :, :, 0, tf.newaxis]
    amplitudes = tf.transpose(  # Shape [N_RX, N_TX, N_ARR_RX, N_ARR_TX, N_PATH, N_POINTS=1]
        amplitudes,
        [0, 2, 1, 3, 4, 5]
    )

    n_elements_rx_array = amplitudes.shape[2]
    n_elements_tx_array = amplitudes.shape[3]
    n_paths = amplitudes.shape[4]

    # Get steered power for receiver
    # ------------------------------
    # To shape [N_RX, N_TX, N_ARR_RX, N_ARR_TX=1, N_PATH=1, N_POINTS]
    steering_rx = steering_rx[:, :, :, tf.newaxis, tf.newaxis, :]
    rx_steered_complex_amplitudes = dot(
        tf.math.conj(steering_rx),
        amplitudes,
        axis=2
    )
    rx_steered_power = squared_norm(  # Shape [N_RX, N_TX, N_PATH, N_POINTS]
        rx_steered_complex_amplitudes,
        axis=2
    )

    # Get steered power for transmitter
    # ---------------------------------
    # To shape [N_RX, N_TX, N_ARR_RX=1, N_ARR_TX, N_PATH=1, N_POINTS]
    steering_tx = steering_tx[:, :, tf.newaxis, :, tf.newaxis, :]
    tx_steered_complex_amplitudes = dot(
        tf.math.conj(steering_tx),
        amplitudes,
        axis=3
    )
    tx_steered_power = squared_norm(  # Shape [N_RX, N_TX, N_PATH, N_POINTS]
        tx_steered_complex_amplitudes,
        axis=2
    )

    return elevation, azimuth, rx_steered_power, tx_steered_power


def plot_steered_power_heatmap(
    paths: Paths,
    elevation,
    azimuth,
    rx_steered_power,
    tx_steered_power,
    idx_path: int,
    idx_rx: int,
    idx_tx: int
):
    # Init
    # ----
    min_el, max_el = 0, np.pi
    min_az, max_az = -np.pi, np.pi
    rx_aoa_el = paths.theta_r[0, idx_rx, idx_tx, idx_path]
    rx_aoa_az = paths.phi_r[0, idx_rx, idx_tx, idx_path]
    tx_aod_el = paths.theta_t[0, idx_rx, idx_tx, idx_path]
    tx_aod_az = paths.phi_t[0, idx_rx, idx_tx, idx_path]
    n_points_el = elevation.shape[0]
    n_points_az = azimuth.shape[0]

    rx_angular_power_map = tf.reshape(
        rx_steered_power[idx_rx, idx_tx, idx_path, :],
        [n_points_el, n_points_az]
    )
    tx_angular_power_map = tf.reshape(
        tx_steered_power[idx_rx, idx_tx, idx_path, :],
        [n_points_el, n_points_az]
    )

    # Plot heatmaps
    # -------------
    fig, axs = plt.subplots(2, 1)
    fig.set_size_inches((20, 40))

    # Plot projections
    axs[0].set_title("Receiver Angular steering Power Heatmap")
    sns.heatmap(
        rx_angular_power_map.numpy(),
        xticklabels=azimuth.numpy(),
        yticklabels=elevation.numpy(),
        ax=axs[0]
    )
    axs[1].set_title("Transmitter Angular steering Power Heatmap")
    sns.heatmap(
        tx_angular_power_map.numpy(),
        xticklabels=azimuth.numpy(),
        yticklabels=elevation.numpy(),
        ax=axs[1]
    )

    # Plot AoA and AoD paths
    pos_rx = [  # x-axis and y-axis values are given by the position of the lattice
        n_points_el * (rx_aoa_el.numpy() - min_el) / np.abs(max_el - min_el),
        n_points_az * (rx_aoa_az.numpy() - min_az) / np.abs(max_az - min_az)
    ]
    pos_tx = [
        n_points_el * (tx_aod_el.numpy() - min_el) / np.abs(max_el - min_el),
        n_points_az * (tx_aod_az.numpy() - min_az) / np.abs(max_az - min_az)
    ]
    axs[0].plot(pos_rx[1], pos_rx[0], marker="X", markersize=30, color="blue")
    axs[1].plot(pos_tx[1], pos_tx[0], marker="X", markersize=30, color="blue")

    for ax in axs:
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Elevation")

    print(f"Receiver angles: el={rx_aoa_el}, az={rx_aoa_az}")
    print(f"Transmitter angles: el={tx_aod_el}, az={tx_aod_az}")
