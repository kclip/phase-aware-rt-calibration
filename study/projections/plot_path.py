import tensorflow as tf
import matplotlib.pyplot as plt
from sionna.rt import Scene, Paths

from src.utils.tensor_utils import angles_to_unit_vec


def plot_path(
        scene: Scene,
        paths: Paths,
        idx_path: int,
        idx_rx: int,
        idx_tx: int
):
    pos_rx = list(scene.receivers.values())[idx_rx].position
    pos_tx = list(scene.transmitters.values())[idx_tx].position

    scale = tf.sqrt(tf.reduce_sum(tf.pow(pos_rx - pos_tx, 2), axis=0)) / 3

    dir_path_rx = scale * angles_to_unit_vec(
        elevation=paths.theta_r[0, idx_rx, idx_tx, idx_path],
        azimuth=paths.phi_r[0, idx_rx, idx_tx, idx_path]
    )
    dir_path_tx = scale * angles_to_unit_vec(
        elevation=paths.theta_t[0, idx_rx, idx_tx, idx_path],
        azimuth=paths.phi_t[0, idx_rx, idx_tx, idx_path]
    )

    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)
    width_arrow = 0.002
    # XY plane
    axs[0].set_title("XY Plane")
    axs[0].plot(pos_rx[0], pos_rx[1], marker="X", color="blue")
    axs[0].arrow(pos_rx[0], pos_rx[1], dir_path_rx[0], dir_path_rx[1], color="blue", width=width_arrow * scale)
    axs[0].plot(pos_tx[0], pos_tx[1], marker="X", color="red")
    axs[0].arrow(pos_tx[0], pos_tx[1], dir_path_tx[0], dir_path_tx[1], color="red", width=width_arrow * scale)
    # XZ plane
    axs[1].set_title("XZ Plane")
    axs[1].plot(pos_rx[0], pos_rx[2], marker="X", color="blue")
    axs[1].arrow(pos_rx[0], pos_rx[2], dir_path_rx[0], dir_path_rx[2], color="blue", width=width_arrow * scale)
    axs[1].plot(pos_tx[0], pos_tx[2], marker="X", color="red")
    axs[1].arrow(pos_tx[0], pos_tx[2], dir_path_tx[0], dir_path_tx[2], color="red", width=width_arrow * scale)

    for ax in axs:
        ax.axis('equal')

    return axs