import matplotlib.pyplot as plt
from sionna.rt import Paths

from src.utils.tensor_utils import angles_to_unit_vec
from src.projections.angle_lattice import get_angle_bounds_paths, get_fibonacci_angle_lattices


def plot_fibonacci_lattice(
        paths: Paths,
        n_points_per_lattice: int,
        idx_rx: int,
        idx_tx: int
):
    angle_bounds = get_angle_bounds_paths(paths)
    rx_aoa_angle_lattices = get_fibonacci_angle_lattices(  # Shape [N_RX, N_TX, N_POINTS, 2]
        n_points=n_points_per_lattice,
        min_elevation=angle_bounds.rx_min_el,
        max_elevation=angle_bounds.rx_max_el,
        min_azimuth=angle_bounds.rx_min_az,
        max_azimuth=angle_bounds.rx_max_az,
    )
    tx_aod_angle_lattices = get_fibonacci_angle_lattices(  # Shape [N_RX, N_TX, N_POINTS, 2]
        n_points=n_points_per_lattice,
        min_elevation=angle_bounds.tx_min_el,
        max_elevation=angle_bounds.tx_max_el,
        min_azimuth=angle_bounds.tx_min_az,
        max_azimuth=angle_bounds.tx_max_az,
    )

    rx_points_lattice = angles_to_unit_vec(
        elevation=rx_aoa_angle_lattices[..., 0],
        azimuth=rx_aoa_angle_lattices[..., 1]
    )
    tx_points_lattice = angles_to_unit_vec(
        elevation=tx_aod_angle_lattices[..., 0],
        azimuth=tx_aod_angle_lattices[..., 1]
    )

    rx_paths_directions = angles_to_unit_vec(
        elevation=paths.theta_r[0],
        azimuth=paths.phi_r[0]
    )
    tx_paths_directions = angles_to_unit_vec(
        elevation=paths.theta_t[0],
        azimuth=paths.phi_t[0]
    )

    fig, axs = plt.subplots(2, 1, subplot_kw=dict(projection="3d"))
    fig.set_size_inches((10, 20))
    for ax in axs:
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

    for label, idx_ax, paths_directions, points_lattice in [
        ("Receiver", 0, rx_paths_directions[idx_rx, idx_tx], rx_points_lattice[idx_rx, idx_tx]),
        ("Transmitter", 1, tx_paths_directions[idx_rx, idx_tx], tx_points_lattice[idx_rx, idx_tx])
    ]:
        axs[idx_ax].set_title(f"{label} angle projection lattice")
        axs[idx_ax].quiver(
            # starting point of vector
            0, 0, 0,
            # directions of vector
            paths_directions[:, 0], paths_directions[:, 1], paths_directions[:, 2],
            color='red', alpha=.8, lw=1,
        )
        axs[idx_ax].scatter(
            points_lattice[:, 0],
            points_lattice[:, 1],
            points_lattice[:, 2]
        )


def plot_fibonacci_points(
        n_points: int,  # Number of points in the lattice
        min_elevation: float,  # Minimum elevation in [0, pi], in [rad]
        max_elevation: float,  # Maximum elevation in (<min_elevation>, pi], in [rad]
        min_azimuth: float,  # Minimum azimuth in [-pi, pi], in [rad]
        max_azimuth: float,  # Maximum azimuth in (<min_azimuth>, pi], in [rad]
):
    fibonacci_angles = get_fibonacci_angle_lattices(  # Shape [N_POINTS, 2]
        n_points=n_points,
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        min_azimuth=min_azimuth,
        max_azimuth=max_azimuth,
    )
    fibonacci_points = angles_to_unit_vec(  # Shape [N_POINTS, 3]
        elevation=fibonacci_angles[:, 0],
        azimuth=fibonacci_angles[:, 1]
    )

    fig = plt.figure()
    fig.set_size_inches((10, 10))
    fig.add_subplot(111, projection='3d').scatter(
        fibonacci_points[:, 0],
        fibonacci_points[:, 1],
        fibonacci_points[:, 2]
    )
    ax = fig.gca()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    # .axis('equal')
    plt.show()