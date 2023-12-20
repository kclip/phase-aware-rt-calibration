from typing import Tuple
import numpy as np
import tensorflow as tf
from sionna.rt import Scene, Paths

from const import SPEED_OF_LIGHT
from src.utils.sionna_utils import get_mask_paths
from src.utils.tensor_utils import dot, cast_to_complex, cast_to_pure_imag, angles_to_unit_vec
from src.projections.relative_position_array_elements import get_relative_position_array_elements
from src.projections.angle_lattice import get_paths_angles, get_angle_bounds_paths, get_fibonacci_angle_lattices


def get_steering_vectors(
    # Position of the array elements in cartesian coordinates (with origin located at the center of the array)
    rx_array_elements_positions: tf.Tensor,  # Shape [N_RX, N_ARR_RX, 3]
    tx_array_elements_positions: tf.Tensor,  # Shape [N_TX, N_ARR_TX, 3]
    rx_aoa_steering_angles: tf.Tensor,  # Rx AoA (el, az) projection angles in [rad] ; Shape [N_RX, N_TX, N_POINTS, 2]
    tx_aod_steering_angles: tf.Tensor,  # Tx AoD (el, az) projection angles in [rad] ; Shape [N_RX, N_TX, N_POINTS, 2]
    frequency: float  # Carrier frequency in [Hz]
) -> Tuple[
     tf.Tensor,  # Rx AoA steering coefficients ; Shape [N_RX, N_TX, N_ARR_RX, N_POINTS]
     tf.Tensor  # Tx AoD steering coefficients ; Shape [N_RX, N_TX, N_ARR_TX, N_POINTS]
]:
    """
    Compute steering vectors complex coefficients for each specified angular projection at the receiver and transmitter.
    Note: the coefficients must be conjugated before multiplying it with the CFR
    """

    # Get direction vectors (unitary 3D vector) for angular projections
    rx_aoa_steering_directions = angles_to_unit_vec(  # Shape [N_RX, N_TX, N_POINTS, 3]
        elevation=rx_aoa_steering_angles[..., 0],
        azimuth=rx_aoa_steering_angles[..., 1]
    )
    # To shape [N_RX, N_TX, N_ARR_RX=1, N_POINTS, 3]
    rx_aoa_steering_directions = rx_aoa_steering_directions[:, :, tf.newaxis, :, :]

    tx_aod_steering_directions = angles_to_unit_vec(  # Shape [N_RX, N_TX, N_POINTS, 3]
        elevation=tx_aod_steering_angles[..., 0],
        azimuth=tx_aod_steering_angles[..., 1]
    )
    # To shape [N_RX, N_TX, N_ARR_TX=1, N_POINTS, 3]
    tx_aod_steering_directions = tx_aod_steering_directions[:, :, tf.newaxis, :, :]

    # Format array elements positions
    # To shape [N_RX, N_TX=1, N_ARR_RX, N_POINTS=1, 3]
    rx_array_elements_positions_tiled = rx_array_elements_positions[:, tf.newaxis, :, tf.newaxis, :]
    # To shape [N_RX=1, N_TX, N_ARR_TX, N_POINTS=1, 3]
    tx_array_elements_positions_tiled = tx_array_elements_positions[tf.newaxis, :, :, tf.newaxis, :]

    # Compute steering vectors
    two_pi_over_wavelength = tf.constant(2 * np.pi * frequency / SPEED_OF_LIGHT, dtype=tf.float32)
    rx_steering_phase = (
        two_pi_over_wavelength *
        dot(rx_aoa_steering_directions, rx_array_elements_positions_tiled, axis=-1)
    )
    tx_steering_phase = (
        two_pi_over_wavelength *
        dot(tx_aod_steering_directions, tx_array_elements_positions_tiled, axis=-1)
    )

    return (
        tf.exp(cast_to_pure_imag(rx_steering_phase)),
        tf.exp(cast_to_pure_imag(tx_steering_phase))
    )


def get_paths_angles_projections(
    scene: Scene,
    paths: Paths
) -> Tuple[
     tf.Tensor,  # Rx AoA steering coefficients ; Shape [N_RX, N_TX, N_ARR_RX, N_POINTS]
     tf.Tensor  # Tx AoD steering coefficients ; Shape [N_RX, N_TX, N_ARR_TX, N_POINTS]
]:
    """
    Get angular projections for all the (Rx, Tx) pairs in the scene, both on the receiver and transmitter side.
    The angles of the corresponding projection points are obtained as the angles of arrival and departure of the 
    predicted paths.
    If the number of projection points is superior to the number of paths for a given (Rx, Tx) pair, extra points
    are set to 0 (projection to a null value).

    Note: values must be conjugated before being multiplied with the CFR.
    """
    # Rx/Tx (elevation, azimuth) angles (shape [N_RX, N_TX, N_PATHS, 2])
    rx_angles, tx_angles = get_paths_angles(paths)

    # Position array elements
    (
        rx_pos_array_elements,  # Shape [N_RX, N_ARR_RX, 3]
        tx_pos_array_elements  # Shape [N_TX, N_ARR_TX, 3]
    ) = get_relative_position_array_elements(scene=scene)

    # Steering vectors ; shape [N_RX, N_TX, N_ARR_RX/N_ARR_TX, N_POINTS]
    steering_rx, steering_tx = get_steering_vectors(
        rx_array_elements_positions=rx_pos_array_elements,
        tx_array_elements_positions=tx_pos_array_elements,
        rx_aoa_steering_angles=rx_angles,
        tx_aod_steering_angles=tx_angles,
        frequency=scene.frequency
    )

    # Remove projections corresponding to non-existent paths
    mask_paths = get_mask_paths(paths)  # Shape [N_RX, N_TX, N_PATHS]
    mask_paths = cast_to_complex(tf.cast(mask_paths, tf.float32))
    # To shape [N_RX, N_TX, N_ARR_RX/N_ARR_TX=1, N_POINTS=N_PATH]
    mask_paths = mask_paths[:, :, tf.newaxis, :]

    steering_rx = steering_rx * mask_paths
    steering_tx = steering_tx * mask_paths

    return steering_rx, steering_tx


def get_evenly_spaced_angle_projections(
    scene: Scene,
    paths: Paths,
    angle_resolution_az: float,  # in [rad]
    angle_resolution_el: float  # in [rad]
) -> Tuple[
     tf.Tensor,  # Rx AoA steering coefficients ; Shape [N_RX, N_TX, N_ARR_RX, N_POINTS]
     tf.Tensor  # Tx AoD steering coefficients ; Shape [N_RX, N_TX, N_ARR_TX, N_POINTS]
]:
    """
    Get angular projections for all the (Rx, Tx) pairs in the scene, both on the receiver and transmitter side.

    The angular domain of the angular projections at a given device (for a given (Rx, Tx) pair) is computed as follows:
        - Range elevation: [min_p angle_p_el ; max_p angle_p_el] where angle_p_el in [0, pi] for all paths p
        - Mean azimuth angle: mu_az = 1/N_PATHS sum_{p in Paths} angle_az_p
        - Mean shifted angles: angle_p_az_shift = (angle_p_az - mu_az) mod [-pi, pi]
        - Range azimuth: [(min_p angle_p_az_shift) + mu_az ; (max_p angle_p_az_shift) + mu_az]

    The positions of the projections are obtained by computing a Fibonacci lattice on the fraction of the unit sphere
    spanned by the predicted angles, resulting in angular projections approximately equidistant on the unit sphere for
    the angle-window of interest according to the simulated paths.

    The number of projections points is computed based on the azimuth and elevation resolution of the arrays and on the
    maximum angle spread across all (Rx, Tx) pairs as: max_angle_spread / angle_res.
    We assume that the azimuth-angular resolution is independent of the elevation of the considered point.
    In practice, the azimuth resolution should be set as its minimum across all possible elevation angles to avoid
    under-sampling the number of angular projections.

    Note: values must be conjugated before being multiplied with the CFR.
    """
    # Get angle bounds and number of points
    # -------------------------------------
    angle_bounds = get_angle_bounds_paths(
        paths=paths,
        angle_resolution_az=angle_resolution_az,
        angle_resolution_el=angle_resolution_el
    )

    max_angle_spread_az = max(
        tf.reduce_max(angle_bounds.rx_max_az - angle_bounds.rx_min_az).numpy(),
        tf.reduce_max(angle_bounds.tx_max_az - angle_bounds.tx_min_az).numpy()
    )
    if (max_angle_spread_az < 0) or (max_angle_spread_az > 2*np.pi):
        raise ValueError(f"Invalid azimuth angle spread: '{max_angle_spread_az}'")
    n_points_az = np.ceil(max_angle_spread_az / angle_resolution_az)
    max_angle_spread_el = max(
        tf.reduce_max(angle_bounds.rx_max_el - angle_bounds.rx_min_el).numpy(),
        tf.reduce_max(angle_bounds.tx_max_el - angle_bounds.tx_min_el).numpy()
    )
    if (max_angle_spread_el < 0) or (max_angle_spread_el > np.pi):
        raise ValueError(f"Invalid elevation angle spread: '{max_angle_spread_el}'")
    n_points_el = np.ceil(max_angle_spread_el / angle_resolution_el)

    n_points_per_device = int(n_points_az * n_points_el)

    # Get Fibonacci angle lattice for each (Rx, Tx) pair
    # --------------------------------------------------
    rx_aoa_angle_lattices = get_fibonacci_angle_lattices(  # (El, Az) Rx proj ; Shape [N_RX, N_TX, N_POINTS, 2]
        n_points=n_points_per_device,
        min_elevation=angle_bounds.rx_min_el,
        max_elevation=angle_bounds.rx_max_el,
        min_azimuth=angle_bounds.rx_min_az,
        max_azimuth=angle_bounds.rx_max_az,
    )
    tx_aod_angle_lattices = get_fibonacci_angle_lattices(  # (El, Az) Tx proj ; Shape [N_RX, N_TX, N_POINTS, 2]
        n_points=n_points_per_device,
        min_elevation=angle_bounds.tx_min_el,
        max_elevation=angle_bounds.tx_max_el,
        min_azimuth=angle_bounds.tx_min_az,
        max_azimuth=angle_bounds.tx_max_az,
    )

    # Compute steering vectors for angles in the lattice
    # --------------------------------------------------
    (
        rx_pos_array_elements,  # Shape [N_RX, N_ARR_RX, 3]
        tx_pos_array_elements  # Shape [N_TX, N_ARR_TX, 3]
    ) = get_relative_position_array_elements(scene=scene)

    steering_rx, steering_tx = get_steering_vectors(
        rx_array_elements_positions=rx_pos_array_elements,
        tx_array_elements_positions=tx_pos_array_elements,
        rx_aoa_steering_angles=rx_aoa_angle_lattices,
        tx_aod_steering_angles=tx_aod_angle_lattices,
        frequency=scene.frequency
    )

    return steering_rx, steering_tx
