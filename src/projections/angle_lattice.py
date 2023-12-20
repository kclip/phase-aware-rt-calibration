from typing import Tuple, Union
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from sionna.rt import Paths

from src.utils.sionna_utils import get_mask_paths
from src.utils.tensor_utils import reduce_masked_max, reduce_masked_min, angle_in_minus_pi_plus_pi, \
    get_mean_angles


def _check_angle_bounds(  # All tensors have same shape [dim1, ..., dimN]
    min_elevation: Union[float, tf.Tensor],  # Minimum elevation in [0, pi], in [rad]
    max_elevation: Union[float, tf.Tensor],  # Maximum elevation in (<min_elevation>, pi], in [rad]
    min_azimuth: Union[float, tf.Tensor],  # Minimum azimuth in [-pi, pi], in [rad]
    max_azimuth: Union[float, tf.Tensor],  # Maximum azimuth in (<min_azimuth>, pi], in [rad]
):
    """
    Check angle bounds:
       - Min and max angle bounds should be specified in the right order
       - The angle span of the azimuth angles (max_az - min_az) should be lower than 2pi
       - Bounds for elevation should be in the [0, pi] range
    """
    # Check order
    for label, min_angle, max_angle in [
        ("elevation", min_elevation, max_elevation),
        ("azimuth", min_azimuth, max_azimuth),
    ]:
        if tf.reduce_any(min_angle > max_angle):
            raise ValueError(f"Angle bounds for {label} are not defined in the right order")
    # Check angle span azimuth
    if tf.reduce_any((max_azimuth - min_azimuth) > (2 * np.pi)):
        raise ValueError("Angle span of azimuth angles is larger than 2 pi")
    # Check domain elevation
    if (
        tf.reduce_any(min_elevation < 0.0) or
        tf.reduce_any(max_elevation > np.pi)
    ):
        raise ValueError(f"Angle bounds for elevation are not in the [0, pi] domain")


@dataclass
class AngleBounds(object):
    # Rx
    rx_min_el: tf.Tensor  # Min AoA elevation in [0, pi] ; Shape [N_RX, N_TX]
    rx_max_el: tf.Tensor  # Max AoA elevation in [0, pi] ; Shape [N_RX, N_TX]
    rx_min_az: tf.Tensor  # Min AoA azimuth in [-2pi, 2pi] ; Shape [N_RX, N_TX]
    rx_max_az: tf.Tensor  # Max AoA azimuth in [-2pi, 2pi] ; Shape [N_RX, N_TX]

    # Tx
    tx_min_el: tf.Tensor  # Min AoD elevation in [0, pi] ; Shape [N_RX, N_TX]
    tx_max_el: tf.Tensor  # Max AoD elevation in [0, pi] ; Shape [N_RX, N_TX]
    tx_min_az: tf.Tensor  # Min AoD azimuth in [-2pi, 2pi] ; Shape [N_RX, N_TX]
    tx_max_az: tf.Tensor  # Max AoD azimuth in [-2pi, 2pi] ; Shape [N_RX, N_TX]


def get_paths_angles(paths: Paths) -> Tuple[
    tf.Tensor,  # Rx (elevation, azimuth) angles (in this order) ; shape [N_RX, N_TX, N_PATHS, 2]
    tf.Tensor  # Tx (elevation, azimuth) angles (in this order) ; shape [N_RX, N_TX, N_PATHS, 2]
]:
    # Individual angles (expected shape [N_RX, N_TX, N_PATHS])
    rx_az = paths.phi_r[0]
    tx_az = paths.phi_t[0]
    rx_el = paths.theta_r[0]
    tx_el = paths.theta_t[0]

    # Check shape corresponds to a synthetic array (single angle per path)
    for angle in (rx_az, tx_az, rx_el, tx_el):
        if len(angle.shape) > 3:
            raise ValueError("Projection functions are only defined for synthetic arrays")

    # Format shape
    rx_angles = tf.stack([rx_el, rx_az], axis=-1)
    tx_angles = tf.stack([tx_el, tx_az], axis=-1)

    return rx_angles, tx_angles


def get_angle_bounds_paths(
    paths: Paths,
    angle_resolution_az: float = None,  # in [rad]
    angle_resolution_el: float = None  # in [rad]
) -> AngleBounds:
    """
    Get the azimuth-elevation domain spanned by the simulated propagation paths.
    If angular resolutions are provided, this domain is extended by angle_res/2 on each side to take into account
    border effects due to resolutions of the array.
    """
    rx_angles, tx_angles = get_paths_angles(paths)

    # Mask of paths to consider
    mask_path = get_mask_paths(paths)

    # Elevation bounds in [0, pi]
    rx_el = rx_angles[:, :, :, 0]
    rx_min_el = reduce_masked_min(rx_el, mask=mask_path, axis=-1)
    rx_max_el = reduce_masked_max(rx_el, mask=mask_path, axis=-1)
    tx_el = tx_angles[:, :, :, 0]
    tx_min_el = reduce_masked_min(tx_el, mask=mask_path, axis=-1)
    tx_max_el = reduce_masked_max(tx_el, mask=mask_path, axis=-1)
    # If elevation resolution is provided, extend bounds by res/2 on each side
    if angle_resolution_el is not None:
        mid_res_el = tf.constant(angle_resolution_el / 2, dtype=tf.float32)
        rx_min_el = tf.maximum(0, rx_min_el - mid_res_el)
        tx_min_el = tf.maximum(0, tx_min_el - mid_res_el)
        rx_max_el = tf.minimum(np.pi, rx_max_el + mid_res_el)
        tx_max_el = tf.minimum(np.pi, tx_max_el + mid_res_el)

    # For azimuth, we first get the mean angle and then get the min and max angles as the
    # minimal and maximal deviations from the mean within [-pi, 0] and [0, pi] respectively
    def get_min_max_azimuth(angles):
        mean_angle = get_mean_angles(tensor_angles=angles, mask=mask_path, axis=-1)
        # To shape [N_RX, N_TX, N_PATH=1]
        mean_angle_tiled = mean_angle[..., tf.newaxis]
        mean_shifted_angles = angle_in_minus_pi_plus_pi(angles - mean_angle_tiled)
        min_angle = reduce_masked_min(
            mean_shifted_angles,
            mask=mask_path,
            axis=-1
        ) + mean_angle
        max_angle = reduce_masked_max(
            mean_shifted_angles,
            mask=mask_path,
            axis=-1
        ) + mean_angle

        # If azimuth resolution is provided, extend bounds by res/2 on each side
        if angle_resolution_el is not None:
            mid_res_az = tf.constant(angle_resolution_az / 2, dtype=tf.float32)
            min_angle = min_angle - mid_res_az
            max_angle = max_angle + mid_res_az
            # Bound max such that max_angle-min_angle does not exceed 2pi
            max_angle = min_angle + tf.minimum(2*np.pi, max_angle - min_angle)

        return min_angle, max_angle

    rx_min_az, rx_max_az = get_min_max_azimuth(rx_angles[:, :, :, 1])
    tx_min_az, tx_max_az = get_min_max_azimuth(tx_angles[:, :, :, 1])

    return AngleBounds(
        rx_min_el=rx_min_el,
        rx_max_el=rx_max_el,
        rx_min_az=rx_min_az,
        rx_max_az=rx_max_az,
        tx_min_el=tx_min_el,
        tx_max_el=tx_max_el,
        tx_min_az=tx_min_az,
        tx_max_az=tx_max_az
    )


def get_angle_linspace(
    n_points: int,  # Number of points in the lattice
    min_elevation: float,  # Minimum elevation in [0, pi], in [rad]
    max_elevation: float,  # Maximum elevation in (<min_elevation>, pi], in [rad]
    min_azimuth: float,  # Minimum azimuth in [-pi, pi], in [rad]
    max_azimuth: float,  # Maximum azimuth in (<min_azimuth>, pi], in [rad]
) -> Tuple[
    tf.Tensor,  # Elevation angles in [rad], shape [N_EL_POINTS]
    tf.Tensor  # Azimuth angles in [rad], shape [N_AZ_POINTS]
]:
    """
    Generate angle lattice corresponding to <n_points> with equidistant angles value in the 2D linear space.
    Note: this does NOT mean the corresponding points on the unit-sphere in 3D space are equidistant.
    """
    _check_angle_bounds(
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        min_azimuth=min_azimuth,
        max_azimuth=max_azimuth
    )

    angular_length_el = tf.math.abs(max_elevation - min_elevation)
    angular_length_az = tf.math.abs(max_azimuth - min_azimuth)

    # Get number of points along each angle coordinate
    if (angular_length_el == 0.0) and (angular_length_az == 0.0):
        print("WARNING : lattice contains only a single point")
        n_points_el = 1
        n_points_az = n_points
    elif angular_length_el == 0.0:
        n_points_el = 1
        n_points_az = n_points
    elif angular_length_az == 0.0:
        n_points_el = n_points
        n_points_az = 1
    else:
        ratio_el_az = angular_length_el / angular_length_az
        n_points_el = max(
            int(np.ceil(np.sqrt(ratio_el_az * n_points))),
            1
        )
        n_points_az = max(
            int(np.ceil(np.sqrt(n_points / ratio_el_az))),
            1
        )

    # Generate angles lattice
    el_angles = tf.linspace(
        tf.constant(min_elevation, dtype=tf.float32),
        tf.constant(max_elevation, dtype=tf.float32),
        n_points_el,
        axis=-1
    )
    az_angles = tf.linspace(
        tf.constant(min_azimuth, dtype=tf.float32),
        tf.constant(max_azimuth, dtype=tf.float32),
        n_points_az,
        axis=-1
    )

    return el_angles, az_angles


def get_fibonacci_angle_lattices(
    n_points: int,  # Number of points in the lattice
    min_elevation: tf.Tensor,  # Minimum elevation in [0, pi], in [rad] ; Shape [dim1, ..., dimN]
    max_elevation: tf.Tensor,  # Maximum elevation in (<min_elevation>, pi], in [rad] ; Shape [dim1, ..., dimN]
    min_azimuth: tf.Tensor,  # Minimum azimuth in [-pi, pi], in [rad] ; Shape [dim1, ..., dimN]
    max_azimuth: tf.Tensor,  # Maximum azimuth in (<min_azimuth>, pi], in [rad] ; Shape [dim1, ..., dimN]
) -> tf.Tensor:  # (Elevation, Azimuth) angles in [rad] ;  Shape [dim1, ..., dimN, N_POINTS, 2]
    """
    Generate the elevation and azimuth angles of <n_points> approximately uniformly spaced on the unit sphere
    (Fibonacci lattice) for the given angle bounds.
    """
    _check_angle_bounds(
        min_elevation=min_elevation,
        max_elevation=max_elevation,
        min_azimuth=min_azimuth,
        max_azimuth=max_azimuth
    )

    golden_ratio = (1 + tf.sqrt(tf.constant(5, dtype=tf.float32))) / 2

    indices = tf.range(0, n_points, dtype=tf.float32)  # Shape [N_POINTS]

    # Tile bounds (shape [*<bounds_shape>, N_POINTS=1])
    min_elevation = min_elevation[..., tf.newaxis]  # tile_newdim(min_elevation, tile_dim=-1, tile_n=n_points)
    max_elevation = max_elevation[..., tf.newaxis]  # tile_newdim(max_elevation, tile_dim=-1, tile_n=n_points)
    min_azimuth = min_azimuth[..., tf.newaxis]  # tile_newdim(min_azimuth, tile_dim=-1, tile_n=n_points)
    max_azimuth = max_azimuth[..., tf.newaxis]  # tile_newdim(max_azimuth, tile_dim=-1, tile_n=n_points)

    # Build lattice (note: indices are broadcasted to shape [*<bounds_shape>, N_POINTS])
    rotations = tf.math.mod(indices / golden_ratio, tf.constant(1, dtype=tf.float32))
    z_min = tf.math.cos(max_elevation)
    z_max = tf.math.cos(min_elevation)
    z = z_max + ((z_min - z_max) * (indices / n_points))

    azimuth = min_azimuth + ((max_azimuth - min_azimuth) * rotations)
    elevation = tf.math.acos(z)

    return tf.stack([elevation, azimuth], axis=-1)
