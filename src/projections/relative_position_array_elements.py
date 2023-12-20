from typing import Tuple

import tensorflow as tf
from sionna.rt import Scene, rotation_matrix


def get_relative_position_array_elements(scene: Scene) -> Tuple[
    tf.Tensor,  # Relative position of receiver array elements ; shape [N_RX, N_ARR_RX, 3]
    tf.Tensor  # Relative position of transmitter array elements ; shape [N_TX, N_ARR_TX, 3]
]:
    """
    Return the relative positions of array elements for all receivers and transmitters.
    The relative position is expressed in the global cartesian coordinates (taking into account the orientation of the
    receiver/transmitter), with its origin located at the center of the respective array (i.e., the position of the
    respective receiver/transmitter)
    """
    rx_rot_mat, tx_rot_mat = _get_tx_rx_rotation_matrices(scene)
    return _get_antennas_relative_positions(scene, rx_rot_mat, tx_rot_mat)


def _get_tx_rx_rotation_matrices(scene: Scene) -> Tuple[
    tf.Tensor,  # Rx orientation rotation matrix ; shape [N_RX, 3, 3]
    tf.Tensor  # Tx orientation rotation matrix ; shape [N_TX, 3, 3]
]:
    r"""
    This function is taken from Sionna source code at:
    https://github.com/NVlabs/sionna/blob/f5ea373b948bfc367eaccdb8907889cfe62badd9/sionna/rt/solver_paths.py#L3735

    Computes and returns the rotation matrices for rotating according to
    the orientations of the transmitters and receivers rotation matrices,

    Output
    -------
    rx_rot_mat : [num_rx, 3, 3], tf.float
        Matrices for rotating according to the receivers orientations

    tx_rot_mat : [num_tx, 3, 3], tf.float
        Matrices for rotating according to the receivers orientations
    """

    transmitters = scene.transmitters.values()
    receivers = scene.receivers.values()

    # Rotation matrices for transmitters
    # [num_tx, 3]
    tx_orientations = [tx.orientation for tx in transmitters]
    tx_orientations = tf.stack(tx_orientations, axis=0)
    # [num_tx, 3, 3]
    tx_rot_mat = rotation_matrix(tx_orientations)

    # Rotation matrices for receivers
    # [num_rx, 3]
    rx_orientations = [rx.orientation for rx in receivers]
    rx_orientations = tf.stack(rx_orientations, axis=0)
    # [num_rx, 3, 3]
    rx_rot_mat = rotation_matrix(rx_orientations)

    return rx_rot_mat, tx_rot_mat


def _get_antennas_relative_positions(
    scene: Scene,
    rx_rot_mat: tf.Tensor,  # Rx orientation rotation matrix ; shape [N_RX, 3, 3]
    tx_rot_mat: tf.Tensor  # Tx orientation rotation matrix ; shape [N_TX, 3, 3]
) -> Tuple[
    tf.Tensor,  # Relative position of receiver array elements ; shape [N_RX, N_ARR_RX, 3]
    tf.Tensor  # Relative position of transmitter array elements ; shape [N_TX, N_ARR_TX, 3]
]:
    r"""
    This function is taken from Sionna source code at:
    https://github.com/NVlabs/sionna/blob/f5ea373b948bfc367eaccdb8907889cfe62badd9/sionna/rt/solver_paths.py#L3768

    Returns the positions of the antennas of the transmitters and receivers.
    The positions are relative to the center of the radio devices, but
    rotated to the GCS.

    Input
    ------
    rx_rot_mat : [num_rx, 3, 3], tf.float
        Matrices for rotating according to the receivers orientations

    tx_rot_mat : [num_tx, 3, 3], tf.float
        Matrices for rotating according to the receivers orientations

    Output
    -------
    rx_rel_ant_pos: [num_rx, rx_array_size, 3], tf.float
        Relative positions of the receivers antennas

    tx_rel_ant_pos: [num_tx, rx_array_size, 3], tf.float
        Relative positions of the transmitters antennas
    """

    # Rotated position of the TX and RX antenna elements
    # [1, tx_array_size, 3]
    tx_rel_ant_pos = tf.expand_dims(scene.tx_array.positions, axis=0)
    # [num_tx, 1, 3, 3]
    tx_rot_mat = tf.expand_dims(tx_rot_mat, axis=1)
    # [num_tx, tx_array_size, 3]
    tx_rel_ant_pos = tf.linalg.matvec(tx_rot_mat, tx_rel_ant_pos)

    # [1, rx_array_size, 3]
    rx_rel_ant_pos = tf.expand_dims(scene.rx_array.positions, axis=0)
    # [num_rx, 1, 3, 3]
    rx_rot_mat = tf.expand_dims(rx_rot_mat, axis=1)
    # [num_tx, tx_array_size, 3]
    rx_rel_ant_pos = tf.linalg.matvec(rx_rot_mat, rx_rel_ant_pos)

    return rx_rel_ant_pos, tx_rel_ant_pos
