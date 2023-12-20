from typing import Tuple, List

import numpy as np
import tensorflow as tf
from sionna.rt import Scene, Paths

from src.utils.tensor_utils import sample_uniform_unitary_cartesian_coordinates_tensor
from src.utils.sionna_utils import select_rx_tx_pairs
from src.ofdm_measurements.compute_paths import compute_paths
from src.ofdm_measurements.channel_frequency_response import compute_cfr_per_mpc_from_paths


def sample_position_noise(
    # Scenario parameters
    carrier_wavelength: float,
    n_measurements: int,
    n_rx: int,
    n_tx: int,
    # Position error parameters
    displacement_amplitude: float,  # In multiple of the wavelength; amplitude of random displacement
    displace_rx: bool,  # Add random displacements to receivers' positions
    displace_tx: bool,  # Add random displacements to transmitters' positions
) -> Tuple[
    tf.Tensor,  # Sampled position noise of receiver ; shape [N_MEASUREMENTS, N_RX, N_COORDS=3]
    tf.Tensor  # Sampled position noise of transmitter ; shape [N_MEASUREMENTS, N_TX, N_COORDS=3]
]:
    absolute_displacement = tf.constant(displacement_amplitude * carrier_wavelength, dtype=tf.float32)
    shape_rx = (n_measurements, n_rx)
    shape_tx = (n_measurements, n_tx)
    if displace_rx:
        position_noise_rx = absolute_displacement * sample_uniform_unitary_cartesian_coordinates_tensor(shape_rx)
    else:
        position_noise_rx = tf.zeros((*shape_rx, 3), dtype=tf.float32)
    if displace_tx:
        position_noise_tx = absolute_displacement * sample_uniform_unitary_cartesian_coordinates_tensor(shape_tx)
    else:
        position_noise_tx = tf.zeros((*shape_tx, 3), dtype=tf.float32)

    return position_noise_rx, position_noise_tx


def simulate_position_noise_paths(
    scene: Scene,
    n_measurements: int,
    # Paths parameters
    max_depth_path: int,
    num_samples_path: int,
    # Position noise
    position_noise_rx: tf.Tensor,  # Sampled position noise of receiver ; shape [N_MEASUREMENTS, N_RX, N_COORDS=3]
    position_noise_tx: tf.Tensor  # Sampled position noise of transmitter ; shape [N_MEASUREMENTS, N_RX, N_COORDS=3]
) -> List[Paths]:  # Paths for each measurement
    # Save receiver and transmitter positions
    rx_name_pos = dict()
    tx_name_pos = dict()
    for idx, (rx_name, rx) in enumerate(scene.receivers.items()):
        rx_name_pos[rx_name] = {"idx": idx, "pos": rx.position}
    for idx, (tx_name, tx) in enumerate(scene.transmitters.items()):
        tx_name_pos[tx_name] = {"idx": idx, "pos": tx.position}

    # Compute paths for each set of sampled position noise
    measured_paths = []
    for idx_meas in range(n_measurements):
        # Displace Rx/Tx positions along randomly sampled directions
        for rx_name, rx_info in rx_name_pos.items():
            scene.receivers.get(rx_name).position = rx_info["pos"] + position_noise_rx[idx_meas, rx_info["idx"]]
        for tx_name, tx_info in tx_name_pos.items():
            scene.transmitters.get(tx_name).position = tx_info["pos"] + position_noise_tx[idx_meas, tx_info["idx"]]
        # Compute and store propagation paths
        paths = compute_paths(
            scene=scene,
            max_depth=max_depth_path,
            num_samples=num_samples_path
        )
        measured_paths.append(paths)

    # Reset positions to default
    for rx_name, rx_info in rx_name_pos.items():
        scene.receivers.get(rx_name).position = rx_info["pos"]
    for tx_name, tx_info in tx_name_pos.items():
        scene.transmitters.get(tx_name).position = tx_info["pos"]

    return measured_paths


def simulate_position_noise_cfr(
    scene: Scene,
    n_measurements: int,
    # Paths parameters
    max_depth_path: int,
    num_samples_path: int,
    # CFR parameters
    num_subcarriers: int,
    subcarrier_spacing: float,
    rx_tx_indexes: List[Tuple[int]],  # List of (receiver, transimitter) index pairs
    # Position noise
    position_noise_rx: tf.Tensor,  # Sampled position noise of receiver ; shape [N_MEASUREMENTS, N_RX, N_COORDS=3]
    position_noise_tx: tf.Tensor,  # Sampled position noise of transmitter ; shape [N_MEASUREMENTS, N_RX, N_COORDS=3]
    raise_error_on_different_number_of_paths: bool = False
) -> tf.Tensor:  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATH]
    # Get paths
    paths_list = simulate_position_noise_paths(
        scene=scene,
        n_measurements=n_measurements,
        max_depth_path=max_depth_path,
        num_samples_path=num_samples_path,
        position_noise_rx=position_noise_rx,
        position_noise_tx=position_noise_tx
    )

    # Compute selected CFRs
    cfr_list = []
    for paths in paths_list:
        cfr_per_mpc = compute_cfr_per_mpc_from_paths(
            paths=paths,
            num_subcarriers=num_subcarriers,
            subcarrier_spacing=subcarrier_spacing,
            normalization_constant=1.0,
        )
        selected_cfr = select_rx_tx_pairs(
            rx_tx_indexed_tensor=cfr_per_mpc,
            rx_tx_indexes=rx_tx_indexes
        )
        cfr_list.append(selected_cfr)

    # Pad CFRs if the number of predicted paths are different between positions
    n_paths_arr = np.array([cfr.shape[-1] for cfr in cfr_list])
    n_paths_max = np.max(n_paths_arr)
    if (n_paths_arr != n_paths_max).any():
        msg = "The number of paths is not the same for each simulated position error..."
        if raise_error_on_different_number_of_paths:
            raise ValueError(msg)
        else:
            print(f"WARNING: {msg}")
            print("Padding simulated responses with zeros to equalize path-dimensions across CFRs at each position...")
            n_cfr = len(cfr_list)
            padding_value = tf.constant(0, dtype=cfr_list[0].dtype)
            for i in range(n_cfr):
                n_paths = cfr_list[i].shape[-1]
                n_pad = n_paths_max - n_paths
                cfr_list[i] = tf.pad(
                    cfr_list[i],  # Shape [N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX, N_PATHS]
                    [[0, 0], [0, 0], [0, 0], [0, 0], [0, n_pad]],  # Paddings per dimension
                    mode="CONSTANT",
                    constant_values=padding_value
                )

    # Stack CFRs
    return tf.stack(cfr_list, axis=0)
