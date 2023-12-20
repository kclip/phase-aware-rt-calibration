from typing import Tuple
import numpy as np
import tensorflow as tf
from sionna.rt import Scene, Receiver

from src.utils.python_utils import batch
from src.utils.telecom_utils import from_db
from src.utils.channel_power import compute_channel_power_rt_simulated_phases, compute_channel_power_uniform_phase_error
from src.utils.tensor_utils import reduce_mean_ignore_null
from src.coverage_map import CoverageMapMetadata
from src.data_classes import CoverageMapCFRData, CoverageMapPowerData
from src.ofdm_measurements.compute_paths import compute_paths
from src.ofdm_measurements.channel_frequency_response import compute_cfr_per_mpc_from_paths


def compute_coverage_map_data(
    scene: Scene,
    cm_metadata: CoverageMapMetadata,
    ignore_cfr_data: bool = False
) -> Tuple[CoverageMapCFRData, CoverageMapPowerData]:
    """Discretize the scene into cells and compute the channel frequency response at each cell"""

    # Set the carrier frequency
    scene.frequency = cm_metadata.frequency

    # Remove all pre-existing receivers
    for rx_name in list(scene.receivers.keys()):
        scene.remove(rx_name)

    # Compute coverage map without noise
    cm_center = cm_metadata.coverage_map_center_with_default(scene=scene)
    cm_orientation = cm_metadata.coverage_map_orientation_with_default()
    cm_size = cm_metadata.coverage_map_size_with_default(scene=scene)
    print("Computing initial coverage map...")
    coverage_map = scene.coverage_map(
        max_depth=cm_metadata.max_depth_path,
        cm_cell_size=cm_metadata.cell_size,
        cm_center=cm_center,
        cm_orientation=cm_orientation,
        cm_size=cm_size,
        combining_vec=None,
        precoding_vec=None,
        num_samples=cm_metadata.num_samples_path
    )

    # Normalize coverage map
    cm_tensor = coverage_map.as_tensor()[cm_metadata.index_transmitter]
    cm_max = tf.reduce_max(cm_tensor)
    cm_tensor /= cm_max

    # Simulate receivers at coverage map cell centers
    # Keep only cells with a given minimal signal strength
    if cm_metadata.rx_min_normalized_power_dbm is None:
        min_power_mask = (cm_tensor > 0.0)  # All positions with signal coverage
    else:
        min_power_val = from_db(cm_metadata.rx_min_normalized_power_dbm, input_dbm=True)
        min_power_mask = (cm_tensor > min_power_val)
    min_power_positions_mask = tf.tile(min_power_mask[:, :, tf.newaxis], [1, 1, 3])
    rx_positions = tf.reshape(
        tf.boolean_mask(coverage_map.cell_centers, min_power_positions_mask),
        [-1, 3]
    )

    n_rx = rx_positions.shape[0]
    n_batch = 1 if cm_metadata.rx_batch_size is None else int(np.ceil(n_rx / cm_metadata.rx_batch_size))
    cfr_per_mpc_all_batches = []
    power_uniform_phase_errors_all_batches = []
    power_rt_simulated_phases_all_batches = []
    print(f"Computing paths for {n_rx} cells in {n_batch} batches")
    n_prints = 10
    prints_freq = max(n_batch // n_prints, 1)
    n_paths_max = 0
    for batch_idx, rx_indexes in enumerate(batch(tf.range(n_rx, dtype=tf.int32), batch_size=cm_metadata.rx_batch_size)):
        # Add receivers
        for rx_idx in rx_indexes:
            scene.add(Receiver(name=f"_cm_rx_{rx_idx}", position=rx_positions[rx_idx]))

        # Compute rays and channel frequency response at each receiver
        paths_batch = compute_paths(
            scene=scene,
            max_depth=cm_metadata.max_depth_path,
            num_samples=cm_metadata.num_samples_path,
            check_scene=False
        )
        cfr_per_mpc_batch = compute_cfr_per_mpc_from_paths(
            paths=paths_batch,
            num_subcarriers=cm_metadata.num_subcarriers,
            subcarrier_spacing=cm_metadata.subcarrier_spacing,
            normalization_constant=1.0
        )
        if not ignore_cfr_data:
            # Store CFR
            cfr_per_mpc_all_batches.append(cfr_per_mpc_batch)
            # Update max number of paths
            n_paths_max = max(n_paths_max, cfr_per_mpc_batch.shape[-1])

        # Compute coverage map average powers (shape [N_RX, N_TX])
        power_uniform_phase_errors = compute_channel_power_uniform_phase_error(cfr_per_mpc_batch)
        power_uniform_phase_errors = reduce_mean_ignore_null(
            power_uniform_phase_errors,
            axis=[2, 3, 4]
        )
        power_uniform_phase_errors_all_batches.append(power_uniform_phase_errors)

        power_rt_simulated_phases = compute_channel_power_rt_simulated_phases(cfr_per_mpc_batch)
        power_rt_simulated_phases = reduce_mean_ignore_null(
            power_rt_simulated_phases,
            axis=[2, 3, 4]
        )
        power_rt_simulated_phases_all_batches.append(power_rt_simulated_phases)

        # Remove receivers
        for rx_idx in rx_indexes:
            scene.remove(name=f"_cm_rx_{rx_idx}")

        # Log progress
        if (batch_idx + 1) % prints_freq == 0:
            print(f"{batch_idx + 1} / {n_batch} batches done")

    # Concatenate receivers CFRs
    if not ignore_cfr_data:
        dims_no_padding = [  # Pad all CFRs with 0 on the path dimension before concatenating
            [0, 0],  # N_RX
            [0, 0],  # N_TX
            [0, 0],  # N_SUBCARRIERS
            [0, 0],  # N_ARR_RX
            [0, 0],  # N_ARR_TX
        ]
        for i in range(len(cfr_per_mpc_all_batches)):
            n_paths = cfr_per_mpc_all_batches[i].shape[-1]
            padding = dims_no_padding + [[0, n_paths_max - n_paths]]
            cfr_per_mpc_all_batches[i] = tf.pad(cfr_per_mpc_all_batches[i], padding, mode="CONSTANT")
        cfr_per_mpc = tf.concat(cfr_per_mpc_all_batches, axis=0)
    else:
        cfr_per_mpc = None

    # Concatenate receivers powers
    uniform_phases_power = tf.concat(power_uniform_phase_errors_all_batches, axis=0)
    simulated_phases_power = tf.concat(power_rt_simulated_phases_all_batches, axis=0)

    # Store and return data
    common_kwargs = dict(
        cm_cell_size=cm_metadata.cell_size,
        cm_center=list(cm_center),
        cm_orientation=list(cm_orientation),
        cm_size=list(cm_size),
        sionna_cm_normalization_constant=cm_max.numpy(),
        sionna_cm_values=coverage_map.as_tensor(),
        cell_centers=coverage_map.cell_centers.numpy(),
        rx_cells_indexes=tf.where(min_power_mask).numpy(),
    )
    cfr_data = CoverageMapCFRData(
        **common_kwargs,
        cfr_per_mpc=cfr_per_mpc
    )
    power_data = CoverageMapPowerData(
        **common_kwargs,
        uniform_phases_power=uniform_phases_power,
        simulated_phases_power=simulated_phases_power
    )

    return cfr_data, power_data
