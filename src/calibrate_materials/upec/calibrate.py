from IPython.display import clear_output

from typing import Tuple

import numpy as np
import tensorflow as tf
from sionna.rt import Scene

from src.utils.sionna_utils import get_original_and_trainable_materials
from src.data_classes import MaterialsCalibrationInfo
from src.optimizer.optimizer import get_optimizer
from src.ofdm_measurements import _MeasurementMetadataBase
from src.ofdm_measurements.main import compute_paths_traces_from_metadata, compute_selected_cfr_from_paths_traces
from src.calibrate_materials import CalibrateMaterialsUPECMetadata
from src.calibrate_materials.utils import check_material
from src.calibrate_materials.upec.get_projections import get_predicted_paths_projections, get_evenly_spaced_projections
from src.calibrate_materials.upec.compute_projections import compute_projected_power_cfr,\
    compute_projected_power_measurements


def _backward_pass(
    tape: tf.GradientTape,
    loss: tf.Tensor,
    variables: Tuple[tf.Variable],
    optimizer: tf.keras.optimizers.Optimizer
) -> tf.Tensor:
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return grads


def uniform_phase_error_calibration(
    channel_measurements: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX]
    scene: Scene,
    measurement_metadata: _MeasurementMetadataBase,
    calibration_metadata: CalibrateMaterialsUPECMetadata,
    print_freq: int = 1
) -> MaterialsCalibrationInfo:
    # Init Optimizers
    optimizer_conductivity = get_optimizer(calibration_metadata.optimizer_conductivity_metadata)
    optimizer_permittivity = get_optimizer(calibration_metadata.optimizer_permittivity_metadata)

    # Init materials
    original_materials, trainable_materials = get_original_and_trainable_materials(scene)
    materials_conductivity = tuple(mat._conductivity for mat in trainable_materials)
    materials_permittivity = tuple(mat._relative_permittivity for mat in trainable_materials)

    # Track values training
    track_losses = np.zeros([calibration_metadata.n_steps], dtype=np.float32)
    track_materials_conductivity = np.zeros([calibration_metadata.n_steps, len(trainable_materials)], dtype=np.float32)
    track_materials_permittivity = np.zeros([calibration_metadata.n_steps, len(trainable_materials)], dtype=np.float32)

    # Get time and angle projections
    if calibration_metadata.paths_projections:
        (
            time_projections,  # Shape [N_RX_TX_PAIRS, N_SUBCARRIERS, N_POINTS_TIME]
            rx_angle_projections,  # Shape [N_RX_TX_PAIRS, N_ARR_RX, N_POINTS_ANGLE]
            tx_angle_projections  # Shape [N_RX_TX_PAIRS, N_ARR_TX, N_POINTS_ANGLE]
        ) = get_predicted_paths_projections(
            scene=scene,
            measurement_metadata=measurement_metadata
        )
    else:
        (
            time_projections,  # Shape [N_RX_TX_PAIRS, N_SUBCARRIERS, N_POINTS_TIME]
            rx_angle_projections,  # Shape [N_RX_TX_PAIRS, N_ARR_RX, N_POINTS_ANGLE]
            tx_angle_projections  # Shape [N_RX_TX_PAIRS, N_ARR_TX, N_POINTS_ANGLE]
        ) = get_evenly_spaced_projections(
            scene=scene,
            measurement_metadata=measurement_metadata,
            calibration_metadata=calibration_metadata
        )

    # Projected measurements power
    # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_POINTS_TIME, N_POINTS_ANGLE, N_POINTS_ANGLE]
    projected_power_measurements = compute_projected_power_measurements(
        channel_measurements=channel_measurements,
        time_projections=time_projections,
        rx_angle_projections=rx_angle_projections,
        tx_angle_projections=tx_angle_projections
    )
    mse_normalizer = tf.reduce_mean(
        tf.pow(projected_power_measurements, 2)
    )

    # Init paths
    traced_paths = compute_paths_traces_from_metadata(
        scene=scene,
        measurement_metadata=measurement_metadata,
        check_scene=True
    )

    for step in range(calibration_metadata.n_steps):
        with tf.GradientTape(persistent=True) as tape:
            # Get CFR projected power under uniform phase error assumption
            cfr_train = compute_selected_cfr_from_paths_traces(
                scene=scene,
                traced_paths=traced_paths,
                measurement_metadata=measurement_metadata,
                normalization_constant=calibration_metadata.normalization_constant,
                check_paths=False
            )
            # Shape [N_MEASUREMENTS=1, N_RX_TX_PAIRS, N_POINTS_TIME, N_POINTS_ANGLE, N_POINTS_ANGLE]
            projected_power_cfr = compute_projected_power_cfr(
                cfr_per_mpc=cfr_train,
                time_projections=time_projections,
                rx_angle_projections=rx_angle_projections,
                tx_angle_projections=tx_angle_projections
            )

            # Compute normalized MSE
            loss = tf.reduce_mean(
                tf.pow(
                    (projected_power_measurements - projected_power_cfr),
                    2
                )
            ) / mse_normalizer

        # Compute gradients and apply through the optimizer
        grads_conductivity = _backward_pass(
            tape=tape,
            loss=loss,
            variables=materials_conductivity,
            optimizer=optimizer_conductivity
        )
        grads_permittivity = _backward_pass(
            tape=tape,
            loss=loss,
            variables=materials_permittivity,
            optimizer=optimizer_permittivity
        )

        for i, mat in enumerate(trainable_materials):
            # Clip undefined material values
            check_material(mat)
            # Track loss/material values
            track_materials_conductivity[step, i] = mat.conductivity.numpy()
            track_materials_permittivity[step, i] = mat.relative_permittivity.numpy()
        track_losses[step] = loss.numpy()

        # Display
        if step % print_freq == 0:
            clear_output(wait=True)
            print(
                f"Training step {step}:\n" +
                f"  - Loss: {loss.numpy()}\n" +
                f"  - Grads Conductivity: {tuple(grad_var.numpy() for grad_var in grads_conductivity)}\n" +
                f"  - Grads Permittivity: {tuple(grad_var.numpy() for grad_var in grads_permittivity)}\n" +
                f"  - Conductivities: {track_materials_conductivity[step, :]}\n" +
                f"  - Permittivities: {track_materials_permittivity[step, :]}"
            )

    return MaterialsCalibrationInfo(
        track_losses=track_losses,
        track_materials_conductivity=track_materials_conductivity,
        track_materials_permittivity=track_materials_permittivity,
        calibrated_materials_conductivity=np.asarray([mat.conductivity.numpy() for mat in trainable_materials]),
        calibrated_materials_permittivity=np.asarray(
            [mat.relative_permittivity.numpy() for mat in trainable_materials]
        ),
        ground_truth_materials_conductivity=np.asarray([mat.conductivity.numpy() for mat in original_materials]),
        ground_truth_materials_permittivity=np.asarray(
            [mat.relative_permittivity.numpy() for mat in original_materials]
        )
    )
