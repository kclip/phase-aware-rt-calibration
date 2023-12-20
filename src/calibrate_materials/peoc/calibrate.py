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
from src.calibrate_materials import CalibrateMaterialsPEOCMetadata
from src.calibrate_materials.utils import check_material
from src.calibrate_materials.peoc.least_squares_loss import least_squares_loss


def _backward_pass(
    tape: tf.GradientTape,
    loss: tf.Tensor,
    variables: Tuple[tf.Variable],
    optimizer: tf.keras.optimizers.Optimizer
) -> tf.Tensor:
    grads = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return grads


def phase_error_oblivious_calibration(
    channel_measurements: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX]
    scene: Scene,
    measurement_metadata: _MeasurementMetadataBase,
    calibration_metadata: CalibrateMaterialsPEOCMetadata,
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

    # Init paths
    traced_paths = compute_paths_traces_from_metadata(
        scene=scene,
        measurement_metadata=measurement_metadata,
        check_scene=True
    )

    for step in range(calibration_metadata.n_steps):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([materials_conductivity, materials_permittivity])
            cfr_train = compute_selected_cfr_from_paths_traces(
                scene=scene,
                traced_paths=traced_paths,
                measurement_metadata=measurement_metadata,
                normalization_constant=calibration_metadata.normalization_constant,
                check_paths=False
            )
            loss = least_squares_loss(
                cfr_train=cfr_train,
                channel_measurements=channel_measurements
            )

        # Compute gradients and apply gradients through the optimizer
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
