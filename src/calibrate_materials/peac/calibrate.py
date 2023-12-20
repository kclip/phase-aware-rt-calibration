from IPython.display import clear_output

from typing import Tuple, Union
import numpy as np
import tensorflow as tf
from sionna.rt import Scene

from src.utils.sionna_utils import get_original_and_trainable_materials
from src.data_classes import MaterialsCalibrationInfo, VonMisesCalibrationInfo
from src.ofdm_measurements import _MeasurementMetadataBase
from src.ofdm_measurements.main import compute_paths_traces_from_metadata, compute_selected_cfr_from_paths_traces
from src.calibrate_materials import CalibrateMaterialsPEACMetadata
from src.calibrate_materials.utils import check_material
from src.calibrate_materials.peac.init import init_von_mises_parameters, init_optimizers_peac, \
    init_von_mises_global_concentration_variable, init_trackers_peac
from src.calibrate_materials.peac.free_energy import free_energy_von_mises, expected_nll_normalized
from src.calibrate_materials.peac.von_mises_parameters import get_valid_paths_indices_per_rx_tx_pair, \
    compute_von_mises_posterior_mean, compute_von_mises_posterior_concentration, compute_von_mises_global_concentration


CLIP_GRADS = 1.0


def _print(disp_list):
    clear_output(wait=True)
    print("\n".join(disp_list))


def _backward_pass(
    tape: tf.GradientTape,
    loss: tf.Tensor,
    variables: Union[tf.Variable, Tuple[tf.Variable]],
    optimizer: tf.keras.optimizers.Optimizer
) -> tf.Tensor:
    grads = tape.gradient(loss, variables)
    grads = tf.clip_by_value(grads, -CLIP_GRADS, CLIP_GRADS)
    if isinstance(variables, tf.Variable):
        optimizer.apply_gradients([(grads, variables)])
    else:
        optimizer.apply_gradients(zip(grads, variables))
    return grads


def phase_error_aware_calibration(
    channel_measurements: tf.Tensor,  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX, N_ARR_TX]
    scene: Scene,
    measurement_metadata: _MeasurementMetadataBase,
    calibration_metadata: CalibrateMaterialsPEACMetadata,
    print_freq: int = 1,
    print_von_mises_params: bool = False,
    use_normalised_nll_for_m_step: bool = True
) -> Tuple[MaterialsCalibrationInfo, VonMisesCalibrationInfo]:
    # Init
    # ----
    n_measurements = channel_measurements.shape[0]
    n_rx_tx_pairs = channel_measurements.shape[1]
    measurement_noise_std = tf.constant(calibration_metadata.measurement_noise_std, dtype=tf.float32)

    # Init paths and CFR
    # ------------------
    traced_paths = compute_paths_traces_from_metadata(
        scene=scene,
        measurement_metadata=measurement_metadata,
        check_scene=True
    )
    cfr_train = compute_selected_cfr_from_paths_traces(
        scene=scene,
        traced_paths=traced_paths,
        measurement_metadata=measurement_metadata,
        normalization_constant=calibration_metadata.normalization_constant,
        check_paths=True
    )
    valid_paths_indices = get_valid_paths_indices_per_rx_tx_pair(cfr_per_mpc=cfr_train)

    # Get original and trainable materials
    # ------------------------------------
    original_materials, trainable_materials = get_original_and_trainable_materials(scene)
    materials_conductivity = tuple(mat._conductivity for mat in trainable_materials)
    materials_permittivity = tuple(mat._relative_permittivity for mat in trainable_materials)

    # Init posterior and global/prior von Mises params
    # ------------------------------------------------
    n_paths = cfr_train.shape[-1]
    vm_mean_variables, vm_concentration_variables = init_von_mises_parameters(
        n_measurements=n_measurements,
        n_rx_tx_pairs=n_rx_tx_pairs,
        n_paths=n_paths,
        min_mean=calibration_metadata.init_von_mises_mean_min,
        max_mean=calibration_metadata.init_von_mises_mean_max,
        min_concentration=calibration_metadata.init_von_mises_concentration_min,
        max_concentration=calibration_metadata.init_von_mises_concentration_max,
        amortize_concentration=calibration_metadata.von_mises_amortize_concentration
    )
    vm_global_mean = tf.constant(calibration_metadata.von_mises_global_mean, dtype=tf.float32)
    vm_global_concentration_variable = init_von_mises_global_concentration_variable(
        calibration_metadata.init_von_mises_global_concentration
    )

    # Init optimizers and watched variables
    # -------------------------------------
    optimizers = init_optimizers_peac(calibration_metadata)
    watched_variables_e_step = []
    optimizers_e_step = []
    if calibration_metadata.use_gradient_von_mises_mean:
        watched_variables_e_step.append(vm_mean_variables)
        optimizers_e_step.append(optimizers["von_mises_mean"])
    if calibration_metadata.use_gradient_von_mises_concentration:
        watched_variables_e_step.append(vm_concentration_variables)
        optimizers_e_step.append(optimizers["von_mises_concentration"])
    variable_names_m_step = ["conductivity", "permittivity"]
    watched_variables_m_step = [materials_conductivity, materials_permittivity]
    optimizers_m_step = [optimizers["conductivity"], optimizers["permittivity"]]
    if (
        calibration_metadata.use_gradient_von_mises_global_concentration and
        (not calibration_metadata.von_mises_fixed_prior_concentration)
    ):
        variable_names_m_step.append("von Mises global concentration")
        watched_variables_m_step.append(vm_global_concentration_variable)
        optimizers_m_step.append(optimizers["von_mises_global_concentration"])

    # Init values trackers
    # --------------------
    trackers = init_trackers_peac(
        calibration_metadata=calibration_metadata,
        n_materials=len(trainable_materials),
        von_mises_mean_shape=vm_mean_variables.shape,
        von_mises_concentration_shape=vm_concentration_variables.shape
    )

    display = ["", ""]

    n_e_step = 0
    n_m_step = 0
    loss_iter_m_step = tf.constant(0, dtype=tf.float32)
    bypass_grads_e_step = (
        (not calibration_metadata.use_gradient_von_mises_mean) and
        (not calibration_metadata.use_gradient_von_mises_concentration)
    )
    for step in range(calibration_metadata.n_steps):

        # Expectation step
        # ----------------
        # Analytical updates
        if not calibration_metadata.use_gradient_von_mises_mean:
            vm_mean_variables.assign(
                compute_von_mises_posterior_mean(
                    valid_paths_indices=valid_paths_indices,
                    cfr_per_mpc=cfr_train,
                    channel_measurements=channel_measurements,
                    von_mises_global_concentration=vm_global_concentration_variable,
                    measurement_noise_std=measurement_noise_std
                )
            )
        if not calibration_metadata.use_gradient_von_mises_concentration:
            vm_concentration_variables.assign(
                compute_von_mises_posterior_concentration(
                    valid_paths_indices=valid_paths_indices,
                    cfr_per_mpc=cfr_train,
                    n_measurements=channel_measurements.shape[0],
                    measurement_noise_std=measurement_noise_std,
                    amortize_concentration=calibration_metadata.von_mises_amortize_concentration
                )
            )
        # Gradient-based updates
        for e_iter in range(calibration_metadata.n_iter_e_step):
            grads_e_step = dict()
            if bypass_grads_e_step:
                loss_iter_e_step = loss_iter_m_step
            else:
                # Forward pass
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape_e_step:
                    tape_e_step.watch(watched_variables_e_step)
                    free_energy_iter_e_step = free_energy_von_mises(
                        cfr_train=cfr_train,
                        channel_measurements=channel_measurements,
                        von_mises_mean_params=vm_mean_variables,
                        von_mises_concentration_params=vm_concentration_variables,
                        von_mises_amortize_concentration=calibration_metadata.von_mises_amortize_concentration,
                        measurement_noise_std=measurement_noise_std,
                        von_mises_global_mean=vm_global_mean,
                        von_mises_global_concentration_param=vm_global_concentration_variable,
                    )
                    loss_iter_e_step = tf.reduce_mean(free_energy_iter_e_step)
                # Backward pass
                for variables, optimizer in zip(watched_variables_e_step, optimizers_e_step):
                    grads_e_step[variables.name] = _backward_pass(
                        tape=tape_e_step,
                        loss=loss_iter_e_step,
                        variables=variables,
                        optimizer=optimizer
                    ).numpy().tolist()
            # Track progress
            trackers["losses"][n_e_step + n_m_step] = loss_iter_e_step.numpy()
            trackers["von_mises_means"][n_e_step] = vm_mean_variables.numpy()
            trackers["von_mises_concentrations"][n_e_step] = vm_concentration_variables.numpy()
            # Display info
            display[0] = (
                f"Training step {step}:\n" +
                f"  - E-step loss: {loss_iter_e_step}"
            )
            if print_von_mises_params:
                display[0] += (
                    f"\n  - Gradients: {grads_e_step}\n" +
                    f"  - Von Mises means (mean over measurements): {np.mean(trackers['von_mises_means'][n_e_step], axis=0)}\n" +
                    f"  - Von Mises concentration (mean over measurements): {np.mean(trackers['von_mises_concentrations'][n_e_step], axis=0)}"
                )
            if (n_e_step + n_m_step) % print_freq == 0:
                _print(display)
            # Update index
            n_e_step += 1

        # Maximization step
        # -----------------
        # Analytical updates
        if (
            (not calibration_metadata.use_gradient_von_mises_global_concentration) and
            (not calibration_metadata.von_mises_fixed_prior_concentration)
        ):
            vm_global_concentration_variable.assign(
                compute_von_mises_global_concentration(
                    valid_paths_indices=valid_paths_indices,
                    von_mises_posterior_mean=vm_mean_variables.value(),
                    von_mises_posterior_concentration=vm_concentration_variables.value()
                )
            )
        # Gradient-based updates
        for m_iter in range(calibration_metadata.n_iter_m_step):
            # Forward pass
            with tf.GradientTape(persistent=True) as tape_m_step:
                tape_m_step.watch(watched_variables_m_step)
                cfr_train = compute_selected_cfr_from_paths_traces(
                    scene=scene,
                    traced_paths=traced_paths,
                    measurement_metadata=measurement_metadata,
                    normalization_constant=calibration_metadata.normalization_constant,
                    check_paths=False
                )
                if use_normalised_nll_for_m_step:
                    # Only compute terms of the free-energy that have an impact on the M-step gradient, normalized
                    # by the overall measured channel power
                    free_energy_iter_m_step = expected_nll_normalized(
                        cfr_train=cfr_train,
                        channel_measurements=channel_measurements,
                        von_mises_mean_params=vm_mean_variables,
                        von_mises_concentration_params=vm_concentration_variables,
                        von_mises_amortize_concentration=calibration_metadata.von_mises_amortize_concentration
                    )
                else:
                    # Compute the full free-energy term
                    free_energy_iter_m_step = free_energy_von_mises(
                        cfr_train=cfr_train,
                        channel_measurements=channel_measurements,
                        von_mises_mean_params=vm_mean_variables,
                        von_mises_concentration_params=vm_concentration_variables,
                        von_mises_amortize_concentration=calibration_metadata.von_mises_amortize_concentration,
                        measurement_noise_std=measurement_noise_std,
                        von_mises_global_mean=vm_global_mean,
                        von_mises_global_concentration_param=vm_global_concentration_variable,
                    )

                loss_iter_m_step = tf.reduce_mean(free_energy_iter_m_step)
            # Backward pass
            grads_m_step = dict()
            for variable_name, variables, optimizer in zip(
                variable_names_m_step,
                watched_variables_m_step,
                optimizers_m_step
            ):
                grads_m_step[variable_name] = _backward_pass(
                    tape=tape_m_step,
                    loss=loss_iter_m_step,
                    variables=variables,
                    optimizer=optimizer
                ).numpy().tolist()
            # Check params
            _mat_conductivities = []
            _mat_permittivities = []
            for mat in trainable_materials:
                check_material(mat)
                _mat_conductivities.append(mat.conductivity.numpy())
                _mat_permittivities.append(mat.relative_permittivity.numpy())
            # Track progress
            trackers["losses"][n_e_step + n_m_step] = loss_iter_m_step.numpy()
            trackers["materials_conductivity"][n_m_step] = np.asarray(_mat_conductivities, dtype=np.float32)
            trackers["materials_permittivity"][n_m_step] = np.asarray(_mat_permittivities, dtype=np.float32)
            trackers["von_mises_global_concentration"][n_m_step] = vm_global_concentration_variable.numpy()
            # Display info
            display[1] = (
                f"  - M-step loss: {loss_iter_m_step}\n" +
                f"  - Gradients: {grads_m_step}\n" +
                f"  - Materials conductivity: {_mat_conductivities}\n" +
                f"  - Materials permittivity: {_mat_permittivities}\n" +
                f"  - VM global concentration: {trackers['von_mises_global_concentration'][n_m_step]}"
            )
            if (n_e_step + n_m_step) % print_freq == 0:
                _print(display)
            # Update index
            n_m_step += 1

    # Return training info
    materials_calibration_info = MaterialsCalibrationInfo(
        track_losses=trackers["losses"],
        track_materials_conductivity=trackers["materials_conductivity"],
        track_materials_permittivity=trackers["materials_permittivity"],
        calibrated_materials_conductivity=np.asarray([mat.conductivity.numpy() for mat in trainable_materials]),
        calibrated_materials_permittivity=np.asarray(
            [mat.relative_permittivity.numpy() for mat in trainable_materials]
        ),
        ground_truth_materials_conductivity=np.asarray([mat.conductivity.numpy() for mat in original_materials]),
        ground_truth_materials_permittivity=np.asarray(
            [mat.relative_permittivity.numpy() for mat in original_materials]
        )
    )
    von_mises_calibration_info = VonMisesCalibrationInfo(
        track_von_mises_means=trackers["von_mises_means"],
        track_von_mises_concentrations=trackers["von_mises_concentrations"],
        track_von_mises_global_concentration=trackers["von_mises_global_concentration"],
        von_mises_means=vm_mean_variables.numpy(),
        von_mises_concentrations=vm_concentration_variables.numpy(),
        von_mises_global_concentration=vm_global_concentration_variable.numpy(),
    )
    return materials_calibration_info, von_mises_calibration_info
