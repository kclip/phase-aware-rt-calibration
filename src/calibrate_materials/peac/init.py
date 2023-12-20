from typing import Tuple, Dict
import numpy as np
import tensorflow as tf

from src.calibrate_materials import CalibrateMaterialsPEACMetadata
from src.optimizer.optimizer import get_optimizer


# Von Mises Posterior parameters
# ------------------------------

# Keep concentration slightly above 0 to avoid unstable gradient
MIN_VON_MISES_CONCENTRATION = 1e-7


def check_von_mises_concentration_params(von_mises_concentration_params):
    """Von Mises concentrations parameters must be positive"""
    return tf.clip_by_value(
        von_mises_concentration_params,
        clip_value_min=MIN_VON_MISES_CONCENTRATION,
        clip_value_max=von_mises_concentration_params.dtype.max
    )


def init_von_mises_parameters(
    n_measurements: int,
    n_rx_tx_pairs: int,
    n_paths: int,
    min_mean: float = 0.0,
    max_mean: float = 2 * np.pi,
    min_concentration: float = 1.0,
    max_concentration: float = 5.0,
    amortize_concentration: bool = False
) -> Tuple[
    tf.Variable,  # Mean params [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    tf.Variable  # Concentration params [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS] or [1]
]:
    shape = (n_measurements, n_rx_tx_pairs, n_paths)
    means = tf.random.uniform(
        shape=shape,
        minval=min_mean,
        maxval=max_mean,
        dtype=tf.float32
    )
    concentrations = tf.random.uniform(
        shape=[] if amortize_concentration else shape,
        minval=min_concentration,
        maxval=max_concentration,
        dtype=tf.float32
    )

    return (
        tf.Variable(means, trainable=False, name="Von Mises Posterior Mean"),
        tf.Variable(
            concentrations,
            trainable=False,
            constraint=check_von_mises_concentration_params,
            name="Von Mises Posterior Concentration"
        )
    )


# Von Mises Prior parameters
# ------------------------------

def init_von_mises_global_concentration_variable(init_von_mises_global_concentration: float) -> tf.Variable:
    return tf.Variable(
        init_von_mises_global_concentration,
        trainable=False,
        constraint=check_von_mises_concentration_params,
        name="Von Mises Prior Concentration"
    )


# Optimizers
# ----------

def init_optimizers_peac(
    calibration_metadata: CalibrateMaterialsPEACMetadata
) -> Dict[str, tf.keras.optimizers.Optimizer]:
    all_optimizers_metadata = [
        ("von_mises_mean", calibration_metadata.optimizer_von_mises_mean_metadata),
        ("von_mises_concentration", calibration_metadata.optimizer_von_mises_concentration_metadata),
        ("conductivity", calibration_metadata.optimizer_conductivity_metadata),
        ("permittivity", calibration_metadata.optimizer_permittivity_metadata),
        ("von_mises_global_concentration", calibration_metadata.optimizer_von_mises_global_concentration_metadata)
    ]
    optimizers = dict()
    for optimizer_name, optimizer_metadata in all_optimizers_metadata:
        if optimizer_metadata is not None:
            optimizers[optimizer_name] = get_optimizer(optimizer_metadata)
    return optimizers


# Trackers
# --------

def init_trackers_peac(
    calibration_metadata: CalibrateMaterialsPEACMetadata,
    n_materials: int,
    von_mises_mean_shape: tuple,
    von_mises_concentration_shape: tuple,
) -> Dict[str, np.ndarray]:
    total_e_steps = calibration_metadata.n_steps * calibration_metadata.n_iter_e_step
    total_m_steps = calibration_metadata.n_steps * calibration_metadata.n_iter_m_step
    trackers = dict()
    trackers["losses"] = np.zeros([total_e_steps + total_m_steps], dtype=np.float32)
    trackers["materials_conductivity"] = np.zeros([total_m_steps, n_materials], dtype=np.float32)
    trackers["materials_permittivity"] = np.zeros([total_m_steps, n_materials], dtype=np.float32)
    trackers["von_mises_global_concentration"] = np.zeros([total_m_steps], dtype=np.float32)
    trackers["von_mises_means"] = np.zeros([total_e_steps, *von_mises_mean_shape], dtype=np.float32)
    trackers["von_mises_concentrations"] = np.zeros([total_e_steps, *von_mises_concentration_shape], dtype=np.float32)

    return trackers
