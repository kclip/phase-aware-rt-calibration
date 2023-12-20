import tensorflow as tf

from src.data_classes import MeasurementPhaseNoiseType
from src.utils.tensor_utils import sample_circular_uniform_tensor, sample_circular_von_mises_tensor, \
    sample_complex_standard_normal_tensor, cast_to_pure_imag


def simulate_phase_noise_per_mpc(
    n_measurements: int,  # Number of measurements per (Rx, Tx) pair
    n_rx_tx_pairs: int,  # Number of (Rx, Tx) pairs
    n_paths: int,  # Maximal number of paths across all considered (Rx, Tx) pairs
    component_noise_type: str,  # See <MeasurementPhaseNoiseType> class
    # Von Mises parameters for MeasurementPhaseNoiseType.VON_MISES_PHASE component noise
    von_mises_prior_mean: float = None,  # in [-pi, pi[
    von_mises_prior_concentration: float = None,  # greater than 0
    # Specify the component noise phases manually for each path and measurement (MeasurementPhaseNoiseType.MANUAL_PHASE)
    manual_phases: tf.Tensor = None,  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATH]
    seed: float = None
) -> tf.Tensor:  # Phase noise per path; shape [N_MEASURMENTS, N_RX_TX_PAIRS, N_PATH]
    """
    Generate a set of <n_measurements> phase noise samples for each (Rx, Tx) pair and each path independently.
    The generated phase noise terms can be used to generate synthetic channel observations exhibiting errors between
    the RT-predicted phases and the ground-truth path phases during measurements.
    """
    # Validate input
    if (component_noise_type == MeasurementPhaseNoiseType.VON_MISES_PHASE) and (
        (von_mises_prior_mean is None) or (von_mises_prior_concentration is None)
    ):
        raise ValueError("Von Mises component noise type used but prior von Mises parameter were not provided")

    # Multipath components phase noise
    component_noise_shape = (n_measurements, n_rx_tx_pairs, n_paths)
    if component_noise_type == MeasurementPhaseNoiseType.PERFECT_PHASE:
        return tf.ones(component_noise_shape, dtype=tf.complex64)
    elif component_noise_type == MeasurementPhaseNoiseType.UNIFORM_PHASE:
        return sample_circular_uniform_tensor(component_noise_shape, seed=seed)
    elif component_noise_type == MeasurementPhaseNoiseType.VON_MISES_PHASE:
        return sample_circular_von_mises_tensor(
            component_noise_shape,
            mean=von_mises_prior_mean,
            concentration=von_mises_prior_concentration,
            seed=seed
        )
    elif component_noise_type == MeasurementPhaseNoiseType.DENSE_COMPONENTS:
        return sample_complex_standard_normal_tensor(
            shape=component_noise_shape,
            std=1.0,
            seed=seed
        )
    elif component_noise_type == MeasurementPhaseNoiseType.MANUAL_PHASE:
        return tf.math.exp(cast_to_pure_imag(manual_phases))
    else:
        raise ValueError(f"Unknown component noise type '{component_noise_type}'...")
