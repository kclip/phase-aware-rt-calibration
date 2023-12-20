from dataclasses import dataclass
from typing import List, Tuple
import tensorflow as tf

from src.data_classes import MeasurementType, MeasurementNoiseType, MeasurementPhaseNoiseType, StorableDataclass


@dataclass()
class _MeasurementMetadataBase(StorableDataclass):
    __measurement_type__ = None

    # Ray tracing
    max_depth_path: int  # Maximum number of ray-bounces
    num_samples_path: int  # Number of launched rays
    
    # Channel Frequency Response
    num_subcarriers: int
    subcarrier_spacing: float  # in [Hz]
    rx_tx_indexes: List[Tuple[int]]  # List device tuples (rx_idx, tx_idx) to measure

    # Measurements
    n_measurements_per_channel: int  # Number of samples per (Rx, Tx) pair/channel
    with_additive_noise: bool  # Additive noise due to measurement/model mismatch
    normalize: bool  # Normalize measurements by average power
    measurement_snr: float

    seed: float


@dataclass()
class MeasurementMaxwellSimulationMetadata(_MeasurementMetadataBase):
    __measurement_type__ = MeasurementType.MAXWELL_SIMULATION


@dataclass()
class _MeasurementRayTracingMetadataBase(_MeasurementMetadataBase):
    __measurement_type__ = MeasurementType.RAY_TRACING
    __noise_type__ = None  # See <MeasurementNoiseType> class


@dataclass()
class MeasurementPositionNoiseMetadata(_MeasurementRayTracingMetadataBase):
    __noise_type__ = MeasurementNoiseType.POSITION

    displace_rx: bool  # Add noise/random displacements to receivers' positions
    displace_tx: bool  # Add noise/random displacements to transmitters' positions
    displacement_amplitude: float  # In multiple of the wavelength; amplitude of the random displacement direction


@dataclass()
class _MeasurementPhaseNoiseMetadataBase(_MeasurementRayTracingMetadataBase):
    __noise_type__ = MeasurementNoiseType.PHASE
    __component_noise_type__ = None  # See <MeasurementPhaseNoiseType> class

@dataclass()
class MeasurementWithoutPhaseNoiseMetadata(_MeasurementPhaseNoiseMetadataBase):
    __component_noise_type__ = MeasurementPhaseNoiseType.PERFECT_PHASE


@dataclass()
class MeasurementDMCPhaseNoiseMetadata(_MeasurementPhaseNoiseMetadataBase):
    __component_noise_type__ = MeasurementPhaseNoiseType.DENSE_COMPONENTS


@dataclass()
class MeasurementUniformPhasePhaseNoiseMetadata(_MeasurementPhaseNoiseMetadataBase):
    __component_noise_type__ = MeasurementPhaseNoiseType.UNIFORM_PHASE


@dataclass()
class MeasurementVonMisesPhasePhaseNoiseMetadata(_MeasurementPhaseNoiseMetadataBase):
    __component_noise_type__ = MeasurementPhaseNoiseType.VON_MISES_PHASE

    von_mises_prior_mean: float  # in [0, 2 pi[
    von_mises_prior_concentration: float  # must be greater than 0


@dataclass()
class MeasurementManualPhasePhaseNoiseMetadata(_MeasurementPhaseNoiseMetadataBase):
    __component_noise_type__ = MeasurementPhaseNoiseType.MANUAL_PHASE

    manual_phases: tf.Tensor  # Shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATH]
