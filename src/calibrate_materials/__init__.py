from typing import List, Tuple
from dataclasses import dataclass
import numpy as np

from src.data_classes import StorableDataclass
from src.optimizer import _OptimizerMetadataBase


@dataclass()
class _CalibrateMaterialsMetadataBase(StorableDataclass):
    __calibration_method__ = None
    seed = None

    normalization_constant: float  # Normalization constant of measured data


@dataclass()
class CalibrateMaterialsPEOCMetadata(_CalibrateMaterialsMetadataBase):
    __calibration_method__ = "phase_error_oblivious_calibration"

    n_steps: int
    optimizer_conductivity_metadata: _OptimizerMetadataBase
    optimizer_permittivity_metadata: _OptimizerMetadataBase


@dataclass()
class CalibrateMaterialsUPECMetadata(_CalibrateMaterialsMetadataBase):
    __calibration_method__ = "uniform_phase_error_calibration"

    # If True, get projections as predicted angles/times of arrival ; else sample projection points evenly, with
    # steps between points equal to the resolution of the system in angle/time domain
    paths_projections: bool

    # Angular projection
    num_rows_rx_array: int
    num_cols_rx_array: int
    num_rows_tx_array: int
    num_cols_tx_array: int
    spacing_array_elements: float  # in multiples of the carrier wavelength

    # Optimization
    n_steps: int
    optimizer_conductivity_metadata: _OptimizerMetadataBase
    optimizer_permittivity_metadata: _OptimizerMetadataBase


@dataclass()
class CalibrateMaterialsPEACMetadata(_CalibrateMaterialsMetadataBase):
    __calibration_method__ = "phase_error_aware_calibration"

    measurement_noise_std: List[float]  # Normalized measurement noise std ; shape [N_RX_TX_PAIR]

    # Training params
    # ---------------
    n_steps: int
    n_iter_e_step: int
    n_iter_m_step: int
    optimizer_conductivity_metadata: _OptimizerMetadataBase
    optimizer_permittivity_metadata: _OptimizerMetadataBase
    # Prior/Global von Mises distribution
    von_mises_global_mean: float = 0.0
    # One concentration parameter per latent variable, or amortized concentration (1 parameter for all variables)
    von_mises_amortize_concentration: bool = False

    # Fixed Prior Concentration
    init_von_mises_global_concentration: float = 0.0
    von_mises_fixed_prior_concentration: bool = False

    # Gradient-based von Mises parameters optimization (Legacy)
    # ------------------------------------------------
    # Prior/Global concentration (M-step)
    use_gradient_von_mises_global_concentration: bool = False
    optimizer_von_mises_global_concentration_metadata: _OptimizerMetadataBase = None
    # Posterior von Mises mean (E-step)
    use_gradient_von_mises_mean: bool = False
    init_von_mises_mean_min: float = 0.0
    init_von_mises_mean_max: float = 2 * np.pi
    optimizer_von_mises_mean_metadata: _OptimizerMetadataBase = None
    # Posterior von Mises concentration (E-step)
    use_gradient_von_mises_concentration: bool = False
    init_von_mises_concentration_min: float = 1.0
    init_von_mises_concentration_max: float = 2.0
    optimizer_von_mises_concentration_metadata: _OptimizerMetadataBase = None
