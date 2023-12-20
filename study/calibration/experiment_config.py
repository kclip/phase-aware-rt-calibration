from typing import Union
import numpy as np
from typing import List

from settings import STUDY_EXPERIMENT_VERSION
from src.data_classes import MeasurementType, MeasurementNoiseType
from src.scenarios import ScenarioTheStrandMetadata, ScenarioToyExampleMetadata, ScenarioToyExampleMaxwellMetadata
from src.coverage_map import CoverageMapMetadata
from src.ofdm_measurements import MeasurementPositionNoiseMetadata, MeasurementWithoutPhaseNoiseMetadata, \
    MeasurementVonMisesPhasePhaseNoiseMetadata, MeasurementMaxwellSimulationMetadata
from src.calibrate_materials import CalibrateMaterialsPEOCMetadata, CalibrateMaterialsPEACMetadata, \
    CalibrateMaterialsUPECMetadata
from src.optimizer import OptimizerAdamMetadata, SchedulerConstantMetadata
from study.calibration.configs import get_experiment_config_from_version


EXPERIMENT_NAME = f"calibration_{STUDY_EXPERIMENT_VERSION}"

_EXPERIMENT_CONFIG = get_experiment_config_from_version(STUDY_EXPERIMENT_VERSION)

# Constants
# ---------
SEED = None  # Affects seed of measurement and phase noise

# Scenario
SCENARIO_NAME = _EXPERIMENT_CONFIG.scenario_name
N_RX = _EXPERIMENT_CONFIG.n_rx
FORCE_RX_POSITIONS = _EXPERIMENT_CONFIG.force_rx_positions
NUM_ROWS_RX_ARRAY = _EXPERIMENT_CONFIG.num_rows_rx_array
NUM_COLS_RX_ARRAY = _EXPERIMENT_CONFIG.num_cols_rx_array
NUM_ROWS_TX_ARRAY = _EXPERIMENT_CONFIG.num_rows_tx_array
NUM_COLS_TX_ARRAY = _EXPERIMENT_CONFIG.num_cols_tx_array
SPACING_ARRAY_ELEMENTS = 0.5  # in multiples of the carrier wavelength

# Ray tracing
MAX_DEPTH_PATH = _EXPERIMENT_CONFIG.max_depth_path
NUM_SAMPLES_PATH = int(1e6)

# Measurements
NUM_MEASUREMENTS = _EXPERIMENT_CONFIG.num_measurements
CARRIER_FREQUENCY = 6e9  # in Hz
NUM_SUBCARRIERS = _EXPERIMENT_CONFIG.num_subcarriers
SUBCARRIER_SPACING = _EXPERIMENT_CONFIG.subcarrier_spacing
NORMALIZE_MEASUREMENTS = True
MEASUREMENT_TYPE = _EXPERIMENT_CONFIG.measurement_type
MEASURMENT_NOISE_TYPE = _EXPERIMENT_CONFIG.measurement_noise_type

# Phase Error-Aware Calibration
PEAC_AMORTIZE_CONCENTRATION = False
PEAC_N_STEPS = _EXPERIMENT_CONFIG.peac_n_steps
PEAC_N_ITER_E_STEP = 1
PEAC_N_ITER_M_STEP = 1
PEAC_BETA_1 = 0.9
PEAC_BETA_2 = 0.999
PEAC_LR_CONDUCTIVITY = 0.15
PEAC_LR_PERMITTIVITY = 0.15
# For PEAC calibration with fixed prior
PEAC_FIXED_PRIOR_INIT_PRIOR_CONCENTRATION = 0.0

# Phase Error-Oblivious Calibration
PEOC_CAL_N_STEPS = _EXPERIMENT_CONFIG.peoc_n_steps
PEOC_BETA_1 = 0.9
PEOC_BETA_2 = 0.999
PEOC_LR_CONDUCTIVITY = 0.15
PEOC_LR_PERMITTIVITY = 0.15
PEOC_PRINT_FREQ = 5

# Uniform Phase Error Calibration
UPEC_N_STEPS = _EXPERIMENT_CONFIG.upec_n_steps
UPEC_BETA_1 = 0.9
UPEC_BETA_2 = 0.999
UPEC_LR_CONDUCTIVITY = 0.15
UPEC_LR_PERMITTIVITY = 0.15
UPEC_PRINT_FREQ = 5

# Coverage map
CM_CELL_SIZE = _EXPERIMENT_CONFIG.cm_cell_size
CM_NUM_SAMPLES = _EXPERIMENT_CONFIG.cm_num_samples
CM_RX_MIN_NORMALIZED_POWER_DBM = None
CM_RX_HEIGHT = 1.0  # in [m]
CM_NUM_SUBCARRIERS = 1
CM_SIZE = [600, 600]  # Enough to capture all buildings, can be made smaller for zoomed camera
CM_RX_BATCH_SIZE = 300

# Print progress
PRINT_VM_PARAMS = False
PRINT_FREQ = 1 if "light" in STUDY_EXPERIMENT_VERSION else (PEAC_N_ITER_E_STEP + PEAC_N_ITER_M_STEP) * 5

# Plot params
PLOT_QUANTILE_METHOD = "linear"
PLOT_QUANTILE_VALUE = 0.75
PLOT_CM_SHOW_DEVICES = True
PLOT_CM_VMAX = 0
PLOT_CM_VMIN = -20
PLOT_CM_NUM_SAMPLES = 512
PLOT_RENDER_HIGH_RES = (2000, 1600)  # Resolution of scenario rendering


# Scenario Metadata
# -----------------

def get_scenario_metadata(ground_truth_geometry: bool, array_config_name: str = "mimo"):
    # Number of antennas
    antenna_kwargs = dict()
    if array_config_name == "mimo":
        antenna_kwargs.update(dict(
            num_rows_rx_array=NUM_ROWS_RX_ARRAY,
            num_cols_rx_array=NUM_COLS_RX_ARRAY,
            num_rows_tx_array=NUM_ROWS_TX_ARRAY,
            num_cols_tx_array=NUM_COLS_TX_ARRAY
        ))
    elif array_config_name == "miso":
        antenna_kwargs.update(dict(
            num_rows_rx_array=1,
            num_cols_rx_array=1,
            num_rows_tx_array=NUM_ROWS_TX_ARRAY,
            num_cols_tx_array=NUM_COLS_TX_ARRAY
        ))
    elif array_config_name == "siso":
        antenna_kwargs.update(dict(
            num_rows_rx_array=1,
            num_cols_rx_array=1,
            num_rows_tx_array=NUM_ROWS_TX_ARRAY,
            num_cols_tx_array=NUM_COLS_TX_ARRAY
        ))
    else:
        raise ValueError(f"Unknown array config '{array_config_name}'")
    
    # Scenario
    if SCENARIO_NAME == ScenarioTheStrandMetadata.__scenario_name__:
        # Note: all "the_strand" scenarios use the ground-truth geometry
        return ScenarioTheStrandMetadata(
            carrier_frequency=CARRIER_FREQUENCY,
            nb_receivers=N_RX,
            force_rx_positions=FORCE_RX_POSITIONS,
            **antenna_kwargs
        )
    elif SCENARIO_NAME == ScenarioToyExampleMetadata.__scenario_name__:
        return ScenarioToyExampleMetadata(
            load_ground_truth=ground_truth_geometry,
            num_cols_rx_array=antenna_kwargs["num_cols_rx_array"],
            num_cols_tx_array=antenna_kwargs["num_cols_tx_array"]
        )
    elif SCENARIO_NAME == ScenarioToyExampleMaxwellMetadata.__scenario_name__:
        return ScenarioToyExampleMaxwellMetadata(
            load_ground_truth=ground_truth_geometry
        )
    else:
        raise ValueError(f"Unknown scenario name '{SCENARIO_NAME}'")


# Coverage Map Metadata
# ---------------------

coverage_map_metadata = CoverageMapMetadata(
    max_depth_path=MAX_DEPTH_PATH,
    num_samples_path=CM_NUM_SAMPLES,
    cell_size=CM_CELL_SIZE,
    rx_min_normalized_power_dbm=CM_RX_MIN_NORMALIZED_POWER_DBM,
    rx_height=CM_RX_HEIGHT,
    frequency=CARRIER_FREQUENCY,
    num_subcarriers=CM_NUM_SUBCARRIERS,
    subcarrier_spacing=SUBCARRIER_SPACING,
    coverage_map_size=CM_SIZE,
    rx_batch_size=CM_RX_BATCH_SIZE
)


# Measurement Metadata
# --------------------

def get_rx_tx_indexes(n_rx: int) -> List[tuple]:
    return [(i, 0) for i in range(n_rx)]


def get_num_subcarriers_from_bandwidth(
    bandwidth: float,
    subcarrier_spacing: float,
    default_num_subcarriers: int
) -> int:
    if bandwidth is None:  # Default (no bandwidth specified)
        return default_num_subcarriers
    return int(np.ceil(bandwidth / subcarrier_spacing))


def get_measurement_metadata(
    measurement_snr: float,
    measurement_additive_noise: bool,
    measurement_perfect_phase: bool,
    measurement_von_mises_mean: float,
    measurement_von_mises_concentration: float,
    bandwidth: float,
    position_noise_amplitude: float = None
) -> Union[
    MeasurementVonMisesPhasePhaseNoiseMetadata,
    MeasurementWithoutPhaseNoiseMetadata,
    MeasurementPositionNoiseMetadata,
    MeasurementMaxwellSimulationMetadata
]:
    num_subcarriers = get_num_subcarriers_from_bandwidth(
        bandwidth=bandwidth,
        default_num_subcarriers=NUM_SUBCARRIERS,
        subcarrier_spacing=SUBCARRIER_SPACING,
    )

    common_params = dict(
        # Ray tracer
        max_depth_path=MAX_DEPTH_PATH,
        num_samples_path=NUM_SAMPLES_PATH,
        # Channel Frequency Response
        num_subcarriers=num_subcarriers,
        subcarrier_spacing=SUBCARRIER_SPACING,
        rx_tx_indexes=get_rx_tx_indexes(n_rx=N_RX),
        # Measurements
        n_measurements_per_channel=NUM_MEASUREMENTS,
        with_additive_noise=measurement_additive_noise,
        normalize=NORMALIZE_MEASUREMENTS,
        measurement_snr=measurement_snr,
        seed=SEED,
    )
    if MEASUREMENT_TYPE == MeasurementType.RAY_TRACING:
        if MEASURMENT_NOISE_TYPE == MeasurementNoiseType.PHASE:
            if measurement_perfect_phase:
                return MeasurementWithoutPhaseNoiseMetadata(**common_params)
            else:
                return MeasurementVonMisesPhasePhaseNoiseMetadata(
                    **common_params,
                    von_mises_prior_mean=measurement_von_mises_mean,
                    von_mises_prior_concentration=measurement_von_mises_concentration
                )
        elif MEASURMENT_NOISE_TYPE == MeasurementNoiseType.POSITION:
            return MeasurementPositionNoiseMetadata(
                **common_params,
                displace_rx=True,
                displace_tx=False,
                displacement_amplitude=position_noise_amplitude
            )
        else:
            raise ValueError(f"Unknown measurement noise type '{MEASURMENT_NOISE_TYPE}'...")
    elif MEASUREMENT_TYPE == MeasurementType.MAXWELL_SIMULATION:
        return MeasurementMaxwellSimulationMetadata(**common_params)
    else:
        raise ValueError(f"Unknown measurement type '{MEASUREMENT_TYPE}'...")


# Calibration Metadata
# --------------------

# PEOC
# ====

def get_peoc_calibration_metadata(normalization_constant: float) -> CalibrateMaterialsPEOCMetadata:
    return CalibrateMaterialsPEOCMetadata(
        normalization_constant=normalization_constant,
        n_steps=PEOC_CAL_N_STEPS,
        optimizer_conductivity_metadata=OptimizerAdamMetadata(
            beta_1=PEOC_BETA_1,
            beta_2=PEOC_BETA_2,
            scheduler_metadata=SchedulerConstantMetadata(PEOC_LR_CONDUCTIVITY)
        ),
        optimizer_permittivity_metadata=OptimizerAdamMetadata(
            beta_1=PEOC_BETA_1,
            beta_2=PEOC_BETA_2,
            scheduler_metadata=SchedulerConstantMetadata(PEOC_LR_PERMITTIVITY)
        )
    )


# UPEC
# ====

def _get_upec_calibration_metadata_base(
    normalization_constant: float,
    paths_projections: bool
) -> CalibrateMaterialsUPECMetadata:
    return CalibrateMaterialsUPECMetadata(
        normalization_constant=normalization_constant,
        paths_projections=paths_projections,
        num_rows_rx_array=NUM_ROWS_RX_ARRAY,
        num_cols_rx_array=NUM_COLS_RX_ARRAY,
        num_rows_tx_array=NUM_ROWS_TX_ARRAY,
        num_cols_tx_array=NUM_COLS_TX_ARRAY,
        spacing_array_elements=SPACING_ARRAY_ELEMENTS,
        n_steps=UPEC_N_STEPS,
        optimizer_conductivity_metadata=OptimizerAdamMetadata(
            beta_1=UPEC_BETA_1,
            beta_2=UPEC_BETA_2,
            scheduler_metadata=SchedulerConstantMetadata(UPEC_LR_CONDUCTIVITY)
        ),
        optimizer_permittivity_metadata=OptimizerAdamMetadata(
            beta_1=UPEC_BETA_1,
            beta_2=UPEC_BETA_2,
            scheduler_metadata=SchedulerConstantMetadata(UPEC_LR_PERMITTIVITY)
        )
    )


def get_upec_calibration_metadata(normalization_constant: float) -> CalibrateMaterialsUPECMetadata:
    return _get_upec_calibration_metadata_base(
        normalization_constant=normalization_constant,
        paths_projections=False
    )


def get_upec_paths_proj_calibration_metadata(normalization_constant: float) -> CalibrateMaterialsUPECMetadata:
    return _get_upec_calibration_metadata_base(
        normalization_constant=normalization_constant,
        paths_projections=True
    )


# PEAC
# ====

def _get_peac_calibration_metadata_base(
    normalization_constant: float,
    normalized_measurement_noise_std: List[float],  # Shape [N_RX_TX_PAIR]
    von_mises_fixed_prior_concentration: bool,
    init_von_mises_global_concentration: float
):
    optimizer_conductivity_metadata = OptimizerAdamMetadata(
        beta_1=PEAC_BETA_1,
        beta_2=PEAC_BETA_2,
        scheduler_metadata=SchedulerConstantMetadata(PEAC_LR_CONDUCTIVITY)
    )
    optimizer_permittivity_metadata = OptimizerAdamMetadata(
        beta_1=PEAC_BETA_1,
        beta_2=PEAC_BETA_2,
        scheduler_metadata=SchedulerConstantMetadata(PEAC_LR_PERMITTIVITY)
    )

    return CalibrateMaterialsPEACMetadata(
        normalization_constant=normalization_constant,
        measurement_noise_std=normalized_measurement_noise_std,
        # Training params
        n_steps=PEAC_N_STEPS,
        n_iter_e_step=PEAC_N_ITER_E_STEP,
        n_iter_m_step=PEAC_N_ITER_M_STEP,
        von_mises_amortize_concentration=PEAC_AMORTIZE_CONCENTRATION,
        # Optimizers
        optimizer_conductivity_metadata=optimizer_conductivity_metadata,
        optimizer_permittivity_metadata=optimizer_permittivity_metadata,
        # Prior concentration
        von_mises_fixed_prior_concentration=von_mises_fixed_prior_concentration,
        init_von_mises_global_concentration=init_von_mises_global_concentration
    )


def get_peac_calibration_metadata(
    normalization_constant: float,
    normalized_measurement_noise_std: List[float]  # Shape [N_RX_TX_PAIR]
):
    return _get_peac_calibration_metadata_base(
        normalization_constant=normalization_constant,
        normalized_measurement_noise_std=normalized_measurement_noise_std,
        # Prior concentration
        von_mises_fixed_prior_concentration=False,
        init_von_mises_global_concentration=1.0
    )


def get_peac_fixed_prior_calibration_metadata(
    normalization_constant: float,
    normalized_measurement_noise_std: List[float]  # Shape [N_RX_TX_PAIR]
):
    return _get_peac_calibration_metadata_base(
        normalization_constant=normalization_constant,
        normalized_measurement_noise_std=normalized_measurement_noise_std,
        # Prior concentration
        von_mises_fixed_prior_concentration=True,
        init_von_mises_global_concentration=PEAC_FIXED_PRIOR_INIT_PRIOR_CONCENTRATION
    )
