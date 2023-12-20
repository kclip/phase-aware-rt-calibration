from dataclasses import dataclass
from typing import List, Tuple


@dataclass()
class CalibrationExperimentConfig(object):
    experiment_version: str

    # Scenario
    scenario_name: str
    n_rx: int
    num_rows_rx_array: int
    num_cols_rx_array: int
    num_rows_tx_array: int
    num_cols_tx_array: int
    
    # Measurement
    num_measurements: int
    num_subcarriers: int
    subcarrier_spacing: float
    measurement_type: str
    measurement_noise_type: str
    
    # Ray tracing
    max_depth_path: int

    # Calibration
    peac_n_steps: int
    peoc_n_steps: int
    upec_n_steps: int

    # Maps
    cm_cell_size: Tuple[float, float]
    cm_num_samples: int

    force_rx_positions: List[List[float]] = None
