from src.data_classes import MeasurementType
from study.calibration.configs.experiment_config_dataclass import CalibrationExperimentConfig


# Config for calibration experiments with different bandwidths

V4_CALIBRATION_CONFIG = CalibrationExperimentConfig(
    experiment_version="v4",
    scenario_name="toy_example_maxwell",
    n_rx=2,  # 2 out of 3 available RXs are used for calibration
    num_rows_rx_array=1,
    num_cols_rx_array=1,
    num_rows_tx_array=1,
    num_cols_tx_array=1,
    num_measurements=50,
    num_subcarriers=64,
    subcarrier_spacing=30e3,  # 30kHz
    measurement_type=MeasurementType.MAXWELL_SIMULATION,
    measurement_noise_type=None,
    max_depth_path=3,
    peac_n_steps=800,
    peoc_n_steps=200,
    upec_n_steps=800,
    cm_cell_size=(1.0, 1.0),
    cm_num_samples=int(1e6),
    # Useless params
    force_rx_positions=None,
)
