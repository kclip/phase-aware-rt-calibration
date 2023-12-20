from src.data_classes import MeasurementType, MeasurementNoiseType
from study.calibration.configs.experiment_config_dataclass import CalibrationExperimentConfig


# Config for calibration experiments with different bandwidths

V2_CALIBRATION_CONFIG = CalibrationExperimentConfig(
    experiment_version="v2",
    scenario_name="toy_example",
    num_rows_rx_array=1,
    num_cols_rx_array=1,
    num_rows_tx_array=1,
    num_cols_tx_array=1,
    num_measurements=50,
    num_subcarriers=64,
    subcarrier_spacing=30e3,  # 30kHz
    measurement_type=MeasurementType.RAY_TRACING,
    measurement_noise_type=MeasurementNoiseType.PHASE,
    max_depth_path=1,
    peac_n_steps=800,
    peoc_n_steps=200,
    upec_n_steps=800,
    cm_cell_size=(1.0, 1.0),
    cm_num_samples=int(1e6),
    # Useless params
    n_rx=1,
    force_rx_positions=[[-78.09, -56.81, 1.]],
)
