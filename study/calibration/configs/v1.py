from src.data_classes import MeasurementType, MeasurementNoiseType
from study.calibration.configs.experiment_config_dataclass import CalibrationExperimentConfig


# Config for calibration experiments for different SNRs and von Mises std.

V1_CALIBRATION_CONFIG = CalibrationExperimentConfig(
    experiment_version="v1",
    scenario_name="the_strand",
    n_rx=1,
    force_rx_positions=[[-78.09, -56.81, 1.]],
    num_rows_rx_array=8,
    num_cols_rx_array=8,
    num_rows_tx_array=8,
    num_cols_tx_array=8,
    num_measurements=50,
    num_subcarriers=64,
    subcarrier_spacing=30e3,
    measurement_type=MeasurementType.RAY_TRACING,
    measurement_noise_type=MeasurementNoiseType.PHASE,
    max_depth_path=10,
    peac_n_steps=900,
    peoc_n_steps=200,
    upec_n_steps=800,
    cm_cell_size=(5.0, 5.0),
    cm_num_samples=int(1e6)
)
