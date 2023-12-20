from study.calibration.configs.experiment_config_dataclass import CalibrationExperimentConfig
from study.calibration.configs.v1 import V1_CALIBRATION_CONFIG
from study.calibration.configs.v2 import V2_CALIBRATION_CONFIG
from study.calibration.configs.v3 import V3_CALIBRATION_CONFIG
from study.calibration.configs.v4 import V4_CALIBRATION_CONFIG


_ALL_CONFIGS_MAP = {
    config.experiment_version: config
    for config in [
        V1_CALIBRATION_CONFIG,
        V2_CALIBRATION_CONFIG,
        V3_CALIBRATION_CONFIG,
        V4_CALIBRATION_CONFIG,
    ]
}


def get_experiment_config_from_version(experiment_version: str) -> CalibrationExperimentConfig:
    if experiment_version not in _ALL_CONFIGS_MAP.keys():
        raise ValueError(f"Unknown experiment version '{experiment_version}'")
    return _ALL_CONFIGS_MAP[experiment_version]
