from dataclasses import dataclass

from src.utils.save_utils import StorableDataclass
from src.coverage_map import CoverageMapMetadata
from src.ofdm_measurements import _MeasurementMetadataBase
from src.calibrate_materials import _CalibrateMaterialsMetadataBase
from src.scenarios import _ScenarioMetadataBase


@dataclass()
class Metadata(StorableDataclass):
    __filename__ = "metadata"

    measurement_metadata: _MeasurementMetadataBase = None
    calibrate_materials_metadata: _CalibrateMaterialsMetadataBase = None
    scenario_metadata: _ScenarioMetadataBase = None
    coverage_map_metadata: CoverageMapMetadata = None
