from dataclasses import dataclass
from typing import Tuple, List
from sionna.rt import Scene

from src.utils.save_utils import StorableDataclass


@dataclass()
class CoverageMapMetadata(StorableDataclass):
    # Ray tracing
    max_depth_path: int  # Maximum number of ray-bounces
    num_samples_path: int  # Number of launched rays

    # Coverage Map
    frequency: float  # in [Hz]
    cell_size: Tuple[float, float]  # in [m]

    # Simulate receivers only at positions where (normalized) signal strength is above <min_normalized_power_dbm>
    rx_min_normalized_power_dbm: float  # in [dBm], if set to 'None' simulate receivers at all positions with signal
    rx_height: float  # in [m]

    # Channel frequency response
    num_subcarriers: int
    subcarrier_spacing: float  # in multiples of the wavelength

    # Other
    index_transmitter: int = 0  # Index of transmitter for which to compute the coverage map

    # Coverage Map (parameters with default)
    coverage_map_center: List[float] = None  # Shape [2] ; if None, (x, y) center is taken as the origin scene center
    coverage_map_size: List[float] = None  # Shape [2] ; (x, y) size of coverage map ; if None, the whole scene is taken
    coverage_map_orientation: List[float] = None  # Shape [3] ; angles of coverage map orientation

    # Simulate receivers (parameters with default)
    rx_batch_size: int = None  # If None, the CFR is estimated for all Rx at once (can consume a lot of memory)

    def coverage_map_center_with_default(self, scene: Scene) -> List[float]:  # Shape [3]
        _cm_xy_center = (
            self.coverage_map_center
            if self.coverage_map_center is not None else
            scene.center.numpy().tolist()[:2]
        )
        return [*_cm_xy_center, self.rx_height]

    def coverage_map_size_with_default(self, scene: Scene) -> List[float]:  # Shape [2]
        return (
            self.coverage_map_size
            if self.coverage_map_size is not None else
            scene.size.numpy().tolist()[:2]
        )

    def coverage_map_orientation_with_default(self) -> List[float]:  # Shape [3]
        return (
            self.coverage_map_orientation
            if self.coverage_map_orientation is not None else
            [0.0, 0.0, 0.0]
        )
