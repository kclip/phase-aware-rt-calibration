import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import tensorflow as tf
from sionna.rt import Scene

from src.utils.save_utils import StorableDataclass


# Materials
# ---------

@dataclass()
class MaterialInfo(object):
    conductivity: float
    permittivity: float


@dataclass()
class MaterialsMapping(StorableDataclass):
    __filename__ = "materials_mapping"
    _scene_object_to_material = None

    frequency: float  # in Hz
    materials_info: Dict[str, MaterialInfo]  # Material name -> Material Info
    materials_to_scene_objects: Dict[str, List[str]]  # Material name -> Scene objects names

    @property
    def scene_objects_to_materials(self) -> Dict[str, str]:
        return self._scene_object_to_material

    def __post_init__(self):
        # Init material info
        for mat_name in self.materials_info.keys():
            if isinstance(self.materials_info[mat_name], dict):
                self.materials_info[mat_name] = MaterialInfo(**self.materials_info[mat_name])
        # Init object to material mapping
        self._scene_object_to_material = {
            obj_name: mat_name
            for mat_name, obj_list in self.materials_to_scene_objects.items()
            for obj_name in obj_list
        }

    @classmethod
    def from_scene(cls, scene: Scene):
        # Radio material mapping
        material_to_obj = dict()
        for obj_name, obj in scene.objects.items():
            mat_name = obj.radio_material.name
            if mat_name in material_to_obj.keys():
                material_to_obj[mat_name].append(obj_name)
            else:
                material_to_obj[mat_name] = [obj_name]
        # Used radio materials
        materials_info = dict()
        for mat_name in material_to_obj.keys():
            material = scene.radio_materials[mat_name]
            materials_info[mat_name] = MaterialInfo(
                conductivity=material.conductivity.numpy(),
                permittivity=material.relative_permittivity.numpy()
            )
        return cls(
            frequency=scene.frequency.numpy(),
            materials_info=materials_info,
            materials_to_scene_objects=material_to_obj
        )


# Measurements
# ------------

class MeasurementType(object):
    MAXWELL_SIMULATION = "maxwell_simulation"  # Measurements are simulated by solving Maxwell's equations
    RAY_TRACING = "ray_tracing"  # Simulate measurement via ray-tracing


class MeasurementNoiseType(object):
    PHASE = "phase"  # Simulate measurement by adding noise to the predicted phase
    POSITION = "position"  # Simulate measurement by adding a small displacement to the devices positions


class MeasurementPhaseNoiseType(object):
    PERFECT_PHASE = "perfect_phase"  # CFR has perfect phase info
    DENSE_COMPONENTS = "dense_components"  # Each ray is a large sum of random phases; complex gaussian latent
    UNIFORM_PHASE = "uniform_phase"  # The phase of each path is taken uniformly in [0; 2 pi[
    VON_MISES_PHASE = "von_mises_phase"  # The phase of each path is taken according to a von Mises distribution
    MANUAL_PHASE = "manual_phase"  # Specify the phase of each path manually


@dataclass()
class _MeasurementDataBase(StorableDataclass):
    __filename__ = None

    measurement: tf.Tensor  # Measurement [N_MEASUREMENTS, N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]
    normalization_constant: float  # Normalization constant / Average (noiseless) measurement power
    normalized_measurement_noise_std: tf.Tensor  # Normalized measurement noise standard deviation per Rx-Tx pair ; shape [N_RX_TX_PAIRS]
    measurement_noise: tf.Tensor  # Measurement noise [N_MEASUREMENTS, N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]

    def __post_init__(self):
        # Convert legacy data to the right format
        if isinstance(self.normalized_measurement_noise_std, float):
            self.normalized_measurement_noise_std = tf.constant(
                [self.normalized_measurement_noise_std],
                dtype=tf.float32
            )


@dataclass()
class MeasurementDataMaxwell(_MeasurementDataBase):
    __filename__ = "measurement_data_maxwell"

    freq_axis_maxwell: tf.Tensor  # Set of available frequencies in the Maxwell simulation ; shape [N_POINTS]
    freq_response_maxwell: tf.Tensor  # Frequency responses of the Maxwell simulation ; shape [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX=1, N_ARR_TX=1]


@dataclass()
class MeasurementDataPhaseNoise(_MeasurementDataBase):
    __filename__ = "measurement_data_phase_noise"

    cfr_per_mpc: tf.Tensor  # Frequency Responses [N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX, N_PATHS]
    components_noise: tf.Tensor  # Components noise [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]


@dataclass()
class MeasurementDataPositionNoise(_MeasurementDataBase):
    __filename__ = "measurement_data_pos_noise"

    position_noise_rx: tf.Tensor  # Sampled position noise at receiver [N_MEASUREMENTS, N_RX, N_COORDS=3]
    position_noise_tx: tf.Tensor  # Sampled position noise at transmitter [N_MEASUREMENTS, N_TX, N_COORDS=3]


def load_measurement_data(directory) -> Union[
    MeasurementDataMaxwell,
    MeasurementDataPhaseNoise,
    MeasurementDataPositionNoise
]:
    list_filenames = os.listdir(directory)
    if f"{MeasurementDataMaxwell.__filename__}.json" in list_filenames:
        return MeasurementDataMaxwell.load(directory)
    elif f"{MeasurementDataPositionNoise.__filename__}.json" in list_filenames:
        return MeasurementDataPositionNoise.load(directory)
    elif f"{MeasurementDataPhaseNoise.__filename__}.json" in list_filenames:
        return MeasurementDataPhaseNoise.load(directory)
    else:
        raise ValueError(f"No measurement file was found...")


# Power at calibration locations
# ------------------------------

@dataclass()
class CalibrationChannelsPower(StorableDataclass):
    __filename__ = "calibration_channels_power"
    
    uniform_phases_power: tf.Tensor  # [N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]
    simulated_phases_power: tf.Tensor  # [N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]


# Coverage Map
# ------------

@dataclass()
class _CoverageMapDataBase(StorableDataclass):
    __filename__ = None

    # Coverage map parameters
    cm_cell_size: Tuple[float]
    cm_center: List[float]  # Shape [3], (x, y, z) position of coverage map center
    cm_orientation: List[float]  # Shape [3], rotation angles of coverage map
    cm_size: List[float]  # Shape [2], (x, y) sizes of the coverage map

    # Sionna coverage map
    sionna_cm_normalization_constant: float
    sionna_cm_values: tf.Tensor  # Shape [N_TX=1, N_CELLS_Y, N_CELLS_X]

    # Cells
    cell_centers: np.ndarray  # Shape [N_CELLS_Y, N_CELLS_X, 3], (x, y, z) positions of cell centers
    rx_cells_indexes: np.ndarray  # [N_RX, 2], (n_cell_y, n_cell_x) indexes of cells where a receiver is simulated


@dataclass()
class CoverageMapCFRData(_CoverageMapDataBase):
    __filename__ = "coverage_map_cfr"

    # Channel frequency response
    cfr_per_mpc: tf.Tensor  # Frequency Responses [N_RX, N_TX=1, N_CARRIERS, N_ARR_RX, N_ARR_TX, N_PATHS]


@dataclass()
class CoverageMapPowerData(_CoverageMapDataBase):
    __filename__ = "coverage_map_power"

    # Powers are averaged across all subcarriers and all antenna pairs

    uniform_phases_power: tf.Tensor  # Power under uniform phases at each receiver [N_RX, N_TX=1]
    simulated_phases_power: tf.Tensor  # Power under RT-predicted phases at each receiver [N_RX, N_TX=1]


# Materials Calibration
# ---------------------

@dataclass()
class MaterialsCalibrationInfo(StorableDataclass):
    __filename__ = "materials_calibration_info"

    track_losses: np.ndarray  # [N_STEPS]
    track_materials_conductivity: np.ndarray  # [N_STEPS, N_MATERIALS]
    track_materials_permittivity: np.ndarray  # [N_STEPS, N_MATERIALS]
    calibrated_materials_conductivity: np.ndarray  # [N_MATERIALS]
    calibrated_materials_permittivity: np.ndarray  # [N_MATERIALS]
    ground_truth_materials_conductivity: np.ndarray  # [N_MATERIALS]
    ground_truth_materials_permittivity: np.ndarray  # [N_MATERIALS]


@dataclass()
class VonMisesCalibrationInfo(StorableDataclass):
    __filename__ = "von_mises_calibration_info"

    track_von_mises_means: np.ndarray  # [N_STEPS, N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    track_von_mises_concentrations: np.ndarray  # [N_STEPS, N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    track_von_mises_global_concentration: np.ndarray  # [N_STEPS]
    von_mises_means: np.ndarray  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    von_mises_concentrations: np.ndarray  # [N_MEASUREMENTS, N_RX_TX_PAIRS, N_PATHS]
    von_mises_global_concentration: np.float32
