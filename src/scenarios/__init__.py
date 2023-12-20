from dataclasses import dataclass
from typing import List
from sionna.rt import Scene, RadioMaterial

from src.data_classes import StorableDataclass, MaterialsMapping
from src.scenarios.the_strand.main import get_scenario_the_strand
from src.scenarios.toy_example.main import get_scenario_toy_example
from src.scenarios.toy_example_maxwell.main import get_scenario_toy_example_maxwell


@dataclass()
class _ScenarioMetadataBase(StorableDataclass):
    __scenario_name__ = None


@dataclass()
class ScenarioTheStrandMetadata(_ScenarioMetadataBase):
    __scenario_name__ = "the_strand"

    carrier_frequency: float  # in Hz
    num_rows_rx_array: int
    num_cols_rx_array: int
    num_rows_tx_array: int
    num_cols_tx_array: int
    nb_receivers: int
    force_rx_positions: List[List[float]] = None  # [N_RX, 3]


@dataclass()
class ScenarioToyExampleMetadata(_ScenarioMetadataBase):
    __scenario_name__ = "toy_example"

    load_ground_truth: bool  # If True -> Load ground-truth model; else -> load DT model (lower wall position error)
    num_cols_rx_array: int
    num_cols_tx_array: int


@dataclass()
class ScenarioToyExampleMaxwellMetadata(_ScenarioMetadataBase):
    __scenario_name__ = "toy_example_maxwell"

    load_ground_truth: bool  # If True -> Load ground-truth model; else -> load DT model (lower wall position error)


def get_scenario(
    metadata: _ScenarioMetadataBase,
    materials_mapping: MaterialsMapping = None
) -> Scene:
    # Load given scenario
    if metadata.__scenario_name__ == ScenarioTheStrandMetadata.__scenario_name__:
        scene = get_scenario_the_strand(
            carrier_frequency=metadata.carrier_frequency,
            num_rows_rx_array=metadata.num_rows_rx_array,
            num_cols_rx_array=metadata.num_cols_rx_array,
            num_rows_tx_array=metadata.num_rows_tx_array,
            num_cols_tx_array=metadata.num_cols_tx_array,
            nb_receivers=metadata.nb_receivers,
            force_rx_positions=metadata.force_rx_positions
        )
    elif metadata.__scenario_name__ == ScenarioToyExampleMetadata.__scenario_name__:
        scene = get_scenario_toy_example(
            load_ground_truth=metadata.load_ground_truth,
            num_cols_rx_array=metadata.num_cols_rx_array,
            num_cols_tx_array=metadata.num_cols_tx_array
        )
    elif metadata.__scenario_name__ == ScenarioToyExampleMaxwellMetadata.__scenario_name__:
        scene = get_scenario_toy_example_maxwell(
            load_ground_truth=metadata.load_ground_truth
        )
    else:
        raise ValueError(f"Unknown scenario '{metadata.__scenario_name__}'...")

    # If a material mapping is given, replace material parameters in the loaded scene with the ones given
    if materials_mapping is not None:
        # Create/update materials
        for mat_name, mat_info in materials_mapping.materials_info.items():
            if mat_name in scene.radio_materials.keys():
                mat = scene.radio_materials[mat_name]
            else:
                mat = RadioMaterial(name=mat_name)
                scene.add(mat)
            mat.conductivity = mat_info.conductivity
            mat.relative_permittivity = mat_info.permittivity
        # Re-assign scene objects to materials
        for obj_name, scene_obj in scene.objects.items():
            scene_obj.radio_material = materials_mapping.scene_objects_to_materials[obj_name]

    return scene
