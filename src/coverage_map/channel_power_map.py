import tensorflow as tf
from sionna.rt import CoverageMap, Scene

from src.data_classes import CoverageMapPowerData
from src.utils.channel_power import weight_rt_predicted_and_uniform_phase_power_maps_by_von_mises_concentration


# Signal power from CFR and interference types
# --------------------------------------------

class InterferenceType(object):
    UNIFORM_PHASE = "uniform_phase"  # Components have uniformly distributed phase errors
    RAY_TRACER_SIMULATED = "ray_tracer_simulated"  # Components phases are given by the ray-tracer
    VON_MISES_PHASE = "von_mises_phase"  # Components have von Mises distributed phase errors


def get_channel_power_tensor(
    coverage_map_power_data: CoverageMapPowerData,
    interference_type: str,
    von_mises_concentration: float = None,
    idx_tx: int = 0
) -> tf.Tensor:  # Shape [N_RX]
    if interference_type == InterferenceType.UNIFORM_PHASE:
        return coverage_map_power_data.uniform_phases_power[:, idx_tx]
    elif interference_type == InterferenceType.RAY_TRACER_SIMULATED:
        return coverage_map_power_data.simulated_phases_power[:, idx_tx]
    elif interference_type == InterferenceType.VON_MISES_PHASE:
        if von_mises_concentration is None:
            raise ValueError(f"Interference type '{interference_type}' requires a concentration to be specified")
        return weight_rt_predicted_and_uniform_phase_power_maps_by_von_mises_concentration(
            power_uniform_phases=coverage_map_power_data.uniform_phases_power[:, idx_tx],
            power_rt_simulated_phases=coverage_map_power_data.simulated_phases_power[:, idx_tx],
            von_mises_concentration=von_mises_concentration
        )
    else:
        raise ValueError(f"Unknown interference type '{interference_type}'...")


def _to_coverage_map(
    scene: Scene,
    channel_power_tensor: tf.Tensor,  # Shape [N_RX]
    coverage_map_data: CoverageMapPowerData,
) -> CoverageMap:
    cells_dim = coverage_map_data.cell_centers.shape[:2]
    cm_values = tf.zeros(cells_dim, dtype=tf.float32)
    cm_values = tf.tensor_scatter_nd_update(
        tensor=cm_values,
        indices=coverage_map_data.rx_cells_indexes,
        updates=channel_power_tensor
    )
    cm_values = cm_values[tf.newaxis, :, :]

    return CoverageMap(
        center=coverage_map_data.cm_center,
        orientation=coverage_map_data.cm_orientation,
        size=coverage_map_data.cm_size,
        cell_size=coverage_map_data.cm_cell_size,
        value=cm_values,
        scene=scene
    )


def get_channel_power_map(
    scene: Scene,
    coverage_map_power_data: CoverageMapPowerData,
    interference_type: str,
    von_mises_concentration: float = None,
    idx_tx: int = 0
) -> CoverageMap:
    channel_power_tensor = get_channel_power_tensor(
        coverage_map_power_data=coverage_map_power_data,
        interference_type=interference_type,
        von_mises_concentration=von_mises_concentration,
        idx_tx=idx_tx
    )

    return _to_coverage_map(
        scene=scene,
        channel_power_tensor=channel_power_tensor,
        coverage_map_data=coverage_map_power_data
    )
