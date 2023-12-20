from typing import Tuple
import tensorflow as tf
from sionna.rt import Scene

from src.utils.telecom_utils import get_lower_bound_angular_resolution_ula
from src.utils.sionna_utils import select_rx_tx_pairs
from src.ofdm_measurements import _MeasurementMetadataBase
from src.ofdm_measurements.main import compute_paths_from_metadata
from src.calibrate_materials import CalibrateMaterialsUPECMetadata
from src.projections.time_projection import get_paths_delays_time_projections, get_evenly_spaced_time_projections
from src.projections.angle_projection import get_paths_angles_projections, get_evenly_spaced_angle_projections


def get_predicted_paths_projections(
    scene: Scene,
    measurement_metadata: _MeasurementMetadataBase
) -> Tuple[
    tf.Tensor,  # Time projections ; Shape [N_RX_TX_PAIRS, N_SUBCARRIERS, N_POINTS_TIME]
    tf.Tensor,  # Rx angle projections ; Shape [N_RX_TX_PAIRS, N_ARR_RX, N_POINTS_ANGLE]
    tf.Tensor  # Tx angle projections ; Shape [N_RX_TX_PAIRS, N_ARR_TX, N_POINTS_ANGLE]
]:
    paths = compute_paths_from_metadata(
        scene=scene,
        measurement_metadata=measurement_metadata,
        check_paths=True
    )

    # Time projections (computed from Measurement Metadata)
    # -----------------------------------------------------
    time_projections = get_paths_delays_time_projections(  # Shape [N_RX, N_TX, N_SUBCARRIERS, N_POINTS_TIME]
        paths=paths,
        num_subcarriers=measurement_metadata.num_subcarriers,
        subcarrier_spacing=measurement_metadata.subcarrier_spacing
    )
    time_projections = select_rx_tx_pairs(
        rx_tx_indexed_tensor=time_projections,
        rx_tx_indexes=measurement_metadata.rx_tx_indexes
    )

    # Angle projections
    # -----------------
    # Shape [N_RX, N_TX, N_ARR_RX/N_ARR_TX, N_POINTS_ANGLE]
    rx_angle_projections, tx_angle_projections = get_paths_angles_projections(
        scene=scene,
        paths=paths,
    )
    rx_angle_projections = select_rx_tx_pairs(
        rx_tx_indexed_tensor=rx_angle_projections,
        rx_tx_indexes=measurement_metadata.rx_tx_indexes
    )
    tx_angle_projections = select_rx_tx_pairs(
        rx_tx_indexed_tensor=tx_angle_projections,
        rx_tx_indexes=measurement_metadata.rx_tx_indexes
    )

    return time_projections, rx_angle_projections, tx_angle_projections


def get_evenly_spaced_projections(
    scene: Scene,
    measurement_metadata: _MeasurementMetadataBase,
    calibration_metadata: CalibrateMaterialsUPECMetadata,
) -> Tuple[
    tf.Tensor,  # Time projections ; Shape [N_RX_TX_PAIRS, N_SUBCARRIERS, N_POINTS_TIME]
    tf.Tensor,  # Rx angle projections ; Shape [N_RX_TX_PAIRS, N_ARR_RX, N_POINTS_ANGLE]
    tf.Tensor  # Tx angle projections ; Shape [N_RX_TX_PAIRS, N_ARR_TX, N_POINTS_ANGLE]
]:
    paths = compute_paths_from_metadata(
        scene=scene,
        measurement_metadata=measurement_metadata,
        check_paths=True
    )

    # Time projections (computed from Measurement Metadata)
    # -----------------------------------------------------
    time_projections = get_evenly_spaced_time_projections(  # Shape [N_RX, N_TX, N_SUBCARRIERS, N_POINTS_TIME]
        paths=paths,
        num_subcarriers=measurement_metadata.num_subcarriers,
        subcarrier_spacing=measurement_metadata.subcarrier_spacing
    )
    time_projections = select_rx_tx_pairs(
        rx_tx_indexed_tensor=time_projections,
        rx_tx_indexes=measurement_metadata.rx_tx_indexes
    )

    # Angle projections (computed from Calibration Metadata)
    # ------------------------------------------------------
    # We assume that the azimuth-angle resolution is independent of the considered elevation angle, and that it is
    # equal to its maximal resolution (at elevation = pi/ 2).
    min_n_elements_per_axis = min(  # Minimal number of elements on an array-axis
        calibration_metadata.num_rows_rx_array,
        calibration_metadata.num_cols_rx_array,
        calibration_metadata.num_rows_tx_array,
        calibration_metadata.num_cols_tx_array,
    )
    angular_resolution = get_lower_bound_angular_resolution_ula(
        n_array_elements=min_n_elements_per_axis,
        spacing_array_elements=calibration_metadata.spacing_array_elements
    )
    # Shape [N_RX, N_TX, N_ARR_RX/N_ARR_TX, N_POINTS_ANGLE]
    rx_angle_projections, tx_angle_projections = get_evenly_spaced_angle_projections(
        scene=scene,
        paths=paths,
        angle_resolution_az=angular_resolution,
        angle_resolution_el=angular_resolution
    )
    rx_angle_projections = select_rx_tx_pairs(
        rx_tx_indexed_tensor=rx_angle_projections,
        rx_tx_indexes=measurement_metadata.rx_tx_indexes
    )
    tx_angle_projections = select_rx_tx_pairs(
        rx_tx_indexed_tensor=tx_angle_projections,
        rx_tx_indexes=measurement_metadata.rx_tx_indexes
    )

    return time_projections, rx_angle_projections, tx_angle_projections
