from typing import List, Tuple
import tensorflow as tf
from sionna.channel import subcarrier_frequencies
from sionna.rt import Scene, Paths, RadioMaterial

KNOWN_MATERIAL_SUFFIX = "__known"
TRAIN_MATERIAL_SUFFIX = "__train"


# Paths masks
# -----------

def get_mask_paths(paths: Paths) -> tf.Tensor:  # Mask of paths per (Rx, Tx) pair ; Shape [N_RX, N_TX, N_PATHS_MAX]
    """Return a mask of shape [N_RX, N_TX, N_PATHS_MAX] indicating the valid path indexes for each (Rx, Tx) pair"""
    return tf.logical_not(
        tf.reduce_all(
            paths.a[0, :, :, :, :, :, 0] == 0.0,
            axis=[1, 3]  # Reduce over Rx and Tx antenna dimensions
        )
    )


def get_mask_paths_from_cfr(
    # Shape [N_RX, N_TX, N_CARRIERS, N_ARR_RX, N_ARR_TX, N_PATHS_MAX] or
    # [N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX, N_PATHS_MAX]
    cfr_per_mpc: tf.Tensor
) -> tf.Tensor:  # Mask of paths per (Rx, Tx) pair ; Shape [N_RX, N_TX, N_PATHS_MAX] or [N_RX_TX_PAIRS, N_PATHS_MAX]
    """Return a mask of shape [N_RX, N_TX, N_PATHS_MAX] indicating the valid path indexes for each (Rx, Tx) pair"""
    return tf.logical_not(
        tf.reduce_all(
            cfr_per_mpc == 0.0,
            axis=[-1, -2, -3, -4]  # Reduce over paths, Tx antenna, Rx antenna and subcarriers dimensions
        )
    )


# Channel responses and measurements utils
# ----------------------------------------

def set_path_delays_normalization(
    paths: Paths,
    normalize_delays: bool
):
    if paths.normalize_delays and (not normalize_delays):
        print("WARNING: REMOVING PATH DELAY NORMALIZATION !")
        paths.normalize_delays = False
    if (not paths.normalize_delays) and normalize_delays:
        print("WARNING: ACTIVATING PATH DELAY NORMALIZATION !")
        paths.normalize_delays = True


def select_rx_tx_pairs(
    rx_tx_indexed_tensor: tf.Tensor,  # Shape [N_RX, N_TX, dim1, ..., dimN]
    rx_tx_indexes: List[Tuple[int]],  # List of (receiver, transimitter) index pairs
) -> tf.Tensor:  # Shape [N_SELECTED_RX_TX_PAIRS, dim1, ..., dimN]
    """
    Select specific (Rx, Tx) for a tensor whose first dimensions are the indexes of receiver and transmitter devices
    (i.e., shape [N_RX, N_TX, ...]).
    Output is given as a tensor with dimension [N_SELECTED_RX_TX_PAIRS, ...].
    """
    return tf.gather_nd(
        params=rx_tx_indexed_tensor,
        indices=rx_tx_indexes
    )


# Checks utils
# ------------

def check_all_rx_tx_pairs_have_at_least_one_path(
    paths: Paths,
    rx_tx_indexes: List[Tuple[int]] = None
):
    # Check that all (Rx, Tx) pairs have at least one path linking them
    mask_paths = get_mask_paths(paths)
    has_no_path = tf.reduce_all(tf.logical_not(mask_paths), axis=-1)
    if rx_tx_indexes is not None:
        has_no_path = select_rx_tx_pairs(
            rx_tx_indexed_tensor=has_no_path,
            rx_tx_indexes=rx_tx_indexes
        )
    if tf.reduce_any(has_no_path):
        raise ValueError("Some (Rx, Tx) pairs do not have a propagation path linking them !")


# Materials utils
# ---------------

def _is_training_material(mat: RadioMaterial):
    return mat.name.endswith(TRAIN_MATERIAL_SUFFIX)


def _get_original_material_name(trainable_mat: RadioMaterial):
    return trainable_mat.name.split(TRAIN_MATERIAL_SUFFIX)[0]


def reset_scene_materials(
    scene: Scene,
    radio_material: RadioMaterial
) -> List[RadioMaterial]:
    """Set all Scene objects to the same given radio_material"""

    # Remove material if it was already set
    temp_material = RadioMaterial(
        name="__temp_material__",
        conductivity=0.0,
        relative_permittivity=1.0
    )
    if radio_material.name in scene.radio_materials:
        if temp_material.name not in scene.radio_materials:
            scene.add(temp_material)
        for name, scene_object in scene.objects.items():
            # Remove material from all objects
            if scene_object.radio_material.name == radio_material.name:
                scene_object.radio_material = temp_material.name
        scene.remove(radio_material.name)

    # Add material
    scene.add(radio_material)
    # Change all scene objects to material
    for name, scene_object in scene.objects.items():
        scene_object.radio_material = radio_material.name

    # Remove temp material
    if temp_material.name in scene.radio_materials:
        scene.remove(temp_material.name)

    return [radio_material]


def setup_training_materials(
    scene: Scene,
    default_relative_permittivity=3.0,
    default_conductivity=0.1,
    trainable: bool = True  # If set to False, the gradient tape must be manually set to watch the parameters
):
    len_suffix = len(TRAIN_MATERIAL_SUFFIX)

    # Cleanup previous training parameters
    for obj in scene.objects.values():
        if _is_training_material(obj.radio_material):
            obj.radio_material = obj.radio_material.name[:-len_suffix]
    for mat in scene.radio_materials.values():
        if _is_training_material(mat):
            scene.remove(mat.name)

    # Set training parameters
    original_materials = []
    trainable_materials = []

    for mat in list(scene.radio_materials.values()):
        if mat.is_used and (not mat.name.endswith(KNOWN_MATERIAL_SUFFIX)):
            # Create new trainable material with some default values
            new_mat_name = mat.name + TRAIN_MATERIAL_SUFFIX
            new_mat = RadioMaterial(
                new_mat_name,
                relative_permittivity=tf.Variable(
                    default_relative_permittivity,
                    name=f"{new_mat_name}__permittivity",
                    trainable=trainable
                ),
                conductivity=tf.Variable(
                    default_conductivity,
                    name=f"{new_mat_name}__conductivity",
                    trainable=trainable
                )
            )
            scene.add(new_mat)
            trainable_materials.append(new_mat)
            original_materials.append(mat)

    # Assign trainable materials to the corresponding objects
    original_materials_names = [mat.name for mat in original_materials]
    for obj in scene.objects.values():
        if obj.radio_material.name in original_materials_names:
            obj.radio_material = obj.radio_material.name + TRAIN_MATERIAL_SUFFIX

    return original_materials, trainable_materials


def get_original_and_trainable_materials(scene: Scene) -> Tuple[
    List[RadioMaterial],  # Original materials
    List[RadioMaterial]  # Trainable materials
]:
    trainable_materials = [
        mat
        for mat in scene.radio_materials.values()
        if _is_training_material(mat)
    ]
    original_materials = [scene.radio_materials[_get_original_material_name(mat)] for mat in trainable_materials]

    return original_materials, trainable_materials
