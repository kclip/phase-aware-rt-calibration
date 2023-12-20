import os
import numpy as np
from sionna.rt import Scene, Camera, Transmitter, Receiver, PlanarArray, RadioMaterial, load_scene

from settings import PROJECT_FOLDER
from src.material import CONCRETE
from src.scenarios.toy_example.const import TOP_CAM_NAME, RX_NAME, TX_NAME
from src.utils.sionna_utils import reset_scene_materials


def get_scenario_toy_example(
    load_ground_truth: bool,
    num_cols_rx_array: int,
    num_cols_tx_array: int,
) -> Scene:
    # Load 3D scene
    if load_ground_truth:  # Load ground-truth geometry
        filepath_scene = os.path.join(
            PROJECT_FOLDER, "assets", "toy_example_ground_truth", "toy_example_ground_truth.xml"
        )
    else:  # Load DT model
        filepath_scene = os.path.join(
            PROJECT_FOLDER, "assets", "toy_example_dt_model", "toy_example_dt_model.xml"
        )
    scene = load_scene(filepath_scene)

    # Add cameras
    top_cam = Camera(TOP_CAM_NAME, position=[0, -2, 50], orientation=[np.pi / 2, np.pi / 2, 0])
    scene.add(top_cam)

    # Set carrier frequency
    scene.frequency = 6e9

    # Configure antenna arrays
    # ------------------------

    # Uniform planar array
    pattern = "iso"
    polarization = "V"
    spacing_array = 0.5  # in multiple of the wavelength

    # Simulate point antenna and use steering vectors to model the array
    scene.synthetic_array = True

    # Single antenna on both sides
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=num_cols_rx_array,
        vertical_spacing=spacing_array,
        horizontal_spacing=spacing_array,
        pattern=pattern,
        polarization=polarization
    )
    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=num_cols_tx_array,
        vertical_spacing=spacing_array,
        horizontal_spacing=spacing_array,
        pattern=pattern,
        polarization=polarization
    )

    # Position of receivers and transmitter (in [m])
    # ----------------------------------------------
    height_devices = 1

    # Receiver
    rx_pos = (11.99169921875, 0, height_devices)
    if RX_NAME in scene.receivers:
        scene.remove(RX_NAME)
    rx = Receiver(
        name=RX_NAME,
        position=rx_pos,
        orientation=[np.pi, 0, 0],  # Face Tx
    )
    scene.add(rx)

    # Transmitter
    tx_pos = (-11.99169921875, 0, height_devices)
    if TX_NAME in scene.transmitters:
        scene.remove(TX_NAME)
    tx = Transmitter(
        name=TX_NAME,
        position=tx_pos,
        orientation=[0, 0, 0],  # Face Rx
    )
    scene.add(tx)

    # Set scene materials
    # -------------------
    # Unique material
    material_name = "material"
    radio_material = RadioMaterial(
        name=material_name,
        frequency_update_callback=CONCRETE.material_callback
    )
    reset_scene_materials(
        scene=scene,
        radio_material=radio_material
    )

    return scene
