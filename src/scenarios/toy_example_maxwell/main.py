import os
import numpy as np
from sionna.rt import Scene, Camera, Transmitter, Receiver, PlanarArray, RadioMaterial, load_scene

from settings import PROJECT_FOLDER
from src.material import CONCRETE, METAL
from src.scenarios.toy_example_maxwell.const import TOP_CAM_NAME, RX_NAME_PREFIX, TX_NAME
from src.utils.sionna_utils import reset_scene_materials


# Device distance from blocker (origin) in number of wavelengths
D_TX_LAMBDA = 30
D_RX_LAMBDA_LIST = [20, 30, 40]
N_RX_MAXWELL_TOY_EXAMPLE = len(D_RX_LAMBDA_LIST)


def get_scenario_toy_example_maxwell(load_ground_truth: bool) -> Scene:
    # Load 3D scene
    asset_name = "toy_example_maxwell"
    asset_name = f"{asset_name}_ground_truth" if load_ground_truth else f"{asset_name}_dt_model"
    filepath_scene = os.path.join(PROJECT_FOLDER, "assets", asset_name, f"{asset_name}.xml")
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
        num_cols=1,
        vertical_spacing=spacing_array,
        horizontal_spacing=spacing_array,
        pattern=pattern,
        polarization=polarization
    )
    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=spacing_array,
        horizontal_spacing=spacing_array,
        pattern=pattern,
        polarization=polarization
    )

    # Position of receivers and transmitter (in [m])
    # ----------------------------------------------
    height_devices = 0.4
    wavelength = scene.wavelength

    # Receiver
    for rx_idx, rx_x_pos in enumerate(D_RX_LAMBDA_LIST):
        rx_name = f"{RX_NAME_PREFIX}_{rx_idx}"
        rx_pos = (rx_x_pos * wavelength, 0, height_devices)
        if rx_name in scene.receivers:
            scene.remove(rx_name)
        rx = Receiver(
            name=rx_name,
            position=rx_pos,
        )
        scene.add(rx)

    # Transmitter
    tx_pos = (-D_TX_LAMBDA * wavelength, 0, height_devices)
    if TX_NAME in scene.transmitters:
        scene.remove(TX_NAME)
    tx = Transmitter(
        name=TX_NAME,
        position=tx_pos,
    )
    scene.add(tx)

    # Set scene materials
    # -------------------
    # Concrete material by default
    material_name = "material"
    radio_material_concrete = RadioMaterial(
        name=material_name,
        frequency_update_callback=CONCRETE.material_callback
    )
    reset_scene_materials(
        scene=scene,
        radio_material=radio_material_concrete
    )
    # Metal blocker
    radio_material_metal = RadioMaterial(
        name="metal__known",  # Assumed to be known (does not appear in the list of trainable materials)
        frequency_update_callback=METAL.material_callback
    )
    scene.add(radio_material_metal)
    scene.objects["blocker"].radio_material = radio_material_metal.name

    return scene
