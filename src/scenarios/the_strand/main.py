import os
from typing import List
import numpy as np
from sionna.rt import Scene, Camera, Transmitter, Receiver, PlanarArray, RadioMaterial, load_scene

from settings import PROJECT_FOLDER
from src.material import CONCRETE
from src.scenarios.the_strand.const import TOP_CAM_NAME, TOP_CAM_ZOOM_NAME, TOP_CAM_CM_NAME, TX_NAME, RX_NAME_PREFIX

from src.utils.bezier_curve import MultipleSegmentsBezierCurve
from src.utils.sionna_utils import reset_scene_materials


def get_scenario_the_strand(
    carrier_frequency: float,
    num_rows_rx_array: int,
    num_cols_rx_array: int,
    num_rows_tx_array: int,
    num_cols_tx_array: int,
    nb_receivers: int,
    force_rx_positions: List[List[float]] = None  # [N_RX, 3]
) -> Scene:
    # Load 3D scene
    filepath_scene = os.path.join(PROJECT_FOLDER, "assets", "the_strand", "the_strand.xml")
    scene = load_scene(filepath_scene)

    # Add cameras
    top_cam = Camera(TOP_CAM_NAME, position=[2.61, -40.22, 700], orientation=[np.pi / 2, np.pi / 2, 0])
    top_cam_zoom = Camera(
        TOP_CAM_ZOOM_NAME,
        position=[-8.77, -17.97, 475],
        orientation=[np.pi / 2, np.pi / 2, 0]
    )
    top_cam_cm = Camera(  # Camera optimized for coverage maps
        TOP_CAM_CM_NAME,
        position=[0, 0, 475],
        orientation=[np.pi / 2, np.pi / 2, 0]
    )
    for cam in [top_cam, top_cam_zoom, top_cam_cm]:
        scene.add(cam)

    # Set carrier frequency
    scene.frequency = carrier_frequency

    # Configure antenna arrays
    # ------------------------

    # Uniform planar array
    pattern = "iso"
    polarization = "V"
    spacing_array = 0.5  # in multiple of the wavelength

    # Simulate point antenna and use steering vectors to model the array
    scene.synthetic_array = True

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(
        num_rows=num_rows_rx_array,
        num_cols=num_cols_rx_array,
        vertical_spacing=spacing_array,
        horizontal_spacing=spacing_array,
        pattern=pattern,
        polarization=polarization
    )
    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(
        num_rows=num_rows_tx_array,
        num_cols=num_cols_tx_array,
        vertical_spacing=spacing_array,
        horizontal_spacing=spacing_array,
        pattern=pattern,
        polarization=polarization
    )

    # Position of receivers and transmitter
    # -------------------------------------

    # Get curve along which to place the receivers
    filepath_curves = os.path.join(PROJECT_FOLDER, "assets", "the_strand", "rx_curves.json")
    rx_curve = MultipleSegmentsBezierCurve.from_json(filepath_curves)["RxCurve0"]

    # Rx position config
    if force_rx_positions is None:
        rx_height = 1.0  # in m
        rx_positions = rx_curve.evaluate(np.linspace(0, 1, nb_receivers)).transpose()
        rx_positions[:, 2] = rx_height
    else:
        rx_positions = np.array(force_rx_positions, dtype=np.float32)
    nb_receivers = len(rx_positions)
    rx_names = [f"{RX_NAME_PREFIX}_{n}" for n in range(nb_receivers)]

    # Transmitter
    if TX_NAME in scene.transmitters:
        scene.remove(TX_NAME)
    tx = Transmitter(
        name=TX_NAME,
        position=[-2, -26., 5.]
    )
    scene.add(tx)
    # Receivers
    rxs = []
    for rx_name, rx_pos in zip(rx_names, rx_positions):
        if rx_name in scene.receivers:
            scene.remove(rx_name)
        rx = Receiver(
            name=rx_name,
            position=rx_pos,
            orientation=[0, 0, 0]
        )
        rxs.append(rx)
        scene.add(rx)

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

