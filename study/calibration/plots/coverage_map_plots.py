import os
import numpy as np
from sionna.rt import CoverageMap

from src.utils.save_utils import SafeOpen
from src.utils.plot_utils import set_rc_params
from src.scenarios import get_scenario
from src.scenarios.the_strand.const import TOP_CAM_CM_NAME, TOP_CAM_CM_RESOLUTION
from study.calibration.experiment_protocol import load_experiment_protocol
from study.calibration.utils import protocol_plots_folder, power_map_data_subfolder, \
    power_map_plots_subfolder, coverage_map_filename
from study.calibration.experiment_config import get_scenario_metadata, coverage_map_metadata


def run_coverage_map_plots(
    protocol_name: str,
    n_loaded_runs: int,
    array_config: str,
    cm_show_devices: bool = True,
    cm_show_tx_only: bool = True,
    cm_vmin: float = -30,  # in dB
    cm_vmax: float = 0,  # in dB
    cm_num_samples: int = 128,
    save_plots: bool = True,
):
    protocol_plots_dir = protocol_plots_folder(protocol_name=protocol_name, n_loaded_runs=n_loaded_runs)

    # Load scene
    scenario_metadata = get_scenario_metadata(ground_truth_geometry=True)
    scene = get_scenario(scenario_metadata)
    
    # Remove Rx
    if cm_show_tx_only:
        for rx_name in scene.receivers.keys():
            scene.remove(rx_name)

    # Coverage map default kwargs
    coverage_map_kwargs = dict(
        center=coverage_map_metadata.coverage_map_center_with_default(scene=scene),
        orientation=coverage_map_metadata.coverage_map_orientation_with_default(),
        size=coverage_map_metadata.coverage_map_size_with_default(scene=scene),
        cell_size=coverage_map_metadata.cell_size,
        scene=scene
    )

    # Setup general plot config
    set_rc_params(scale=1.5)

    # Plot coverage maps in protocol
    for run_parameters in load_experiment_protocol(protocol_name):
        for calibration_type in run_parameters.run_calibration_types:
            # Load data
            cm_data_filename = coverage_map_filename(
                calibration_type=calibration_type,
                run_parameters=run_parameters
            )
            with SafeOpen(
                os.path.join(protocol_plots_dir, power_map_data_subfolder(array_config=array_config)),
                cm_data_filename,
                "rb"
            ) as file:
                cm_values = np.load(file)

            # Plot coverage map
            coverage_map = CoverageMap(value=cm_values, **coverage_map_kwargs)
            fig = scene.render(
                TOP_CAM_CM_NAME,
                coverage_map=coverage_map,
                show_devices=cm_show_devices,
                num_samples=cm_num_samples,
                cm_vmin=cm_vmin,
                cm_vmax=cm_vmax,
                resolution=TOP_CAM_CM_RESOLUTION
            )

            if save_plots:
                cm_plot_filename = coverage_map_filename(
                    calibration_type=calibration_type,
                    run_parameters=run_parameters,
                    ext=".png"
                )
                with SafeOpen(
                    os.path.join(protocol_plots_dir, power_map_plots_subfolder(array_config=array_config)),
                    cm_plot_filename,
                    "wb"
                ) as file:
                    fig.savefig(file, dpi=300, bbox_inches="tight")
