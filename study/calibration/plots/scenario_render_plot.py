from src.utils.save_utils import SafeOpen
from src.utils.plot_utils import set_rc_params
from src.scenarios import get_scenario
from src.scenarios.the_strand.const import TOP_CAM_ZOOM_NAME
from study.calibration.utils import protocol_plots_folder
from study.calibration.experiment_config import MAX_DEPTH_PATH, NUM_SAMPLES_PATH, PLOT_RENDER_HIGH_RES, \
    get_scenario_metadata


def run_scenario_render_plot(
    protocol_name: str,
    n_loaded_runs: int,
    array_config: str = None,  # Placeholder to match signature of other coverage map plots
    show_devices: bool = True,
    show_paths: bool = True,
    num_samples: int = 128,
    save_plot: bool = True
):
    protocol_plots_dir = protocol_plots_folder(protocol_name=protocol_name, n_loaded_runs=n_loaded_runs)

    # Load scene
    scenario_metadata = get_scenario_metadata(ground_truth_geometry=True)
    scene = get_scenario(scenario_metadata)

    # Setup general plot config
    set_rc_params()
    
    # Compute paths
    if show_paths:
        paths = scene.compute_paths(
            max_depth=MAX_DEPTH_PATH,
            method="fibonacci",
            num_samples=NUM_SAMPLES_PATH,
            diffraction=False, scattering=False, edge_diffraction=False,
            check_scene=False
        )
    else:
        paths = None

    # Plot coverage maps in protocol
    fig = scene.render(
        TOP_CAM_ZOOM_NAME,
        show_devices=show_devices,
        paths=paths,
        show_paths=show_paths,
        num_samples=num_samples,
        resolution=PLOT_RENDER_HIGH_RES
    )
    if save_plot:
        filename = (
            "scenario_render" +
            ("_with_devices" if show_devices else "") +
            ("_with_paths" if show_paths else "") +
            ".png"
        )
        with SafeOpen(protocol_plots_dir, filename, "wb") as file:
            fig.savefig(file, dpi=300)
