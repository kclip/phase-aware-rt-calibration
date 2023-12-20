import os
import matplotlib as mpl

from settings import PROJECT_FOLDER, STUDY_EXPERIMENT_VERSION


# Dataclasses and enums
# ---------------------

class RunType(object):
    MEASUREMENT = "measurement"
    CALIBRATION = "calibration"
    COVERAGE_MAP = "coverage_map"


class CalibrationType(object):
    PEOC = "PEOC"
    UPEC = "UPEC"
    UPEC_PATHS_PROJ = "UPEC_PATHS_PROJ"
    PEAC = "PEAC"
    PEAC_FIXED_PRIOR = "PEAC_FIXED_PRIOR"

    _folder_map = {
        PEOC: "peoc",
        UPEC: "upec",
        UPEC_PATHS_PROJ: "upec_paths_proj",
        PEAC: "peac_learned_prior",
        PEAC_FIXED_PRIOR: "peac_fixed_prior"
    }

    _plot_kwargs_colormap = "tab10"
    _plot_kwargs_map = {
        PEOC: dict(
            label="Phase Error-Oblivious Calibration",
            color=mpl.colormaps[_plot_kwargs_colormap].colors[1],
            linestyle="-",
            marker="|"
        ),
        UPEC: dict(
            label="Uniform Phase Error Calibration",
            color=mpl.colormaps[_plot_kwargs_colormap].colors[2],
            linestyle="--",
            marker="x"
        ),
        UPEC_PATHS_PROJ: dict(
            label="Uniform Phase Error Calibration",
            color=mpl.colormaps[_plot_kwargs_colormap].colors[2],
            linestyle="-",
            marker="x"
        ),
        PEAC_FIXED_PRIOR: dict(
            label="Phase Error-Aware Calibration",
            color=mpl.colormaps[_plot_kwargs_colormap].colors[0],
            linestyle="-",
            marker="o"
        ),
        PEAC: dict(
            label="Phase Error-Aware Calibration",
            color=mpl.colormaps[_plot_kwargs_colormap].colors[0],
            linestyle="-",
            marker="o"
        ),
    }
    _stem_plot_kwargs_map = {
        PEOC: dict(
            label="Phase Error-Oblivious Calibration",
            linefmt=f"tab:orange",
            markerfmt="_"
        ),
        UPEC: dict(
            label="Uniform Phase Error Calibration",
            linefmt=f"tab:green--",
            markerfmt="x"
        ),
        UPEC_PATHS_PROJ: dict(
            label="Uniform Phase Error Calibration",
            linefmt=f"tab:green",
            markerfmt="x"
        ),
        PEAC_FIXED_PRIOR: dict(
            label="Phase Error-Aware Calibration",
            linefmt=f"tab:blue--",
            markerfmt="o"
        ),
        PEAC: dict(
            label="Phase Error-Aware Calibration",
            linefmt=f"tab:blue",
            markerfmt="o"
        ),
    }

    @staticmethod
    def _check_calibration_type(calibration_type, map):
        if calibration_type not in map.keys():
            raise ValueError(f"Unknown calibration type '{calibration_type}'")

    @classmethod
    def all_types(cls):
        return list(cls._folder_map.keys())

    @classmethod
    def get_folder_name(cls, calibration_type: str):
        cls._check_calibration_type(calibration_type, cls._folder_map)
        return cls._folder_map.get(calibration_type)

    @classmethod
    def get_plot_kwargs(cls, calibration_type: str):
        cls._check_calibration_type(calibration_type, cls._plot_kwargs_map)
        return cls._plot_kwargs_map.get(calibration_type)

    @classmethod
    def get_stemplot_kwargs(cls, calibration_type: str):
        cls._check_calibration_type(calibration_type, cls._stem_plot_kwargs_map)
        return cls._stem_plot_kwargs_map.get(calibration_type)


# Folders and filenames
# ---------------------

def format_parameter_folder(parameter_value: str) -> str:
    """Format parameter value to fit in folder name"""
    str_param_list = str(float(parameter_value)).split(".")
    return "_".join(str_param_list)


EXPERIMENT_PLOTS_FOLDER = os.path.join(PROJECT_FOLDER, "logs", "saved_plots")


def protocol_plots_folder(protocol_name, n_loaded_runs: int):
    return os.path.join(
        EXPERIMENT_PLOTS_FOLDER,
        f"{STUDY_EXPERIMENT_VERSION}_{protocol_name}_protocol_{n_loaded_runs}_loaded_runs"
    )


def calibration_plots_folder(protocol_name, n_loaded_runs: int, annotate: bool = None):
    plots_dir = os.path.join(
        protocol_plots_folder(protocol_name=protocol_name, n_loaded_runs=n_loaded_runs),
        "calibration"
    )
    if annotate is None:
        return plots_dir
    else:
        return os.path.join(
            plots_dir,
            "plots_annotated" if annotate else "plots"
        )


PROTOCOL_CALIBRATION_DATA_FILENAME = "calibration_data.csv"
PROTOCOL_POWER_MAP_DATA_FILENAME = "power_maps_estimation_error_data.csv"


def power_map_data_subfolder(array_config: str) -> str:
    return f"power_map_{array_config}_data"


def power_map_plots_subfolder(array_config: str) -> str:
    return f"power_map_{array_config}_plots"


def coverage_map_filename(
    calibration_type: str,
    run_parameters,  # RunParameters
    ext: str = ".npy"
) -> str:
    calibration_type_folder_name = CalibrationType.get_folder_name(calibration_type)
    return (
        calibration_type_folder_name +
        f"_mean_error_coverage_map" +
        f"_snr_{run_parameters.measurement_snr}" +
        (
            f"_meas_perfect_phase"
            if run_parameters.measurement_perfect_phase else
            f"_meas_vm_concentration_{run_parameters.measurement_von_mises_concentration}"
        ) +
        ext
    )


PROTOCOL_POWER_PROFILES_DATA_FILENAME = "power_profiles_data.parquet.gzip"


def power_profiles_plots_folder(protocol_name, n_loaded_runs: int):
    return os.path.join(
        protocol_plots_folder(protocol_name=protocol_name, n_loaded_runs=n_loaded_runs),
        "power_profiles"
    )
