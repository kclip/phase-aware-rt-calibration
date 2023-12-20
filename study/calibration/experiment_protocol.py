from itertools import product
from typing import List
from dataclasses import dataclass
import numpy as np

from src.utils.telecom_utils import from_db
from src.mappings.von_mises_std_concentration_mapping.mapping_funcs import map_von_mises_std_to_concentration
from study.calibration.utils import CalibrationType


_ROUND_VON_MISES_CONCENTRATION = 3
_MAX_STD_VON_MISES_DEG = 103.923  # pi / sqrt(3) in degrees


# Protocol dataclasses
# --------------------

@dataclass()
class ExperimentProtocol(object):
    protocol_name: str

    # Mandatory params
    run_calibration_types: List[str]  # Calibration schemes to run
    measurement_snr_list: List[float]
    measurement_von_mises_std_deg_list: List[float]
    # Optional params
    bandwidth_list: List[float] = None  # Bandwidth in [Hz] (changes number of subcarriers)
    position_noise_amplitude_list: List[float] = None  # Position noise amplitude in multiples of the wavelength
    n_channel_measurements_list: List[int] = None  # Number of channel measurements for calibration (if None, use all available measurements)

    # Default parameters
    measurement_additive_noise: bool = True
    measurement_von_mises_mean: float = 0.0

    def __post_init__(self):
        if self.bandwidth_list is None:
            self.bandwidth_list = [None]
        if self.position_noise_amplitude_list is None:
            self.position_noise_amplitude_list = [None]
        if self.n_channel_measurements_list is None:
            self.n_channel_measurements_list = [None]


@dataclass()
class RunParameters(object):
    run_calibration_types: List[str]  # Calibration schemes to run
    measurement_snr: float
    measurement_additive_noise: bool
    measurement_perfect_phase: bool
    measurement_von_mises_mean: float
    measurement_von_mises_concentration: float
    bandwidth: float
    position_noise_amplitude: float  # in multiples of the wavelength
    n_channel_measurements: int

    def __str__(self):
        info = [
            f"SNR={self.measurement_snr if self.measurement_additive_noise else 'infinite'}",
            f"CONCENTRATION={'infinite' if self.measurement_perfect_phase else self.measurement_von_mises_concentration}",
        ] + (
            [f"BANDWIDTH={self.bandwidth}"] if self.bandwidth is not None else []
        ) + (
            [f"POSITION NOISE={self.position_noise_amplitude}"] if self.position_noise_amplitude is not None else []
        ) + (
            [f"NUM MEASURMENTS={self.n_channel_measurements}"] if self.n_channel_measurements is not None else []
        )
        n_char_per_line = 10 + max([len(s) for s in info])
        delim = "=" * n_char_per_line
        info_delim = [
            ("=" * int(np.floor((n_char_per_line - len(s)) / 2))) +
            s +
            ("=" * int(np.ceil((n_char_per_line - len(s)) / 2)))
            for s in info
        ]
        return (
            "\n" +
            delim + "\n" +
            "\n".join(info_delim) + "\n" +
            delim + "\n" +
            "\n"
        )


# Protocol methods
# ----------------

def get_protocol_by_name(protocol_name: str) -> ExperimentProtocol:
    for experiment_protocol in _ALL_PROTOCOLS:
        if experiment_protocol.protocol_name == protocol_name:
            return experiment_protocol
    raise ValueError(f"Protocol with name '{protocol_name}' not found ...")


def load_experiment_protocol(protocol_name: str) -> List[RunParameters]:
    experiment_protocol = get_protocol_by_name(protocol_name)
    # Load von Mises concentration parameters
    vm_concentration_params = map_von_mises_std_to_concentration(
        std=np.array(experiment_protocol.measurement_von_mises_std_deg_list, dtype=np.float32),
        std_in_degrees=True,
        round_concentration=_ROUND_VON_MISES_CONCENTRATION,
        map_infinite_concentration=True,
        print_errors=True
    )

    # Parse perfect phase scenario
    data_von_mises_concentration_and_perfect_phase_list = [
        (concentration, False) if concentration != "_infinity" else (None, True)
        for concentration in vm_concentration_params
    ]

    return [
        RunParameters(
            run_calibration_types=experiment_protocol.run_calibration_types,
            measurement_snr=data_snr,
            measurement_additive_noise=experiment_protocol.measurement_additive_noise,
            measurement_perfect_phase=concentration_and_perfect_phase[1],
            measurement_von_mises_mean=experiment_protocol.measurement_von_mises_mean,
            measurement_von_mises_concentration=concentration_and_perfect_phase[0],
            bandwidth=bandwidth,
            position_noise_amplitude=position_noise_amplitude,
            n_channel_measurements=n_channel_measurements
        )
        for (
            data_snr,
            concentration_and_perfect_phase,
            bandwidth,
            position_noise_amplitude,
            n_channel_measurements
        ) in product(
            experiment_protocol.measurement_snr_list,
            data_von_mises_concentration_and_perfect_phase_list,
            experiment_protocol.bandwidth_list,
            experiment_protocol.position_noise_amplitude_list,
            experiment_protocol.n_channel_measurements_list
        )
    ]


def get_all_protocol_names() -> List[str]:
    return [
        experiment_protocol.protocol_name for experiment_protocol in _ALL_PROTOCOLS
    ]


# Saved Protocols
# ---------------

_ALL_PROTOCOLS = [
    ExperimentProtocol(  # Study calibration for different SNR parameters
        protocol_name="snr_paper",
        measurement_snr_list=(
            [  # Positive SNR
                round(from_db(snr_db), 3)
                for snr_db in range(0, 33, 3)
            ] + [  # Negative SNR
                round(from_db(snr_db), 3)
                for snr_db in range(-30, 0, 3)
            ]
        ),
        measurement_von_mises_std_deg_list=[_MAX_STD_VON_MISES_DEG],  # Uniform phase error
        run_calibration_types=[
            CalibrationType.PEAC,
            CalibrationType.UPEC_PATHS_PROJ,
            CalibrationType.PEOC,
        ],
    ),
    ExperimentProtocol(  # Study calibration for different concentration parameters
        protocol_name="prior_concentration_paper",
        measurement_snr_list=[100.0], # 20dB
        measurement_von_mises_std_deg_list=[
            0.0,
            11.573,
            23.116,
            34.66,
            46.204,
            57.748,
            69.292,
            80.835,
            92.379,
            _MAX_STD_VON_MISES_DEG
        ],
        run_calibration_types=[
            CalibrationType.PEAC,
            CalibrationType.UPEC_PATHS_PROJ,
            CalibrationType.PEOC,
        ],
    ),
    ExperimentProtocol(
        protocol_name="bandwidth_paper",
        measurement_snr_list=[100],
        measurement_von_mises_std_deg_list=[0.0],  # No phase noise
        bandwidth_list=[  # 1 MHz to 500 MHz
            1e6,
            2e6,
            5e6,
            1e7,
            2e7,
            5e7,
            1e8,
            2e8,
            5e8
        ],
        run_calibration_types=[
            CalibrationType.PEAC,
            CalibrationType.UPEC_PATHS_PROJ,
            CalibrationType.PEOC,
        ],
    ),
    ExperimentProtocol(
        protocol_name="pos_noise_paper",
        measurement_snr_list=[100],
        measurement_von_mises_std_deg_list=[0.0],  # Note: useless param for pos noise (affects folder name only)
        position_noise_amplitude_list=[
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5
        ],
        run_calibration_types=[
            CalibrationType.PEAC,
            CalibrationType.UPEC_PATHS_PROJ,
            CalibrationType.PEOC,
        ],
    ),
    ExperimentProtocol(
        protocol_name="maxwell_paper",
        measurement_snr_list=[100],
        measurement_von_mises_std_deg_list=[0.0],  # No phase noise
        bandwidth_list=[  # 100 MHz to 500 MHz
            1e8,
            2e8,
            5e8
        ],
        run_calibration_types=[
            CalibrationType.PEAC,
            CalibrationType.UPEC_PATHS_PROJ,
            CalibrationType.PEOC,
        ],
    ),
]
