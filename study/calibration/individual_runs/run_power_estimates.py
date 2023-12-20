import os
import argparse
import timeit
import tensorflow as tf

from settings import LOGS_FOLDER
from src.data_classes import MaterialsMapping, CalibrationChannelsPower, MeasurementType, MeasurementNoiseType, \
    load_measurement_data
from src.utils.channel_power import compute_channel_power_rt_simulated_phases, \
    compute_channel_power_uniform_phase_error
from src.scenarios import get_scenario
from src.ofdm_measurements.main import compute_selected_cfr_from_scratch
from src.ofdm_measurements.simulate_position_noise import simulate_position_noise_cfr
from src.ofdm_measurements.maxwell_simulation_toy_example import get_channel_power_uniform_phases_maxwell_simulation
from study.calibration.utils import RunType, CalibrationType
from study.calibration.experiment_config import get_scenario_metadata, get_measurement_metadata
from study.calibration.experiment_protocol import load_experiment_protocol, get_all_protocol_names
from study.calibration.individual_runs.run_measurements import get_measurements_run_name
from study.calibration.individual_runs.run_calibration import get_calibration_run_name


def compute_channel_power_at_calibration_locations(
    run_type: str,
    measurement_snr: float,
    measurement_additive_noise: bool,
    measurement_perfect_phase: bool,
    measurement_von_mises_mean: float,
    measurement_von_mises_concentration: float,
    bandwidth: float,
    position_noise_amplitude: float,
    n_channel_measurements: int,
    n_run: int,
    calibration_type: str = None
):
    # Get scene
    # ---------
    measurement_run_name = get_measurements_run_name(
        measurement_snr=measurement_snr,
        measurement_additive_noise=measurement_additive_noise,
        measurement_perfect_phase=measurement_perfect_phase,
        measurement_von_mises_mean=measurement_von_mises_mean,
        measurement_von_mises_concentration=measurement_von_mises_concentration,
        bandwidth=bandwidth,
        position_noise_amplitude=position_noise_amplitude,
        n_run=n_run,
    )
    # Get material properties
    if run_type == RunType.MEASUREMENT:
        loaded_run_name = measurement_run_name
    elif run_type == RunType.CALIBRATION:
        loaded_run_name = get_calibration_run_name(
            measurement_snr=measurement_snr,
            calibration_type=calibration_type,
            measurement_additive_noise=measurement_additive_noise,
            measurement_perfect_phase=measurement_perfect_phase,
            measurement_von_mises_mean=measurement_von_mises_mean,
            measurement_von_mises_concentration=measurement_von_mises_concentration,
            bandwidth=bandwidth,
            position_noise_amplitude=position_noise_amplitude,
            n_channel_measurements=n_channel_measurements,
            n_run=n_run
        )
    else:
        raise ValueError(f"Run type '{run_type}' not supported")

    loaded_run_dir = os.path.join(LOGS_FOLDER, loaded_run_name)
    materials_mapping = MaterialsMapping.load(loaded_run_dir)

    # Get measurement metadata
    measurement_metadata = get_measurement_metadata(
        measurement_snr=measurement_snr,
        measurement_additive_noise=measurement_additive_noise,
        measurement_perfect_phase=measurement_perfect_phase,
        measurement_von_mises_mean=measurement_von_mises_mean,
        measurement_von_mises_concentration=measurement_von_mises_concentration,
        bandwidth=bandwidth,
        position_noise_amplitude=position_noise_amplitude
    )

    # Get scenario (use ground-truth scenario for measurement runs only)
    ground_truth_geometry = (run_type == RunType.MEASUREMENT)
    scenario_metadata = get_scenario_metadata(ground_truth_geometry=ground_truth_geometry)
    scene = get_scenario(scenario_metadata, materials_mapping=materials_mapping)

    # Get power
    # ---------
    if (  # For random position displacements in measurement runs, average the power at each position
        (run_type == RunType.MEASUREMENT) and
        (measurement_metadata.__measurement_type__ == MeasurementType.RAY_TRACING) and
        (measurement_metadata.__noise_type__ == MeasurementNoiseType.POSITION)
    ):
        # Compute the CFR at each sampled position (due to position noise)
        measurement_data = load_measurement_data(os.path.join(LOGS_FOLDER, measurement_run_name))
        all_pos_cfr_per_mpc = simulate_position_noise_cfr(
            scene=scene,
            n_measurements=measurement_metadata.n_measurements_per_channel,
            max_depth_path=measurement_metadata.max_depth_path,
            num_samples_path=measurement_metadata.num_samples_path,
            num_subcarriers=measurement_metadata.num_subcarriers,
            subcarrier_spacing=measurement_metadata.subcarrier_spacing,
            rx_tx_indexes=measurement_metadata.rx_tx_indexes,
            position_noise_rx=measurement_data.position_noise_rx,
            position_noise_tx=measurement_data.position_noise_tx
        )
        # Compute power at each location
        all_pos_uniform_phases_power = compute_channel_power_uniform_phase_error(cfr_per_mpc=all_pos_cfr_per_mpc)
        all_pos_simulated_phases_power = compute_channel_power_rt_simulated_phases(cfr_per_mpc=all_pos_cfr_per_mpc)
        # Average the powers at each simulated location
        uniform_phases_power = tf.reduce_mean(all_pos_uniform_phases_power, axis=0)
        simulated_phases_power = tf.reduce_mean(all_pos_simulated_phases_power, axis=0)
    elif (  # For Maxwell simulations, the ground-truth channel power is pre-computed over a larger bandwidth
        (run_type == RunType.MEASUREMENT) and
        (measurement_metadata.__measurement_type__ == MeasurementType.MAXWELL_SIMULATION)
    ):
        # Simulated phases power
        measurement_data = load_measurement_data(os.path.join(LOGS_FOLDER, measurement_run_name))
        measurement_without_noise = measurement_data.normalization_constant * (
            measurement_data.measurement[0] - measurement_data.measurement_noise[0]
        )
        simulated_phases_power = tf.pow(tf.abs(measurement_without_noise), 2)
        # Uniform phases power
        uniform_phases_power = get_channel_power_uniform_phases_maxwell_simulation(  # [N_RX_TX_PAIRS, N_ARR_RX, N_ARR_TX]
            freq_axis=measurement_data.freq_axis_maxwell,
            freq_response=measurement_data.freq_response_maxwell
        )
        n_carriers = measurement_data.measurement.shape[2]
        uniform_phases_power = tf.tile(  # [N_RX_TX_PAIRS, N_CARRIERS, N_ARR_RX, N_ARR_TX]
            uniform_phases_power[:, tf.newaxis, :, :],
            [1, n_carriers, 1, 1],
        )
    else:  # In all other cases, the CFR needs only to be computed once at the original position
        selected_cfr_per_mpc = compute_selected_cfr_from_scratch(
            scene=scene,
            measurement_metadata=measurement_metadata,
            normalization_constant=1.0,
            check_scene=True,
            check_paths=True
        )
        uniform_phases_power = compute_channel_power_uniform_phase_error(cfr_per_mpc=selected_cfr_per_mpc)
        simulated_phases_power = compute_channel_power_rt_simulated_phases(cfr_per_mpc=selected_cfr_per_mpc)

    # Save
    channels_power_dataclass = CalibrationChannelsPower(
        uniform_phases_power=uniform_phases_power,
        simulated_phases_power=simulated_phases_power
    )
    channels_power_dataclass.store(loaded_run_dir)


def run_power_estimates(
    n_run: int,
    protocol_name: str,
    skip_measurement: bool = False,
    # Force runs that are not specified in the given protocol
    force_calibration_type: str = None
):
    protocol_parameters = load_experiment_protocol(protocol_name)

    # Run experiments
    for run_parameters in protocol_parameters:
        print(run_parameters)
        start_timer = timeit.default_timer()

        # Run measurements
        if not skip_measurement:
            compute_channel_power_at_calibration_locations(
                run_type=RunType.MEASUREMENT,
                measurement_snr=run_parameters.measurement_snr,
                measurement_additive_noise=run_parameters.measurement_additive_noise,
                measurement_perfect_phase=run_parameters.measurement_perfect_phase,
                measurement_von_mises_mean=run_parameters.measurement_von_mises_mean,
                measurement_von_mises_concentration=run_parameters.measurement_von_mises_concentration,
                bandwidth=run_parameters.bandwidth,
                position_noise_amplitude=run_parameters.position_noise_amplitude,
                n_channel_measurements=run_parameters.n_channel_measurements,
                n_run=n_run
            )

        calibration_types_to_run = (
            [force_calibration_type]  # Run manually specified calibration type
            if force_calibration_type is not None else
            run_parameters.run_calibration_types  # Run calibration types specified in the protocol
        )
        for calibration_type in calibration_types_to_run:
            compute_channel_power_at_calibration_locations(
                run_type=RunType.CALIBRATION,
                measurement_snr=run_parameters.measurement_snr,
                measurement_additive_noise=run_parameters.measurement_additive_noise,
                measurement_perfect_phase=run_parameters.measurement_perfect_phase,
                measurement_von_mises_mean=run_parameters.measurement_von_mises_mean,
                measurement_von_mises_concentration=run_parameters.measurement_von_mises_concentration,
                bandwidth=run_parameters.bandwidth,
                position_noise_amplitude=run_parameters.position_noise_amplitude,
                n_channel_measurements=run_parameters.n_channel_measurements,
                n_run=n_run,
                calibration_type=calibration_type,
            )

        # End run
        end_timer = timeit.default_timer()
        duration_run = end_timer - start_timer
        print(f"Run executed in {duration_run // 60} min and {duration_run % 60:.1f} s")


if __name__ == '__main__':
    protocol_names = get_all_protocol_names()

    parser = argparse.ArgumentParser(
        "Run series of measurement and calibration experiments for the different SNRs and phase-error concentrations "
        "specified in the protocol"
    )
    parser.add_argument("--n-run", dest="n_run", type=int, help="Select run number")
    parser.add_argument("--protocol-name", dest="protocol_name", type=str,
                        help=f"Select protocol to run among: {'|'.join(protocol_names)}")
    parser.add_argument("--skip-measurement", dest="skip_measurement", default=False, action="store_true",
                        help="If set, skip measurement runs")
    parser.add_argument("--force-calibration-type", dest="force_calibration_type",  type=str, default=None,
                        help=f"Select calibration type to run among: {'|'.join(CalibrationType.all_types())}")
    args = parser.parse_args()

    _force_calibration_type = None if args.force_calibration_type in (None, "") else args.force_calibration_type

    run_power_estimates(
        n_run=args.n_run,
        protocol_name=args.protocol_name,
        skip_measurement=args.skip_measurement,
        force_calibration_type=_force_calibration_type
    )
