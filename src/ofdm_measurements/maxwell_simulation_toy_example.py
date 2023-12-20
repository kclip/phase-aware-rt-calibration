import os
import math
from typing import Tuple, List
import h5py
import numpy as np
import tensorflow as tf
from sionna.channel import subcarrier_frequencies

from settings import PROJECT_FOLDER
from const import SPEED_OF_LIGHT, FREE_SPACE_IMPEDANCE
from src.utils.fft_utils import get_power_profile, filter_freq_response
from src.ofdm_measurements import _MeasurementMetadataBase


MAXWELL_SIMULATION_ASSET_FILEPATH = os.path.join(
    PROJECT_FOLDER,
    "assets",
    "maxwell_simulation",
    "toy_example_maxwell.out"
)

MAXWELL_SIMULATION_CENTRAL_FREQUENCY = 6e9
MAXWELL_SIMULATION_BANDWIDTH = 2e9  # The bandwidth should be large enough to resolve each path


# Frequency Response
# ------------------

def _get_input_ricker_impulse(time_axis: np.ndarray, frequency: float, amplitude: float):
    chi = np.sqrt(2) / frequency
    zeta = (np.pi * frequency) ** 2
    delay = chi
    delayed_time_axis = time_axis - delay
    ricker_val = -amplitude * (
        (2 * zeta * (delayed_time_axis ** 2)) - 1
    ) * np.exp(
        -zeta * (delayed_time_axis ** 2)
    )
    return ricker_val, delay


def _get_simulated_field_at_rx(
    simulation_data,
    rx_tx_indexes: List[Tuple[int]],
    n_points_target: int = None,
    undersampling_factor: int = 1
) -> np.ndarray:  # Shape [N_RX, N_POINTS]
    time_E_field = np.stack(
        [
            simulation_data.get("rxs").get(f"rx{rx_idx+1}").get("Ez")[::undersampling_factor]
            for rx_idx, _ in rx_tx_indexes
        ],
        axis=0
    )
    if n_points_target is not None:
        time_E_field = np.pad(time_E_field, ((0, 0), (0, n_points_target - time_E_field.shape[1])))
    return time_E_field


def _get_frequency_response(
    time_input_current: np.ndarray,  # shape [N_POINTS]
    time_output_field: np.ndarray,  # shape [N_RX, N_POINTS]
    time_step: float,
    dipole_length: float,
    fill_na: float = 0
) -> Tuple[
    np.ndarray,  # Frequencies axis ; shape [N_POINTS]
    np.ndarray  # Frequency response at each Rx ; shape [N_RX, N_POINTS]
]:
    """Compute E fields frequency response as E_out(f) / E_in(f) for an time impulse starting at t=0"""
    # Input/Output FFTs
    n_points = time_input_current.size
    freq_input_current = np.fft.fftshift(np.fft.fft(time_input_current))
    freq_output_field = np.fft.fftshift(np.fft.fft(time_output_field, axis=1), axes=(1,))
    freq_axis = np.fft.fftshift(np.fft.fftfreq(n_points, d=time_step))

    # Coefficient E_0 [V] of dipole electric field E(r) = E_0 sin(\theta) e^{-j k r} / r
    wave_number = (2 * np.pi * freq_axis) / SPEED_OF_LIGHT
    freq_input_field_E0 = (
        (-1j * FREE_SPACE_IMPEDANCE * freq_input_current * dipole_length * wave_number) /
        (4 * np.pi)
    )

    # Compute frequency response
    with np.errstate(divide='ignore', invalid='ignore'):
        freq_response = np.true_divide(freq_output_field, freq_input_field_E0[np.newaxis, :])
    freq_response[~np.isfinite(freq_response)] = fill_na

    return freq_axis, freq_response


# Power
# -----

def get_channel_power_uniform_phases_maxwell_simulation(
    freq_axis: tf.Tensor,  # Shape [N_POINTS]
    freq_response: tf.Tensor,  # Shape [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX, N_ARR_TX]
) -> tf.Tensor:  # Shape [N_RX_TX_PAIRS, N_ARR_RX, N_ARR_TX]
    """
    Compute the channel power from the channel frequency responses assuming that phases are uniformly distributed
    at each path and that the available bandwidth is large enough to identify each path.
    """
    _, power_profile = get_power_profile(
        freq_axis=freq_axis,
        freq_response=freq_response
    )

    return tf.reduce_sum(power_profile, axis=1)


def get_mean_channel_power_maxwell_simulation(
    freq_axis: tf.Tensor,  # Shape [N_POINTS]
    freq_response: tf.Tensor,  # Shape [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX, N_ARR_TX]
) -> tf.Tensor:  # Shape [N_RX_TX_PAIRS]
    power_uniform_phases = get_channel_power_uniform_phases_maxwell_simulation(
        freq_axis=freq_axis,
        freq_response=freq_response
    )

    return tf.reduce_mean(power_uniform_phases, axis=[1, 2])


# Channel frequency response
# --------------------------

def cfr_and_freq_response_maxwell_simulation_toy_example(
    measurement_metadata: _MeasurementMetadataBase,
    carrier_frequency: float
) -> Tuple[
    tf.Tensor,  # measured CFR ; shape [N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX=1, N_ARR_TX=1]
    tf.Tensor,  # frequency axis of Maxwell simulation ; shape [N_POINTS]
    tf.Tensor  # Frequency responses of the Maxwell simulation ; shape [N_RX_TX_PAIRS, N_POINTS, N_ARR_RX=1, N_ARR_TX=1]
]:
    print(
        "WARNING: MAXWELL SIMULATION MEASUREMENT IS ONLY SUPPORTED FOR THE 'TOY EXAMPLE MAXWELL' SCENARIO UNDER THE SETTINGS:"
        "   - 2D SIMULATION (XY-PLANE) WITH A SINGLE TRANSMITTER"
        "   - VERTICAL POLARIZATION OF TRANSMITTED SIGNAL (Z-AXIS)"
        "   - DEVICES CONSISTING OF A SINGLE ISOTROPIC ANTENNA"
        "   - MAXWELL SIMULATION WITH RICKER IMPULSE INPUT WITH AMPLITUDE=1.0 AND FREQUENCY=6GHz"
    )

    # Get frequency responses of electric fields
    # ------------------------------------------
    # Open gprMax simulation output
    simulation_data = h5py.File(MAXWELL_SIMULATION_ASSET_FILEPATH, 'r')
    spatial_resolution = min(simulation_data.attrs['dx_dy_dz'])
    time_resolution_base = simulation_data.attrs['dt']

    # Pad signal to get same freq resolution as subcarriers
    # (assumes FDTD simulation is long enough to capture all propagated signals)
    undersampling_factor = 10
    time_resolution = undersampling_factor * time_resolution_base
    time_window = 1 / measurement_metadata.subcarrier_spacing
    n_points = math.ceil(time_window / time_resolution)

    time_axis = np.linspace(0, (n_points - 1) * time_resolution, num=n_points)

    # Get electric field generated at Tx ; shape [N_POINTS]
    time_input_current, _ = _get_input_ricker_impulse(
        time_axis=time_axis,
        frequency=MAXWELL_SIMULATION_CENTRAL_FREQUENCY,
        amplitude=1.0
    )

    # Get electric field at Rx ; shape [N_RX, N_POINTS]
    time_E_field_output = _get_simulated_field_at_rx(
        simulation_data=simulation_data,
        rx_tx_indexes=measurement_metadata.rx_tx_indexes,
        n_points_target=n_points,
        undersampling_factor=undersampling_factor
    )

    # Get electric field frequency response E_out(f) / E_in(f) ; shape [N_RX, N_POINTS]
    freq_axis, freq_response_E_field = _get_frequency_response(
        time_input_current=time_input_current,
        time_output_field=time_E_field_output,
        time_step=time_resolution,
        dipole_length=spatial_resolution
    )

    # Get channel frequency response (for isotropic Tx and Rx antennas)
    # -----------------------------------------------------------------
    antenna_gain_rx = 1  # Isotropic antenna
    antenna_gain_tx = 1  # Isotropic antenna
    antennas_coef = np.sqrt(antenna_gain_rx * antenna_gain_tx)

    with np.errstate(divide='ignore', invalid='ignore'):
        wavelength_axis = np.true_divide(SPEED_OF_LIGHT, freq_axis)
        wavelength_axis[~np.isfinite(wavelength_axis)] = 0.0

    freq_channel_responses = (wavelength_axis / (4 * np.pi)) * antennas_coef * freq_response_E_field

    # Interpolate broadband frequency response to the subcarriers frequencies
    f_subcarriers = carrier_frequency + subcarrier_frequencies(
        num_subcarriers=measurement_metadata.num_subcarriers,
        subcarrier_spacing=measurement_metadata.subcarrier_spacing
    )
    cfr_maxwell = np.stack(  # Shape [N_RX, N_SUBCARRIERS]
        [
            np.interp(f_subcarriers, freq_axis, freq_channel_response_rx)
            for freq_channel_response_rx in freq_channel_responses
        ],
        axis=0
    )

    # Format responses to tf.Tensor[N_MEASUREMENTS, N_RX_TX_PAIRS, N_SUBCARRIERS, N_ARR_RX=1, N_ARR_TX=1]
    cfr_maxwell = tf.constant(cfr_maxwell, dtype=tf.complex64)
    cfr_maxwell = tf.tile(
        cfr_maxwell[tf.newaxis, :, :, tf.newaxis, tf.newaxis],
        [measurement_metadata.n_measurements_per_channel, 1, 1, 1, 1]
    )

    # Get Maxwell simulation frequency responses (larger bandwidth)
    # -------------------------------------------------------------
    # Filter around carrier
    selected_freq_axis = tf.constant(freq_axis, dtype=tf.float32)
    selected_freq_response = tf.constant(freq_channel_responses, dtype=tf.complex64)
    selected_freq_axis, selected_freq_response = filter_freq_response(
        freq_axis=selected_freq_axis,
        freq_response=selected_freq_response,
        axis=1,
        central_frequency=MAXWELL_SIMULATION_CENTRAL_FREQUENCY,
        bandwidth=MAXWELL_SIMULATION_BANDWIDTH
    )
    # Format responses to tf.Tensor[N_RX_TX_PAIRS, N_POINTS, N_ARR_RX=1, N_ARR_TX=1]
    selected_freq_response = selected_freq_response[:, :, tf.newaxis, tf.newaxis]

    return cfr_maxwell, selected_freq_axis, selected_freq_response


