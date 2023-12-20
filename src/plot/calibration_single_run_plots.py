import numpy as np
import matplotlib.pyplot as plt

from src.utils.plot_utils import set_rc_params
from src.utils.telecom_utils import get_material_reflection_coefficients


def plot_training_loss(
    training_loss: np.ndarray,  # [NUM_STEPS]
):
    # Setup plot params
    set_rc_params()

    # Plot
    fig, ax = plt.subplots()
    fig.set_size_inches((8, 6))
    ax.set_title("Training Loss")
    ax.set_xlabel("Calibration Step")
    ax.set_ylabel("Loss")
    ax.plot(training_loss)

    return ax


def plot_calibration_materials(
    track_permittivities: np.ndarray,  # [NUM_STEPS, N_MATERIALS]
    track_conductivities: np.ndarray,  # [NUM_STEPS, N_MATERIALS]
    true_permittivities: np.ndarray = None,  # [N_MATERIALS]
    true_conductivities: np.ndarray = None,  # [N_MATERIALS]
):
    num_steps, n_materials = track_permittivities.shape

    # Setup plot params
    set_rc_params()

    fig, axs = plt.subplots(n_materials, 2)
    axs = axs.reshape((n_materials, 2))
    fig.set_size_inches((16, 6 * n_materials))
    # fig.tight_layout()
    colors = ["g", "b", "red"]

    for i in range(n_materials):
        # Relative permittivity
        axs[i, 0].plot(track_permittivities[:, i], '-', color=colors[i], label="Learned value")
        if true_permittivities is not None:
            axs[i, 0].plot(
                np.repeat(true_permittivities[i], num_steps),
                '--',
                color=colors[i],
                label="True value"
            )
        axs[i, 0].set_title(f"Relative Permittivity for Material {i}")
        axs[i, 0].set_xlabel("Calibration Step")
        axs[i, 0].set_ylabel(r"Relative permittivity")
        axs[i, 0].legend()

        # Conductivity
        axs[i, 1].plot(track_conductivities[:, i], '-', color=colors[i], label="Learned value")
        if true_conductivities is not None:
            axs[i, 1].plot(
                np.repeat(true_conductivities[i], num_steps),
                '--',
                color=colors[i],
                label="True value"
            )
        axs[i, 1].set_title(f"Conductivity for Material {i}")
        axs[i, 1].set_xlabel("Calibration step")
        axs[i, 1].set_ylabel(r"Conductivity")
        axs[i, 1].legend()

    return axs


def plot_calibrated_reflection_coefficients(
    frequency: float,
    calibrated_permittivities: np.ndarray,  # Shape [N_MATERIALS]
    calibrated_conductivities: np.ndarray,  # Shape [N_MATERIALS]
    n_points: int = 100,
    true_permittivities: np.ndarray = None,  # Shape [N_MATERIALS]
    true_conductivities: np.ndarray = None,  # Shape [N_MATERIALS]
):
    n_materials = calibrated_permittivities.shape[0]
    angles = np.linspace(0, np.pi/2, n_points)
    angles_deg = (180 / np.pi) * angles

    # Setup plot params
    set_rc_params()

    # Plot
    fig, axs = plt.subplots(n_materials, 2)
    axs = axs.reshape((n_materials, 2))
    fig.set_size_inches((20, 8 * n_materials))
    plot_kwargs = dict(
        lw=3
    )

    for i in range(n_materials):
        # Get data
        calibrated_r_p, calibrated_r_s = get_material_reflection_coefficients(
            real_permittivity=calibrated_permittivities[i],
            conductivity=calibrated_conductivities[i],
            frequency=frequency,
            angle_of_incidence=angles
        )
        true_r_s, true_r_p = None, None
        if (true_permittivities is not None) and (true_conductivities is not None):
            true_r_p, true_r_s = get_material_reflection_coefficients(
                real_permittivity=true_permittivities[i],
                conductivity=true_conductivities[i],
                frequency=frequency,
                angle_of_incidence=angles
            )

        # S-polarization
        axs[i, 0].plot(angles_deg, np.abs(calibrated_r_s), '-', color="red", label=r"Calibrated $r_s$", **plot_kwargs)
        if true_r_s is not None:
            axs[i, 0].plot(angles_deg, np.abs(true_r_s), '--', color="black", label=r"True $r_s$", **plot_kwargs)
        axs[i, 0].set_title(f"S-polarized Reflection Coefficient for Material {i}")
        axs[i, 0].set_xlabel("Angle of incidence [deg]")
        axs[i, 0].set_ylabel(r"$|r_s|$")
        axs[i, 0].legend()

        # P-polarization
        axs[i, 1].plot(angles_deg, np.abs(calibrated_r_p), '-', color="red", label=r"Calibrated $r_p$", **plot_kwargs)
        if true_r_p is not None:
            axs[i, 1].plot(angles_deg, np.abs(true_r_p), '--', color="black", label=r"True $r_p$", **plot_kwargs)
        axs[i, 1].set_title(f"P-polarized Reflection Coefficient for Material {i}")
        axs[i, 1].set_xlabel("Angle of incidence [deg]")
        axs[i, 1].set_ylabel(r"$|r_p|$")
        axs[i, 1].legend()

    return axs