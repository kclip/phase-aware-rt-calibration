#!/usr/bin/env bash

# Calibration error vs. prior phase-error concentration plots

export STUDY_EXPERIMENT_VERSION="v1"
PROTOCOL_NAME="prior_concentration_paper"
N_RUNS_TO_LOAD=10
PLOT_TYPE="calibration_error_vs_std"

python study/calibration/run_plots_protocol.py \
    --n-runs=$N_RUNS_TO_LOAD \
    --protocol-name=$PROTOCOL_NAME \
    --plot-type=$PLOT_TYPE


# Calibration error vs. SNR plots

export STUDY_EXPERIMENT_VERSION="v1"
PROTOCOL_NAME="snr_paper"
N_RUNS_TO_LOAD=10
PLOT_TYPE="calibration_error_vs_snr"

python study/calibration/run_plots_protocol.py \
    --n-runs=$N_RUNS_TO_LOAD \
    --protocol-name=$PROTOCOL_NAME \
    --plot-type=$PLOT_TYPE


# Calibration error vs. position noise plots

export STUDY_EXPERIMENT_VERSION="v3"
PROTOCOL_NAME="pos_noise_paper"
N_RUNS_TO_LOAD=10
PLOT_TYPE="calibration_error_vs_position_noise"

python study/calibration/run_plots_protocol.py \
    --n-runs=$N_RUNS_TO_LOAD \
    --protocol-name=$PROTOCOL_NAME \
    --plot-type=$PLOT_TYPE