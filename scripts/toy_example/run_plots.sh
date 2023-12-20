#!/usr/bin/env bash

export STUDY_EXPERIMENT_VERSION="v2"
PROTOCOL_NAME="bandwidth_paper"
N_RUNS_TO_LOAD=10
PLOT_TYPE="calibration_error_vs_bandwidth"

python study/calibration/run_plots_protocol.py \
    --n-runs=$N_RUNS_TO_LOAD \
    --protocol-name=$PROTOCOL_NAME \
    --plot-type=$PLOT_TYPE