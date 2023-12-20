#!/usr/bin/env bash

export STUDY_EXPERIMENT_VERSION="v4"
PROTOCOL_NAME="maxwell_paper"
N_RUNS_TO_LOAD=10
PLOT_TYPE="power_profiles"

python study/calibration/run_plots_protocol.py \
    --n-runs=$N_RUNS_TO_LOAD \
    --protocol-name=$PROTOCOL_NAME \
    --plot-type=$PLOT_TYPE \
    --skip-gather-data-calibration \
    --gather-data-maxwell