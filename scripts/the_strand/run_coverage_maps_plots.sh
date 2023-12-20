#!/usr/bin/env bash

export STUDY_EXPERIMENT_VERSION="v1"
N_RUNS_TO_LOAD=10
ARRAY_CONFIG="mimo"
GROUND_TRUTH_COVERAGE_MAP_RUN_NAME="calibration_v1_coverage_map_mimo_from_mesurement_run_0_snr_100_0_vm_data_mean_0_0_vm_data_concentration_0_0"


# Coverage Map Error vs. SNR

PROTOCOL_NAME="snr_paper"
PLOT_TYPE="coverage_map_error_vs_snr"

python study/calibration/run_coverage_map_plots_protocol.py \
    --n-runs=$N_RUNS_TO_LOAD \
    --protocol-name=$PROTOCOL_NAME \
    --array-config=$ARRAY_CONFIG \
    --ground-truth-coverage-map-run-name=$GROUND_TRUTH_COVERAGE_MAP_RUN_NAME \
    --plot-type=$PLOT_TYPE


# Coverage Map Error vs. phase-error concentration

PROTOCOL_NAME="prior_concentration_paper"
PLOT_TYPE="coverage_map_error_vs_std"

python study/calibration/run_coverage_map_plots_protocol.py \
    --n-runs=$N_RUNS_TO_LOAD \
    --protocol-name=$PROTOCOL_NAME \
    --array-config=$ARRAY_CONFIG \
    --ground-truth-coverage-map-run-name=$GROUND_TRUTH_COVERAGE_MAP_RUN_NAME \
    --plot-type=$PLOT_TYPE


# Coverage Map plots for SNR

PROTOCOL_NAME="snr_paper"
PLOT_TYPE="coverage_map"

python study/calibration/run_coverage_map_plots_protocol.py \
    --n-runs=$N_RUNS_TO_LOAD \
    --protocol-name=$PROTOCOL_NAME \
    --array-config=$ARRAY_CONFIG \
    --ground-truth-coverage-map-run-name=$GROUND_TRUTH_COVERAGE_MAP_RUN_NAME \
    --plot-type=$PLOT_TYPE \
    --skip-gather-data


# Coverage Map plots for phase-error concentrations

PROTOCOL_NAME="prior_concentration_paper"
PLOT_TYPE="coverage_map"

python study/calibration/run_coverage_map_plots_protocol.py \
    --n-runs=$N_RUNS_TO_LOAD \
    --protocol-name=$PROTOCOL_NAME \
    --array-config=$ARRAY_CONFIG \
    --ground-truth-coverage-map-run-name=$GROUND_TRUTH_COVERAGE_MAP_RUN_NAME \
    --plot-type=$PLOT_TYPE \
    --skip-gather-data