#!/usr/bin/env bash

export TF_ENABLE_AUTO_GC=1
export TF_GPU_ALLOCATOR="cuda_malloc_async"

export STUDY_EXPERIMENT_VERSION="v1"
N_RUNS=10
ARRAY_CONFIG="mimo"


# Ground-Truth Coverage Map

python study/calibration/individual_runs/run_coverage_map.py \
    --load-materials-run-type="measurement" \
    --meas-snr=100.0 \
    --meas-vm-mean=0.0 \
    --meas-vm-concentration=0.0 \
    --array-config="mimo" \
    --ignore-cfr-data \
    --n-run=0


# Calibrated Coverage Maps for different SNR levels

PROTOCOL_NAME="snr_paper"

for RUN_ID in $(seq 0 $((N_RUNS - 1)))
do
  python study/calibration/run_coverage_map_protocol.py \
    --n-run=$RUN_ID \
    --protocol-name=$PROTOCOL_NAME \
    --array-config=$ARRAY_CONFIG \
    --ignore-cfr-data
done


# Calibrated Coverage Maps for different phase-error concentration levels

PROTOCOL_NAME="prior_concentration_paper"

for RUN_ID in $(seq 0 $((N_RUNS - 1)))
do
  python study/calibration/run_coverage_map_protocol.py \
    --n-run=$RUN_ID \
    --protocol-name=$PROTOCOL_NAME \
    --array-config=$ARRAY_CONFIG \
    --ignore-cfr-data
done