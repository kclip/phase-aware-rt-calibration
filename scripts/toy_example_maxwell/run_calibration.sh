#!/usr/bin/env bash

export STUDY_EXPERIMENT_VERSION="v4"
N_RUNS=10
PROTOCOL_NAME="maxwell_paper"

for RUN_ID in $(seq 0 $((N_RUNS - 1)))
do
  python study/calibration/run_experiment_protocol.py \
      --n-run=$RUN_ID \
      --protocol-name=$PROTOCOL_NAME
done