#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

CONFIG_PATH="configs/utr3/hani_rna_activity/utr_bassetvl/utr3__hani_rna_activity__utr_bassetvl__bayes.yml"
TASK_FAMILY="utr3"
TARGET_FAMILY="hani_rna_activity"
COMPARISON_GROUP="utr3__hani_rna_activity__baseline_architecture"
LAUNCH_SCRIPT="launch/utr3_hani_utr_bassetvl_sweep.sh"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"

NUM_AGENTS="${NUM_AGENTS:-4}"
NUM_RUNS="${NUM_RUNS:-8}"
read -r -a GPU_ARRAY <<< "${GPU_LIST:-0 1 2 3}"

launch_wandb_agents \
  "${CONFIG_PATH}" \
  "${TASK_FAMILY}" \
  "${TARGET_FAMILY}" \
  "${COMPARISON_GROUP}" \
  "${LAUNCH_SCRIPT}" \
  "${NUM_AGENTS}" \
  "${NUM_RUNS}" \
  "${GPU_ARRAY[@]}"
