#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

CONFIG_PATH="configs/enhancer/malinois_mpra/basset_branched/enhancer__malinois_mpra__basset_branched__transfer_baseline.yml"
TASK_FAMILY="enhancer"
TARGET_FAMILY="malinois_mpra"
COMPARISON_GROUP="enhancer__malinois_mpra__baseline_transfer"
LAUNCH_SCRIPT="launch/enhancer_malinois_basset_branched_baseline.sh"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"

NUM_AGENTS="${NUM_AGENTS:-1}"
NUM_RUNS="${NUM_RUNS:-1}"
read -r -a GPU_ARRAY <<< "${GPU_LIST:-0}"

launch_wandb_agents \
  "${CONFIG_PATH}" \
  "${TASK_FAMILY}" \
  "${TARGET_FAMILY}" \
  "${COMPARISON_GROUP}" \
  "${LAUNCH_SCRIPT}" \
  "${NUM_AGENTS}" \
  "${NUM_RUNS}" \
  "${GPU_ARRAY[@]}"
