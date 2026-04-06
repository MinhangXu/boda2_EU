#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

MODE="${MODE:-basic}"
TASK_FAMILY="enhancer"
TARGET_FAMILY="bashor_lib1"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"
NUM_AGENTS="${NUM_AGENTS:-4}"
NUM_RUNS="${NUM_RUNS:-8}"
read -r -a GPU_ARRAY <<< "${GPU_LIST:-0 1 2 3 4 5 6 7}"

case "${MODE}" in
  basic)
    CONFIG_PATH="/home/minhang/synBio_AL/boda2_EU/src/learn/configs/enhancer/bashor_in_house/lib1_enhancer__scratch_basic__bayes.yml"
    COMPARISON_GROUP="enhancer__bashor_lib1__scratch_basic__bassetvl_vs_resnet1d"
    LAUNCH_SCRIPT="launch/lib1_enhancer_scratch_compare_loss_modes.sh"
    ;;
  weighted)
    CONFIG_PATH="/home/minhang/synBio_AL/boda2_EU/src/learn/configs/enhancer/bashor_in_house/lib1_enhancer__scratch_weighted__bayes.yml"
    COMPARISON_GROUP="enhancer__bashor_lib1__scratch_weighted__bassetvl_vs_resnet1d"
    LAUNCH_SCRIPT="launch/lib1_enhancer_scratch_weighted_sweep.sh"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use basic or weighted." >&2
    exit 1
    ;;
esac

launch_wandb_agents \
  "${CONFIG_PATH}" \
  "${TASK_FAMILY}" \
  "${TARGET_FAMILY}" \
  "${COMPARISON_GROUP}" \
  "${LAUNCH_SCRIPT}" \
  "${NUM_AGENTS}" \
  "${NUM_RUNS}" \
  "${GPU_ARRAY[@]}"
