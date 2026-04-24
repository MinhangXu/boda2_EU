#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

LIBRARY="${LIBRARY:-egfp_1}"
TASK_FAMILY="utr5"
TARGET_FAMILY="polysome"
LAUNCH_SCRIPT="launch/utr5_polysome_utr_bassetvl_sweep.sh"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"
NUM_AGENTS="${NUM_AGENTS:-4}"
NUM_RUNS="${NUM_RUNS:-8}"
read -r -a GPU_ARRAY <<< "${GPU_LIST:-0 1 2 3}"

case "${LIBRARY}" in
  egfp_1)
    CONFIG_PATH="configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__bayes__egfp_1.yml"
    COMPARISON_GROUP="utr5__polysome__within_library__egfp_1"
    ;;
  egfp_2)
    CONFIG_PATH="configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__bayes__egfp_2.yml"
    COMPARISON_GROUP="utr5__polysome__within_library__egfp_2"
    ;;
  mcherry_1)
    CONFIG_PATH="configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__bayes__mcherry_1.yml"
    COMPARISON_GROUP="utr5__polysome__within_library__mcherry_1"
    ;;
  mcherry_2)
    CONFIG_PATH="configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__bayes__mcherry_2.yml"
    COMPARISON_GROUP="utr5__polysome__within_library__mcherry_2"
    ;;
  *)
    echo "Unknown LIBRARY=${LIBRARY}. Use egfp_1, egfp_2, mcherry_1, or mcherry_2." >&2
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
