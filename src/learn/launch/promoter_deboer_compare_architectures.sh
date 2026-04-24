#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

MODE="${MODE:-utr_bassetvl}"
TASK_FAMILY="promoter"
TARGET_FAMILY="deboer_core"
COMPARISON_GROUP="promoter__deboer_core__architecture_comparison"
LAUNCH_SCRIPT="launch/promoter_deboer_compare_architectures.sh"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"
NUM_AGENTS="${NUM_AGENTS:-4}"
NUM_RUNS="${NUM_RUNS:-8}"
read -r -a GPU_ARRAY <<< "${GPU_LIST:-0 1 2 3}"

case "${MODE}" in
  utr_bassetvl)
    CONFIG_PATH="configs/promoter/deboer_core/utr_bassetvl/promoter__deboer_core__utr_bassetvl__bayes.yml"
    ;;
  bassetvl)
    CONFIG_PATH="configs/promoter/deboer_core/bassetvl/promoter__deboer_core__bassetvl__bayes.yml"
    ;;
  resnet1d)
    CONFIG_PATH="configs/promoter/deboer_core/resnet1d/promoter__deboer_core__resnet1d__bayes.yml"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use utr_bassetvl, bassetvl, or resnet1d." >&2
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
