#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

MODE="${MODE:-basic}"
TASK_FAMILY="enhancer"
TARGET_FAMILY="bashor_in_house_fastqs1_5_filtered"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"
NUM_AGENTS="${NUM_AGENTS:-4}"
NUM_RUNS="${NUM_RUNS:-8}"
PREPARE_DATASET="${PREPARE_DATASET:-1}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPU_ARRAY <<< "${GPU_LIST:-0 1 2 3}"

if [[ "${PREPARE_DATASET}" == "1" ]]; then
  python "${LEARN_DIR}/prepare_lib1_enhancer_fastqs1_5_dataset.py"
fi

case "${MODE}" in
  basic)
    CONFIG_PATH="configs/enhancer/bashor_in_house/lib1_enhancer_fastqs1_5__scratch_basic__bayes.yml"
    COMPARISON_GROUP="enhancer__bashor_in_house__fastqs1_5_filtered__scratch_basic__bassetvl_vs_resnet1d"
    ;;
  weighted)
    CONFIG_PATH="configs/enhancer/bashor_in_house/lib1_enhancer_fastqs1_5__scratch_weighted__bayes.yml"
    COMPARISON_GROUP="enhancer__bashor_in_house__fastqs1_5_filtered__scratch_weighted__bassetvl_vs_resnet1d"
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use basic or weighted." >&2
    exit 1
    ;;
esac

LAUNCH_SCRIPT="launch/lib1_enhancer_fastqs1_5_scratch_compare_loss_modes.sh"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "LIB1 FASTQS1-5 SCRATCH HPO"
  echo "MODE:             ${MODE}"
  echo "CONFIG_PATH:      ${CONFIG_PATH}"
  echo "TASK_FAMILY:      ${TASK_FAMILY}"
  echo "TARGET_FAMILY:    ${TARGET_FAMILY}"
  echo "COMPARISON_GROUP: ${COMPARISON_GROUP}"
  echo "NUM_AGENTS:       ${NUM_AGENTS}"
  echo "NUM_RUNS:         ${NUM_RUNS}"
  echo "GPU_LIST:         ${GPU_ARRAY[*]}"
  echo "PREPARE_DATASET:  ${PREPARE_DATASET}"
  echo "Dry run only; no W&B sweep or agents were started."
  exit 0
fi

launch_wandb_agents \
  "${CONFIG_PATH}" \
  "${TASK_FAMILY}" \
  "${TARGET_FAMILY}" \
  "${COMPARISON_GROUP}" \
  "${LAUNCH_SCRIPT}" \
  "${NUM_AGENTS}" \
  "${NUM_RUNS}" \
  "${GPU_ARRAY[@]}"
