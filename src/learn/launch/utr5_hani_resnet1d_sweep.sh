#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

CONFIG_PATH="configs/utr5/hani_rna_activity/resnet1d/utr5__hani_rna_activity__resnet1d__bayes.yml"
TASK_FAMILY="utr5"
TARGET_FAMILY="hani_rna_activity"
COMPARISON_GROUP="utr5__hani_rna_activity__observed_head_resnet1d_challenger"
LAUNCH_SCRIPT="launch/utr5_hani_resnet1d_sweep.sh"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"

if [[ -n "${GPU_LIST:-}" ]]; then
  read -r -a GPU_ARRAY <<< "${GPU_LIST}"
else
  mapfile -t GPU_ARRAY < <(detect_idle_gpus)
fi

if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: no idle GPUs detected for ${LAUNCH_SCRIPT}." >&2
  echo "       Recheck with nvidia-smi or set GPU_LIST explicitly, e.g. GPU_LIST=\"3 4 6 7\"." >&2
  exit 1
fi

NUM_AGENTS="${NUM_AGENTS:-${#GPU_ARRAY[@]}}"
NUM_RUNS="${NUM_RUNS:-16}"

echo "Selected GPU_LIST: ${GPU_ARRAY[*]}"
echo "NUM_AGENTS=${NUM_AGENTS}, NUM_RUNS=${NUM_RUNS}"

launch_wandb_agents \
  "${CONFIG_PATH}" \
  "${TASK_FAMILY}" \
  "${TARGET_FAMILY}" \
  "${COMPARISON_GROUP}" \
  "${LAUNCH_SCRIPT}" \
  "${NUM_AGENTS}" \
  "${NUM_RUNS}" \
  "${GPU_ARRAY[@]}"
