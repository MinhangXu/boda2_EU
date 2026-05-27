#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

CONFIG_PATH="configs/utr5/hani_rna_activity/resnet1d/utr5__hani_rna_activity__resnet1d__cell_conditioned_delta_aux_bayes.yml"
TASK_FAMILY="utr5"
TARGET_FAMILY="hani_rna_activity_cell_conditioned_delta_aux"
COMPARISON_GROUP="utr5__hani_rna_activity__cell_conditioned_delta_aux_resnet1d"
LAUNCH_SCRIPT="launch/utr5_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"
TOTAL_RUNS="${TOTAL_RUNS:-64}"

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

choose_default_num_agents() {
  local total_runs="$1"
  local max_agents="$2"
  local candidate
  for ((candidate=max_agents; candidate>=1; candidate--)); do
    if (( total_runs % candidate == 0 )); then
      echo "${candidate}"
      return 0
    fi
  done
}

if [[ -n "${NUM_AGENTS:-}" ]]; then
  if (( NUM_AGENTS < 1 )); then
    echo "ERROR: NUM_AGENTS must be >= 1." >&2
    exit 1
  fi
  if (( NUM_AGENTS > ${#GPU_ARRAY[@]} )); then
    echo "ERROR: NUM_AGENTS=${NUM_AGENTS} exceeds available GPU count ${#GPU_ARRAY[@]}." >&2
    exit 1
  fi
  if (( TOTAL_RUNS % NUM_AGENTS != 0 )); then
    echo "ERROR: TOTAL_RUNS=${TOTAL_RUNS} must divide evenly by NUM_AGENTS=${NUM_AGENTS}." >&2
    exit 1
  fi
else
  NUM_AGENTS="$(choose_default_num_agents "${TOTAL_RUNS}" "${#GPU_ARRAY[@]}")"
fi

RUNS_PER_AGENT="$((TOTAL_RUNS / NUM_AGENTS))"
NUM_RUNS="${RUNS_PER_AGENT}"

echo "Selected GPU_LIST: ${GPU_ARRAY[*]}"
echo "TOTAL_RUNS=${TOTAL_RUNS}, NUM_AGENTS=${NUM_AGENTS}, RUNS_PER_AGENT=${RUNS_PER_AGENT}"

launch_wandb_agents \
  "${CONFIG_PATH}" \
  "${TASK_FAMILY}" \
  "${TARGET_FAMILY}" \
  "${COMPARISON_GROUP}" \
  "${LAUNCH_SCRIPT}" \
  "${NUM_AGENTS}" \
  "${NUM_RUNS}" \
  "${GPU_ARRAY[@]}"
