#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

CONFIG_PATH="configs/introns/seelig_2015/basset_branched/introns__seelig_2015_a5ss_sd1__scratch__basset_branched.yml"
TASK_FAMILY="introns"
TARGET_FAMILY="seelig_2015_a5ss_sd1"
COMPARISON_GROUP="introns__seelig_2015_a5ss_sd1__scratch_basset_branched"
LAUNCH_SCRIPT="launch/introns_seelig_a5ss_sd1_basset_branched_sweep.sh"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"

if [[ -n "${GPU_LIST:-}" ]]; then
  read -r -a GPU_ARRAY <<< "${GPU_LIST}"
else
  mapfile -t GPU_ARRAY < <(detect_idle_gpus)
fi

if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: no idle GPUs detected for ${LAUNCH_SCRIPT}." >&2
  echo "       Recheck with nvidia-smi or set GPU_LIST explicitly, e.g. GPU_LIST=\"0 1\"." >&2
  exit 1
fi

NUM_AGENTS="${NUM_AGENTS:-${#GPU_ARRAY[@]}}"
NUM_RUNS="${NUM_RUNS:-8}"

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
