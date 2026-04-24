#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

CONFIG_PATH="configs/enhancer/bashor_in_house/lib1_enhancer__scratch_weighted__bayes.yml"
TASK_FAMILY="enhancer"
TARGET_FAMILY="bashor_in_house"
COMPARISON_GROUP="enhancer__bashor_in_house__scratch_weighted__bassetvl_vs_resnet1d"
LAUNCH_SCRIPT="launch/lib1_enhancer_scratch_weighted_sweep.sh"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"

# Optional override for a different local dataset path if needed.
export LIB1_DATA_PATH="${LIB1_DATA_PATH:-/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/enhancers/20251218_np_fastq1_500000NPreads_enh_variants_bc_sum_avg_expression.txt}"

if [[ ! -f "${LIB1_DATA_PATH}" ]]; then
  echo "LIB1 dataset not found: ${LIB1_DATA_PATH}" >&2
  exit 1
fi

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
