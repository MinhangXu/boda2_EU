#!/bin/bash

# set -e: exit the script if any command fails
# set -u: exit the script if any variable is used uninitialized
# set -o pipefail: exit the script if any command in a pipeline fails
set -euo pipefail

# get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# get the directory of the learn directory
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# source the wandb helpers
source "${SCRIPT_DIR}/_wandb_helpers.sh"

# get the path to the data preparation script
DATA_PREP_SCRIPT="${LEARN_DIR}/prepare_enhancer_single_head_dataset.py"
# get the path to the derived dataset
DERIVED_DATASET="${LEARN_DIR}/derived_data/enhancer/malinois_mpra/MPRA_ALL_HD_v2__single_head_combined.tsv"

# if the derived dataset does not exist or the FORCE_REBUILD_DATASET flag is set, run the data preparation script
if [[ ! -f "${DERIVED_DATASET}" || "${FORCE_REBUILD_DATASET:-0}" == "1" ]]; then
  echo "Preparing derived enhancer single-head dataset..."
  python "${DATA_PREP_SCRIPT}"   # run the data preparation script
fi

# path of the yml config file 
CONFIG_PATH="configs/legacy/enhancer/malinois_mpra/basset_nonbranched/enhancer__malinois_mpra__basset_nonbranched__single_head_combined__bayes.yml"
TASK_FAMILY="enhancer"
TARGET_FAMILY="malinois_mpra"
COMPARISON_GROUP="enhancer__malinois_mpra__single_head_combined__basset_nonbranched"
LAUNCH_SCRIPT="launch/enhancer_malinois_basset_nonbranched_single_head_combined_sweep.sh"
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
