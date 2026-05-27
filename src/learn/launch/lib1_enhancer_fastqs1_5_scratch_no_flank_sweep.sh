#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

CONFIG_PATH="configs/enhancer/bashor_in_house/lib1_enhancer_fastqs1_5__scratch_no_flank_basic__bayes.yml"
TASK_FAMILY="enhancer"
TARGET_FAMILY="bashor_in_house_fastqs1_5_filtered"
COMPARISON_GROUP="enhancer__bashor_in_house__fastqs1_5_filtered__scratch_no_flank__bassetvl_vs_resnet1d"
LAUNCH_SCRIPT="launch/lib1_enhancer_fastqs1_5_scratch_no_flank_sweep.sh"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"
PREPARE_DATASET="${PREPARE_DATASET:-1}"
DRY_RUN="${DRY_RUN:-0}"

SOURCE_DATA="/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/enhancers/L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.csv"
LEARN_READY_DATA="${LEARN_DIR}/derived_data/enhancer/bashor_in_house/lib1_fastqs1_5_0filtered_out__learn_ready.tsv"

if [[ "${PREPARE_DATASET}" == "1" ]]; then
  python "${LEARN_DIR}/prepare_lib1_enhancer_fastqs1_5_dataset.py" \
    --input_path "${SOURCE_DATA}" \
    --output_path "${LEARN_READY_DATA}"
fi

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
NUM_RUNS="${NUM_RUNS:-8}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "LIB1 FASTQS1-5 SCRATCH NO-FLANK HPO"
  echo "CONFIG_PATH:      ${CONFIG_PATH}"
  echo "SOURCE_DATA:      ${SOURCE_DATA}"
  echo "LEARN_READY_DATA: ${LEARN_READY_DATA}"
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
