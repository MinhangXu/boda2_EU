#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

CONFIG_PATH="configs/enhancer/bashor_in_house/resnet1d/lib1_enhancer_no_flank_hq8__scratch_resnet1d__bayes.yml"
TASK_FAMILY="enhancer"
TARGET_FAMILY="bashor_in_house_lib1_enhancer_no_flank_hq8_fastqs1_5"
COMPARISON_GROUP="enhancer__bashor_in_house__no_flank_hq8__scratch_resnet1d_fp32"
LAUNCH_SCRIPT="launch/lib1_enhancer_no_flank_hq8_scratch_resnet1d_sweep.sh"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"

PREPARE_DATASET="${PREPARE_DATASET:-1}"
HELDOUT_MIN_BARCODES="${HELDOUT_MIN_BARCODES:-8}"
VAL_FRAC_WITHIN_HQ="${VAL_FRAC_WITHIN_HQ:-0.2}"
TEST_FRAC_WITHIN_HQ="${TEST_FRAC_WITHIN_HQ:-0.2}"
VAL_SIZE_WITHIN_HQ="${VAL_SIZE_WITHIN_HQ:-250}"
TEST_SIZE_WITHIN_HQ="${TEST_SIZE_WITHIN_HQ:-250}"
DRY_RUN="${DRY_RUN:-0}"

SOURCE_DATA="/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/single_part_variant_level/enhancers/L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.csv"
LEARN_READY_DATA="${LEARN_DIR}/derived_data/enhancer/bashor_in_house/lib1_fastqs1_5_0filtered_out__learn_ready.tsv"

if [[ "${PREPARE_DATASET}" == "1" ]]; then
  python "${LEARN_DIR}/prepare_lib1_enhancer_fastqs1_5_dataset.py" \
    --input_path "${SOURCE_DATA}" \
    --output_path "${LEARN_READY_DATA}"
fi

if [[ ! -f "${LEARN_READY_DATA}" && "${DRY_RUN}" != "1" ]]; then
  echo "ERROR: expected derived data file not found: ${LEARN_READY_DATA}" >&2
  echo "Run: python ${LEARN_DIR}/prepare_lib1_enhancer_fastqs1_5_dataset.py" >&2
  exit 1
fi

if [[ -n "${GPU_LIST:-}" ]]; then
  read -r -a GPU_ARRAY <<< "${GPU_LIST}"
else
  mapfile -t GPU_ARRAY < <(detect_idle_gpus)
fi

if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: no idle GPUs detected for ${LAUNCH_SCRIPT}." >&2
  echo "       Recheck with nvidia-smi or set GPU_LIST explicitly, e.g. GPU_LIST=\"0 1 2 3\"." >&2
  exit 1
fi

NUM_AGENTS="${NUM_AGENTS:-${#GPU_ARRAY[@]}}"
NUM_RUNS="${NUM_RUNS:-8}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "LIB1 ENHANCER NO-FLANK HQ8 SCRATCH RESNET1D HPO"
  echo "CONFIG_PATH:          ${CONFIG_PATH}"
  echo "SOURCE_DATA:          ${SOURCE_DATA}"
  echo "LEARN_READY_DATA:     ${LEARN_READY_DATA}"
  echo "TASK_FAMILY:          ${TASK_FAMILY}"
  echo "TARGET_FAMILY:        ${TARGET_FAMILY}"
  echo "COMPARISON_GROUP:     ${COMPARISON_GROUP}"
  echo "NUM_AGENTS:           ${NUM_AGENTS}"
  echo "NUM_RUNS:             ${NUM_RUNS}"
  echo "GPU_LIST:             ${GPU_ARRAY[*]}"
  echo "PREPARE_DATASET:      ${PREPARE_DATASET}"
  echo "HELDOUT_MIN_BARCODES: ${HELDOUT_MIN_BARCODES}"
  echo "VAL_FRAC_WITHIN_HQ:   ${VAL_FRAC_WITHIN_HQ}"
  echo "TEST_FRAC_WITHIN_HQ:  ${TEST_FRAC_WITHIN_HQ}"
  echo "VAL_SIZE_WITHIN_HQ:   ${VAL_SIZE_WITHIN_HQ}"
  echo "TEST_SIZE_WITHIN_HQ:  ${TEST_SIZE_WITHIN_HQ}"
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
