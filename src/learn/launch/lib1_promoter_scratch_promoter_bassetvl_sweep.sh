#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LEARN_DIR}/../.." && pwd)"
WORK_ROOT="${BODA_WORK_ROOT:-$(dirname "${REPO_ROOT}")}"
LIB1_VARIANT_ROOT="${BODA_LIB1_VARIANT_ROOT:-${WORK_ROOT}/opt_EU_learn_n_design/MattLee_lib1/single_part_variant_level}"
source "${SCRIPT_DIR}/_wandb_helpers.sh"

CONFIG_PATH="configs/promoter/bashor_in_house/promoter_bassetvl/lib1_promoter__scratch_promoter_bassetvl__bayes.yml"
TASK_FAMILY="promoter"
TARGET_FAMILY="bashor_in_house_lib1_promoter_allvalid_fastqs1_5_dedup_exact"
COMPARISON_GROUP="promoter__bashor_in_house__lib1_allvalid__scratch_promoter_bassetvl"
LAUNCH_SCRIPT="launch/lib1_promoter_scratch_promoter_bassetvl_sweep.sh"
WANDB_SWEEP_ENTITY="${WANDB_SWEEP_ENTITY:-}"
WANDB_SWEEP_PROJECT="${WANDB_SWEEP_PROJECT:-}"

PREPARE_DATASET="${PREPARE_DATASET:-1}"
HELDOUT_MIN_BARCODES="${HELDOUT_MIN_BARCODES:-8}"
VAL_FRAC_WITHIN_HQ="${VAL_FRAC_WITHIN_HQ:-0.1295}"
TEST_FRAC_WITHIN_HQ="${TEST_FRAC_WITHIN_HQ:-0.1295}"
VAL_SIZE_WITHIN_HQ="${VAL_SIZE_WITHIN_HQ:-250}"
TEST_SIZE_WITHIN_HQ="${TEST_SIZE_WITHIN_HQ:-250}"
DRY_RUN="${DRY_RUN:-0}"

SOURCE_DATA="${SOURCE_DATA:-${LIB1_VARIANT_ROOT}/L1_final_fastqs1-5_sublibrary_Promoter_subset.dedup_exact.csv}"
LEARN_READY_DATA="${LEARN_DIR}/derived_data/promoter/bashor_in_house/lib1_promoter_allvalid_fastqs1_5_dedup_exact__learn_ready.tsv"

if [[ "${PREPARE_DATASET}" == "1" ]]; then
  python "${LEARN_DIR}/prepare_lib1_promoter_inhouse_dataset.py" \
    --input-path "${SOURCE_DATA}" \
    --output-path "${LEARN_READY_DATA}" \
    --heldout-min-barcodes "${HELDOUT_MIN_BARCODES}" \
    --val-frac-within-hq "${VAL_FRAC_WITHIN_HQ}" \
    --test-frac-within-hq "${TEST_FRAC_WITHIN_HQ}" \
    --val-size-within-hq "${VAL_SIZE_WITHIN_HQ}" \
    --test-size-within-hq "${TEST_SIZE_WITHIN_HQ}"
fi

if [[ ! -f "${LEARN_READY_DATA}" ]]; then
  echo "ERROR: expected derived data file not found: ${LEARN_READY_DATA}" >&2
  echo "Run: python ${LEARN_DIR}/prepare_lib1_promoter_inhouse_dataset.py" >&2
  exit 1
fi

if [[ -n "${GPU_LIST:-}" ]]; then
  read -r -a GPU_ARRAY <<< "${GPU_LIST}"
else
  mapfile -t GPU_ARRAY < <(detect_idle_gpus)
fi

if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: no idle GPUs detected for ${LAUNCH_SCRIPT}." >&2
  echo "       Recheck with nvidia-smi or set GPU_LIST explicitly, e.g. GPU_LIST=\"4 5 6 7\"." >&2
  exit 1
fi

NUM_AGENTS="${NUM_AGENTS:-${#GPU_ARRAY[@]}}"
NUM_RUNS="${NUM_RUNS:-8}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "LIB1 PROMOTER SCRATCH PROMOTER_BASSETVL HPO"
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
