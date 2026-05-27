#!/usr/bin/env bash
set -euo pipefail

# GPU-parallel launcher for the May 2026 B2 threshold HPO rerun.
#
# Shape:
#   - training thresholds: 1+, 2+, 3+
#   - held-out val/test pool: number_of_barcodes >= 8
#   - val/test fractions: 0.20 each, which is ~246 val and ~246 test rows
#     for the current filtered lib1 collaborator CSV
#   - train sizes: 50, 400, 1000, 2000, and full eligible train pool
#   - all pretrained init heads: K562, HepG2, SKNSH
#   - setting: B2_with_RC only
#
# Usage:
#   bash run_lib1_threshold_hq8_multiseed_hpo_parallel.sh
#
# Useful overrides:
#   GPU_IDS="0 1 2 3" SEED_LIST="17 19 23 31" bash run_lib1_threshold_hq8_multiseed_hpo_parallel.sh
#   PREVIEW_ONLY=1 bash run_lib1_threshold_hq8_multiseed_hpo_parallel.sh
#   UNFREEZE_SCOPES="branched_only full" bash run_lib1_threshold_hq8_multiseed_hpo_parallel.sh
#   PYTHON_CMD="python" bash run_lib1_threshold_hq8_multiseed_hpo_parallel.sh

REPO_ROOT="/home/minhang/synBio_AL"
SCRIPT="${REPO_ROOT}/boda2_EU/src/finetune/finetune_sweep_scripts/lib1_enhancer_learning_curve_filtered_raw_ratio_split_options.py"
COMBINE_SCRIPT="${REPO_ROOT}/boda2_EU/src/finetune/finetune_sweep_scripts/combine_learning_curve_seed_outputs.py"
OUTDIR="${OUTDIR:-${REPO_ROOT}/boda2_EU/src/finetune/learning_curve/lib1_enhancer_threshold_hq8_random_mixed_b2_allheads_8seed_absgrid_may2026}"
PYTHON_CMD="${PYTHON_CMD:-conda run -n boda_env python}"

SEED_LIST="${SEED_LIST:-17 19 23 31 37 43 47 53}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-random_hq_val_test_per_seed}"
HELDOUT_MIN_BARCODES="${HELDOUT_MIN_BARCODES:-8}"
VAL_FRAC_WITHIN_HQ="${VAL_FRAC_WITHIN_HQ:-0.20}"
TEST_FRAC_WITHIN_HQ="${TEST_FRAC_WITHIN_HQ:-0.20}"
TRAIN_THRESHOLDS="${TRAIN_THRESHOLDS:-1 2 3}"
TRAIN_SIZES="${TRAIN_SIZES:-50 400 1000 2000}"
MIN_TRAIN_SIZE="${MIN_TRAIN_SIZE:-50}"
TRAIN_SAMPLING_MODE="${TRAIN_SAMPLING_MODE:-random}"
SETTING_FLAGS="${SETTING_FLAGS:---include_b2}"
UNFREEZE_SCOPES="${UNFREEZE_SCOPES:-branched_only conv3_plus full}"

# Later May runs used 5e-4 / 1e-4 as the focused B2/B3 LR pair. To mirror
# the archived Mar 25 B2-only notebook exactly, override with:
#   HEAD_LRS="2e-4" BACKBONE_LRS="5e-5"
HEAD_LRS="${HEAD_LRS:-5e-4}"
BACKBONE_LRS="${BACKBONE_LRS:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-250}"
PATIENCE="${PATIENCE:-40}"
FROZEN_EPOCHS="${FROZEN_EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
PREVIEW_ONLY="${PREVIEW_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

read -r -a SEEDS <<< "${SEED_LIST}"
read -r -a GPUS <<< "${GPU_IDS}"
read -r -a PYTHON <<< "${PYTHON_CMD}"
read -r -a THRESHOLDS <<< "${TRAIN_THRESHOLDS}"
read -r -a SIZES <<< "${TRAIN_SIZES}"
read -r -a SETTINGS <<< "${SETTING_FLAGS}"
read -r -a SCOPES <<< "${UNFREEZE_SCOPES}"
read -r -a HLRS <<< "${HEAD_LRS}"
read -r -a BLRS <<< "${BACKBONE_LRS}"
read -r -a EXTRA <<< "${EXTRA_ARGS}"

PREVIEW_ARGS=()
if [[ "${PREVIEW_ONLY}" == "1" || "${PREVIEW_ONLY}" == "true" || "${PREVIEW_ONLY}" == "TRUE" ]]; then
  PREVIEW_ARGS=(--preview_only)
fi

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs provided. Set GPU_IDS, for example GPU_IDS=\"0 1\"." >&2
  exit 1
fi

mkdir -p "${OUTDIR}/logs" "${OUTDIR}/per_seed"

echo "Repo root: ${REPO_ROOT}"
echo "Output dir: ${OUTDIR}"
echo "Seeds: ${SEEDS[*]}"
echo "GPU slots: ${GPUS[*]}"
echo "Split strategy: ${SPLIT_STRATEGY}"
echo "Heldout min barcodes: ${HELDOUT_MIN_BARCODES}"
echo "Val/test fractions within HQ: ${VAL_FRAC_WITHIN_HQ} / ${TEST_FRAC_WITHIN_HQ}"
echo "Train thresholds: ${THRESHOLDS[*]}"
echo "Train sizes: ${SIZES[*]} + full eligible pool"
echo "Settings: ${SETTINGS[*]}"
echo "Unfreeze scopes: ${SCOPES[*]}"
echo "Head LRs: ${HLRS[*]}"
echo "Backbone LRs: ${BLRS[*]}"
echo "Max epochs / patience: ${MAX_EPOCHS} / ${PATIENCE}"

running=0
pids=()
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  seed_outdir="${OUTDIR}/per_seed/seed_${seed}"
  log_path="${OUTDIR}/logs/seed_${seed}.log"

  echo "Launching seed ${seed} on visible GPU ${gpu}; log: ${log_path}"
  (
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON[@]}" "${SCRIPT}" \
      --device cuda \
      --outdir "${seed_outdir}" \
      --seeds "${seed}" \
      --split_strategy "${SPLIT_STRATEGY}" \
      --train_priority_min_barcodes "${HELDOUT_MIN_BARCODES}" \
      --val_frac_within_hq "${VAL_FRAC_WITHIN_HQ}" \
      --test_frac_within_hq "${TEST_FRAC_WITHIN_HQ}" \
      --train_thresholds "${THRESHOLDS[@]}" \
      --train_sizes "${SIZES[@]}" \
      --min_train_size "${MIN_TRAIN_SIZE}" \
      --train_sampling_mode "${TRAIN_SAMPLING_MODE}" \
      "${SETTINGS[@]}" \
      --unfreeze_scopes "${SCOPES[@]}" \
      --head_lrs "${HLRS[@]}" \
      --backbone_lrs "${BLRS[@]}" \
      --max_epochs "${MAX_EPOCHS}" \
      --patience "${PATIENCE}" \
      --frozen_epochs "${FROZEN_EPOCHS}" \
      --train_batch_size "${TRAIN_BATCH_SIZE}" \
      "${PREVIEW_ARGS[@]}" \
      "${EXTRA[@]}"
  ) > "${log_path}" 2>&1 &
  pids+=("$!")

  running=$((running + 1))
  if [[ "${running}" -ge "${#GPUS[@]}" ]]; then
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
    pids=()
    running=0
  fi
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "All seed jobs finished. Combining CSV outputs..."
"${PYTHON[@]}" "${COMBINE_SCRIPT}" "${OUTDIR}" --seeds "${SEEDS[@]}"
echo "Combined outputs are in: ${OUTDIR}/combined"
