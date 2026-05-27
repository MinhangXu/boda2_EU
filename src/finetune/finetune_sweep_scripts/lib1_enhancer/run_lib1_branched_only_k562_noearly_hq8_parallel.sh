#!/usr/bin/env bash
set -euo pipefail

# GPU-parallel diagnostic launcher for the May 2026 Lib1 enhancer HPO follow-up.
#
# Goal:
#   Check whether branched_only can eventually overfit / memorize when early
#   stopping is disabled.
#
# Shape:
#   - training thresholds: 1+, 2+, 3+
#   - held-out val/test pool: number_of_barcodes >= 8
#   - val/test fractions: 0.20 each, randomized per seed
#   - train sizes: 400, 1000, and full eligible train pool
#   - init head: K562 only
#   - unfreeze scope: branched_only only
#   - setting: B2_with_RC only
#   - epochs: 250, no early stopping
#   - default parallelism: 4 seeds on 4 GPUs
#
# Usage:
#   bash run_lib1_branched_only_k562_noearly_hq8_parallel.sh
#
# Useful overrides:
#   GPU_IDS="0 1 2 3" SEED_LIST="17 19 23 31" bash run_lib1_branched_only_k562_noearly_hq8_parallel.sh
#   PREVIEW_ONLY=1 bash run_lib1_branched_only_k562_noearly_hq8_parallel.sh
#   PYTHON_CMD="python" bash run_lib1_branched_only_k562_noearly_hq8_parallel.sh
#   LAUNCH_DELAY_SECONDS=0 bash run_lib1_branched_only_k562_noearly_hq8_parallel.sh

REPO_ROOT="/home/minhang/synBio_AL"
SCRIPT="${REPO_ROOT}/boda2_EU/src/finetune/finetune_sweep_scripts/lib1_enhancer_learning_curve_filtered_raw_ratio_split_options.py"
COMBINE_SCRIPT="${REPO_ROOT}/boda2_EU/src/finetune/finetune_sweep_scripts/combine_learning_curve_seed_outputs.py"
OUTDIR="${OUTDIR:-${REPO_ROOT}/boda2_EU/src/finetune/learning_curve/lib1_enhancer_branched_only_k562_hq8_4seed_noearly_250epoch_may2026}"
PYTHON_CMD="${PYTHON_CMD:-conda run --no-capture-output -n boda_env python}"

SEED_LIST="${SEED_LIST:-17 19 23 31}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-random_hq_val_test_per_seed}"
HELDOUT_MIN_BARCODES="${HELDOUT_MIN_BARCODES:-8}"
VAL_FRAC_WITHIN_HQ="${VAL_FRAC_WITHIN_HQ:-0.20}"
TEST_FRAC_WITHIN_HQ="${TEST_FRAC_WITHIN_HQ:-0.20}"
TRAIN_THRESHOLDS="${TRAIN_THRESHOLDS:-1 2 3}"
TRAIN_SIZES="${TRAIN_SIZES:-400 1000}"
MIN_TRAIN_SIZE="${MIN_TRAIN_SIZE:-50}"
TRAIN_SAMPLING_MODE="${TRAIN_SAMPLING_MODE:-random}"
SETTING_FLAGS="${SETTING_FLAGS:---include_b2}"
INIT_HEADS="${INIT_HEADS:-K562}"
UNFREEZE_SCOPES="${UNFREEZE_SCOPES:-branched_only}"

HEAD_LRS="${HEAD_LRS:-5e-4}"
BACKBONE_LRS="${BACKBONE_LRS:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-250}"
PATIENCE="${PATIENCE:-40}"
FROZEN_EPOCHS="${FROZEN_EPOCHS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
DISABLE_EARLY_STOPPING="${DISABLE_EARLY_STOPPING:-1}"
PREVIEW_ONLY="${PREVIEW_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LAUNCH_DELAY_SECONDS="${LAUNCH_DELAY_SECONDS:-15}"

read -r -a SEEDS <<< "${SEED_LIST}"
read -r -a GPUS <<< "${GPU_IDS}"
read -r -a PYTHON <<< "${PYTHON_CMD}"
read -r -a THRESHOLDS <<< "${TRAIN_THRESHOLDS}"
read -r -a SIZES <<< "${TRAIN_SIZES}"
read -r -a SETTINGS <<< "${SETTING_FLAGS}"
read -r -a HEADS <<< "${INIT_HEADS}"
read -r -a SCOPES <<< "${UNFREEZE_SCOPES}"
read -r -a HLRS <<< "${HEAD_LRS}"
read -r -a BLRS <<< "${BACKBONE_LRS}"
read -r -a EXTRA <<< "${EXTRA_ARGS}"

PREVIEW_ARGS=()
if [[ "${PREVIEW_ONLY}" == "1" || "${PREVIEW_ONLY}" == "true" || "${PREVIEW_ONLY}" == "TRUE" ]]; then
  PREVIEW_ARGS=(--preview_only)
fi

EARLY_STOPPING_ARGS=()
if [[ "${DISABLE_EARLY_STOPPING}" == "1" || "${DISABLE_EARLY_STOPPING}" == "true" || "${DISABLE_EARLY_STOPPING}" == "TRUE" ]]; then
  EARLY_STOPPING_ARGS=(--disable_early_stopping)
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
echo "Init heads: ${HEADS[*]}"
echo "Unfreeze scopes: ${SCOPES[*]}"
echo "Head LRs: ${HLRS[*]}"
echo "Backbone LRs: ${BLRS[*]}"
echo "Max epochs / patience / disable early stopping: ${MAX_EPOCHS} / ${PATIENCE} / ${DISABLE_EARLY_STOPPING}"
echo "Launch delay seconds: ${LAUNCH_DELAY_SECONDS}"

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
      --init_heads "${HEADS[@]}" \
      --unfreeze_scopes "${SCOPES[@]}" \
      --head_lrs "${HLRS[@]}" \
      --backbone_lrs "${BLRS[@]}" \
      --max_epochs "${MAX_EPOCHS}" \
      --patience "${PATIENCE}" \
      "${EARLY_STOPPING_ARGS[@]}" \
      --frozen_epochs "${FROZEN_EPOCHS}" \
      --train_batch_size "${TRAIN_BATCH_SIZE}" \
      "${PREVIEW_ARGS[@]}" \
      "${EXTRA[@]}"
  ) > "${log_path}" 2>&1 &
  pids+=("$!")

  running=$((running + 1))
  if [[ "${LAUNCH_DELAY_SECONDS}" != "0" && "${idx}" -lt "$((${#SEEDS[@]} - 1))" ]]; then
    sleep "${LAUNCH_DELAY_SECONDS}" || true
  fi
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
