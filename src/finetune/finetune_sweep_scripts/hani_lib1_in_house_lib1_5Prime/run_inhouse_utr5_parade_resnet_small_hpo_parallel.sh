#!/usr/bin/env bash
set -euo pipefail

# GPU-parallel launcher for the first in-house 5'UTR fine-tuning HPO.
#
# Default shape:
#   - model families: BODA ResNet1D 1mmy39ku and PARADE public UTR5 checkpoint
#   - training barcode thresholds: 1+, 2+, 3+
#   - high-quality pool: number_of_barcodes >= 8
#   - heldout pool: 15% of the high-quality pool, split 50/50 into val/test
#   - training pool: all non-heldout rows passing the threshold, including
#     the remaining 85% of high-quality rows
#   - cell-type heads/conditions: c1, c2, c4, c6, c17
#   - unfreeze scopes: head_only, last_stage_plus_head, full
#
# Usage from any shell with conda available:
#   bash run_inhouse_utr5_parade_resnet_small_hpo_parallel.sh
#
# Useful overrides:
#   PREVIEW_ONLY=1 bash run_inhouse_utr5_parade_resnet_small_hpo_parallel.sh
#   GPU_IDS="0 1 2 3" SEED_LIST="17 19" bash run_inhouse_utr5_parade_resnet_small_hpo_parallel.sh
#   TRAIN_SIZES="512 2048 full" bash run_inhouse_utr5_parade_resnet_small_hpo_parallel.sh
#   MODEL_FAMILIES="parade" CELL_HEADS="c2 c4" bash run_inhouse_utr5_parade_resnet_small_hpo_parallel.sh

REPO_ROOT="${REPO_ROOT:-/home/minhang/synBio_AL/boda2_EU}"
SCRIPT="${REPO_ROOT}/src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/inhouse_utr5_parade_resnet_finetune.py"
COMBINE_SCRIPT="${REPO_ROOT}/src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/combine_inhouse_utr5_finetune_outputs.py"
OUTDIR="${OUTDIR:-${REPO_ROOT}/src/finetune/learning_curve/inhouse_utr5_parade_resnet_small_hpo_jun2026}"
PYTHON_CMD="${PYTHON_CMD:-conda run -n boda_env python}"

GPU_IDS="${GPU_IDS:-0 1 2 3}"
MODEL_FAMILIES="${MODEL_FAMILIES:-boda_resnet1d parade}"
SEED_LIST="${SEED_LIST:-17}"
TRAIN_THRESHOLDS="${TRAIN_THRESHOLDS:-1 2 3}"
TRAIN_SIZES="${TRAIN_SIZES:-full}"
CELL_HEADS="${CELL_HEADS:-c1 c2 c4 c6 c17}"
UNFREEZE_SCOPES="${UNFREEZE_SCOPES:-head_only last_stage_plus_head full}"
HEAD_LRS="${HEAD_LRS:-1e-4}"
BACKBONE_LRS="${BACKBONE_LRS:-1e-5}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-2}"
WEIGHT_DECAYS="${WEIGHT_DECAYS:-1e-4}"

HELDOUT_MIN_BARCODES="${HELDOUT_MIN_BARCODES:-8}"
HELDOUT_FRAC_WITHIN_HQ="${HELDOUT_FRAC_WITHIN_HQ:-0.20}"
HELDOUT_VAL_FRAC="${HELDOUT_VAL_FRAC:-0.20}"
SPLIT_SEED="${SPLIT_SEED:-20260603}"
MAX_EPOCHS="${MAX_EPOCHS:-120}"
MIN_EPOCHS="${MIN_EPOCHS:-8}"
PATIENCE="${PATIENCE:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-512}"
MONITOR_METRIC="${MONITOR_METRIC:-val_spearman}"
PREVIEW_ONLY="${PREVIEW_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

read -r -a PYTHON <<< "${PYTHON_CMD}"
read -r -a GPUS <<< "${GPU_IDS}"
read -r -a FAMILIES <<< "${MODEL_FAMILIES}"
read -r -a SEEDS <<< "${SEED_LIST}"
read -r -a THRESHOLDS <<< "${TRAIN_THRESHOLDS}"
read -r -a SIZES <<< "${TRAIN_SIZES}"
read -r -a HEADS <<< "${CELL_HEADS}"
read -r -a SCOPES <<< "${UNFREEZE_SCOPES}"
read -r -a HLRS <<< "${HEAD_LRS}"
read -r -a BLRS <<< "${BACKBONE_LRS}"
read -r -a FREEZE_EPOCHS <<< "${FREEZE_BACKBONE_EPOCHS}"
read -r -a WDS <<< "${WEIGHT_DECAYS}"
read -r -a EXTRA <<< "${EXTRA_ARGS}"

PREVIEW_ARGS=()
if [[ "${PREVIEW_ONLY}" == "1" || "${PREVIEW_ONLY}" == "true" || "${PREVIEW_ONLY}" == "TRUE" ]]; then
  PREVIEW_ARGS=(--preview_only)
fi

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs provided. Set GPU_IDS, for example GPU_IDS=\"0 1\"." >&2
  exit 1
fi

mkdir -p "${OUTDIR}/logs" "${OUTDIR}/per_job"

echo "Repo root: ${REPO_ROOT}"
echo "Output dir: ${OUTDIR}"
echo "GPU slots: ${GPUS[*]}"
echo "Model families: ${FAMILIES[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Train thresholds: ${THRESHOLDS[*]}"
echo "Train sizes: ${SIZES[*]}"
echo "Cell heads: ${HEADS[*]}"
echo "Unfreeze scopes: ${SCOPES[*]}"
echo "Head LRs: ${HLRS[*]}"
echo "Backbone LRs: ${BLRS[*]}"
echo "Heldout min barcodes / HQ fraction / val fraction: ${HELDOUT_MIN_BARCODES} / ${HELDOUT_FRAC_WITHIN_HQ} / ${HELDOUT_VAL_FRAC}"
echo "Max epochs / patience: ${MAX_EPOCHS} / ${PATIENCE}"

running=0
pids=()
job_idx=0
for family in "${FAMILIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for threshold in "${THRESHOLDS[@]}"; do
      for cell_head in "${HEADS[@]}"; do
        if [[ "${family}" == "parade" && "${cell_head}" == "average" ]]; then
          continue
        fi
        gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
        job_outdir="${OUTDIR}/per_job/${family}/threshold_${threshold}/cell_${cell_head}/seed_${seed}"
        log_path="${OUTDIR}/logs/${family}__thr${threshold}__cell${cell_head}__seed${seed}.log"

        echo "Launching ${family} threshold ${threshold} cell ${cell_head} seed ${seed} on visible GPU ${gpu}; log: ${log_path}"
        (
          cd "${REPO_ROOT}"
          CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON[@]}" "${SCRIPT}" \
            --device cuda \
            --outdir "${job_outdir}" \
            --model_families "${family}" \
            --seeds "${seed}" \
            --train_thresholds "${threshold}" \
            --train_sizes "${SIZES[@]}" \
            --cell_heads "${cell_head}" \
            --unfreeze_scopes "${SCOPES[@]}" \
            --head_lrs "${HLRS[@]}" \
            --backbone_lrs "${BLRS[@]}" \
            --freeze_backbone_epochs_list "${FREEZE_EPOCHS[@]}" \
            --weight_decays "${WDS[@]}" \
            --heldout_min_barcodes "${HELDOUT_MIN_BARCODES}" \
            --heldout_frac_within_hq "${HELDOUT_FRAC_WITHIN_HQ}" \
            --heldout_val_frac "${HELDOUT_VAL_FRAC}" \
            --split_seed "${SPLIT_SEED}" \
            --max_epochs "${MAX_EPOCHS}" \
            --min_epochs "${MIN_EPOCHS}" \
            --patience "${PATIENCE}" \
            --train_batch_size "${TRAIN_BATCH_SIZE}" \
            --pred_batch_size "${PRED_BATCH_SIZE}" \
            --monitor_metric "${MONITOR_METRIC}" \
            "${PREVIEW_ARGS[@]}" \
            "${EXTRA[@]}"
        ) > "${log_path}" 2>&1 &
        pids+=("$!")

        running=$((running + 1))
        job_idx=$((job_idx + 1))
        if [[ "${running}" -ge "${#GPUS[@]}" ]]; then
          for pid in "${pids[@]}"; do
            wait "${pid}"
          done
          pids=()
          running=0
        fi
      done
    done
  done
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

if [[ "${PREVIEW_ONLY}" == "1" || "${PREVIEW_ONLY}" == "true" || "${PREVIEW_ONLY}" == "TRUE" ]]; then
  echo "Preview jobs finished. Combining preview manifests."
else
  echo "All jobs finished. Combining CSV outputs..."
fi
"${PYTHON[@]}" "${COMBINE_SCRIPT}" "${OUTDIR}"
echo "Combined outputs are in: ${OUTDIR}/combined"
