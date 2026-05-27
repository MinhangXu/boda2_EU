#!/usr/bin/env bash
set -euo pipefail

# GPU-parallel launcher for Phase 2 v2 BODA-first 5'UTR Lib2 fine-tuning.
#
# Default v2 shape:
#   - source model: current canonical BODA 5'UTR ResNet1D run 1mmy39ku
#   - outer split: fixed stratified 90% HPO pool / 10% untouched final test
#   - inner HPO: three stratified train/validation splits inside the HPO pool
#   - screening: one training seed per inner split
#   - final-test evaluation is disabled unless STAGE=final_eval
#
# Usage from an already activated boda_env:
#   bash run_hani_utr5_lib2_finetune_parallel.sh
#
# Useful overrides:
#   PREVIEW_ONLY=1 bash run_hani_utr5_lib2_finetune_parallel.sh
#   GPU_IDS="0 1 2 3" bash run_hani_utr5_lib2_finetune_parallel.sh
#   STAGE=confirmation TRAINING_SEEDS="7 11 13" bash run_hani_utr5_lib2_finetune_parallel.sh
#   UNFREEZE_SCOPES="head_only last_stage_plus_head full" bash run_hani_utr5_lib2_finetune_parallel.sh
#   STAGE=final_eval TRAINING_SEEDS="7 11 13" UNFREEZE_SCOPES="full" HEAD_LRS="3e-4" BACKBONE_LRS="1e-5" TARGET_SCALER_SOURCES="pretrained_lib1_train" FREEZE_BACKBONE_EPOCHS="3" bash run_hani_utr5_lib2_finetune_parallel.sh

REPO_ROOT="${REPO_ROOT:-/home/minhang/synBio_AL/boda2_EU}"
SCRIPT="${REPO_ROOT}/src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/hani_utr5_lib2_finetune.py"
COMBINE_SCRIPT="${REPO_ROOT}/src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/combine_hani_utr5_lib2_outputs.py"
OUTDIR="${OUTDIR:-${REPO_ROOT}/src/finetune/learning_curve/hani_utr5_lib2_resnet1d_1mmy39ku_phase2_v2_may2026}"
SPLIT_MANIFEST_DIR="${SPLIT_MANIFEST_DIR:-${OUTDIR}/split_manifests}"
PYTHON_CMD="${PYTHON_CMD:-python}"

STAGE="${STAGE:-screening}"
GPU_IDS="${GPU_IDS:-0 1 2 5}"
TRAINING_SEEDS="${TRAINING_SEEDS:-7}"
OUTER_SPLIT_SEED="${OUTER_SPLIT_SEED:-20260526}"
FINAL_TEST_FRAC="${FINAL_TEST_FRAC:-0.10}"
INNER_SPLIT_SEEDS="${INNER_SPLIT_SEEDS:-101 202 303}"
INNER_VAL_FRAC="${INNER_VAL_FRAC:-0.10}"
ACTIVITY_QUANTILE_BINS="${ACTIVITY_QUANTILE_BINS:-5}"
GC_QUANTILE_BINS="${GC_QUANTILE_BINS:-5}"

UNFREEZE_SCOPES="${UNFREEZE_SCOPES:-last_stage_plus_head full}"
HEAD_LRS="${HEAD_LRS:-1e-4 3e-4}"
BACKBONE_LRS="${BACKBONE_LRS:-3e-6 1e-5 3e-5}"
TARGET_SCALER_SOURCES="${TARGET_SCALER_SOURCES:-pretrained_lib1_train lib2_train}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
MIN_EPOCHS="${MIN_EPOCHS:-8}"
PATIENCE="${PATIENCE:-30}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-0 3 5}"
WEIGHT_DECAYS="${WEIGHT_DECAYS:-1e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
PRED_BATCH_SIZE="${PRED_BATCH_SIZE:-512}"
INHOUSE_MIN_BARCODES="${INHOUSE_MIN_BARCODES:-8}"
MONITOR_METRIC="${MONITOR_METRIC:-val_average_activity_pearson}"
PREVIEW_ONLY="${PREVIEW_ONLY:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

read -r -a PYTHON <<< "${PYTHON_CMD}"
read -r -a GPUS <<< "${GPU_IDS}"
read -r -a SEEDS <<< "${TRAINING_SEEDS}"
read -r -a INNER_SEEDS <<< "${INNER_SPLIT_SEEDS}"
read -r -a SCOPES <<< "${UNFREEZE_SCOPES}"
read -r -a HLRS <<< "${HEAD_LRS}"
read -r -a BLRS <<< "${BACKBONE_LRS}"
read -r -a SCALERS <<< "${TARGET_SCALER_SOURCES}"
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

mkdir -p "${OUTDIR}/logs" "${OUTDIR}/per_job" "${SPLIT_MANIFEST_DIR}"

echo "Repo root: ${REPO_ROOT}"
echo "Output dir: ${OUTDIR}"
echo "Split manifest dir: ${SPLIT_MANIFEST_DIR}"
echo "Stage: ${STAGE}"
echo "Training seeds: ${SEEDS[*]}"
echo "GPU slots: ${GPUS[*]}"
echo "Outer split seed / final-test frac: ${OUTER_SPLIT_SEED} / ${FINAL_TEST_FRAC}"
echo "Inner split seeds / val frac: ${INNER_SEEDS[*]} / ${INNER_VAL_FRAC}"
echo "Scopes: ${SCOPES[*]}"
echo "Head LRs: ${HLRS[*]}"
echo "Backbone LRs: ${BLRS[*]}"
echo "Target scalers: ${SCALERS[*]}"
echo "Freeze epochs: ${FREEZE_EPOCHS[*]}"
echo "Weight decays: ${WDS[*]}"
echo "Max epochs / patience: ${MAX_EPOCHS} / ${PATIENCE}"

split_ids=()
split_seeds=()
if [[ "${STAGE}" == "final_eval" ]]; then
  split_ids=("hpo_pool_to_final_test")
  split_seeds=("${INNER_SEEDS[0]}")
elif [[ "${STAGE}" == "legacy_v1" ]]; then
  split_ids=("legacy_v1")
  split_seeds=("${INNER_SEEDS[0]}")
else
  for idx in "${!INNER_SEEDS[@]}"; do
    split_seed="${INNER_SEEDS[$idx]}"
    split_ids+=("inner${idx}_seed_${split_seed}")
    split_seeds+=("${split_seed}")
  done
fi

if [[ "${STAGE}" != "legacy_v1" ]]; then
  echo "Preparing shared v2 split manifests before parallel jobs..."
  (
    cd "${REPO_ROOT}"
    "${PYTHON[@]}" "${SCRIPT}" \
      --device cpu \
      --outdir "${OUTDIR}/split_prepare" \
      --stage screening \
      --split_manifest_dir "${SPLIT_MANIFEST_DIR}" \
      --outer_split_seed "${OUTER_SPLIT_SEED}" \
      --final_test_frac "${FINAL_TEST_FRAC}" \
      --inner_split_seeds "${INNER_SEEDS[@]}" \
      --inner_split_seed "${INNER_SEEDS[0]}" \
      --inner_val_frac "${INNER_VAL_FRAC}" \
      --split_id "inner0_seed_${INNER_SEEDS[0]}" \
      --activity_quantile_bins "${ACTIVITY_QUANTILE_BINS}" \
      --gc_quantile_bins "${GC_QUANTILE_BINS}" \
      --seeds "${SEEDS[0]}" \
      --unfreeze_scopes "${SCOPES[0]}" \
      --head_lrs "${HLRS[0]}" \
      --backbone_lrs "${BLRS[0]}" \
      --target_scaler_sources "${SCALERS[0]}" \
      --freeze_backbone_epochs_list "${FREEZE_EPOCHS[0]}" \
      --weight_decays "${WDS[0]}" \
      --prepare_splits_only
  ) > "${OUTDIR}/logs/split_prepare.log" 2>&1
fi

running=0
pids=()
job_idx=0
for split_idx in "${!split_ids[@]}"; do
  split_id="${split_ids[$split_idx]}"
  inner_seed="${split_seeds[$split_idx]}"
  for seed in "${SEEDS[@]}"; do
    gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
    job_outdir="${OUTDIR}/per_job/${STAGE}/${split_id}/training_seed_${seed}"
    log_path="${OUTDIR}/logs/${STAGE}__${split_id}__tseed_${seed}.log"

    echo "Launching ${STAGE} ${split_id} training seed ${seed} on visible GPU ${gpu}; log: ${log_path}"
    (
      cd "${REPO_ROOT}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON[@]}" "${SCRIPT}" \
        --device cuda \
        --outdir "${job_outdir}" \
        --stage "${STAGE}" \
        --split_manifest_dir "${SPLIT_MANIFEST_DIR}" \
        --outer_split_seed "${OUTER_SPLIT_SEED}" \
        --final_test_frac "${FINAL_TEST_FRAC}" \
        --inner_split_seeds "${INNER_SEEDS[@]}" \
        --inner_split_seed "${inner_seed}" \
        --inner_val_frac "${INNER_VAL_FRAC}" \
        --split_id "${split_id}" \
        --activity_quantile_bins "${ACTIVITY_QUANTILE_BINS}" \
        --gc_quantile_bins "${GC_QUANTILE_BINS}" \
        --seeds "${seed}" \
        --unfreeze_scopes "${SCOPES[@]}" \
        --head_lrs "${HLRS[@]}" \
        --backbone_lrs "${BLRS[@]}" \
        --target_scaler_sources "${SCALERS[@]}" \
        --max_epochs "${MAX_EPOCHS}" \
        --min_epochs "${MIN_EPOCHS}" \
        --patience "${PATIENCE}" \
        --freeze_backbone_epochs_list "${FREEZE_EPOCHS[@]}" \
        --weight_decays "${WDS[@]}" \
        --train_batch_size "${TRAIN_BATCH_SIZE}" \
        --pred_batch_size "${PRED_BATCH_SIZE}" \
        --inhouse_min_barcodes "${INHOUSE_MIN_BARCODES}" \
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

for pid in "${pids[@]}"; do
  wait "${pid}"
done

if [[ "${PREVIEW_ONLY}" == "1" || "${PREVIEW_ONLY}" == "true" || "${PREVIEW_ONLY}" == "TRUE" ]]; then
  echo "Preview jobs finished. Skipping combine."
else
  echo "All jobs finished. Combining CSV outputs..."
  "${PYTHON[@]}" "${COMBINE_SCRIPT}" "${OUTDIR}"
  echo "Combined outputs are in: ${OUTDIR}/combined"
fi
