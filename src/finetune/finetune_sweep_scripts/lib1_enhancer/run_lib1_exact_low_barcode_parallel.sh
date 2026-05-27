#!/usr/bin/env bash
set -euo pipefail

# GPU-parallel launcher for the exact-low-barcode follow-up.
#
# Usage:
#   GPU_IDS="0 1 2" bash run_lib1_exact_low_barcode_parallel.sh
#
# Optional knobs:
#   SEED_LIST="23 19 31"
#   TRAIN_BARCODE_BINS="bc_eq1 bc_eq2 bc_eq3 bc_4_6 bc_ge7"
#   SETTING_FLAGS="--include_b1 --include_b2"
#   UNFREEZE_SCOPES="branched_only"
#   PYTHON_CMD="conda run -n boda_env python"

REPO_ROOT="/home/minhang/synBio_AL"
SCRIPT="${REPO_ROOT}/boda2_EU/src/finetune/finetune_sweep_scripts/lib1_enhancer_exact_low_barcode_finetuning.py"
COMBINE_SCRIPT="${REPO_ROOT}/boda2_EU/src/finetune/finetune_sweep_scripts/combine_learning_curve_seed_outputs.py"
OUTDIR="${OUTDIR:-${REPO_ROOT}/boda2_EU/src/finetune/learning_curve/lib1_enhancer_exact_low_barcode_hq4_hq8_cap500_b1_b2_may2026}"

SEED_LIST="${SEED_LIST:-23 19 31}"
GPU_IDS="${GPU_IDS:-0}"
HELDOUT_MIN_BARCODES="${HELDOUT_MIN_BARCODES:-4 8}"
TRAIN_BARCODE_BINS="${TRAIN_BARCODE_BINS:-bc_eq1 bc_eq2 bc_eq3 bc_4_6 bc_ge7}"
TRAIN_SIZE_FRACS="${TRAIN_SIZE_FRACS:-0.02 0.05 0.1 0.2 0.5 1.0}"
SETTING_FLAGS="${SETTING_FLAGS:---include_b1 --include_b2}"
B3_BCAPS="${B3_BCAPS:-10}"
UNFREEZE_SCOPES="${UNFREEZE_SCOPES:-branched_only}"
PRETRAINED_HEADS="${PRETRAINED_HEADS:-K562}"
HEAD_LRS="${HEAD_LRS:-5e-4}"
BACKBONE_LRS="${BACKBONE_LRS:-1e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-70}"
PATIENCE="${PATIENCE:-10}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON_CMD="${PYTHON_CMD:-python}"

read -r -a SEEDS <<< "${SEED_LIST}"
read -r -a GPUS <<< "${GPU_IDS}"
read -r -a PYTHON <<< "${PYTHON_CMD}"
read -r -a HELDOUTS <<< "${HELDOUT_MIN_BARCODES}"
read -r -a BINS <<< "${TRAIN_BARCODE_BINS}"
read -r -a FRACS <<< "${TRAIN_SIZE_FRACS}"
read -r -a SETTINGS <<< "${SETTING_FLAGS}"
read -r -a BCAPS <<< "${B3_BCAPS}"
read -r -a SCOPES <<< "${UNFREEZE_SCOPES}"
read -r -a HEADS <<< "${PRETRAINED_HEADS}"
read -r -a HLRS <<< "${HEAD_LRS}"
read -r -a BLRS <<< "${BACKBONE_LRS}"
read -r -a EXTRA <<< "${EXTRA_ARGS}"

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs provided. Set GPU_IDS, for example GPU_IDS=\"0 1\"." >&2
  exit 1
fi

mkdir -p "${OUTDIR}/logs" "${OUTDIR}/per_seed"

echo "Repo root: ${REPO_ROOT}"
echo "Output dir: ${OUTDIR}"
echo "Seeds: ${SEEDS[*]}"
echo "GPU slots: ${GPUS[*]}"
echo "Train bins: ${BINS[*]}"
echo "Train fractions: ${FRACS[*]}"
echo "Settings: ${SETTINGS[*]}"

running=0
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
      --heldout_min_barcodes "${HELDOUTS[@]}" \
      --seeds "${seed}" \
      --train_pool_cap 500 \
      --train_barcode_bins "${BINS[@]}" \
      --train_size_fracs "${FRACS[@]}" \
      --min_train_size 10 \
      --pretrained_heads "${HEADS[@]}" \
      "${SETTINGS[@]}" \
      --b3_bcaps "${BCAPS[@]}" \
      --min_weight 0.1 \
      --unfreeze_scopes "${SCOPES[@]}" \
      --head_lrs "${HLRS[@]}" \
      --backbone_lrs "${BLRS[@]}" \
      --max_epochs "${MAX_EPOCHS}" \
      --patience "${PATIENCE}" \
      --frozen_epochs 2 \
      --train_batch_size "${TRAIN_BATCH_SIZE}" \
      "${EXTRA[@]}"
  ) > "${log_path}" 2>&1 &

  running=$((running + 1))
  if [[ "${running}" -ge "${#GPUS[@]}" ]]; then
    wait
    running=0
  fi
done

wait

echo "All seed jobs finished. Combining CSV outputs..."
"${PYTHON[@]}" "${COMBINE_SCRIPT}" "${OUTDIR}" --seeds "${SEEDS[@]}"
echo "Combined outputs are in: ${OUTDIR}/combined"
