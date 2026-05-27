#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/minhang/synBio_AL/boda2_EU"
RUN_NAME="lib1_enhancer_barcode_range_stage1_hq4_hq8_apr2026"
OUTDIR="${REPO_ROOT}/src/finetune/learning_curve/${RUN_NAME}"
LOG_PATH="${OUTDIR}/barcode_range_stage1.log"

mkdir -p "${OUTDIR}"
cd "${REPO_ROOT}"

python src/finetune/finetune_sweep_scripts/lib1_enhancer_barcode_range_finetuning.py \
  --outdir "${OUTDIR}" \
  --split_strategy random_hq_val_test_per_seed \
  --heldout_min_barcodes 4 8 \
  --pretrained_heads K562
  --seeds 23 19 31 \
  --split_seed 7 \
  --val_frac_within_hq 0.10 \
  --test_frac_within_hq 0.10 \
  --train_barcode_bins bc_eq1 bc_2_3 bc_4_10 bc_gt10 bc_ge4 \
  --train_size_fracs 0.25 0.50 0.75 1.0 \
  --train_sampling_mode random \
  --include_b2 \
  --unfreeze_scopes branched_only full \
  --head_lrs 5e-4 \
  --backbone_lrs 1e-4 \
  --weight_decay 1e-4 \
  --frozen_epochs 2 \
  --patience 10 \
  --train_batch_size 256 \
  2>&1 | tee "${LOG_PATH}"
