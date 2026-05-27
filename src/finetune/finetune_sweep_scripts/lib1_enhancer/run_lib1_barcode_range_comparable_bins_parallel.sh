#!/usr/bin/env bash
set -euo pipefail

# GPU-parallel launcher for the comparable barcode-bin run.
#
# Usage:
#   GPU_IDS="0 1 2 3" bash run_lib1_barcode_range_comparable_bins_parallel.sh
#
# To run multiple jobs on the same GPU, repeat that GPU id, for example:
#   GPU_IDS="0 0 1 1" bash run_lib1_barcode_range_comparable_bins_parallel.sh

REPO_ROOT="/home/minhang/synBio_AL"
SCRIPT="${REPO_ROOT}/boda2_EU/src/finetune/finetune_sweep_scripts/lib1_enhancer_barcode_range_comparable_bins_finetuning.py"
OUTDIR="${OUTDIR:-${REPO_ROOT}/boda2_EU/src/finetune/learning_curve/lib1_enhancer_barcode_range_comparable_bins_hq4_hq8_b2_b3_bcap10_30_seed5_may2026}"

SEED_LIST="${SEED_LIST:-23 19 31 37 43}"
GPU_IDS="${GPU_IDS:-0}"

read -r -a SEEDS <<< "${SEED_LIST}"
read -r -a GPUS <<< "${GPU_IDS}"

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPUs provided. Set GPU_IDS, for example GPU_IDS=\"0 1\"." >&2
  exit 1
fi

mkdir -p "${OUTDIR}/logs" "${OUTDIR}/per_seed"

echo "Repo root: ${REPO_ROOT}"
echo "Output dir: ${OUTDIR}"
echo "Seeds: ${SEEDS[*]}"
echo "GPU slots: ${GPUS[*]}"
echo "Each seed runs in its own process/outdir; combined CSVs are written after all seeds finish."

running=0
for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  seed_outdir="${OUTDIR}/per_seed/seed_${seed}"
  log_path="${OUTDIR}/logs/seed_${seed}.log"

  echo "Launching seed ${seed} on visible GPU ${gpu}; log: ${log_path}"
  (
    cd "${REPO_ROOT}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${SCRIPT}" \
      --device cuda \
      --outdir "${seed_outdir}" \
      --heldout_min_barcodes 4 8 \
      --seeds "${seed}" \
      --train_pool_cap 1000 \
      --train_size_fracs 0.25 0.5 0.75 1.0 \
      --pretrained_heads K562 \
      --include_b2 --include_b3 \
      --b3_bcaps 10 30 \
      --min_weight 0.1 \
      --unfreeze_scopes branched_only full \
      --head_lrs 5e-4 \
      --backbone_lrs 1e-4 \
      --max_epochs 90 \
      --patience 20 \
      --frozen_epochs 2 \
      --train_batch_size 256
  ) > "${log_path}" 2>&1 &

  running=$((running + 1))
  if [[ "${running}" -ge "${#GPUS[@]}" ]]; then
    wait
    running=0
  fi
done

wait

echo "All seed jobs finished. Combining CSV outputs..."
python - <<'PY' "${OUTDIR}" "${SEED_LIST}"
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

outdir = Path(sys.argv[1])
seeds = sys.argv[2].split()
combined = outdir / "combined"
combined.mkdir(parents=True, exist_ok=True)

seed_dirs = [outdir / "per_seed" / f"seed_{seed}" for seed in seeds]
csv_names = [
    "learning_curve_runs.csv",
    "learning_curve_histories.csv",
    "zero_shot_evaluations.csv",
    "barcode_range_planned_grid.csv",
    "learning_curve_velocity_segments.csv",
]

combined_frames: dict[str, pd.DataFrame] = {}
for name in csv_names:
    parts = []
    for seed_dir in seed_dirs:
        path = seed_dir / name
        if path.exists():
            part = pd.read_csv(path)
            part["source_seed_dir"] = seed_dir.name
            parts.append(part)
    if parts:
        frame = pd.concat(parts, ignore_index=True)
        frame.to_csv(combined / name, index=False)
        combined_frames[name] = frame
        print(f"Wrote {combined / name} ({frame.shape[0]} rows)")
    else:
        print(f"Skipped {name}: no per-seed files found")

def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cols = []
    for col in out.columns:
        if isinstance(col, tuple):
            cols.append("_".join(str(x) for x in col if x).rstrip("_"))
        else:
            cols.append(str(col))
    out.columns = cols
    return out

def aggregate(frame: pd.DataFrame, group_cols: list[str], metric_cols: list[str], output_name: str) -> None:
    group_cols = [col for col in group_cols if col in frame.columns]
    metric_cols = [col for col in metric_cols if col in frame.columns]
    if not group_cols or not metric_cols:
        return
    summary = frame.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary = flatten_columns(summary).sort_values(group_cols).reset_index(drop=True)
    summary.to_csv(combined / output_name, index=False)
    print(f"Wrote {combined / output_name} ({summary.shape[0]} rows)")

runs = combined_frames.get("learning_curve_runs.csv")
if runs is not None:
    metric_cols = [
        "train_mae", "train_rmse", "train_pearson", "train_spearman", "train_r2", "train_r2_cod",
        "train_pearson_sq", "train_loss_standardized",
        "val_mae", "val_rmse", "val_pearson", "val_spearman", "val_r2", "val_r2_cod",
        "val_pearson_sq", "val_loss_standardized",
        "test_mae", "test_rmse", "test_pearson", "test_spearman", "test_r2", "test_r2_cod",
        "test_pearson_sq", "test_loss_standardized",
        "best_epoch", "best_val_loss_standardized", "initial_trainable_params", "final_trainable_params",
    ]
    aggregate(
        runs,
        [
            "heldout_min_barcodes", "train_barcode_bin", "train_barcode_bin_label", "train_barcode_bin_query",
            "setting", "b_cap", "head_lr", "backbone_lr", "train_sampling_mode", "unfreeze_scope",
            "train_size", "init_head",
        ],
        metric_cols,
        "learning_curve_summary_mean_std.csv",
    )

    full_runs = runs.loc[runs["train_size"] == runs["train_pool_eligible_size"]].copy()
    if not full_runs.empty:
        aggregate(
            full_runs,
            [
                "heldout_min_barcodes", "train_barcode_bin", "train_barcode_bin_label", "train_barcode_bin_query",
                "setting", "b_cap", "head_lr", "backbone_lr", "train_sampling_mode", "unfreeze_scope",
                "init_head",
            ],
            [
                "val_pearson", "val_spearman", "val_r2", "val_r2_cod", "val_loss_standardized",
                "test_pearson", "test_spearman", "test_r2", "test_r2_cod", "test_loss_standardized",
                "test_pearson_sq", "best_epoch",
            ],
            "barcode_bin_full_fraction_summary_mean_std.csv",
        )

segments = combined_frames.get("learning_curve_velocity_segments.csv")
if segments is not None:
    aggregate(
        segments,
        [
            "heldout_min_barcodes", "train_barcode_bin", "train_barcode_bin_label", "train_barcode_bin_query",
            "train_sampling_mode", "setting", "b_cap", "head_lr", "backbone_lr", "unfreeze_scope", "metric",
        ],
        ["delta_metric_mean", "slope_per_construct", "slope_per_100_constructs"],
        "learning_curve_velocity_summary_mean_std.csv",
    )

manifest_parts = []
for seed_dir in seed_dirs:
    path = seed_dir / "run_manifest.json"
    if path.exists():
        with path.open() as handle:
            manifest_parts.append(json.load(handle))

combined_manifest = {
    "source_seed_dirs": [str(path) for path in seed_dirs],
    "n_seed_runs": len(manifest_parts),
    "combined_outdir": str(combined),
}
if manifest_parts:
    combined_manifest["example_manifest"] = manifest_parts[0]
(combined / "run_manifest_combined.json").write_text(json.dumps(combined_manifest, indent=2) + "\n")
print(f"Wrote {combined / 'run_manifest_combined.json'}")
PY

echo "Combined outputs are in: ${OUTDIR}/combined"
