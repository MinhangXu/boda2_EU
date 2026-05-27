#!/usr/bin/env python3
"""Combine per-seed learning-curve outputs into one summary directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


CSV_NAMES = [
    "learning_curve_runs.csv",
    "learning_curve_histories.csv",
    "zero_shot_by_seed.csv",
    "zero_shot_fixed_test.csv",
    "zero_shot_evaluations.csv",
    "threshold_planned_grid.csv",
    "split_membership_rows.csv",
    "split_membership_summary.csv",
    "barcode_range_planned_grid.csv",
    "quality_split_planned_grid.csv",
    "learning_curve_velocity_segments.csv",
]

RUN_METRIC_COLS = [
    "train_mae",
    "train_rmse",
    "train_pearson",
    "train_spearman",
    "train_r2",
    "train_r2_cod",
    "train_pearson_sq",
    "train_loss_standardized",
    "val_mae",
    "val_rmse",
    "val_pearson",
    "val_spearman",
    "val_r2",
    "val_r2_cod",
    "val_pearson_sq",
    "val_loss_standardized",
    "test_mae",
    "test_rmse",
    "test_pearson",
    "test_spearman",
    "test_r2",
    "test_r2_cod",
    "test_pearson_sq",
    "test_loss_standardized",
    "best_epoch",
    "best_val_loss_standardized",
    "initial_trainable_params",
    "final_trainable_params",
]

RUN_GROUP_COLS = [
    "heldout_min_barcodes",
    "test_quality_bin",
    "test_quality_bin_label",
    "test_quality_bin_query",
    "test_quality_bin_sort_order",
    "split_strategy",
    "split_pool",
    "split_val_fraction",
    "split_test_fraction",
    "val_min_barcodes",
    "test_n_per_quality",
    "val_is_fixed_across_seeds",
    "test_is_fixed_across_seeds",
    "train_barcode_bin",
    "train_barcode_bin_label",
    "train_barcode_bin_query",
    "setting",
    "b_cap",
    "head_lr",
    "backbone_lr",
    "train_sampling_mode",
    "unfreeze_scope",
    "train_threshold",
    "train_size",
    "init_head",
]

FULL_GROUP_COLS = [
    col for col in RUN_GROUP_COLS if col not in {"train_size"}
]

VELOCITY_GROUP_COLS = [
    "heldout_min_barcodes",
    "test_quality_bin",
    "test_quality_bin_label",
    "test_quality_bin_query",
    "train_barcode_bin",
    "train_barcode_bin_label",
    "train_barcode_bin_query",
    "train_sampling_mode",
    "setting",
    "b_cap",
    "head_lr",
    "backbone_lr",
    "unfreeze_scope",
    "metric",
]


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    columns = []
    for col in out.columns:
        if isinstance(col, tuple):
            columns.append("_".join(str(item) for item in col if item).rstrip("_"))
        else:
            columns.append(str(col))
    out.columns = columns
    return out


def existing_columns(frame: pd.DataFrame, candidates: Iterable[str]) -> list[str]:
    return [col for col in candidates if col in frame.columns]


def aggregate(frame: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    group_cols = existing_columns(frame, group_cols)
    metric_cols = existing_columns(frame, metric_cols)
    if not group_cols or not metric_cols:
        return pd.DataFrame()
    summary = frame.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary = flatten_columns(summary)
    return summary.sort_values(group_cols).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    parser.add_argument("--seeds", nargs="*", default=None)
    parser.add_argument("--per_seed_subdir", default="per_seed")
    parser.add_argument("--combined_subdir", default="combined")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    combined = outdir / args.combined_subdir
    combined.mkdir(parents=True, exist_ok=True)

    if args.seeds:
        seed_dirs = [outdir / args.per_seed_subdir / f"seed_{seed}" for seed in args.seeds]
    else:
        seed_root = outdir / args.per_seed_subdir
        seed_dirs = sorted(path for path in seed_root.glob("seed_*") if path.is_dir())

    combined_frames: dict[str, pd.DataFrame] = {}
    for csv_name in CSV_NAMES:
        parts = []
        for seed_dir in seed_dirs:
            path = seed_dir / csv_name
            if not path.exists():
                continue
            part = pd.read_csv(path)
            part["source_seed_dir"] = seed_dir.name
            parts.append(part)
        if not parts:
            continue
        frame = pd.concat(parts, ignore_index=True)
        frame.to_csv(combined / csv_name, index=False)
        combined_frames[csv_name] = frame
        print(f"Wrote {combined / csv_name} ({frame.shape[0]} rows)")

    runs = combined_frames.get("learning_curve_runs.csv")
    if runs is not None:
        summary = aggregate(runs, RUN_GROUP_COLS, RUN_METRIC_COLS)
        if not summary.empty:
            summary.to_csv(combined / "learning_curve_summary_mean_std.csv", index=False)
            print(f"Wrote {combined / 'learning_curve_summary_mean_std.csv'} ({summary.shape[0]} rows)")

        if {"train_size", "train_pool_eligible_size"}.issubset(runs.columns):
            full_runs = runs.loc[runs["train_size"] == runs["train_pool_eligible_size"]].copy()
            full_summary = aggregate(
                full_runs,
                FULL_GROUP_COLS,
                [
                    "val_pearson",
                    "val_spearman",
                    "val_r2",
                    "val_r2_cod",
                    "val_loss_standardized",
                    "test_pearson",
                    "test_spearman",
                    "test_r2",
                    "test_r2_cod",
                    "test_loss_standardized",
                    "test_pearson_sq",
                    "best_epoch",
                ],
            )
            if not full_summary.empty:
                full_path = combined / "full_fraction_summary_mean_std.csv"
                full_summary.to_csv(full_path, index=False)
                print(f"Wrote {full_path} ({full_summary.shape[0]} rows)")
                if "train_barcode_bin" in full_summary.columns:
                    barcode_path = combined / "barcode_bin_full_fraction_summary_mean_std.csv"
                    full_summary.to_csv(barcode_path, index=False)
                    print(f"Wrote {barcode_path} ({full_summary.shape[0]} rows)")

    segments = combined_frames.get("learning_curve_velocity_segments.csv")
    if segments is not None:
        velocity_summary = aggregate(
            segments,
            VELOCITY_GROUP_COLS,
            ["delta_metric", "slope_per_construct", "slope_per_100_constructs"],
        )
        if not velocity_summary.empty:
            velocity_summary.to_csv(combined / "learning_curve_velocity_summary_mean_std.csv", index=False)
            print(f"Wrote {combined / 'learning_curve_velocity_summary_mean_std.csv'} ({velocity_summary.shape[0]} rows)")

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


if __name__ == "__main__":
    main()
