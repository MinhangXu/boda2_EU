#!/usr/bin/env python3
"""Generate Lib1 full n=1 barcode-bin follow-up manifest.

This follow-up keeps the matched-N barcode-bin experiment fixed in every
important way: same selected configs, split seeds, heldout policy, model seed,
unweighted loss, and exact n=1 barcode filter. The only intervention is using
the full eligible n=1 training pool instead of downsampling to N=1000.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from generate_lib1_barcode_bin_matched_manifest import (
    DEFAULT_CONFIG_SUMMARY,
    DEFAULT_SOURCE_BASELINE,
    SELECTED_CONFIGS,
    load_part_tables,
    load_selected_baseline,
    load_selected_config_summary,
    normalize_record_types,
    records_from_df,
    train_pool_size_for,
)
from generate_lib1_outer_seed_prior_hpo_manifest import (
    DEFAULT_OUTDIR,
    LEARN_DIR,
    OUTER_SEED_MODEL_SEED,
    OUTER_SEED_SPLIT_SEEDS,
    build_train_command,
)


SOURCE_TAG = "lib1_outer_seed_prior_no_rc_june2026"
MANIFEST_TAG = "lib1_barcode_bin_n1_full_june2026"
REQUESTED_CAP_N = 2000
BC1_BIN = {
    "barcode_bin": "bc1",
    "barcode_bin_label": "n=1",
    "train_min_barcodes": 1,
    "train_max_barcodes": 1,
}


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def n1_full_project_name(project: str) -> str:
    marker = "__outer_seed_prior_no_rc__"
    if marker in project:
        return project.replace(marker, marker + "barcode_bin_n1_full__", 1)
    return project + "__barcode_bin_n1_full"


def build_feasibility(baseline: pd.DataFrame) -> pd.DataFrame:
    part_tables = load_part_tables(baseline)
    rows = []
    for _, row in baseline.iterrows():
        available = train_pool_size_for(row, BC1_BIN, part_tables)
        rows.append(
            {
                "part": row["part"],
                "config_id": row["config_id"],
                "split_seed": int(row["split_seed"]),
                "barcode_bin": BC1_BIN["barcode_bin"],
                "barcode_bin_label": BC1_BIN["barcode_bin_label"],
                "available_train_rows": int(available),
                "requested_cap_n": REQUESTED_CAP_N,
                "is_full_train_pool": True,
                "include_run": True,
                "skip_reason": "",
            }
        )
    return pd.DataFrame(rows)


def build_manifest(baseline: pd.DataFrame, feasibility: pd.DataFrame, tag: str) -> pd.DataFrame:
    availability = {
        (row["part"], row["config_id"], int(row["split_seed"])): int(row["available_train_rows"])
        for _, row in feasibility.iterrows()
    }

    rows = []
    manifest_row = 1
    for _, source_row in baseline.iterrows():
        rec = source_row.to_dict()
        part_slug = rec["part_slug"]
        config_id = rec["config_id"]
        split_seed = int(rec["split_seed"])
        available = availability[(rec["part"], config_id, split_seed)]
        run_name = f"{tag}__{part_slug}__{config_id}__bc1_full__seed{split_seed}"
        logger_project = n1_full_project_name(str(rec["logger_project"]))
        rec.update(
            {
                "source_manifest_tag": rec.get("manifest_tag"),
                "source_manifest_row": rec.get("manifest_row"),
                "source_planned_run_name": rec.get("planned_run_name"),
                "manifest_tag": tag,
                "manifest_row": manifest_row,
                "barcode_bin": BC1_BIN["barcode_bin"],
                "barcode_bin_label": BC1_BIN["barcode_bin_label"],
                "train_size_label": "full",
                "requested_cap_n": REQUESTED_CAP_N,
                "available_train_rows": available,
                "effective_train_size_n": available,
                "is_full_train_pool": True,
                "graph_module": "CNNBasicTraining",
                "barcode_weighting": False,
                "train_min_barcodes": int(BC1_BIN["train_min_barcodes"]),
                "train_max_barcodes": int(BC1_BIN["train_max_barcodes"]),
                "train_size_n": None,
                "train_size_frac": 1.0,
                "train_sampling_mode": "random",
                "logger_project": logger_project,
                "comparison_group": logger_project,
                "planned_run_name": run_name,
                "run_name": run_name,
                "exact_run_name": True,
                "model_seed": OUTER_SEED_MODEL_SEED,
                "use_reverse_complements": False,
                "epoch_eval_splits": ["train", "val", "test"],
                "checkpoint_monitor": "val_pearson",
                "stopping_mode": "max",
                "artifact_path": str(
                    LEARN_DIR
                    / "local_artifacts"
                    / tag
                    / part_slug
                    / config_id
                    / "bc1_full"
                    / f"split_seed_{split_seed}"
                ),
                "default_root_dir": str(
                    LEARN_DIR
                    / "outputs"
                    / "hpo_runs"
                    / tag
                    / part_slug
                    / config_id
                    / "bc1_full"
                    / f"split_seed_{split_seed}"
                ),
                "best_checkpoint_dir": str(
                    LEARN_DIR
                    / "outputs"
                    / "hpo_runs"
                    / "by_project"
                    / logger_project
                    / "best_checkpoint_model"
                ),
                "launcher_status_dir": str(
                    LEARN_DIR / "outputs" / "hpo_runs" / "status" / tag
                ),
            }
        )
        rec = normalize_record_types(rec)
        rec["train_command"] = build_train_command(rec)
        rows.append(rec)
        manifest_row += 1
    return pd.DataFrame(rows)


def validate(manifest: pd.DataFrame, feasibility: pd.DataFrame) -> None:
    expected = len(SELECTED_CONFIGS) * len(next(iter(SELECTED_CONFIGS.values()))) * len(OUTER_SEED_SPLIT_SEEDS)
    if len(manifest) != expected:
        raise RuntimeError(f"Expected {expected} manifest rows, got {len(manifest)}")
    if set(manifest["barcode_bin"]) != {"bc1"}:
        raise RuntimeError("Full n=1 manifest must contain only barcode_bin=bc1")
    if set(manifest["train_min_barcodes"].astype(int)) != {1}:
        raise RuntimeError("Full n=1 manifest train_min_barcodes mismatch")
    if set(manifest["train_max_barcodes"].astype(int)) != {1}:
        raise RuntimeError("Full n=1 manifest train_max_barcodes mismatch")
    if manifest["train_size_n"].notna().any():
        raise RuntimeError("Full n=1 manifest should leave train_size_n empty")
    if set(manifest["train_sampling_mode"]) != {"random"}:
        raise RuntimeError("Full n=1 manifest must use random sampling mode")
    if set(manifest["graph_module"]) != {"CNNBasicTraining"}:
        raise RuntimeError("Full n=1 manifest must use CNNBasicTraining")
    if set(manifest["barcode_weighting"].map(bool)) != {False}:
        raise RuntimeError("Full n=1 manifest must be unweighted")
    if set(manifest["model_seed"].astype(int)) != {OUTER_SEED_MODEL_SEED}:
        raise RuntimeError("Full n=1 manifest model_seed mismatch")
    if set(manifest["use_reverse_complements"].map(bool)) != {False}:
        raise RuntimeError("Full n=1 manifest must not use reverse complements")
    if manifest["train_command"].str.contains("--train_size_n", regex=False).any():
        raise RuntimeError("Full n=1 train commands should not pass --train_size_n")
    if (feasibility["available_train_rows"].astype(int) < 32).any():
        raise RuntimeError("Found an n=1 train pool below the data module minimum")


def write_outputs(
    manifest: pd.DataFrame,
    feasibility: pd.DataFrame,
    selected_summary: pd.DataFrame,
    outdir: Path,
    tag: str,
) -> Dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest_csv": outdir / f"{tag}__run_manifest.csv",
        "manifest_json": outdir / f"{tag}__run_manifest.json",
        "manifest_jsonl": outdir / f"{tag}__run_manifest.jsonl",
        "feasibility_csv": outdir / f"{tag}__bin_feasibility.csv",
        "selected_config_summary_csv": outdir / f"{tag}__selected_config_summary.csv",
        "summary_json": outdir / f"{tag}__summary.json",
    }
    records = records_from_df(manifest)
    manifest.to_csv(paths["manifest_csv"], index=False)
    paths["manifest_json"].write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    write_jsonl(paths["manifest_jsonl"], records)
    feasibility.to_csv(paths["feasibility_csv"], index=False)
    selected_summary.to_csv(paths["selected_config_summary_csv"], index=False)

    part_counts = (
        feasibility.groupby("part", observed=True)["available_train_rows"]
        .agg(["min", "median", "max"])
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "manifest_tag": tag,
        "source_manifest_tag": SOURCE_TAG,
        "selected_configs": SELECTED_CONFIGS,
        "barcode_bin": BC1_BIN,
        "requested_cap_n": REQUESTED_CAP_N,
        "is_full_train_pool": True,
        "run_manifest_rows": int(len(manifest)),
        "split_seeds": OUTER_SEED_SPLIT_SEEDS,
        "model_seed": OUTER_SEED_MODEL_SEED,
        "use_reverse_complements": False,
        "barcode_weighting": False,
        "graph_module": "CNNBasicTraining",
        "min_available_train_rows": int(feasibility["available_train_rows"].min()),
        "max_available_train_rows": int(feasibility["available_train_rows"].max()),
        "available_train_rows_by_part": part_counts,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {key: str(value) for key, value in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-baseline", type=Path, default=DEFAULT_SOURCE_BASELINE)
    parser.add_argument("--config-summary", type=Path, default=DEFAULT_CONFIG_SUMMARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest-tag", default=MANIFEST_TAG)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_selected_baseline(args.source_baseline)
    feasibility = build_feasibility(baseline)
    manifest = build_manifest(baseline, feasibility, args.manifest_tag)
    selected_summary = load_selected_config_summary(args.config_summary)
    validate(manifest, feasibility)

    print(f"Full n=1 barcode-bin follow-up manifest rows: {len(manifest)}")
    print("Available n=1 train rows by part:")
    print(
        feasibility.groupby("part", observed=True)["available_train_rows"]
        .agg(["min", "median", "max"])
        .to_string()
    )
    if not args.no_write:
        paths = write_outputs(manifest, feasibility, selected_summary, args.outdir, args.manifest_tag)
        for label, path in paths.items():
            print(f"{label}: {path}")


if __name__ == "__main__":
    main()
