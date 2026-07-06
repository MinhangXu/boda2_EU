#!/usr/bin/env python3
"""Generate the selected-config Lib1 barcode-weighted loss follow-up manifest.

The unweighted baseline rows already exist in the June 2026 outer split-seed
manifest. This script writes the paired weighted rows for the selected robust
configs, preserving the same config IDs, split seeds, model seed, and heldout
definition.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from generate_lib1_outer_seed_prior_hpo_manifest import (
    DEFAULT_OUTDIR,
    LEARN_DIR,
    OUTER_SEED_MODEL_SEED,
    OUTER_SEED_SPLIT_SEEDS,
    build_train_command,
    normalize_record_types,
)


SOURCE_TAG = "lib1_outer_seed_prior_no_rc_june2026"
MANIFEST_TAG = "lib1_outer_seed_selected_barcode_weighted_june2026"
DEFAULT_SOURCE_MANIFEST = DEFAULT_OUTDIR / f"{SOURCE_TAG}__run_manifest.csv"
DEFAULT_CONFIG_SUMMARY = (
    LEARN_DIR
    / "outputs"
    / "hpo_analyses"
    / SOURCE_TAG
    / "outer_seed_config_summary.csv"
)

SELECTED_CONFIGS: Dict[str, List[str]] = {
    "Promoter": ["promoter_cfg011", "promoter_cfg029", "promoter_cfg018"],
    "Intron": ["intron_cfg011", "intron_cfg013", "intron_cfg009"],
    "3UTR": ["utr3_cfg001", "utr3_cfg009", "utr3_cfg022"],
    "5UTR": ["utr5_cfg007", "utr5_cfg015", "utr5_cfg001"],
}


def records_from_df(df: pd.DataFrame) -> List[dict]:
    return [normalize_record_types(record) for record in df.to_dict(orient="records")]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def weighted_project_name(project: str) -> str:
    marker = "__outer_seed_prior_no_rc__"
    if marker in project:
        return project.replace(marker, marker + "barcode_weighted__", 1)
    return project + "__barcode_weighted"


def load_selected_baseline(source_manifest: Path) -> pd.DataFrame:
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    manifest = pd.read_csv(source_manifest)
    selected_rows = []
    for part_order, (part, config_ids) in enumerate(SELECTED_CONFIGS.items()):
        part_rows = manifest[
            manifest["part"].eq(part) & manifest["config_id"].isin(config_ids)
        ].copy()
        if part_rows.empty:
            raise RuntimeError(f"No baseline rows selected for {part}")
        part_rows["part_order"] = part_order
        part_rows["config_order"] = part_rows["config_id"].map(
            {config_id: idx for idx, config_id in enumerate(config_ids)}
        )
        selected_rows.append(part_rows)
    baseline = pd.concat(selected_rows, ignore_index=True)
    baseline = baseline.sort_values(
        ["part_order", "config_order", "split_seed"], kind="stable"
    ).drop(columns=["part_order", "config_order"])
    return baseline.reset_index(drop=True)


def build_weighted_manifest(baseline: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    for manifest_row, (_, source_row) in enumerate(baseline.iterrows(), start=1):
        rec = source_row.to_dict()
        part_slug = rec["part_slug"]
        config_id = rec["config_id"]
        split_seed = int(rec["split_seed"])
        seed_label = f"split_seed_{split_seed}"
        run_name = f"{tag}__{part_slug}__{config_id}__seed{split_seed}"
        logger_project = weighted_project_name(str(rec["logger_project"]))

        rec.update(
            {
                "baseline_manifest_tag": rec.get("manifest_tag"),
                "baseline_manifest_row": rec.get("manifest_row"),
                "baseline_planned_run_name": rec.get("planned_run_name"),
                "manifest_tag": tag,
                "manifest_row": manifest_row,
                "graph_module": "CNNWeightedRegressionTraining",
                "barcode_weighting": True,
                "barcode_weight_cap": 8.0,
                "barcode_weight_min": 0.1,
                "weighted_loss_reduction": "mean",
                "logger_project": logger_project,
                "comparison_group": logger_project,
                "planned_run_name": run_name,
                "run_name": run_name,
                "exact_run_name": True,
                "model_seed": OUTER_SEED_MODEL_SEED,
                "use_reverse_complements": False,
                "artifact_path": str(
                    LEARN_DIR
                    / "local_artifacts"
                    / tag
                    / part_slug
                    / config_id
                    / seed_label
                ),
                "default_root_dir": str(
                    LEARN_DIR
                    / "outputs"
                    / "hpo_runs"
                    / tag
                    / part_slug
                    / config_id
                    / seed_label
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
    return pd.DataFrame(rows)


def load_selected_config_summary(config_summary: Path) -> pd.DataFrame:
    if not config_summary.exists():
        return pd.DataFrame()
    summary = pd.read_csv(config_summary)
    selected_rows = []
    for part_order, (part, config_ids) in enumerate(SELECTED_CONFIGS.items()):
        rows = summary[summary["part"].eq(part) & summary["config_id"].isin(config_ids)].copy()
        rows["part_order"] = part_order
        rows["config_order"] = rows["config_id"].map(
            {config_id: idx for idx, config_id in enumerate(config_ids)}
        )
        selected_rows.append(rows)
    return (
        pd.concat(selected_rows, ignore_index=True)
        .sort_values(["part_order", "config_order"], kind="stable")
        .drop(columns=["part_order", "config_order"])
        .reset_index(drop=True)
    )


def validate(baseline: pd.DataFrame, weighted: pd.DataFrame) -> None:
    expected = sum(len(ids) for ids in SELECTED_CONFIGS.values()) * len(OUTER_SEED_SPLIT_SEEDS)
    if len(baseline) != expected:
        raise RuntimeError(f"Expected {expected} selected baseline rows, got {len(baseline)}")
    if len(weighted) != expected:
        raise RuntimeError(f"Expected {expected} weighted rows, got {len(weighted)}")
    if set(weighted["graph_module"]) != {"CNNWeightedRegressionTraining"}:
        raise RuntimeError("Weighted manifest graph_module mismatch")
    if set(weighted["barcode_weighting"].map(bool)) != {True}:
        raise RuntimeError("Weighted manifest barcode_weighting mismatch")
    if set(weighted["model_seed"].astype(int)) != {OUTER_SEED_MODEL_SEED}:
        raise RuntimeError("Weighted manifest model_seed mismatch")
    if set(weighted["use_reverse_complements"].map(bool)) != {False}:
        raise RuntimeError("Weighted manifest reverse-complement mismatch")

    expected_seeds = set(OUTER_SEED_SPLIT_SEEDS)
    for part, config_ids in SELECTED_CONFIGS.items():
        got_ids = set(weighted.loc[weighted["part"].eq(part), "config_id"])
        if got_ids != set(config_ids):
            raise RuntimeError(f"{part} config IDs mismatch: {sorted(got_ids)}")
        for config_id in config_ids:
            seeds = set(
                weighted.loc[
                    weighted["part"].eq(part) & weighted["config_id"].eq(config_id),
                    "split_seed",
                ].astype(int)
            )
            if seeds != expected_seeds:
                raise RuntimeError(f"{part} {config_id} split seeds mismatch: {sorted(seeds)}")


def write_outputs(
    baseline: pd.DataFrame,
    weighted: pd.DataFrame,
    selected_summary: pd.DataFrame,
    outdir: Path,
    tag: str,
) -> Dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "selected_unweighted_baseline_csv": outdir / f"{tag}__selected_unweighted_baseline.csv",
        "selected_unweighted_baseline_jsonl": outdir / f"{tag}__selected_unweighted_baseline.jsonl",
        "manifest_csv": outdir / f"{tag}__run_manifest.csv",
        "manifest_json": outdir / f"{tag}__run_manifest.json",
        "manifest_jsonl": outdir / f"{tag}__run_manifest.jsonl",
        "selected_config_summary_csv": outdir / f"{tag}__selected_config_summary.csv",
        "summary_json": outdir / f"{tag}__summary.json",
    }

    baseline_records = records_from_df(baseline)
    weighted_records = records_from_df(weighted)
    baseline.to_csv(paths["selected_unweighted_baseline_csv"], index=False)
    write_jsonl(paths["selected_unweighted_baseline_jsonl"], baseline_records)
    weighted.to_csv(paths["manifest_csv"], index=False)
    paths["manifest_json"].write_text(json.dumps(weighted_records, indent=2, sort_keys=True) + "\n")
    write_jsonl(paths["manifest_jsonl"], weighted_records)
    selected_summary.to_csv(paths["selected_config_summary_csv"], index=False)

    summary = {
        "manifest_tag": tag,
        "source_manifest_tag": SOURCE_TAG,
        "selected_configs": SELECTED_CONFIGS,
        "baseline_rows": int(len(baseline)),
        "weighted_run_manifest_rows": int(len(weighted)),
        "split_seeds": OUTER_SEED_SPLIT_SEEDS,
        "model_seed": OUTER_SEED_MODEL_SEED,
        "use_reverse_complements": False,
        "barcode_weighting": True,
        "barcode_weight_cap": 8.0,
        "barcode_weight_min": 0.1,
        "graph_module": "CNNWeightedRegressionTraining",
        "runs_per_part_split_seed": (
            weighted.groupby(["part", "split_seed"]).size().unstack(fill_value=0).astype(int).to_dict(orient="index")
        ),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {key: str(value) for key, value in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--config-summary", type=Path, default=DEFAULT_CONFIG_SUMMARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest-tag", default=MANIFEST_TAG)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_selected_baseline(args.source_manifest)
    weighted = build_weighted_manifest(baseline, args.manifest_tag)
    selected_summary = load_selected_config_summary(args.config_summary)
    validate(baseline, weighted)

    print(f"Selected unweighted baseline rows: {len(baseline)}")
    print(f"Weighted follow-up rows: {len(weighted)}")
    print(weighted.groupby(["part", "split_seed"]).size().unstack(fill_value=0))
    if not args.no_write:
        paths = write_outputs(baseline, weighted, selected_summary, args.outdir, args.manifest_tag)
        for label, path in paths.items():
            print(f"{label}: {path}")


if __name__ == "__main__":
    main()
