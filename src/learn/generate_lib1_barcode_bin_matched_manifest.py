#!/usr/bin/env python3
"""Generate Lib1 matched-N barcode-bin scratch training manifest.

This experiment trains the robust outer-seed configs on one barcode-count range
at a time, with matched training size N=1000, unweighted MSE, and the same
high-barcode validation/test split policy as the outer-seed run.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
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
MANIFEST_TAG = "lib1_barcode_bin_matched_n1000_june2026"
DEFAULT_SOURCE_BASELINE = (
    DEFAULT_OUTDIR
    / "lib1_outer_seed_selected_barcode_weighted_june2026__selected_unweighted_baseline.csv"
)
DEFAULT_CONFIG_SUMMARY = (
    LEARN_DIR
    / "outputs"
    / "hpo_analyses"
    / SOURCE_TAG
    / "outer_seed_config_summary.csv"
)
TRAIN_SIZE_N = 1000

SELECTED_CONFIGS: Dict[str, List[str]] = {
    "Promoter": ["promoter_cfg011", "promoter_cfg029", "promoter_cfg018"],
    "Intron": ["intron_cfg011", "intron_cfg013", "intron_cfg009"],
    "3UTR": ["utr3_cfg001", "utr3_cfg009", "utr3_cfg022"],
    "5UTR": ["utr5_cfg007", "utr5_cfg015", "utr5_cfg001"],
}

BARCODE_BINS = [
    {"barcode_bin": "bc1", "barcode_bin_label": "n=1", "train_min_barcodes": 1, "train_max_barcodes": 1},
    {"barcode_bin": "bc2", "barcode_bin_label": "n=2", "train_min_barcodes": 2, "train_max_barcodes": 2},
    {"barcode_bin": "bc3", "barcode_bin_label": "n=3", "train_min_barcodes": 3, "train_max_barcodes": 3},
    {"barcode_bin": "bc4_5", "barcode_bin_label": "n=4-5", "train_min_barcodes": 4, "train_max_barcodes": 5},
    {"barcode_bin": "bc_ge6", "barcode_bin_label": "n>=6", "train_min_barcodes": 6, "train_max_barcodes": None},
]


def records_from_df(df: pd.DataFrame) -> List[dict]:
    return [normalize_record_types(record) for record in df.to_dict(orient="records")]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def barcode_bin_project_name(project: str) -> str:
    marker = "__outer_seed_prior_no_rc__"
    if marker in project:
        return project.replace(marker, marker + "barcode_bin_n1000__", 1)
    return project + "__barcode_bin_n1000"


def _sep(value: str) -> str:
    return {"space": " ", "tab": "\t", "comma": ",", " ": " ", "\t": "\t", ",": ","}[str(value)]


def load_selected_baseline(source_baseline: Path) -> pd.DataFrame:
    if not source_baseline.exists():
        raise FileNotFoundError(source_baseline)
    manifest = pd.read_csv(source_baseline)
    selected_rows = []
    for part_order, (part, config_ids) in enumerate(SELECTED_CONFIGS.items()):
        part_rows = manifest[
            manifest["part"].eq(part) & manifest["config_id"].isin(config_ids)
        ].copy()
        if part_rows.empty:
            raise RuntimeError(f"No selected baseline rows found for {part}")
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


def load_part_tables(baseline: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tables = {}
    for _, row in baseline.drop_duplicates("part").iterrows():
        df = pd.read_csv(row["datafile_path"], sep=_sep(row["sep"])).copy()
        target_col = row["target_column"]
        barcode_col = row["barcode_column"]
        sequence_col = row["sequence_column"]
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df[barcode_col] = pd.to_numeric(df[barcode_col], errors="coerce")
        df = df.loc[
            df[sequence_col].notna()
            & df[barcode_col].notna()
            & np.isfinite(df[target_col])
        ].reset_index(drop=True)
        tables[row["part"]] = df
    return tables


def train_pool_size_for(row: pd.Series, bin_spec: dict, part_tables: Dict[str, pd.DataFrame]) -> int:
    df = part_tables[row["part"]]
    barcode_col = row["barcode_column"]
    hq_df = df.loc[df[barcode_col] >= int(row["test_min_barcodes"])].copy()
    rng = np.random.default_rng(int(row["split_seed"]))
    perm = rng.permutation(hq_df.index.to_numpy())
    n_hq = len(perm)

    if pd.isna(row.get("test_size_within_hq")):
        n_test = max(1, int(round(n_hq * float(row["test_frac_within_hq"]))))
    else:
        n_test = int(row["test_size_within_hq"])
    if pd.isna(row.get("val_size_within_hq")):
        n_val = max(1, int(round(n_hq * float(row["val_frac_within_hq"]))))
    else:
        n_val = int(row["val_size_within_hq"])
    if n_test + n_val >= n_hq:
        if not pd.isna(row.get("test_size_within_hq")) or not pd.isna(row.get("val_size_within_hq")):
            raise RuntimeError(f"{row['part']} requested val/test sizes exhaust HQ rows")
        n_test = max(1, n_hq // 5)
        n_val = max(1, n_hq // 5)

    heldout = set(perm[: n_test + n_val].tolist())
    rest = df.loc[~df.index.isin(heldout)].copy()
    mask = rest[barcode_col] >= int(bin_spec["train_min_barcodes"])
    max_bc: Optional[int] = bin_spec.get("train_max_barcodes")
    if max_bc is not None:
        mask = mask & (rest[barcode_col] <= int(max_bc))
    return int(mask.sum())


def build_manifest(baseline: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    manifest_row = 1
    for _, source_row in baseline.iterrows():
        for bin_spec in BARCODE_BINS:
            rec = source_row.to_dict()
            part_slug = rec["part_slug"]
            config_id = rec["config_id"]
            split_seed = int(rec["split_seed"])
            barcode_bin = bin_spec["barcode_bin"]
            run_name = f"{tag}__{part_slug}__{config_id}__{barcode_bin}__seed{split_seed}"
            logger_project = barcode_bin_project_name(str(rec["logger_project"]))
            rec.update(
                {
                    "source_manifest_tag": rec.get("manifest_tag"),
                    "source_manifest_row": rec.get("manifest_row"),
                    "source_planned_run_name": rec.get("planned_run_name"),
                    "manifest_tag": tag,
                    "manifest_row": manifest_row,
                    "barcode_bin": barcode_bin,
                    "barcode_bin_label": bin_spec["barcode_bin_label"],
                    "graph_module": "CNNBasicTraining",
                    "barcode_weighting": False,
                    "train_min_barcodes": int(bin_spec["train_min_barcodes"]),
                    "train_max_barcodes": bin_spec.get("train_max_barcodes"),
                    "train_size_n": TRAIN_SIZE_N,
                    "train_size_frac": 1.0,
                    "train_sampling_mode": "random",
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
                        / barcode_bin
                        / f"split_seed_{split_seed}"
                    ),
                    "default_root_dir": str(
                        LEARN_DIR
                        / "outputs"
                        / "hpo_runs"
                        / tag
                        / part_slug
                        / config_id
                        / barcode_bin
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


def load_selected_config_summary(config_summary: Path) -> pd.DataFrame:
    if not config_summary.exists():
        return pd.DataFrame()
    summary = pd.read_csv(config_summary)
    selected = []
    for part_order, (part, config_ids) in enumerate(SELECTED_CONFIGS.items()):
        rows = summary[summary["part"].eq(part) & summary["config_id"].isin(config_ids)].copy()
        rows["part_order"] = part_order
        rows["config_order"] = rows["config_id"].map(
            {config_id: idx for idx, config_id in enumerate(config_ids)}
        )
        selected.append(rows)
    return (
        pd.concat(selected, ignore_index=True)
        .sort_values(["part_order", "config_order"], kind="stable")
        .drop(columns=["part_order", "config_order"])
        .reset_index(drop=True)
    )


def validate(baseline: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    expected = (
        len(SELECTED_CONFIGS)
        * len(next(iter(SELECTED_CONFIGS.values())))
        * len(BARCODE_BINS)
        * len(OUTER_SEED_SPLIT_SEEDS)
    )
    if len(manifest) != expected:
        raise RuntimeError(f"Expected {expected} manifest rows, got {len(manifest)}")
    if set(manifest["graph_module"]) != {"CNNBasicTraining"}:
        raise RuntimeError("Barcode-bin manifest must use CNNBasicTraining")
    if set(manifest["barcode_weighting"].map(bool)) != {False}:
        raise RuntimeError("Barcode-bin manifest must be unweighted")
    if set(manifest["train_size_n"].astype(int)) != {TRAIN_SIZE_N}:
        raise RuntimeError("Barcode-bin manifest train_size_n mismatch")
    if set(manifest["model_seed"].astype(int)) != {OUTER_SEED_MODEL_SEED}:
        raise RuntimeError("Barcode-bin manifest model_seed mismatch")

    part_tables = load_part_tables(baseline)
    feasibility = []
    for _, row in manifest.iterrows():
        bin_spec = next(spec for spec in BARCODE_BINS if spec["barcode_bin"] == row["barcode_bin"])
        available = train_pool_size_for(row, bin_spec, part_tables)
        feasibility.append(
            {
                "part": row["part"],
                "config_id": row["config_id"],
                "split_seed": int(row["split_seed"]),
                "barcode_bin": row["barcode_bin"],
                "available_train_rows": available,
                "requested_train_size_n": TRAIN_SIZE_N,
            }
        )
    feasibility_df = pd.DataFrame(feasibility)
    too_small = feasibility_df[feasibility_df["available_train_rows"] < TRAIN_SIZE_N]
    if not too_small.empty:
        raise RuntimeError("Insufficient rows for matched-N bins:\n{}".format(too_small.to_string(index=False)))
    return feasibility_df


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

    summary = {
        "manifest_tag": tag,
        "source_manifest_tag": SOURCE_TAG,
        "selected_configs": SELECTED_CONFIGS,
        "barcode_bins": BARCODE_BINS,
        "train_size_n": TRAIN_SIZE_N,
        "run_manifest_rows": int(len(manifest)),
        "split_seeds": OUTER_SEED_SPLIT_SEEDS,
        "model_seed": OUTER_SEED_MODEL_SEED,
        "use_reverse_complements": False,
        "barcode_weighting": False,
        "graph_module": "CNNBasicTraining",
        "min_available_train_rows": int(feasibility["available_train_rows"].min()),
        "runs_per_part_bin_split_seed": (
            manifest.groupby(["part", "barcode_bin", "split_seed"])
            .size()
            .reset_index(name="runs")
            .to_dict(orient="records")
        ),
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
    manifest = build_manifest(baseline, args.manifest_tag)
    feasibility = validate(baseline, manifest)
    selected_summary = load_selected_config_summary(args.config_summary)

    print(f"Barcode-bin matched-N manifest rows: {len(manifest)}")
    print(f"Minimum available train rows across bins/splits: {feasibility['available_train_rows'].min()}")
    print(manifest.groupby(["part", "barcode_bin", "split_seed"]).size().unstack(fill_value=0))
    if not args.no_write:
        paths = write_outputs(manifest, feasibility, selected_summary, args.outdir, args.manifest_tag)
        for label, path in paths.items():
            print(f"{label}: {path}")


if __name__ == "__main__":
    main()
