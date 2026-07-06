#!/usr/bin/env python3
"""Generate Lib1 barcode-threshold downsampling scratch training manifest.

This experiment trains the top robust outer-seed configs on threshold-filtered
barcode pools at nested downsampled training sizes, with the same high-barcode
validation/test split policy as the outer-seed run.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
MANIFEST_TAG = "lib1_barcode_threshold_downsample_june2026"
DEFAULT_SOURCE_MANIFEST = DEFAULT_OUTDIR / f"{SOURCE_TAG}__run_manifest.csv"
DEFAULT_CONFIG_SUMMARY = (
    LEARN_DIR
    / "outputs"
    / "hpo_analyses"
    / SOURCE_TAG
    / "outer_seed_config_summary.csv"
)

SELECTED_CONFIGS: Dict[str, List[str]] = {
    "Promoter": [
        "promoter_cfg011",
        "promoter_cfg029",
        "promoter_cfg014",
        "promoter_cfg018",
        "promoter_cfg013",
    ],
    "Intron": [
        "intron_cfg011",
        "intron_cfg013",
        "intron_cfg009",
        "intron_cfg014",
        "intron_cfg003",
    ],
    "3UTR": [
        "utr3_cfg001",
        "utr3_cfg009",
        "utr3_cfg003",
        "utr3_cfg022",
        "utr3_cfg011",
    ],
    "5UTR": [
        "utr5_cfg007",
        "utr5_cfg005",
        "utr5_cfg015",
        "utr5_cfg008",
        "utr5_cfg019",
    ],
}

BARCODE_THRESHOLDS = [
    {"barcode_threshold": 1, "barcode_threshold_label": "bc_ge1", "barcode_threshold_readable": "n>=1"},
    {"barcode_threshold": 2, "barcode_threshold_label": "bc_ge2", "barcode_threshold_readable": "n>=2"},
    {"barcode_threshold": 3, "barcode_threshold_label": "bc_ge3", "barcode_threshold_readable": "n>=3"},
]
TRAIN_SIZE_N_ARMS = [100, 500, 1500, 2500, 3500]


def records_from_df(df: pd.DataFrame) -> List[dict]:
    return [normalize_record_types(record) for record in df.to_dict(orient="records")]


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def _sep(value: str) -> str:
    return {"space": " ", "tab": "\t", "comma": ",", " ": " ", "\t": "\t", ",": ","}[str(value)]


def threshold_project_name(project: str) -> str:
    marker = "__outer_seed_prior_no_rc__"
    if marker in project:
        return project.replace(marker, marker + "barcode_threshold_downsample__", 1)
    return project + "__barcode_threshold_downsample"


def downsample_seed_for(split_seed: int, threshold: int) -> int:
    split_index = OUTER_SEED_SPLIT_SEEDS.index(int(split_seed))
    return 91000 + split_index * 10 + int(threshold)


def load_selected_source_manifest(source_manifest: Path) -> pd.DataFrame:
    if not source_manifest.exists():
        raise FileNotFoundError(source_manifest)
    manifest = pd.read_csv(source_manifest)
    selected_rows = []
    for part_order, (part, config_ids) in enumerate(SELECTED_CONFIGS.items()):
        part_rows = manifest[
            manifest["part"].eq(part) & manifest["config_id"].isin(config_ids)
        ].copy()
        if part_rows.empty:
            raise RuntimeError(f"No selected source rows found for {part}")
        got_configs = set(part_rows["config_id"])
        missing_configs = [config_id for config_id in config_ids if config_id not in got_configs]
        if missing_configs:
            raise RuntimeError(f"{part} missing selected configs: {missing_configs}")
        part_rows["part_order"] = part_order
        part_rows["config_order"] = part_rows["config_id"].map(
            {config_id: idx for idx, config_id in enumerate(config_ids)}
        )
        selected_rows.append(part_rows)
    selected = pd.concat(selected_rows, ignore_index=True)
    selected = selected.sort_values(
        ["part_order", "config_order", "split_seed"], kind="stable"
    ).drop(columns=["part_order", "config_order"])
    return selected.reset_index(drop=True)


def load_part_tables(source: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    tables = {}
    for _, row in source.drop_duplicates("part").iterrows():
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


def train_pool_size_for(
    row: pd.Series,
    threshold: int,
    part_tables: Dict[str, pd.DataFrame],
) -> int:
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
    return int((rest[barcode_col] >= int(threshold)).sum())


def size_arms_for(available_train_rows: int) -> List[Tuple[str, Optional[int], bool, bool, str]]:
    arms = []
    for train_size_n in TRAIN_SIZE_N_ARMS:
        included = available_train_rows >= train_size_n
        skip_reason = "" if included else f"available_train_rows<{train_size_n}"
        arms.append((f"n{train_size_n}", train_size_n, False, included, skip_reason))
    arms.append(("full", None, True, True, ""))
    return arms


def build_feasibility(source: pd.DataFrame, part_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for _, source_row in source.iterrows():
        for threshold_spec in BARCODE_THRESHOLDS:
            threshold = int(threshold_spec["barcode_threshold"])
            available = train_pool_size_for(source_row, threshold, part_tables)
            for size_label, requested_n, is_full, included, skip_reason in size_arms_for(available):
                rows.append(
                    {
                        "part": source_row["part"],
                        "config_id": source_row["config_id"],
                        "split_seed": int(source_row["split_seed"]),
                        "barcode_threshold": threshold,
                        "barcode_threshold_label": threshold_spec["barcode_threshold_label"],
                        "train_size_label": size_label,
                        "train_size_n_requested": requested_n,
                        "is_full_train_pool": is_full,
                        "available_train_rows": available,
                        "include_run": included,
                        "skip_reason": skip_reason,
                    }
                )
    return pd.DataFrame(rows)


def build_manifest(source: pd.DataFrame, feasibility: pd.DataFrame, tag: str) -> pd.DataFrame:
    feasibility_index = {
        (
            row["part"],
            row["config_id"],
            int(row["split_seed"]),
            int(row["barcode_threshold"]),
            row["train_size_label"],
        ): row
        for _, row in feasibility.iterrows()
    }

    rows = []
    manifest_row = 1
    for _, source_row in source.iterrows():
        for threshold_spec in BARCODE_THRESHOLDS:
            threshold = int(threshold_spec["barcode_threshold"])
            threshold_label = threshold_spec["barcode_threshold_label"]
            train_subsample_seed = downsample_seed_for(int(source_row["split_seed"]), threshold)
            for size_label, requested_n, is_full, _, _ in size_arms_for(
                int(
                    feasibility_index[
                        (
                            source_row["part"],
                            source_row["config_id"],
                            int(source_row["split_seed"]),
                            threshold,
                            "full",
                        )
                    ]["available_train_rows"]
                )
            ):
                feasible = feasibility_index[
                    (
                        source_row["part"],
                        source_row["config_id"],
                        int(source_row["split_seed"]),
                        threshold,
                        size_label,
                    )
                ]
                if not bool(feasible["include_run"]):
                    continue

                rec = source_row.to_dict()
                part_slug = rec["part_slug"]
                config_id = rec["config_id"]
                split_seed = int(rec["split_seed"])
                run_name = (
                    f"{tag}__{part_slug}__{config_id}__{threshold_label}__"
                    f"{size_label}__seed{split_seed}__ds{train_subsample_seed}"
                )
                logger_project = threshold_project_name(str(rec["logger_project"]))
                rec.update(
                    {
                        "source_manifest_tag": rec.get("manifest_tag"),
                        "source_manifest_row": rec.get("manifest_row"),
                        "source_planned_run_name": rec.get("planned_run_name"),
                        "manifest_tag": tag,
                        "manifest_row": manifest_row,
                        "barcode_threshold": threshold,
                        "barcode_threshold_label": threshold_label,
                        "barcode_threshold_readable": threshold_spec["barcode_threshold_readable"],
                        "train_size_label": size_label,
                        "train_size_n_requested": requested_n,
                        "available_train_rows": int(feasible["available_train_rows"]),
                        "is_full_train_pool": bool(is_full),
                        "graph_module": "CNNBasicTraining",
                        "barcode_weighting": False,
                        "train_min_barcodes": threshold,
                        "train_max_barcodes": None,
                        "train_size_n": requested_n,
                        "train_size_frac": 1.0,
                        "train_sampling_mode": "random",
                        "train_subsample_seed": train_subsample_seed,
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
                            / threshold_label
                            / size_label
                            / f"split_seed_{split_seed}"
                            / f"ds_{train_subsample_seed}"
                        ),
                        "default_root_dir": str(
                            LEARN_DIR
                            / "outputs"
                            / "hpo_runs"
                            / tag
                            / part_slug
                            / config_id
                            / threshold_label
                            / size_label
                            / f"split_seed_{split_seed}"
                            / f"ds_{train_subsample_seed}"
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


def validate(source: pd.DataFrame, manifest: pd.DataFrame, feasibility: pd.DataFrame) -> None:
    expected_source_rows = len(SELECTED_CONFIGS) * len(next(iter(SELECTED_CONFIGS.values()))) * len(OUTER_SEED_SPLIT_SEEDS)
    if len(source) != expected_source_rows:
        raise RuntimeError(f"Expected {expected_source_rows} selected source rows, got {len(source)}")

    expected_rows = int(feasibility["include_run"].sum())
    if len(manifest) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} manifest rows, got {len(manifest)}")
    if expected_rows != 1775:
        raise RuntimeError(f"Expected 1775 threshold downsample rows, got {expected_rows}")

    if set(manifest["graph_module"]) != {"CNNBasicTraining"}:
        raise RuntimeError("Threshold downsample manifest must use CNNBasicTraining")
    if set(manifest["barcode_weighting"].map(bool)) != {False}:
        raise RuntimeError("Threshold downsample manifest must be unweighted")
    if set(manifest["train_sampling_mode"]) != {"random"}:
        raise RuntimeError("Threshold downsample manifest must use random training sampling")
    if set(manifest["model_seed"].astype(int)) != {OUTER_SEED_MODEL_SEED}:
        raise RuntimeError("Threshold downsample manifest model_seed mismatch")
    if set(manifest["use_reverse_complements"].map(bool)) != {False}:
        raise RuntimeError("Threshold downsample manifest must not use reverse complements")

    exact = manifest[~manifest["is_full_train_pool"].map(bool)].copy()
    if (exact["train_size_n"].astype(int) > exact["available_train_rows"].astype(int)).any():
        bad = exact[exact["train_size_n"].astype(int) > exact["available_train_rows"].astype(int)]
        raise RuntimeError("Found infeasible exact-N rows:\n{}".format(bad.to_string(index=False)))

    skipped = feasibility[~feasibility["include_run"].map(bool)]
    if len(skipped) != 25:
        raise RuntimeError(f"Expected exactly 25 skipped feasibility rows, got {len(skipped)}")
    expected_skip = skipped[
        skipped["part"].eq("3UTR")
        & skipped["barcode_threshold_label"].eq("bc_ge3")
        & skipped["train_size_label"].eq("n3500")
    ]
    if len(expected_skip) != 25:
        raise RuntimeError("Skipped rows are not exactly 3UTR bc_ge3 n3500")

    missing_seed_flag = ~manifest["train_command"].str.contains("--train_subsample_seed", regex=False)
    if missing_seed_flag.any():
        raise RuntimeError("Some manifest train commands are missing --train_subsample_seed")


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
        "feasibility_csv": outdir / f"{tag}__threshold_feasibility.csv",
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
        "barcode_thresholds": BARCODE_THRESHOLDS,
        "train_size_n_arms": TRAIN_SIZE_N_ARMS,
        "run_manifest_rows": int(len(manifest)),
        "skipped_feasibility_rows": int((~feasibility["include_run"].map(bool)).sum()),
        "split_seeds": OUTER_SEED_SPLIT_SEEDS,
        "model_seed": OUTER_SEED_MODEL_SEED,
        "use_reverse_complements": False,
        "barcode_weighting": False,
        "graph_module": "CNNBasicTraining",
        "min_available_train_rows": int(feasibility["available_train_rows"].min()),
        "runs_per_part_threshold_size": (
            manifest.groupby(["part", "barcode_threshold_label", "train_size_label"])
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
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--config-summary", type=Path, default=DEFAULT_CONFIG_SUMMARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest-tag", default=MANIFEST_TAG)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = load_selected_source_manifest(args.source_manifest)
    part_tables = load_part_tables(source)
    feasibility = build_feasibility(source, part_tables)
    manifest = build_manifest(source, feasibility, args.manifest_tag)
    selected_summary = load_selected_config_summary(args.config_summary)
    validate(source, manifest, feasibility)

    print(f"Barcode-threshold downsample manifest rows: {len(manifest)}")
    print(f"Skipped feasibility rows: {(~feasibility['include_run'].map(bool)).sum()}")
    print(f"Minimum available train rows: {feasibility['available_train_rows'].min()}")
    print(manifest.groupby(["part", "barcode_threshold_label", "train_size_label"]).size().unstack(fill_value=0))
    if not args.no_write:
        paths = write_outputs(manifest, feasibility, selected_summary, args.outdir, args.manifest_tag)
        for label, path in paths.items():
            print(f"{label}: {path}")


if __name__ == "__main__":
    main()
