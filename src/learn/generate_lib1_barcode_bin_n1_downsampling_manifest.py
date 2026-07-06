#!/usr/bin/env python3
"""Generate Lib1 exact n=1 barcode-bin downsampling manifest.

This follow-up extends the matched barcode-bin and full n=1 runs by training on
nested downsampled exact n=1 pools. The already completed matched-bin run
provides the N=1000 n=1 point, and the already completed full n=1 run provides
the full-pool endpoint. The goal is an n=1-only learning curve: estimate
marginal gain from more single-barcode examples and compare the fitted
trajectory against the matched N=1000 n>=6 performance target.

By default this manifest does not rerun the already completed full n=1 arms from
`lib1_barcode_bin_n1_full_june2026`; those are used as the full-pool endpoint in
analysis. The new exact-N arms are intentionally minimal and reuse prior work:
250 and 500, plus the existing N=1000 and full n=1 endpoints.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from generate_lib1_barcode_bin_matched_manifest import (
    DEFAULT_CONFIG_SUMMARY,
    DEFAULT_SOURCE_BASELINE,
    SELECTED_CONFIGS as ALL_SELECTED_CONFIGS,
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
MATCHED_N1000_TAG = "lib1_barcode_bin_matched_n1000_june2026"
FULL_N1_REFERENCE_TAG = "lib1_barcode_bin_n1_full_june2026"
MANIFEST_TAG = "lib1_barcode_bin_n1_downsample_june2026"

PARTS = ["Promoter", "3UTR", "5UTR"]
SELECTED_CONFIGS: Dict[str, List[str]] = {
    part: ALL_SELECTED_CONFIGS[part]
    for part in PARTS
}
TRAIN_SIZE_N_ARMS = [250, 500]
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


def n1_downsample_project_name(project: str) -> str:
    marker = "__outer_seed_prior_no_rc__"
    if marker in project:
        return project.replace(marker, marker + "barcode_bin_n1_downsample__", 1)
    return project + "__barcode_bin_n1_downsample"


def downsample_seed_for(split_seed: int) -> int:
    if int(split_seed) not in OUTER_SEED_SPLIT_SEEDS:
        raise ValueError(f"Unknown split_seed={split_seed}")
    # Match the datamodule default used by the matched N=1000 barcode-bin run,
    # so these exact-N arms are nested around that completed N=1000 point.
    return int(split_seed)


def load_selected_n1_baseline(source_baseline: Path) -> pd.DataFrame:
    baseline = load_selected_baseline(source_baseline)
    selected_rows = []
    for part_order, (part, config_ids) in enumerate(SELECTED_CONFIGS.items()):
        part_rows = baseline[
            baseline["part"].eq(part) & baseline["config_id"].isin(config_ids)
        ].copy()
        if part_rows.empty:
            raise RuntimeError(f"No selected baseline rows found for {part}")
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


def size_arms_for(available_train_rows: int, include_full: bool) -> List[Tuple[str, Optional[int], bool, bool, str]]:
    arms = []
    for train_size_n in TRAIN_SIZE_N_ARMS:
        included = int(available_train_rows) >= train_size_n
        skip_reason = "" if included else f"available_train_rows<{train_size_n}"
        arms.append((f"n{train_size_n}", train_size_n, False, included, skip_reason))
    if include_full:
        arms.append(("full", None, True, True, ""))
    return arms


def build_feasibility(baseline: pd.DataFrame, include_full: bool) -> pd.DataFrame:
    part_tables = load_part_tables(baseline)
    rows = []
    for _, source_row in baseline.iterrows():
        available = train_pool_size_for(source_row, BC1_BIN, part_tables)
        for size_label, requested_n, is_full, included, skip_reason in size_arms_for(available, include_full):
            rows.append(
                {
                    "part": source_row["part"],
                    "config_id": source_row["config_id"],
                    "split_seed": int(source_row["split_seed"]),
                    "barcode_bin": BC1_BIN["barcode_bin"],
                    "barcode_bin_label": BC1_BIN["barcode_bin_label"],
                    "train_size_label": size_label,
                    "train_size_n_requested": requested_n,
                    "is_full_train_pool": is_full,
                    "available_train_rows": int(available),
                    "include_run": included,
                    "skip_reason": skip_reason,
                }
            )
    return pd.DataFrame(rows)


def build_manifest(
    baseline: pd.DataFrame,
    feasibility: pd.DataFrame,
    tag: str,
    include_full: bool,
) -> pd.DataFrame:
    feasibility_index = {
        (
            row["part"],
            row["config_id"],
            int(row["split_seed"]),
            row["train_size_label"],
        ): row
        for _, row in feasibility.iterrows()
    }

    rows = []
    manifest_row = 1
    for _, source_row in baseline.iterrows():
        rec_base = source_row.to_dict()
        part_slug = rec_base["part_slug"]
        config_id = rec_base["config_id"]
        split_seed = int(rec_base["split_seed"])
        train_subsample_seed = downsample_seed_for(split_seed)
        first_size_label = f"n{TRAIN_SIZE_N_ARMS[0]}"
        available = int(
            feasibility_index[(rec_base["part"], config_id, split_seed, first_size_label)][
                "available_train_rows"
            ]
        )

        for size_label, requested_n, is_full, _, _ in size_arms_for(available, include_full):
            feasible = feasibility_index[(rec_base["part"], config_id, split_seed, size_label)]
            if not bool(feasible["include_run"]):
                continue

            rec = source_row.to_dict()
            size_token = "full" if is_full else f"n{int(requested_n)}"
            run_name = (
                f"{tag}__{part_slug}__{config_id}__bc1__{size_token}__"
                f"seed{split_seed}__ds{train_subsample_seed}"
            )
            logger_project = n1_downsample_project_name(str(rec["logger_project"]))
            rec.update(
                {
                    "source_manifest_tag": rec.get("manifest_tag"),
                    "source_manifest_row": rec.get("manifest_row"),
                    "source_planned_run_name": rec.get("planned_run_name"),
                    "manifest_tag": tag,
                    "manifest_row": manifest_row,
                    "barcode_bin": BC1_BIN["barcode_bin"],
                    "barcode_bin_label": BC1_BIN["barcode_bin_label"],
                    "train_size_label": size_label,
                    "train_size_n_requested": requested_n,
                    "available_train_rows": int(feasible["available_train_rows"]),
                    "is_full_train_pool": bool(is_full),
                    "full_n1_reference_tag": FULL_N1_REFERENCE_TAG,
                    "matched_n_ge6_reference_tag": MATCHED_N1000_TAG,
                    "graph_module": "CNNBasicTraining",
                    "barcode_weighting": False,
                    "train_min_barcodes": int(BC1_BIN["train_min_barcodes"]),
                    "train_max_barcodes": int(BC1_BIN["train_max_barcodes"]),
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
                        / "bc1"
                        / size_token
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
                        / "bc1"
                        / size_token
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


def load_selected_n1_config_summary(config_summary: Path) -> pd.DataFrame:
    selected = load_selected_config_summary(config_summary)
    if selected.empty:
        return selected
    return selected[selected["part"].isin(PARTS)].reset_index(drop=True)


def validate(manifest: pd.DataFrame, feasibility: pd.DataFrame, include_full: bool) -> None:
    included = feasibility[feasibility["include_run"].map(bool)]
    expected = int(len(included))
    if len(manifest) != expected:
        raise RuntimeError(f"Expected {expected} manifest rows, got {len(manifest)}")
    if set(manifest["part"]) != set(PARTS):
        raise RuntimeError(f"Manifest parts mismatch: {sorted(manifest['part'].unique())}")
    if set(manifest["barcode_bin"]) != {"bc1"}:
        raise RuntimeError("n=1 downsampling manifest must contain only barcode_bin=bc1")
    if set(manifest["train_min_barcodes"].astype(int)) != {1}:
        raise RuntimeError("n=1 downsampling train_min_barcodes mismatch")
    if set(manifest["train_max_barcodes"].astype(int)) != {1}:
        raise RuntimeError("n=1 downsampling train_max_barcodes mismatch")
    if set(manifest["train_sampling_mode"]) != {"random"}:
        raise RuntimeError("n=1 downsampling manifest must use random sampling mode")
    if set(manifest["graph_module"]) != {"CNNBasicTraining"}:
        raise RuntimeError("n=1 downsampling manifest must use CNNBasicTraining")
    if set(manifest["barcode_weighting"].map(bool)) != {False}:
        raise RuntimeError("n=1 downsampling manifest must be unweighted")
    if set(manifest["model_seed"].astype(int)) != {OUTER_SEED_MODEL_SEED}:
        raise RuntimeError("n=1 downsampling model_seed mismatch")
    if set(manifest["use_reverse_complements"].map(bool)) != {False}:
        raise RuntimeError("n=1 downsampling must not use reverse complements")
    if not include_full and manifest["train_size_label"].astype(str).eq("full").any():
        raise RuntimeError("Full arms present despite include_full=False")
    exact = manifest[manifest["train_size_label"].astype(str).ne("full")]
    if exact["train_size_n"].isna().any():
        raise RuntimeError("Exact-N downsampling rows must pass --train_size_n")
    if not bool(exact["train_command"].str.contains("--train_size_n", regex=False).all()):
        raise RuntimeError("Exact-N train commands should pass --train_size_n")
    if not bool(manifest["train_command"].str.contains("--train_min_barcodes 1", regex=False).all()):
        raise RuntimeError("Train commands should pass --train_min_barcodes 1")
    if not bool(manifest["train_command"].str.contains("--train_max_barcodes 1", regex=False).all()):
        raise RuntimeError("Train commands should pass --train_max_barcodes 1")
    if (included["available_train_rows"].astype(int) < included["train_size_n_requested"].fillna(0).astype(int)).any():
        raise RuntimeError("Included row requests more rows than available")


def write_outputs(
    manifest: pd.DataFrame,
    feasibility: pd.DataFrame,
    selected_summary: pd.DataFrame,
    outdir: Path,
    tag: str,
    include_full: bool,
) -> Dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest_csv": outdir / f"{tag}__run_manifest.csv",
        "manifest_json": outdir / f"{tag}__run_manifest.json",
        "manifest_jsonl": outdir / f"{tag}__run_manifest.jsonl",
        "feasibility_csv": outdir / f"{tag}__bin_downsampling_feasibility.csv",
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
        "matched_n1000_reference_tag": MATCHED_N1000_TAG,
        "full_n1_reference_tag": FULL_N1_REFERENCE_TAG,
        "selected_configs": SELECTED_CONFIGS,
        "parts": PARTS,
        "barcode_bin": BC1_BIN,
        "train_size_n_arms": TRAIN_SIZE_N_ARMS,
        "include_full": bool(include_full),
        "run_manifest_rows": int(len(manifest)),
        "split_seeds": OUTER_SEED_SPLIT_SEEDS,
        "model_seed": OUTER_SEED_MODEL_SEED,
        "use_reverse_complements": False,
        "barcode_weighting": False,
        "graph_module": "CNNBasicTraining",
        "min_available_train_rows": int(feasibility["available_train_rows"].min()),
        "max_available_train_rows": int(feasibility["available_train_rows"].max()),
        "runs_by_part_size": (
            manifest.groupby(["part", "train_size_label"], observed=True)
            .size()
            .reset_index(name="runs")
            .to_dict(orient="records")
        ),
        "skipped_by_part_size": (
            feasibility[~feasibility["include_run"].map(bool)]
            .groupby(["part", "train_size_label", "skip_reason"], observed=True)
            .size()
            .reset_index(name="skipped_rows")
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
    parser.add_argument(
        "--include-full",
        action="store_true",
        help="Also emit full n=1 rows. By default, use the completed full-n1 tag as the full endpoint instead.",
    )
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = load_selected_n1_baseline(args.source_baseline)
    feasibility = build_feasibility(baseline, include_full=args.include_full)
    manifest = build_manifest(baseline, feasibility, args.manifest_tag, include_full=args.include_full)
    selected_summary = load_selected_n1_config_summary(args.config_summary)
    validate(manifest, feasibility, include_full=args.include_full)

    print(f"Exact n=1 downsampling manifest rows: {len(manifest)}")
    print("Available n=1 train rows by part:")
    print(
        feasibility.groupby("part", observed=True)["available_train_rows"]
        .agg(["min", "median", "max"])
        .to_string()
    )
    print("Runnable rows by part x train size:")
    print(manifest.groupby(["part", "train_size_label"], observed=True).size().unstack(fill_value=0))
    skipped = feasibility[~feasibility["include_run"].map(bool)]
    if not skipped.empty:
        print("Skipped infeasible arms:")
        print(
            skipped.groupby(["part", "train_size_label", "skip_reason"], observed=True)
            .size()
            .reset_index(name="rows")
            .to_string(index=False)
        )
    if not args.no_write:
        paths = write_outputs(
            manifest,
            feasibility,
            selected_summary,
            args.outdir,
            args.manifest_tag,
            include_full=args.include_full,
        )
        for label, path in paths.items():
            print(f"{label}: {path}")


if __name__ == "__main__":
    main()
