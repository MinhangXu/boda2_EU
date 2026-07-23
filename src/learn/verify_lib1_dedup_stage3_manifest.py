#!/usr/bin/env python3
"""Fail-closed static verification of the Lib1 dedup Stage 3 dry run.

This verifier hashes immutable files and inspects development-only manifest,
command, prediction, and provenance metadata.  It deliberately does not
import/construct a DataModule and does not enumerate audit IDs or audit strata.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate_lib1_dedup_stage2_manifest as stage2
import generate_lib1_dedup_stage3_manifest as stage3


DEFAULT_PREFIX = HERE / "outputs/hpo_manifests" / stage3.MANIFEST_TAG


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def one(options: dict[str, list[str]], name: str) -> str:
    values = options.get(name)
    if values is None or len(values) != 1:
        raise ValueError(f"Expected one --{name} value; observed {values!r}")
    return values[0]


def assert_file_hash(path_value: str, expected: str, cache: dict[str, str]) -> None:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved = str(path.resolve())
    if resolved not in cache:
        cache[resolved] = stage2.sha256_file(path)
    if cache[resolved] != expected:
        raise ValueError(
            f"File hash mismatch for {resolved}: expected {expected}, observed {cache[resolved]}"
        )


def source_lookup(stage2_path: Path, targeted_path: Path) -> dict[tuple, dict]:
    lookup = {}
    for label, path in (("stage2", stage2_path), ("targeted_utr3", targeted_path)):
        manifest_sha = stage2.sha256_file(path)
        for row in read_jsonl(path):
            key = (
                row["part_slug"], row["base_config_id"], int(row["development_fold"]), row["rc_mode"]
            )
            if key in lookup:
                raise ValueError(f"Duplicate immutable source condition {key}")
            wrapped = dict(row)
            wrapped["_label"] = label
            wrapped["_path"] = str(path.resolve())
            wrapped["_sha"] = manifest_sha
            lookup[key] = wrapped
    return lookup


def validate_launch_command(row: dict) -> None:
    options = stage2.parse_command(row["train_command"])
    expected_graph = (
        "CNNBassetBranchedScopedWeightedTransfer"
        if row["training_regime"] == "transfer"
        else "CNNWeightedRegressionTraining"
    )
    expected = {
        "campaign_id": stage3.CAMPAIGN_ID,
        "campaign_stage": stage3.CAMPAIGN_STAGE,
        "part_slug": row["part_slug"],
        "analysis_lane": row["analysis_lane"],
        "training_regime": row["training_regime"],
        "cell_id": row["cell_id"],
        "loss_pair_id": row["loss_pair_id"],
        "source_unweighted_cell_id": row["source_unweighted_cell_id"],
        "rc_mode": row["rc_mode"],
        "execution_disposition": "launch",
        "data_generation_id": row["data_generation_id"],
        "dataset_sha256": row["dataset_sha256"],
        "split_manifest_id": row["split_manifest_id"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "development_fold": str(row["development_fold"]),
        "split_fold": str(row["development_fold"]),
        "base_config_id": row["base_config_id"],
        "architecture": row["architecture"],
        "model_seed": str(stage3.MODEL_SEED),
        "loss_mode": "barcode_weighted_mse",
        "artifact_retention": "none",
        "evaluate_test_after_fit": "false",
        "use_reverse_complements": "true" if row["rc_mode"] == "on" else "false",
        "barcode_weighting": "true",
        "barcode_weight_cap": "8.0",
        "barcode_weight_min": "0.1",
        "graph_module": expected_graph,
        "loss_criterion": "MSELoss",
        "reduction": "mean",
        "logger_type": "wandb",
        "logger_project": stage3.wandb_project(row["part_slug"]),
        "wandb_entity": stage3.EXPECTED_ENTITY,
        "wandb_group": row["wandb_group"],
        "wandb_job_type": "stage3_weighted_cell",
        "run_name": row["planned_run_name"],
        "exact_run_name": "true",
        "default_root_dir": row["default_root_dir"],
        "enable_progress_bar": "false",
    }
    if row["rc_pair_id"]:
        expected["rc_pair_id"] = row["rc_pair_id"]
    elif "rc_pair_id" in options:
        raise ValueError("3'UTR RC-off-only command unexpectedly carries an RC pair ID")
    for name, value in expected.items():
        observed = one(options, name)
        if observed != value:
            raise ValueError(
                f"Cell {row['cell_id']} --{name}: expected {value!r}, observed {observed!r}"
            )
    if row["training_regime"] == "scratch":
        if one(options, "weighted_loss_reduction") != "mean":
            raise ValueError(f"Cell {row['cell_id']} changed weighted reduction")
    elif "weighted_loss_reduction" in options:
        raise ValueError(f"Transfer cell {row['cell_id']} has a scratch-only graph option")
    if options.get("epoch_eval_splits") != ["train", "val"]:
        raise ValueError(f"Cell {row['cell_id']} does not evaluate train/val only")
    if options.get("prediction_splits") != ["val"]:
        raise ValueError(f"Cell {row['cell_id']} does not export val only")
    if row["part_slug"] == "utr3" and one(options, "use_reverse_complements") != "false":
        raise ValueError("3'UTR Stage 3 command enables RC")

    tokens = shlex.split(row["train_command"])
    forbidden_options = {
        "audit_ids", "audit_id_path", "predict_test", "test_prediction_output_dir"
    }
    if forbidden_options & set(options):
        raise ValueError(f"Cell {row['cell_id']} contains forbidden audit/test options")
    if any("audit_ids" in token.lower() or "test_loader" in token.lower() for token in tokens):
        raise ValueError(f"Cell {row['cell_id']} contains audit-loader material")
    if any(
        value.lower() == "test"
        for name in ("epoch_eval_splits", "prediction_splits")
        for value in options.get(name, [])
    ):
        raise ValueError(f"Cell {row['cell_id']} exposes test/audit evaluation")


def validate(args: argparse.Namespace) -> dict:
    analysis_rows = read_jsonl(args.analysis_manifest)
    launch_rows = read_jsonl(args.manifest)
    reuse_rows = read_jsonl(args.reuse_manifest)
    portfolio = json.loads(args.portfolio.read_text())
    summary = json.loads(args.summary.read_text())
    sources = source_lookup(args.stage2_analysis_manifest, args.targeted_utr3_manifest)

    if len(analysis_rows) != stage3.EXPECTED_ANALYSIS_CELLS:
        raise ValueError(f"Expected 900 analysis cells; found {len(analysis_rows)}")
    if len(launch_rows) != stage3.EXPECTED_WEIGHTED_CELLS:
        raise ValueError(f"Expected 450 weighted cells; found {len(launch_rows)}")
    if len(reuse_rows) != stage3.EXPECTED_REUSE_CELLS:
        raise ValueError(f"Expected 450 reuse cells; found {len(reuse_rows)}")
    if [row["manifest_row"] for row in launch_rows] != list(range(1, 451)):
        raise ValueError("Weighted manifest rows are not contiguous 1..450")
    if [row["analysis_cell"] for row in analysis_rows] != list(range(1, 901)):
        raise ValueError("Analysis cells are not contiguous 1..900")
    if {row["cell_id"] for row in reuse_rows} != {
        row["cell_id"] for row in analysis_rows if row["execution_disposition"] == "reuse_unweighted"
    }:
        raise ValueError("Reuse artifact is not the analysis manifest's exact reuse subset")
    if {row["cell_id"] for row in launch_rows} != {
        row["cell_id"] for row in analysis_rows if row["execution_disposition"] == "launch"
    }:
        raise ValueError("Dry run is not the analysis manifest's exact launch subset")

    expected_configs = [
        (part, rank, base_config_id)
        for part in sorted(stage3.PORTFOLIOS, key=stage3.PART_ORDER.get)
        for rank, base_config_id in enumerate(stage3.PORTFOLIOS[part], 1)
    ]
    observed_configs = [
        (row["part_slug"], row["portfolio_rank"], row["base_config_id"])
        for row in portfolio["configs"]
    ]
    if observed_configs != expected_configs:
        raise ValueError("Portfolio artifact changed exact config IDs or ordering")
    if not all(row.get("eligible_for_final_selection") is True for row in portfolio["configs"]):
        raise ValueError("Enhancer anchors or another portfolio member silently became ineligible")

    expected_margins = stage3.metric_margins(args.stage2_metrics, args.targeted_metrics)
    expected_index, _, expected_source_hashes = stage3.source_index(
        args.stage2_analysis_manifest, args.targeted_utr3_manifest
    )
    expected_portfolio = stage3.portfolio_artifact(
        expected_index, expected_margins, expected_source_hashes, args
    )
    if portfolio != expected_portfolio:
        raise ValueError("Portfolio artifact differs from the frozen generated contract")
    if portfolio.get("metric_margins") != expected_margins:
        raise ValueError("Portfolio metric margins differ from the frozen derivation")
    if summary.get("metric_margins") != expected_margins:
        raise ValueError("Summary metric margins differ from the frozen derivation")

    file_hashes = {}
    provenance_cache = {}
    prediction_evidence_cache = {}
    loss_pairs = defaultdict(list)
    rc_pairs = defaultdict(list)
    arm_keys = set()
    config_counts = Counter()
    for row in analysis_rows:
        if row["row_fingerprint"] != stage3.row_fingerprint(row):
            raise ValueError(f"Cell {row['cell_id']} row fingerprint mismatch")
        if row["manifest_status"] != stage3.MANIFEST_STATUS:
            raise ValueError(f"Cell {row['cell_id']} manifest status changed")
        if row["campaign_id"] != stage3.CAMPAIGN_ID or row["campaign_stage"] != stage3.CAMPAIGN_STAGE:
            raise ValueError(f"Cell {row['cell_id']} campaign identity changed")
        if row["model_seed"] != stage3.MODEL_SEED:
            raise ValueError(f"Cell {row['cell_id']} model seed changed")
        if row["portfolio_rank"] < 1 or row["portfolio_rank"] > 10:
            raise ValueError(f"Cell {row['cell_id']} has invalid portfolio rank")
        if row["base_config_id"] != stage3.PORTFOLIOS[row["part_slug"]][row["portfolio_rank"] - 1]:
            raise ValueError(f"Cell {row['cell_id']} is outside the frozen portfolio")
        if row["portfolio_role"] != stage3.portfolio_role(row["part_slug"], row["portfolio_rank"]):
            raise ValueError(f"Cell {row['cell_id']} portfolio role changed")
        if row["part_slug"] == "utr3":
            if row["rc_mode"] != "off" or row["rc_pair_id"]:
                raise ValueError("3'UTR is not RC-off-only")
        elif row["rc_mode"] not in {"off", "on"} or not row["rc_pair_id"]:
            raise ValueError(f"Cell {row['cell_id']} lacks its RC-factorial identity")
        if row["evaluate_test_after_fit"] is not False:
            raise ValueError(f"Cell {row['cell_id']} enables audit/test evaluation")
        if row["epoch_eval_splits"] != ["train", "val"] or row["prediction_splits"] != ["val"]:
            raise ValueError(f"Cell {row['cell_id']} is not train/val-only")

        source_key = (
            row["part_slug"], row["base_config_id"], int(row["development_fold"]), row["rc_mode"]
        )
        source = sources.get(source_key)
        if source is None:
            raise ValueError(f"Cell {row['cell_id']} has no immutable unweighted source")
        source_invariants = {
            "source_manifest_label": source["_label"],
            "source_manifest_path": source["_path"],
            "source_manifest_sha256": source["_sha"],
            "source_unweighted_cell_id": source["cell_id"],
            "source_unweighted_row_fingerprint": source["row_fingerprint"],
            "dataset_sha256": source["dataset_sha256"],
            "split_manifest_sha256": source["split_manifest_sha256"],
            "architecture": source["architecture"],
            "training_regime": source["training_regime"],
            "base_identity": source["base_identity"],
        }
        for field, expected in source_invariants.items():
            if row[field] != expected:
                raise ValueError(f"Cell {row['cell_id']} source invariant {field} changed")
        assert_file_hash(row["source_manifest_path"], row["source_manifest_sha256"], file_hashes)
        assert_file_hash(row["dataset_path"], row["dataset_sha256"], file_hashes)
        assert_file_hash(row["split_manifest_path"], row["split_manifest_sha256"], file_hashes)
        assert_file_hash(row["source_prediction_path"], row["source_prediction_sha256"], file_hashes)
        assert_file_hash(row["source_provenance_path"], row["source_provenance_sha256"], file_hashes)
        provenance_path = row["source_provenance_path"]
        if provenance_path not in provenance_cache:
            provenance_cache[provenance_path] = json.loads(Path(provenance_path).read_text())
        provenance = provenance_cache[provenance_path]
        split_summary = provenance.get("data_split_summary", {})
        if split_summary.get("n_test") != 0:
            raise ValueError(f"Source cell {row['source_unweighted_cell_id']} exposed test/audit data")
        if split_summary.get("n_val") != row["source_prediction_rows"]:
            raise ValueError(f"Source cell {row['source_unweighted_cell_id']} n_val changed")
        if split_summary.get("val_row_id_hash") != row["source_val_row_id_hash"]:
            raise ValueError(f"Source cell {row['source_unweighted_cell_id']} val-row hash changed")
        prediction_path = row["source_prediction_path"]
        if prediction_path not in prediction_evidence_cache:
            with Path(prediction_path).open(newline="") as handle:
                prediction_records = list(csv.DictReader(handle, delimiter="\t"))
            prediction_header = list(prediction_records[0]) if prediction_records else []
            prediction_rows = len(prediction_records)
            prediction_ids = [str(item.get("construct_id", "")) for item in prediction_records]
            prediction_evidence_cache[prediction_path] = (
                prediction_header,
                prediction_rows,
                len(prediction_ids) == len(set(prediction_ids)),
                stage2.sha256_json(sorted(prediction_ids)),
            )
        (
            prediction_header,
            prediction_rows,
            prediction_ids_unique,
            prediction_id_hash,
        ) = prediction_evidence_cache[prediction_path]
        if not {"construct_id", "log2_RNA_DNA", "prediction_raw"}.issubset(
            prediction_header
        ):
            raise ValueError(f"Source cell {row['source_unweighted_cell_id']} prediction schema changed")
        if prediction_rows != row["source_prediction_rows"]:
            raise ValueError(f"Source cell {row['source_unweighted_cell_id']} prediction count changed")
        if not prediction_ids_unique or prediction_id_hash != row["source_val_row_id_hash"]:
            raise ValueError(f"Source cell {row['source_unweighted_cell_id']} prediction IDs changed")

        loss_pairs[row["loss_pair_id"]].append(row)
        if row["rc_pair_id"]:
            rc_pairs[row["rc_pair_id"]].append(row)
        arm_keys.add((row["part_slug"], row["base_config_id"], row["rc_mode"], row["loss_mode"]))
        config_counts[(row["part_slug"], row["base_config_id"])] += 1
        if row["execution_disposition"] == "launch":
            validate_launch_command(row)
        elif row["execution_disposition"] == "reuse_unweighted":
            if row["train_command"] or row["barcode_weighting"] is not False:
                raise ValueError(f"Reuse cell {row['cell_id']} is not immutable")
        else:
            raise ValueError(f"Unknown execution disposition for {row['cell_id']}")

    if len({row["cell_id"] for row in analysis_rows}) != 900:
        raise ValueError("Stage 3 cell IDs are not unique")
    if len({row["row_fingerprint"] for row in analysis_rows}) != 900:
        raise ValueError("Stage 3 row fingerprints are not unique")
    if len(arm_keys) != stage3.EXPECTED_OOF_ARMS:
        raise ValueError(f"Expected 180 OOF arms; found {len(arm_keys)}")
    expected_config_counts = Counter(
        {
            (part, base_config_id): (10 if part == "utr3" else 20)
            for part, config_ids in stage3.PORTFOLIOS.items()
            for base_config_id in config_ids
        }
    )
    if config_counts != expected_config_counts:
        raise ValueError("Config/fold/RC/loss accounting changed")
    if len(loss_pairs) != 450:
        raise ValueError(f"Expected 450 loss pairs; found {len(loss_pairs)}")
    for pair_id, pair in loss_pairs.items():
        if len(pair) != 2 or {row["loss_mode"] for row in pair} != {
            "unweighted_mse", "barcode_weighted_mse"
        }:
            raise ValueError(f"Loss pair {pair_id} is incomplete")
        invariants = (
            "part_slug", "base_config_id", "development_fold", "rc_mode", "model_seed",
            "dataset_sha256", "split_manifest_sha256", "source_unweighted_cell_id",
            "source_prediction_sha256", "source_val_row_id_hash",
        )
        if any(pair[0][field] != pair[1][field] for field in invariants):
            raise ValueError(f"Loss pair {pair_id} changed a paired invariant")
    if len(rc_pairs) != 400:
        raise ValueError(f"Expected 400 RC pairs; found {len(rc_pairs)}")
    for pair_id, pair in rc_pairs.items():
        if len(pair) != 2 or {row["rc_mode"] for row in pair} != {"off", "on"}:
            raise ValueError(f"RC pair {pair_id} is incomplete")

    expected_summary = {
        "manifest_status": stage3.MANIFEST_STATUS,
        "configs_per_part": 10,
        "total_configs": 50,
        "new_weighted_cells": 450,
        "unweighted_reuse_cells": 450,
        "analysis_cells": 900,
        "complete_oof_arms": 180,
        "fold_level_loss_pairs": 450,
        "fold_level_rc_pairs": 400,
        "audit_loader_instantiated": False,
        "audit_ids_materialized": False,
        "audit_stratum_counts_inspected": False,
        "commands_executed": 0,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"Summary {field}: expected {expected!r}, observed {summary.get(field)!r}")
    artifact_hashes = {
        "portfolio_sha256": (args.portfolio, summary.get("portfolio_sha256")),
        "analysis_manifest_sha256": (args.analysis_manifest, summary.get("analysis_manifest_sha256")),
        "dry_run_manifest_sha256": (args.manifest, summary.get("dry_run_manifest_sha256")),
        "unweighted_reuse_sha256": (args.reuse_manifest, summary.get("unweighted_reuse_sha256")),
    }
    for field, (path, expected) in artifact_hashes.items():
        observed = stage2.sha256_file(path)
        if observed != expected:
            raise ValueError(f"Summary {field} mismatch: {observed} != {expected}")

    return {
        "validation_status": "passed",
        "validated_at_protocol_date": "2026-07-14",
        "manifest_status": stage3.MANIFEST_STATUS,
        "configs": 50,
        "new_weighted_cells": 450,
        "unweighted_reuse_cells": 450,
        "analysis_cells": 900,
        "complete_oof_arms": 180,
        "fold_level_loss_pairs": 450,
        "fold_level_rc_pairs": 400,
        "counts_by_part": dict(Counter(row["part_slug"] for row in launch_rows)),
        "metric_margins": expected_margins,
        "portfolio_path": str(args.portfolio.resolve()),
        "portfolio_sha256": stage2.sha256_file(args.portfolio),
        "analysis_manifest_path": str(args.analysis_manifest.resolve()),
        "analysis_manifest_sha256": stage2.sha256_file(args.analysis_manifest),
        "dry_run_manifest_path": str(args.manifest.resolve()),
        "dry_run_manifest_sha256": stage2.sha256_file(args.manifest),
        "unweighted_reuse_path": str(args.reuse_manifest.resolve()),
        "unweighted_reuse_sha256": stage2.sha256_file(args.reuse_manifest),
        "wandb_entity": stage3.EXPECTED_ENTITY,
        "wandb_projects": {part: stage3.wandb_project(part) for part in stage3.PORTFOLIOS},
        "intron_labels": "inferred_mask_sensitivity_strata_not_verified_sublibraries",
        "audit_loader_instantiated": False,
        "audit_ids_materialized": False,
        "audit_stratum_counts_inspected": False,
        "commands_executed": 0,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(str(DEFAULT_PREFIX) + "__dry_run_manifest.jsonl"))
    parser.add_argument("--analysis-manifest", type=Path, default=Path(str(DEFAULT_PREFIX) + "__analysis_manifest.jsonl"))
    parser.add_argument("--reuse-manifest", type=Path, default=Path(str(DEFAULT_PREFIX) + "__unweighted_reuse.jsonl"))
    parser.add_argument("--portfolio", type=Path, default=Path(str(DEFAULT_PREFIX) + "__portfolio.json"))
    parser.add_argument("--summary", type=Path, default=Path(str(DEFAULT_PREFIX) + "__summary.json"))
    parser.add_argument(
        "--stage2-analysis-manifest", type=Path,
        default=HERE / "outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl",
    )
    parser.add_argument(
        "--targeted-utr3-manifest", type=Path,
        default=HERE / "outputs/hpo_manifests/lib1_dedup_utr3_targeted_hpo_july2026__dry_run_manifest.jsonl",
    )
    parser.add_argument(
        "--stage2-metrics", type=Path,
        default=HERE / "outputs/analysis/lib1_dedup_stage2_july2026/stage2_oof_metrics.csv",
    )
    parser.add_argument(
        "--targeted-metrics", type=Path,
        default=HERE / "outputs/analysis/lib1_dedup_utr3_targeted_hpo_july2026/utr3_targeted_hpo_combined_arm_metrics.csv",
    )
    parser.add_argument("--report", type=Path, default=Path(str(DEFAULT_PREFIX) + "__validation_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
