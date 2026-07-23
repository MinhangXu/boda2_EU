#!/usr/bin/env python3
"""Fail-closed validation for the targeted 3'UTR HPO dry-run manifest.

Validation is static: it hashes files and inspects manifest/command metadata.
It never imports or constructs a DataModule and never reads audit row IDs.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate_lib1_dedup_utr3_targeted_hpo_manifest as target
import generate_lib1_dedup_stage2_manifest as stage2


DEFAULT_PREFIX = HERE / "outputs/hpo_manifests" / target.MANIFEST_TAG


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def one(options: dict[str, list[str]], name: str) -> str:
    values = options.get(name)
    if values is None or len(values) != 1:
        raise ValueError(f"Expected exactly one value for --{name}; found {values!r}")
    return values[0]


def require_file_hash(path_value: str, expected_sha: str, cache: dict[str, str]) -> None:
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    resolved = str(path.resolve())
    cache.setdefault(resolved, target.sha256_file(path))
    if cache[resolved] != expected_sha:
        raise ValueError(
            f"File hash mismatch for {path}: expected {expected_sha}, observed {cache[resolved]}"
        )


def validate_command(row: dict) -> None:
    options = stage2.parse_command(row["train_command"])
    expected = {
        "campaign_id": target.CAMPAIGN_ID,
        "campaign_stage": target.CAMPAIGN_STAGE,
        "part_slug": "utr3",
        "analysis_lane": target.ANALYSIS_LANE,
        "challenger_family": target.CHALLENGER_FAMILY,
        "config_origin": target.CONFIG_ORIGIN,
        "training_regime": "scratch",
        "cell_id": row["cell_id"],
        "rc_pair_id": row["rc_pair_id"],
        "rc_mode": row["rc_mode"],
        "execution_disposition": "launch",
        "initialization": "scratch",
        "input_policy": "exact100_v1",
        "data_generation_id": row["data_generation_id"],
        "dataset_sha256": target.EXPECTED_DATA_SHA256,
        "split_manifest_id": row["split_manifest_id"],
        "split_manifest_sha256": target.EXPECTED_SPLIT_SHA256,
        "development_fold": str(row["development_fold"]),
        "split_fold": str(row["development_fold"]),
        "base_config_id": row["base_config_id"],
        "architecture": "UTR_BassetVL",
        "loss_mode": "unweighted_mse",
        "target_definition": "log2(RNA_bc_counts_sum / DNA_bc_counts_sum)",
        "artifact_retention": "none",
        "evaluate_test_after_fit": "false",
        "checkpoint_monitor": "val_pearson",
        "stopping_mode": "max",
        "logger_type": "wandb",
        "logger_project": target.WANDB_PROJECT,
        "wandb_entity": target.EXPECTED_ENTITY,
        "wandb_group": target.WANDB_GROUP,
        "wandb_job_type": target.WANDB_JOB_TYPE,
        "run_name": row["planned_run_name"],
        "exact_run_name": "true",
        "model_seed": str(target.MODEL_SEED),
        "datafile_path": row["dataset_path"],
        "split_manifest_path": row["split_manifest_path"],
        "expected_data_sha256": target.EXPECTED_DATA_SHA256,
        "expected_split_sha256": target.EXPECTED_SPLIT_SHA256,
        "target_column": "log2_RNA_DNA",
        "normalize": "true",
        "test_min_barcodes": "8",
        "train_min_barcodes": "1",
        "use_reverse_complements": "true" if row["rc_mode"] == "on" else "false",
        "barcode_weighting": "false",
        "model_module": "UTR_BassetVL",
        "graph_module": "CNNBasicTraining",
        "loss_criterion": "MSELoss",
        "reduction": "mean",
        "optimizer": "AdamW",
        "scheduler": "None",
        "batch_size": "64",
        "input_len": "100",
        "max_epochs": "220",
        "min_epochs": "25",
        "stopping_patience": "45",
        "precision": "32",
        "lr": str(row["search_values"]["lr"]),
        "weight_decay": str(row["search_values"]["weight_decay"]),
        "linear_dropout_p": str(row["search_values"]["linear_dropout_p"]),
        "default_root_dir": row["default_root_dir"],
        "enable_progress_bar": "false",
    }
    for name, expected_value in expected.items():
        observed = one(options, name)
        if observed != expected_value:
            raise ValueError(
                f"Cell {row['cell_id']} --{name} mismatch: expected "
                f"{expected_value!r}, observed {observed!r}"
            )
    if options.get("epoch_eval_splits") != ["train", "val"]:
        raise ValueError(f"Cell {row['cell_id']} epoch evaluation is not train/val only")
    if options.get("prediction_splits") != ["val"]:
        raise ValueError(f"Cell {row['cell_id']} prediction export is not val only")

    tokens = shlex.split(row["train_command"])
    if any(token.lower() == "audit" or "audit_ids" in token.lower() for token in tokens):
        raise ValueError(f"Cell {row['cell_id']} command contains audit material")
    forbidden_split_values = [
        value
        for name in ("epoch_eval_splits", "prediction_splits")
        for value in options.get(name, [])
        if value.lower() == "test"
    ]
    if forbidden_split_values:
        raise ValueError(f"Cell {row['cell_id']} exposes the test/audit split")
    forbidden_options = {
        "audit_ids",
        "audit_id_path",
        "predict_test",
        "test_prediction_output_dir",
    }
    if forbidden_options & options.keys():
        raise ValueError(f"Cell {row['cell_id']} contains forbidden audit/test options")


def validate(args: argparse.Namespace) -> dict:
    rows = read_jsonl(args.manifest)
    configs = read_jsonl(args.search_configs)
    summary = json.loads(args.summary.read_text())
    if len(rows) != target.EXPECTED_CELLS:
        raise ValueError(f"Expected {target.EXPECTED_CELLS} rows; found {len(rows)}")
    if len(configs) != target.EXPECTED_CONFIGS:
        raise ValueError(f"Expected {target.EXPECTED_CONFIGS} configs; found {len(configs)}")
    if [row["manifest_row"] for row in rows] != list(range(1, target.EXPECTED_CELLS + 1)):
        raise ValueError("manifest_row values are not contiguous 1..240")

    source = target.load_source_row(args.stage2_analysis_manifest)
    expected_configs = target.config_grid(source)
    if configs != expected_configs:
        raise ValueError("Search-config artifact differs from the frozen 4 x 3 x 2 grid")
    expected_by_id = {row["base_config_id"]: row for row in expected_configs}

    config_groups: dict[str, list[dict]] = defaultdict(list)
    pair_groups: dict[str, list[dict]] = defaultdict(list)
    file_hashes: dict[str, str] = {}
    for row in rows:
        if row["row_fingerprint"] != target.row_fingerprint(row):
            raise ValueError(f"Cell {row['cell_id']} row fingerprint mismatch")
        if row["manifest_status"] != target.MANIFEST_STATUS:
            raise ValueError(f"Cell {row['cell_id']} manifest status changed")
        if row["campaign_id"] != target.CAMPAIGN_ID or row["campaign_stage"] != target.CAMPAIGN_STAGE:
            raise ValueError(f"Cell {row['cell_id']} campaign identity changed")
        if row["part_slug"] != "utr3" or row["architecture"] != "UTR_BassetVL":
            raise ValueError(f"Cell {row['cell_id']} escaped the targeted 3'UTR route")
        if row["model_seed"] != target.MODEL_SEED or row["loss_mode"] != "unweighted_mse":
            raise ValueError(f"Cell {row['cell_id']} fixed seed/loss contract changed")
        if row["evaluate_test_after_fit"] is not False or row["artifact_retention"] != "none":
            raise ValueError(f"Cell {row['cell_id']} enables audit/artifact behavior")
        if row["epoch_eval_splits"] != ["train", "val"] or row["prediction_splits"] != ["val"]:
            raise ValueError(f"Cell {row['cell_id']} is not train/val-only")
        if row["base_config_id"] not in expected_by_id:
            raise ValueError(f"Cell {row['cell_id']} has an unapproved base config")
        expected_config = expected_by_id[row["base_config_id"]]
        if row["base_identity"] != expected_config["base_identity"]:
            raise ValueError(f"Cell {row['cell_id']} base identity changed")
        if row["search_values"] != {
            "lr": expected_config["lr"],
            "weight_decay": expected_config["weight_decay"],
            "linear_dropout_p": expected_config["linear_dropout_p"],
        }:
            raise ValueError(f"Cell {row['cell_id']} grid values changed")
        if row["wandb_entity"] != target.EXPECTED_ENTITY:
            raise ValueError(f"Cell {row['cell_id']} W&B entity changed")
        if row["logger_project"] != target.WANDB_PROJECT or row["wandb_group"] != target.WANDB_GROUP:
            raise ValueError(f"Cell {row['cell_id']} W&B organization changed")
        if row["wandb_job_type"] != target.WANDB_JOB_TYPE:
            raise ValueError(f"Cell {row['cell_id']} W&B job type changed")
        require_file_hash(row["dataset_path"], target.EXPECTED_DATA_SHA256, file_hashes)
        require_file_hash(row["split_manifest_path"], target.EXPECTED_SPLIT_SHA256, file_hashes)
        validate_command(row)
        config_groups[row["base_config_id"]].append(row)
        pair_groups[row["rc_pair_id"]].append(row)

    if len({row["cell_id"] for row in rows}) != target.EXPECTED_CELLS:
        raise ValueError("cell_id values are not unique")
    if len({row["planned_run_name"] for row in rows}) != target.EXPECTED_CELLS:
        raise ValueError("W&B run names are not unique")
    if len({row["row_fingerprint"] for row in rows}) != target.EXPECTED_CELLS:
        raise ValueError("row fingerprints are not unique")

    if len(config_groups) != target.EXPECTED_CONFIGS:
        raise ValueError("The manifest does not contain exactly 24 configs")
    for base_config_id, group in config_groups.items():
        observed_grid = Counter((row["development_fold"], row["rc_mode"]) for row in group)
        expected_grid = Counter((fold, rc) for fold in target.FOLDS for rc in ("off", "on"))
        if observed_grid != expected_grid:
            raise ValueError(f"Config {base_config_id} lacks the complete five-fold paired-RC grid")
    if len(pair_groups) != 120:
        raise ValueError(f"Expected 120 fold-level RC pairs; found {len(pair_groups)}")
    for pair_id, pair in pair_groups.items():
        if len(pair) != 2 or {row["rc_mode"] for row in pair} != {"off", "on"}:
            raise ValueError(f"RC pair {pair_id} is incomplete")
        invariant = (
            "base_config_id",
            "development_fold",
            "model_seed",
            "loss_mode",
            "dataset_sha256",
            "split_manifest_sha256",
        )
        if any(pair[0][field] != pair[1][field] for field in invariant):
            raise ValueError(f"RC pair {pair_id} changed an invariant field")

    expected_summary = {
        "manifest_status": target.MANIFEST_STATUS,
        "new_base_configs": 24,
        "new_training_cells": 240,
        "complete_new_oof_arms": 48,
        "fold_level_rc_pairs": 120,
        "screening_cells": 0,
        "promotion_cells": 0,
        "audit_loader_instantiated": False,
        "audit_ids_materialized": False,
        "commands_executed": 0,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"Summary {field} mismatch: {summary.get(field)!r}")
    if summary.get("dry_run_manifest_sha256") != target.sha256_file(args.manifest):
        raise ValueError("Summary dry-run manifest SHA mismatch")
    if summary.get("search_configs_sha256") != target.sha256_file(args.search_configs):
        raise ValueError("Summary search-config SHA mismatch")
    if summary.get("stage2_analysis_manifest_sha256") != target.sha256_file(args.stage2_analysis_manifest):
        raise ValueError("Summary Stage 2 source manifest SHA mismatch")

    return {
        "validation_status": "passed",
        "validated_at_protocol_date": "2026-07-14",
        "manifest_status": target.MANIFEST_STATUS,
        "new_base_configs": 24,
        "training_cells": 240,
        "complete_oof_arms": 48,
        "base_config_rc_pairs": 24,
        "fold_level_rc_pairs": 120,
        "search_dimensions": {
            "lr": list(target.LEARNING_RATES),
            "weight_decay": list(target.WEIGHT_DECAYS),
            "linear_dropout_p": list(target.DROPOUTS),
        },
        "dataset_sha256": target.EXPECTED_DATA_SHA256,
        "split_manifest_sha256": target.EXPECTED_SPLIT_SHA256,
        "dry_run_manifest_path": str(args.manifest.resolve()),
        "dry_run_manifest_sha256": target.sha256_file(args.manifest),
        "search_configs_sha256": target.sha256_file(args.search_configs),
        "stage2_analysis_manifest_sha256": target.sha256_file(args.stage2_analysis_manifest),
        "wandb_entity": target.EXPECTED_ENTITY,
        "wandb_project": target.WANDB_PROJECT,
        "wandb_group": target.WANDB_GROUP,
        "wandb_job_type": target.WANDB_JOB_TYPE,
        "audit_loader_instantiated": False,
        "audit_ids_materialized": False,
        "commands_executed": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path(str(DEFAULT_PREFIX) + "__dry_run_manifest.jsonl")
    )
    parser.add_argument(
        "--search-configs", type=Path, default=Path(str(DEFAULT_PREFIX) + "__search_configs.jsonl")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path(str(DEFAULT_PREFIX) + "__summary.json")
    )
    parser.add_argument(
        "--stage2-analysis-manifest",
        type=Path,
        default=HERE / "outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl",
    )
    parser.add_argument(
        "--report", type=Path, default=Path(str(DEFAULT_PREFIX) + "__validation_report.json")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
