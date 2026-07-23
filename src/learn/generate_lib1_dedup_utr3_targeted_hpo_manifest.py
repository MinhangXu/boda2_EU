#!/usr/bin/env python3
"""Generate the frozen 24-config targeted 3'UTR UTRBassetVL dry-run manifest.

This program reads one already-validated Stage 2 development row as a command
template. It does not import a DataModule, call W&B, execute training, or
materialize audit IDs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import generate_lib1_dedup_stage2_manifest as stage2


CAMPAIGN_ID = "lib1_dedup_phase1_rerun_july2026"
CAMPAIGN_STAGE = "targeted_utr3_hpo"
MANIFEST_TAG = "lib1_dedup_utr3_targeted_hpo_july2026"
MANIFEST_STATUS = "dry_run_validated_pending_user_launch"
ANALYSIS_LANE = "utr3_utrbasset_targeted_hpo"
CHALLENGER_FAMILY = "utr3_utrbasset_optimizer_regularization"
CONFIG_ORIGIN = "targeted_grid_20260714"
EXPECTED_ENTITY = "minhangxu1998-baylor-college-of-medicine"
WANDB_PROJECT = "utr3__bashor_in_house__dedup_exact_v1__targeted_hpo_development"
WANDB_GROUP = (
    "lib1_dedup_phase1_rerun_july2026__targeted_utr3_hpo__full_oof_rc"
)
WANDB_JOB_TYPE = "targeted_utr3_hpo_cell"
MODEL_SEED = 1701
FOLDS = tuple(range(5))
RC_MODES = (False, True)
LEARNING_RATES = (0.001, 0.002, 0.004, 0.006)
WEIGHT_DECAYS = (0.0001, 0.0007, 0.003)
DROPOUTS = (0.35, 0.50)
EXPECTED_CONFIGS = 24
EXPECTED_CELLS = 240
SOURCE_RUN_ID = "utc3cqzn"
SOURCE_BASE_CONFIG_ID = (
    "basecfg_86969bcf79247695d2c27ce1466d4eab2373e5e1f3645da99f24ebf4c59c0fbe"
)
EXPECTED_DATA_SHA256 = (
    "1bd0f70655cbbc3f47be40b2bb50cc641430a6741145ae85cb27506d512f7cc0"
)
EXPECTED_SPLIT_SHA256 = (
    "c7a5799b8e0c5b92a0041822a6bc5d0d9513a39d97f28061a5d46183ef998e1a"
)


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_fingerprint(row: dict) -> str:
    fields = (
        "manifest_status",
        "analysis_lane",
        "base_config_id",
        "search_config_index",
        "development_fold",
        "rc_mode",
        "dataset_sha256",
        "split_manifest_sha256",
        "planned_run_name",
        "train_command",
    )
    return sha256_json({field: row.get(field) for field in fields})


def pair_and_cell_ids(base_config_id: str, fold: int, rc_on: bool) -> tuple[str, str]:
    pair_payload = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "base_config_id": base_config_id,
        "development_fold": int(fold),
        "model_seed": MODEL_SEED,
        "loss_mode": "unweighted_mse",
    }
    pair_id = "rcpair_" + sha256_json(pair_payload)[:20]
    cell_payload = dict(pair_payload)
    cell_payload["rc_mode"] = "on" if rc_on else "off"
    cell_id = "cell_" + sha256_json(cell_payload)[:20]
    return pair_id, cell_id


def load_source_row(path: Path) -> dict:
    matches = [
        row
        for row in read_jsonl(path)
        if row.get("base_config_id") == SOURCE_BASE_CONFIG_ID
        and row.get("source_run_ids") == [SOURCE_RUN_ID]
        and row.get("development_fold") == 0
        and row.get("rc_mode") == "off"
        and row.get("analysis_lane") == "utr3_utrbasset_challenger"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen Stage 2 architecture anchor; found {len(matches)}")
    row = matches[0]
    if row["dataset_sha256"] != EXPECTED_DATA_SHA256:
        raise ValueError("Stage 2 architecture anchor dataset hash changed")
    if row["split_manifest_sha256"] != EXPECTED_SPLIT_SHA256:
        raise ValueError("Stage 2 architecture anchor split hash changed")
    if row["architecture"] != "UTR_BassetVL" or row["model_seed"] != MODEL_SEED:
        raise ValueError("Stage 2 architecture anchor model contract changed")
    return row


def config_grid(source: dict) -> list[dict]:
    configs = []
    for index, (lr, weight_decay, dropout) in enumerate(
        itertools.product(LEARNING_RATES, WEIGHT_DECAYS, DROPOUTS), 1
    ):
        identity = copy.deepcopy(source["base_identity"])
        identity["lr"] = lr
        identity["weight_decay"] = weight_decay
        identity["linear_dropout_p"] = dropout
        digest = sha256_json(identity)
        configs.append(
            {
                "search_config_index": index,
                "search_config_label": f"cfg{index:02d}",
                "base_config_id": "basecfg_" + digest,
                "base_config_sha256": digest,
                "lr": lr,
                "weight_decay": weight_decay,
                "linear_dropout_p": dropout,
                "base_identity": identity,
                "architecture_anchor_base_config_id": SOURCE_BASE_CONFIG_ID,
                "architecture_anchor_source_run_id": SOURCE_RUN_ID,
            }
        )
    if len(configs) != EXPECTED_CONFIGS:
        raise AssertionError(f"Expected {EXPECTED_CONFIGS} configs; found {len(configs)}")
    if len({row["base_config_id"] for row in configs}) != EXPECTED_CONFIGS:
        raise AssertionError("Targeted grid produced duplicate base identities")
    return configs


def build_cell(source: dict, config: dict, fold: int, rc_on: bool, output_root: Path) -> dict:
    rc_mode = "on" if rc_on else "off"
    base_config_id = config["base_config_id"]
    pair_id, cell_id = pair_and_cell_ids(base_config_id, fold, rc_on)
    root = output_root / base_config_id / f"fold_{fold}" / f"rc_{rc_mode}"
    run_name = (
        f"{MANIFEST_TAG}__{config['search_config_label']}__"
        f"{config['base_config_sha256'][:16]}__fold{fold}__rc_{rc_mode}"
    )

    options = stage2.parse_command(source["train_command"])
    replacements = {
        "artifact_path": str(root / "artifacts"),
        "best_checkpoint_dir": str(root / "published_checkpoint_disabled"),
        "prediction_output_dir": str(root / "predictions"),
        "provenance_output_dir": str(root / "provenance"),
        "artifact_retention": "none",
        "evaluate_test_after_fit": False,
        "logger_project": WANDB_PROJECT,
        "wandb_entity": EXPECTED_ENTITY,
        "wandb_group": WANDB_GROUP,
        "wandb_job_type": WANDB_JOB_TYPE,
        "run_name": run_name,
        "exact_run_name": True,
        "campaign_stage": CAMPAIGN_STAGE,
        "development_fold": int(fold),
        "split_fold": int(fold),
        "base_config_id": base_config_id,
        "use_reverse_complements": bool(rc_on),
        "default_root_dir": str(root),
        "analysis_lane": ANALYSIS_LANE,
        "challenger_family": CHALLENGER_FAMILY,
        "policy_id": base_config_id,
        "config_origin": CONFIG_ORIGIN,
        "training_regime": "scratch",
        "cell_id": cell_id,
        "rc_pair_id": pair_id,
        "rc_mode": rc_mode,
        "execution_disposition": "launch",
        "initialization": "scratch",
        "input_policy": "exact100_v1",
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "linear_dropout_p": config["linear_dropout_p"],
        "checkpoint_monitor": "val_pearson",
        "stopping_mode": "max",
    }
    for name, value in replacements.items():
        stage2.put(options, name, value)
    stage2.put(options, "epoch_eval_splits", ["train", "val"])
    stage2.put(options, "prediction_splits", ["val"])
    stage2.put(
        options,
        "wandb_tags",
        [
            CAMPAIGN_ID,
            CAMPAIGN_STAGE,
            "utr3",
            ANALYSIS_LANE,
            config["search_config_label"],
            f"fold{fold}",
            f"rc_{rc_mode}",
            "seed1701",
            "unweighted_mse",
            "dry_run_manifest_20260714",
        ],
    )
    command = stage2.command_from_options(options)

    row = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "manifest_tag": MANIFEST_TAG,
        "manifest_status": MANIFEST_STATUS,
        "analysis_lane": ANALYSIS_LANE,
        "challenger_family": CHALLENGER_FAMILY,
        "config_origin": CONFIG_ORIGIN,
        "selection_design": "full_five_fold_paired_rc_no_screening",
        "search_config_index": config["search_config_index"],
        "search_config_label": config["search_config_label"],
        "part_slug": "utr3",
        "architecture": "UTR_BassetVL",
        "architecture_slug": "utr_bassetvl",
        "base_config_id": base_config_id,
        "base_config_sha256": config["base_config_sha256"],
        "base_identity": config["base_identity"],
        "search_values": {
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
            "linear_dropout_p": config["linear_dropout_p"],
        },
        "architecture_anchor_base_config_id": SOURCE_BASE_CONFIG_ID,
        "architecture_anchor_source_run_id": SOURCE_RUN_ID,
        "policy_id": base_config_id,
        "initialization": "scratch",
        "training_regime": "scratch",
        "source_run_ids": [SOURCE_RUN_ID],
        "data_generation_id": source["data_generation_id"],
        "dataset_path": source["dataset_path"],
        "dataset_sha256": source["dataset_sha256"],
        "split_manifest_id": source["split_manifest_id"],
        "split_manifest_path": source["split_manifest_path"],
        "split_manifest_sha256": source["split_manifest_sha256"],
        "development_fold": int(fold),
        "model_seed": MODEL_SEED,
        "use_reverse_complements": bool(rc_on),
        "rc_mode": rc_mode,
        "rc_pair_id": pair_id,
        "cell_id": cell_id,
        "loss_mode": "unweighted_mse",
        "target_column": "log2_RNA_DNA",
        "target_definition": "log2(RNA_bc_counts_sum / DNA_bc_counts_sum)",
        "length_policy": "modal_exact_100",
        "input_policy": "exact100_v1",
        "artifact_retention": "none",
        "evaluate_test_after_fit": False,
        "epoch_eval_splits": ["train", "val"],
        "prediction_splits": ["val"],
        "wandb_entity": EXPECTED_ENTITY,
        "logger_project": WANDB_PROJECT,
        "wandb_group": WANDB_GROUP,
        "wandb_job_type": WANDB_JOB_TYPE,
        "planned_run_name": run_name,
        "default_root_dir": str(root),
        "execution_disposition": "launch",
        "train_command": command,
    }
    row["row_fingerprint"] = row_fingerprint(row)
    return row


def validate_generated(rows: list[dict], configs: list[dict]) -> None:
    if len(rows) != EXPECTED_CELLS:
        raise AssertionError(f"Expected {EXPECTED_CELLS} cells; found {len(rows)}")
    if len(configs) != EXPECTED_CONFIGS:
        raise AssertionError(f"Expected {EXPECTED_CONFIGS} configs; found {len(configs)}")
    config_counts = Counter(row["base_config_id"] for row in rows)
    if len(config_counts) != EXPECTED_CONFIGS or set(config_counts.values()) != {10}:
        raise AssertionError("Each targeted config must have five folds x two RC modes")
    pair_counts = Counter(row["rc_pair_id"] for row in rows)
    if len(pair_counts) != 120 or set(pair_counts.values()) != {2}:
        raise AssertionError("Targeted manifest must contain 120 complete RC pairs")
    if len({row["cell_id"] for row in rows}) != EXPECTED_CELLS:
        raise AssertionError("cell_id values are not unique")
    if len({row["planned_run_name"] for row in rows}) != EXPECTED_CELLS:
        raise AssertionError("W&B run names are not unique")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2-analysis-manifest",
        type=Path,
        default=HERE / "outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl",
    )
    parser.add_argument("--manifest-tag", default=MANIFEST_TAG)
    parser.add_argument("--outdir", type=Path, default=HERE / "outputs/hpo_manifests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest_tag != MANIFEST_TAG:
        raise ValueError(
            f"Frozen protocol requires manifest tag {MANIFEST_TAG!r}; got {args.manifest_tag!r}"
        )
    source = load_source_row(args.stage2_analysis_manifest)
    configs = config_grid(source)
    output_root = HERE / "outputs/hpo_runs" / MANIFEST_TAG
    rows = [
        build_cell(source, config, fold, rc_on, output_root)
        for config in configs
        for fold in FOLDS
        for rc_on in RC_MODES
    ]
    rows.sort(
        key=lambda row: (
            row["search_config_index"],
            row["development_fold"],
            row["use_reverse_complements"],
        )
    )
    for manifest_row, row in enumerate(rows, 1):
        row["manifest_row"] = manifest_row
    validate_generated(rows, configs)

    prefix = args.outdir / MANIFEST_TAG
    manifest_path = Path(str(prefix) + "__dry_run_manifest.jsonl")
    manifest_csv = Path(str(prefix) + "__dry_run_manifest.csv")
    configs_path = Path(str(prefix) + "__search_configs.jsonl")
    summary_path = Path(str(prefix) + "__summary.json")
    write_jsonl(manifest_path, rows)
    write_jsonl(configs_path, configs)
    write_csv(
        manifest_csv,
        rows,
        (
            "manifest_row",
            "cell_id",
            "rc_pair_id",
            "search_config_index",
            "search_config_label",
            "base_config_id",
            "development_fold",
            "rc_mode",
            "planned_run_name",
            "logger_project",
            "wandb_group",
            "row_fingerprint",
            "train_command",
        ),
    )

    summary = {
        "schema_version": "lib1_dedup_utr3_targeted_hpo_manifest_v1",
        "generated_for_protocol_date": "2026-07-14",
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "manifest_tag": MANIFEST_TAG,
        "manifest_status": MANIFEST_STATUS,
        "search_design": "fixed_cartesian_grid_full_five_fold_paired_rc",
        "search_dimensions": {
            "lr": list(LEARNING_RATES),
            "weight_decay": list(WEIGHT_DECAYS),
            "linear_dropout_p": list(DROPOUTS),
        },
        "new_base_configs": EXPECTED_CONFIGS,
        "development_folds": list(FOLDS),
        "rc_modes": ["off", "on"],
        "new_training_cells": EXPECTED_CELLS,
        "complete_new_oof_arms": 48,
        "complete_new_rc_pairs": 24,
        "fold_level_rc_pairs": 120,
        "screening_cells": 0,
        "promotion_cells": 0,
        "stage2_comparator_cells_reused_for_analysis_only": 200,
        "architecture_anchor_base_config_id": SOURCE_BASE_CONFIG_ID,
        "architecture_anchor_source_run_id": SOURCE_RUN_ID,
        "stage2_analysis_manifest_path": str(args.stage2_analysis_manifest.resolve()),
        "stage2_analysis_manifest_sha256": sha256_file(args.stage2_analysis_manifest),
        "dataset_sha256": EXPECTED_DATA_SHA256,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA256,
        "wandb": {
            "entity": EXPECTED_ENTITY,
            "project": WANDB_PROJECT,
            "group": WANDB_GROUP,
            "job_type": WANDB_JOB_TYPE,
            "adaptive_sweep": False,
        },
        "fixed_policy": {
            "model_seed": MODEL_SEED,
            "loss_mode": "unweighted_mse",
            "evaluate_test_after_fit": False,
            "epoch_eval_splits": ["train", "val"],
            "prediction_splits": ["val"],
            "artifact_retention": "none",
            "audit_loader": False,
        },
        "audit_loader_instantiated": False,
        "audit_ids_materialized": False,
        "commands_executed": 0,
        "dry_run_manifest_path": str(manifest_path.resolve()),
        "dry_run_manifest_sha256": sha256_file(manifest_path),
        "search_configs_path": str(configs_path.resolve()),
        "search_configs_sha256": sha256_file(configs_path),
    }
    write_json(summary_path, summary)

    print("Generated frozen targeted 3'UTR HPO dry-run manifest")
    print(f"  base configs: {EXPECTED_CONFIGS}")
    print(f"  training cells: {EXPECTED_CELLS}")
    print("  design: all five folds x paired RC; no screening/promotion")
    print(f"  manifest: {manifest_path}")
    print(f"  manifest SHA256: {summary['dry_run_manifest_sha256']}")
    print("  commands executed: 0")
    print("  audit loader instantiated: false")


if __name__ == "__main__":
    main()
