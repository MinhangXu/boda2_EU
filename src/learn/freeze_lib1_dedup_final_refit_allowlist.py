#!/usr/bin/env python3
"""Reconcile all final refits and freeze the 15-checkpoint audit allowlist.

No DataModule is imported or instantiated and no audit row is scored here.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from pathlib import Path

import torch

import boda
from src.learn.run_lib1_dedup_final_refit_campaign import (
    EXPECTED_MANIFEST_SHA256,
    MANIFEST,
    completed,
    read_registry,
    read_rows,
    sha256_file,
    validate_inputs,
    validate_record,
)


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_DIR = HERE / "outputs/audit/lib1_dedup_final_audit_july2026"
ALLOWLIST_JSON = OUT_DIR / "lib1_dedup_final_refit_checkpoint_allowlist.json"
ALLOWLIST_CSV = OUT_DIR / "lib1_dedup_final_refit_checkpoint_allowlist.csv"
SUMMARY = OUT_DIR / "lib1_dedup_final_refit_checkpoint_allowlist_summary.json"
RECONCILIATION = (
    REPO / "plan/phase1_lib1/dedup_phase1_rerun_july2026"
    / "lib1_dedup_final_refit_implementation_reconciliation_july16_2026.md"
)
EXPECTED_RECONCILIATION_SHA256 = (
    "07dc683d292a75cdef228af6065d6f14264f6588a4235f1b9c7f51ba72ee8620"
)
STAGE3_COMPLETION = (
    HERE / "outputs/analysis/lib1_dedup_stage3_weighted_loss_july2026"
    / "stage3_cell_completion.csv"
)
STATUS_DONE = (
    HERE / "outputs/hpo_runs/status/lib1_dedup_final_refit_july2026/done"
)
IMPLEMENTATION_SOURCES = (
    HERE / "train_wandb_log.py",
    HERE / "run_lib1_dedup_final_refit_campaign.py",
    HERE / "generate_lib1_dedup_final_refit_manifest.py",
    REPO / "boda/data/bashor_datamodule.py",
    REPO / "boda/model/basset.py",
    REPO / "boda/model/resnet.py",
    REPO / "boda/graph/cnn_prediction.py",
    REPO / "boda/graph/cnn_weighted_regression.py",
)


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_checkpoint_member(artifact: Path) -> tuple[dict, str]:
    with tarfile.open(str(artifact), "r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
        required = {"artifacts/torch_checkpoint.pt", "artifacts/provenance.json"}
        if not required.issubset(members):
            raise ValueError(f"Artifact {artifact} is missing {sorted(required - members)}")
        handle = archive.extractfile("artifacts/torch_checkpoint.pt")
        if handle is None:
            raise ValueError(f"Cannot read checkpoint member in {artifact}")
        checkpoint_bytes = handle.read()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu")
    model_class = getattr(boda.model, checkpoint["model_module"])
    model = model_class(**vars(checkpoint["model_hparams"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return checkpoint, bytes_sha256(checkpoint_bytes)


def read_key_value_marker(path: Path) -> dict:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def source_fold_provenance(row: dict, registry: dict, completion_rows: list[dict]) -> dict:
    selected = sorted(
        [
            item for item in completion_rows
            if item["part_slug"] == row["part_slug"]
            and item["base_config_id"] == row["base_config_id"]
            and item["rc_mode"] == row["rc_mode"]
            and item["loss_mode"] == row["loss_mode"]
        ],
        key=lambda item: int(item["development_fold"]),
    )
    if len(selected) != 5:
        raise RuntimeError(f"Expected five selected fold sources for {row['part_slug']}")
    selected_runs = [
        {
            "development_fold": int(item["development_fold"]),
            "cell_id": item["cell_id"],
            "run_id": item["resolved_run_id"],
            "execution_disposition": item["execution_disposition"],
        }
        for item in selected
    ]
    unweighted_mates = []
    for item in selected:
        cell_id = item.get("source_unweighted_cell_id", "")
        if not cell_id:
            continue
        completed_records = [
            record for record in registry.get(cell_id, [])
            if record.get("status", "").lower() == "completed"
        ]
        unweighted_mates.append({
            "development_fold": int(item["development_fold"]),
            "cell_id": cell_id,
            "run_id": completed_records[-1].get("run_id", "") if completed_records else "",
            "registry_record_available": bool(completed_records),
        })
    return {
        "source_selected_fold_runs": selected_runs,
        "source_unweighted_mate_runs": unweighted_mates,
    }


def freeze() -> dict:
    if sha256_file(RECONCILIATION) != EXPECTED_RECONCILIATION_SHA256:
        raise ValueError("Final-refit implementation reconciliation changed")
    rows, manifest_sha = validate_inputs()
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Final-refit manifest binding changed")
    registry = read_registry()
    with STAGE3_COMPLETION.open(newline="") as handle:
        completion_rows = list(csv.DictReader(handle))
    implementation_hashes = {
        str(path.relative_to(REPO)): sha256_file(path) for path in IMPLEMENTATION_SOURCES
    }
    records_by_cell = {
        cell: [record for record in records if record.get("status", "").lower() == "completed"]
        for cell, records in registry.items()
    }
    allowlist = []
    normalization_by_part = {}
    for row in rows:
        if not completed(row, registry, manifest_sha):
            raise RuntimeError(f"Final refit row {row['row']} is incomplete")
        records = records_by_cell.get(row["cell_id"], [])
        if len(records) != 1:
            raise RuntimeError(f"Expected one completion for {row['cell_id']}")
        record = records[0]
        provenance_path, artifact_path = validate_record(row, record)
        provenance = json.loads(provenance_path.read_text())
        split = provenance["data_split_summary"]
        source_provenance = source_fold_provenance(row, registry, completion_rows)
        checkpoint, checkpoint_member_sha = load_checkpoint_member(artifact_path)
        checkpoint_path = Path(row["default_root_dir"]) / "torch_checkpoint.pt"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"Missing final checkpoint file {checkpoint_path}")
        if sha256_file(checkpoint_path) != checkpoint_member_sha:
            raise RuntimeError(f"Checkpoint/tar member mismatch for {row['cell_id']}")
        done_path = STATUS_DONE / f"row_{row['row']}.done"
        if not done_path.is_file():
            raise RuntimeError(f"Missing completion marker {done_path}")
        done_values = read_key_value_marker(done_path)
        if done_values.get("cell_id") != row["cell_id"]:
            raise RuntimeError(f"Completion marker cell mismatch for {row['cell_id']}")
        log_path = Path(done_values.get("log", ""))
        if not log_path.is_file():
            raise RuntimeError(f"Missing completion log for {row['cell_id']}")
        normalization = {
            "train_row_id_hash": split["train_row_id_hash"],
            "normalization_row_id_hash": split["normalization_row_id_hash"],
            "normalization_row_count": split["target_normalization_row_count"],
            "target_mean": split["target_normalization_mean"],
            "target_std": split["target_normalization_std"],
            "target_std_ddof": split["target_normalization_std_ddof"],
        }
        previous = normalization_by_part.setdefault(row["part_slug"], normalization)
        if previous != normalization:
            raise RuntimeError(f"Seed normalization mismatch for {row['part_slug']}")
        allowlist.append({
            "allowlist_row": len(allowlist) + 1,
            "manifest_row": row["row"],
            "cell_id": row["cell_id"],
            "part_slug": row["part_slug"],
            "base_config_id": row["base_config_id"],
            "architecture": row["architecture"],
            "model_module": checkpoint["model_module"],
            "graph_module": checkpoint["graph_module"],
            "data_module": checkpoint["data_module"],
            "training_regime": row["training_regime"],
            "unfreeze_scope": row["unfreeze_scope"],
            "rc_mode": row["rc_mode"],
            "loss_mode": row["loss_mode"],
            "model_seed": row["model_seed"],
            "fixed_completed_epochs": row["fixed_epochs"],
            "run_id": record["run_id"],
            "run_url": record["run_url"],
            "wandb_project": record["wandb_project"],
            "dataset_path": row["dataset_path"],
            "dataset_sha256": row["dataset_sha256"],
            "split_manifest_path": row["split_manifest_path"],
            "split_manifest_id": row["split_manifest_id"],
            "split_manifest_sha256": row["split_manifest_sha256"],
            "train_row_id_hash": split["train_row_id_hash"],
            "normalization_row_id_hash": split["normalization_row_id_hash"],
            "non_audit_training_allowlist_hash": split["train_row_id_hash"],
            "isolation_method": "stable_id_only_physical_row_exclusion",
            "normalization_row_count": split["target_normalization_row_count"],
            "target_normalization_mean": split["target_normalization_mean"],
            "target_normalization_std": split["target_normalization_std"],
            "target_normalization_std_ddof": split["target_normalization_std_ddof"],
            "audit_exclusion_row_id_hash": split["audit_row_id_hash"],
            "artifact_path": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_member_sha,
            "compact_provenance_path": str(provenance_path),
            "compact_provenance_sha256": sha256_file(provenance_path),
            "completion_marker_path": str(done_path),
            "completion_marker_sha256": sha256_file(done_path),
            "completion_log_path": str(log_path),
            "completion_log_sha256": sha256_file(log_path),
            "completed_epoch_evidence": "successful_fail_closed_trainer_current_epoch_assertion",
            "source_head": provenance.get("source_head", ""),
            "input_policy": provenance.get("input_policy", ""),
            "pretrained_artifact_sha256": provenance.get("pretrained_artifact_sha256", ""),
            **source_provenance,
            "implementation_source_sha256": implementation_hashes,
            "git_commit": provenance.get("git_commit", ""),
            "selection_manifest_sha256": row["selection_manifest_sha256"],
            "protocol_amendment_sha256": row["protocol_amendment_sha256"],
            "implementation_reconciliation_path": str(RECONCILIATION),
            "implementation_reconciliation_sha256": EXPECTED_RECONCILIATION_SHA256,
            "status": "completed_reconciled_pre_audit",
        })

    if len(allowlist) != 15:
        raise RuntimeError(f"Expected 15 allowlist rows, found {len(allowlist)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "lib1_dedup_final_refit_checkpoint_allowlist_v1",
        "manifest_path": str(MANIFEST),
        "manifest_sha256": manifest_sha,
        "implementation_reconciliation_path": str(RECONCILIATION),
        "implementation_reconciliation_sha256": EXPECTED_RECONCILIATION_SHA256,
        "implementation_source_sha256": implementation_hashes,
        "row_count": len(allowlist),
        "rows": allowlist,
        "audit_loader_instantiated": False,
        "audit_targets_loaded": False,
        "audit_predictions_generated": False,
        "audit_metrics_computed": False,
    }
    ALLOWLIST_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    columns = list(allowlist[0])
    with ALLOWLIST_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(allowlist)
    summary = {
        "schema_version": "lib1_dedup_final_refit_checkpoint_allowlist_summary_v1",
        "allowlist_json_path": str(ALLOWLIST_JSON),
        "allowlist_json_sha256": sha256_file(ALLOWLIST_JSON),
        "allowlist_csv_path": str(ALLOWLIST_CSV),
        "allowlist_csv_sha256": sha256_file(ALLOWLIST_CSV),
        "row_count": len(allowlist),
        "parts": sorted(normalization_by_part),
        "seeds": [1701, 1702, 1703],
        "audit_loader_instantiated": False,
        "audit_targets_loaded": False,
        "audit_predictions_generated": False,
        "audit_metrics_computed": False,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2, sort_keys=True))
