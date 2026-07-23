#!/usr/bin/env python3
"""Static fail-closed verifier for the 15-cell final-refit manifest.

The verifier deliberately does not import a DataModule or read a split
manifest/dataset.  It validates only the frozen manifest, command tokens, and
hash-bound development inputs; therefore it cannot instantiate an audit
loader or inspect audit targets/predictions/metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "outputs/hpo_manifests/lib1_dedup_final_refit_july2026__dry_run_manifest.jsonl"
EXPECTED_EPOCHS = {"enhancer": 6, "promoter": 44, "intron": 21, "utr3": 36, "utr5": 83}
EXPECTED_SEEDS = {1701, 1702, 1703}
EXPECTED_POLICIES = {
    "enhancer": ("basecfg_6e6b2b979116f3e9cd83a8747792d89a97918ce57e72949f810c309afa068036", "on", "unweighted_mse"),
    "promoter": ("basecfg_bff24362f7f5a2013947c22336ec779dc986c42124230dae5ff4fcc9904a5d0d", "off", "barcode_weighted_mse"),
    "intron": ("basecfg_58481a479285bf26af4a9813d37abecc1e6a548795eb3f606fe4d5758ecc4a86", "off", "barcode_weighted_mse"),
    "utr3": ("basecfg_7b1f881265b0fc0aee9e2b601352b93e064e37bee808c65b6b038e6a260e2062", "off", "barcode_weighted_mse"),
    "utr5": ("basecfg_9dd728c0df617152551b366c304a265d52be567ad04fb35dbdcecd406235d315", "off", "barcode_weighted_mse"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def option_map(command: str) -> dict[str, list[str]]:
    tokens = shlex.split(command)
    if tokens[:2] != ["python", "train_wandb_log.py"]:
        raise ValueError(f"Unexpected training entry point: {tokens[:2]}")
    result = {}
    index = 2
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected positional token {token!r}")
        key = token[2:]
        index += 1
        values = []
        while index < len(tokens) and not tokens[index].startswith("--"):
            values.append(tokens[index])
            index += 1
        if key in result:
            raise ValueError(f"Duplicate option --{key}")
        result[key] = values
    return result


def scalar(options: dict[str, list[str]], key: str) -> str:
    values = options.get(key)
    if values is None or len(values) != 1:
        raise ValueError(f"Expected one value for --{key}, found {values}")
    return values[0]


def verify(path: Path) -> dict:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    if len(rows) != 15:
        raise ValueError(f"Expected 15 final-refit rows, found {len(rows)}")
    if [int(row["row"]) for row in rows] != list(range(1, 16)):
        raise ValueError("Manifest rows are not numbered 1..15")
    if len({row["cell_id"] for row in rows}) != 15 or len({row["planned_run_name"] for row in rows}) != 15:
        raise ValueError("cell_id and planned_run_name must be unique")

    observed_pairs = {(row["part_slug"], int(row["model_seed"])) for row in rows}
    expected_pairs = {(part, seed) for part in EXPECTED_POLICIES for seed in EXPECTED_SEEDS}
    if observed_pairs != expected_pairs:
        raise ValueError("Part/seed Cartesian product does not match 5 x 3")

    for row in rows:
        part = row["part_slug"]
        config, rc_mode, loss_mode = EXPECTED_POLICIES[part]
        if (row["base_config_id"], row["rc_mode"], row["loss_mode"]) != (config, rc_mode, loss_mode):
            raise ValueError(f"Frozen policy mismatch for {part}")
        if int(row["fixed_epochs"]) != EXPECTED_EPOCHS[part]:
            raise ValueError(f"Fixed epoch mismatch for {part}")
        for flag in (
            "audit_loader_instantiated", "audit_targets_loaded",
            "audit_predictions_generated", "audit_metrics_computed",
            "evaluate_test_after_fit", "validation_loader", "early_stopping",
        ):
            if bool(row.get(flag)):
                raise ValueError(f"Row {row['row']} unexpectedly enables {flag}")
        for bound_path, bound_hash in (
            (row["selection_manifest_path"], row["selection_manifest_sha256"]),
            (row["protocol_amendment_path"], row["protocol_amendment_sha256"]),
            (row["dataset_path"], row["dataset_sha256"]),
            (row["split_manifest_path"], row["split_manifest_sha256"]),
        ):
            if sha256_file(Path(bound_path)) != bound_hash:
                raise ValueError(f"Hash mismatch for bound input {bound_path}")
        command = row["train_command"]
        if hashlib.sha256(command.encode()).hexdigest() != row["train_command_sha256"]:
            raise ValueError(f"Command hash mismatch in row {row['row']}")
        options = option_map(command)
        expected_scalars = {
            "campaign_stage": "final_refit",
            "part_slug": part,
            "base_config_id": config,
            "rc_mode": rc_mode,
            "loss_mode": loss_mode,
            "model_seed": str(row["model_seed"]),
            "max_epochs": str(EXPECTED_EPOCHS[part]),
            "min_epochs": "0",
            "manifest_mode": "final_refit",
            "train_min_barcodes": "1",
            "train_size_frac": "1.0",
            "evaluate_test_after_fit": "false",
            "artifact_retention": "selected",
            "limit_val_batches": "0",
            "num_sanity_val_steps": "0",
            "enable_checkpointing": "false",
            "use_reverse_complements": "true" if rc_mode == "on" else "false",
            "barcode_weighting": "true" if loss_mode == "barcode_weighted_mse" else "false",
        }
        for key, expected in expected_scalars.items():
            if scalar(options, key) != expected:
                raise ValueError(f"Row {row['row']} --{key} mismatch")
        forbidden = {
            "checkpoint_monitor", "prediction_splits", "development_fold",
            "split_fold", "train_size_n", "rc_pair_id", "loss_pair_id",
            "source_unweighted_cell_id",
        }
        present = forbidden & set(options)
        if present:
            raise ValueError(f"Row {row['row']} has forbidden options {sorted(present)}")
        command_lower = command.lower()
        if "audit_eval" in command_lower or "trainer.test" in command_lower or "prediction_splits test" in command_lower:
            raise ValueError(f"Row {row['row']} contains audit/test execution material")

    return {
        "manifest_path": str(path),
        "manifest_sha256": sha256_file(path),
        "row_count": len(rows),
        "parts": sorted(EXPECTED_POLICIES),
        "seeds": sorted(EXPECTED_SEEDS),
        "audit_loader_instantiated": False,
        "audit_targets_loaded": False,
        "audit_predictions_generated": False,
        "audit_metrics_computed": False,
        "validation_status": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
