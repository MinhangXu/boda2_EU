#!/usr/bin/env python3
"""Generate the locked 15-cell Lib1 dedup final-refit manifest.

This generator is development-only.  It reads frozen development manifests
and commands, never imports a DataModule, never constructs an audit loader,
and forces every refit command to use ``manifest_mode=final_refit`` with test
evaluation disabled.  Audit scoring is a separate, allowlist-bound program.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from collections import OrderedDict
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT_ROOT = HERE / "outputs"
ANALYSIS_ROOT = OUTPUT_ROOT / "analysis" / "lib1_dedup_stage3_weighted_loss_july2026"
MANIFEST_ROOT = OUTPUT_ROOT / "hpo_manifests"
SELECTION_PATH = ANALYSIS_ROOT / "stage3_selected_part_policies.json"
STAGE3_PATH = MANIFEST_ROOT / "lib1_dedup_stage3_weighted_loss_july2026__analysis_manifest.jsonl"
STAGE2_PATH = MANIFEST_ROOT / "lib1_dedup_stage2_july2026__analysis_manifest.jsonl"
AMENDMENT_PATH = (
    REPO
    / "plan/phase1_lib1/dedup_phase1_rerun_july2026"
    / "lib1_dedup_final_refit_and_audit_protocol_amendment_july16_2026.md"
)
DEFAULT_MANIFEST = MANIFEST_ROOT / "lib1_dedup_final_refit_july2026__dry_run_manifest.jsonl"
DEFAULT_SUMMARY = MANIFEST_ROOT / "lib1_dedup_final_refit_july2026__summary.json"

EXPECTED_HASHES = {
    SELECTION_PATH: "f0f818fb6b4b722726e5a98edb1e525f2c66f6ff1155772d07a0b0a71769464c",
    STAGE3_PATH: "7b2d4115e697b8ac9507b3a8e1f5ce22aa55a6da8c2fb826d9b52992932d5995",
    STAGE2_PATH: "167a12b15654d8aa9ea63ca725aafe907e7337bec8846c3be41b8935403ea66a",
    AMENDMENT_PATH: "ff5ca5765f15c270ee33a7098dfb18646c426140be132c538aaff8ba003ec686",
}
PART_ORDER = {"enhancer": 0, "promoter": 1, "intron": 2, "utr3": 3, "utr5": 4}
FIXED_EPOCHS = {"enhancer": 6, "promoter": 44, "intron": 21, "utr3": 36, "utr5": 83}
SEEDS = (1701, 1702, 1703)
ENTITY = "minhangxu1998-baylor-college-of-medicine"
CAMPAIGN_ID = "lib1_dedup_phase1_rerun_july2026"
CAMPAIGN_STAGE = "final_refit"
MANIFEST_TAG = "lib1_dedup_final_refit_july2026"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_command(command: str) -> tuple[list[str], OrderedDict[str, list[str]]]:
    tokens = shlex.split(command)
    first_option = next((index for index, token in enumerate(tokens) if token.startswith("--")), len(tokens))
    prefix = tokens[:first_option]
    options: OrderedDict[str, list[str]] = OrderedDict()
    index = first_option
    while index < len(tokens):
        key = tokens[index]
        if not key.startswith("--"):
            raise ValueError(f"Unexpected positional command token {key!r}")
        index += 1
        values = []
        while index < len(tokens) and not tokens[index].startswith("--"):
            values.append(tokens[index])
            index += 1
        options[key[2:]] = values
    return prefix, options


def build_command(source: str, mutations: dict[str, list[str] | None]) -> str:
    prefix, options = parse_command(source)
    if prefix[-1:] != ["train_wandb_log.py"]:
        raise ValueError(f"Unexpected training entry point: {prefix}")
    for key, values in mutations.items():
        if values is None:
            options.pop(key, None)
        else:
            options[key] = [str(value) for value in values]
    tokens = prefix[:]
    for key, values in options.items():
        tokens.append(f"--{key}")
        tokens.extend(values)
    return " ".join(shlex.quote(token) for token in tokens)


def source_row_for(selection: dict, stage3_rows: list[dict], stage2_rows: list[dict]) -> tuple[dict, list[dict]]:
    matched = [
        row
        for row in stage3_rows
        if row.get("part_slug") == selection["part_slug"]
        and row.get("base_config_id") == selection["base_config_id"]
        and row.get("rc_mode") == selection["rc_mode"]
        and row.get("loss_mode") == selection["loss_mode"]
    ]
    if len(matched) != 5 or sorted(int(row["development_fold"]) for row in matched) != list(range(5)):
        raise ValueError(f"Selected {selection['part_slug']} arm is not a complete five-fold arm")
    command_rows = [row for row in matched if row.get("train_command")]
    if not command_rows and selection["part_slug"] == "enhancer":
        command_rows = [
            row
            for row in stage2_rows
            if row.get("part_slug") == "enhancer"
            and row.get("base_config_id") == selection["base_config_id"]
            and row.get("rc_mode") == selection["rc_mode"]
            and row.get("train_command")
        ]
    if not command_rows:
        raise ValueError(f"No source training command for {selection['part_slug']}")
    representative = min(command_rows, key=lambda row: int(row["development_fold"]))
    return representative, matched


def generate(manifest_path: Path, summary_path: Path) -> dict:
    for path, expected in EXPECTED_HASHES.items():
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Frozen input hash mismatch for {path}: {observed} != {expected}")

    selections = json.loads(SELECTION_PATH.read_text())["part_selections"]
    if len(selections) != 5:
        raise ValueError("Expected exactly five selected part policies")
    selections = sorted(selections, key=lambda row: PART_ORDER[row["part_slug"]])
    stage3_rows = read_jsonl(STAGE3_PATH)
    stage2_rows = read_jsonl(STAGE2_PATH)
    rows = []
    for selection in selections:
        part = selection["part_slug"]
        representative, five_fold_rows = source_row_for(selection, stage3_rows, stage2_rows)
        source_fingerprints = sorted(str(row["row_fingerprint"]) for row in five_fold_rows)
        for seed in SEEDS:
            identity = {
                "manifest_tag": MANIFEST_TAG,
                "part_slug": part,
                "base_config_id": selection["base_config_id"],
                "rc_mode": selection["rc_mode"],
                "loss_mode": selection["loss_mode"],
                "model_seed": seed,
                "fixed_epochs": FIXED_EPOCHS[part],
            }
            cell_id = f"refitcell_{canonical_hash(identity)[:20]}"
            run_root = (
                OUTPUT_ROOT / "hpo_runs" / MANIFEST_TAG / part
                / selection["base_config_id"] / f"seed_{seed}"
            )
            run_name = (
                f"{MANIFEST_TAG}__{part}__{selection['base_config_id'][8:24]}"
                f"__seed{seed}__epochs{FIXED_EPOCHS[part]}"
            )
            project = f"{part}__bashor_in_house__dedup_exact_v1__final_refit_development"
            mutations = {
                "artifact_path": [str(run_root / "artifacts")],
                "best_checkpoint_dir": [str(run_root / "published_checkpoint")],
                "keep_lightning_checkpoints": ["false"],
                "artifact_retention": ["selected"],
                "evaluate_test_after_fit": ["false"],
                "prediction_output_dir": [str(run_root / "predictions_disabled")],
                "prediction_splits": None,
                "provenance_output_dir": [str(run_root / "provenance")],
                "checkpoint_monitor": None,
                "stopping_mode": None,
                "stopping_patience": None,
                "logger_project": [project],
                "wandb_entity": [ENTITY],
                "wandb_group": [f"{CAMPAIGN_ID}__final_refit__{part}"],
                "wandb_job_type": ["final_refit_cell"],
                "run_name": [run_name],
                "exact_run_name": ["true"],
                "model_seed": [str(seed)],
                "campaign_id": [CAMPAIGN_ID],
                "campaign_stage": [CAMPAIGN_STAGE],
                "cell_id": [cell_id],
                "rc_pair_id": None,
                "loss_pair_id": None,
                "source_unweighted_cell_id": None,
                "execution_disposition": ["launch"],
                "development_fold": None,
                "source_run_ids": None,
                "wandb_tags": [
                    CAMPAIGN_ID, CAMPAIGN_STAGE, part,
                    selection["architecture"], selection["rc_mode"],
                    selection["loss_mode"], f"seed{seed}",
                ],
                "epoch_eval_splits": None,
                "manifest_mode": ["final_refit"],
                "split_fold": None,
                "train_size_frac": ["1.0"],
                "train_size_n": None,
                "train_min_barcodes": ["1"],
                "max_epochs": [str(FIXED_EPOCHS[part])],
                "min_epochs": ["0"],
                "limit_val_batches": ["0"],
                "num_sanity_val_steps": ["0"],
                "enable_checkpointing": ["false"],
                "default_root_dir": [str(run_root)],
                "enable_progress_bar": ["false"],
            }
            command = build_command(representative["train_command"], mutations)
            row = {
                **identity,
                "row": len(rows) + 1,
                "cell_id": cell_id,
                "row_fingerprint": canonical_hash({**identity, "command": command}),
                "architecture": selection["architecture"],
                "training_regime": selection["training_regime"],
                "unfreeze_scope": selection["unfreeze_scope"],
                "selection_manifest_path": str(SELECTION_PATH),
                "selection_manifest_sha256": EXPECTED_HASHES[SELECTION_PATH],
                "stage3_analysis_manifest_sha256": EXPECTED_HASHES[STAGE3_PATH],
                "protocol_amendment_path": str(AMENDMENT_PATH),
                "protocol_amendment_sha256": EXPECTED_HASHES[AMENDMENT_PATH],
                "source_fold_row_fingerprints": source_fingerprints,
                "source_command_manifest_path": str(STAGE2_PATH if part == "enhancer" else STAGE3_PATH),
                "source_command_row_fingerprint": representative["row_fingerprint"],
                "dataset_path": representative["dataset_path"],
                "dataset_sha256": representative["dataset_sha256"],
                "split_manifest_path": representative["split_manifest_path"],
                "split_manifest_id": representative["split_manifest_id"],
                "split_manifest_sha256": representative["split_manifest_sha256"],
                "train_min_barcodes": 1,
                "manifest_mode": "final_refit",
                "validation_loader": False,
                "early_stopping": False,
                "checkpoint_policy": "final_epoch_portable_artifact",
                "artifact_retention": "selected",
                "evaluate_test_after_fit": False,
                "audit_loader_instantiated": False,
                "audit_targets_loaded": False,
                "audit_predictions_generated": False,
                "audit_metrics_computed": False,
                "default_root_dir": str(run_root),
                "artifact_dir": str(run_root / "artifacts"),
                "provenance_dir": str(run_root / "provenance"),
                "logger_project": project,
                "wandb_entity": ENTITY,
                "planned_run_name": run_name,
                "train_command": command,
                "train_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
                "manifest_status": "frozen_dry_run_not_launched",
            }
            rows.append(row)

    if len(rows) != 15:
        raise AssertionError(f"Expected 15 final refits, found {len(rows)}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    manifest_sha = sha256_file(manifest_path)
    summary = {
        "schema_version": "lib1_dedup_final_refit_manifest_v1",
        "manifest_tag": MANIFEST_TAG,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "row_count": len(rows),
        "part_count": 5,
        "seeds": list(SEEDS),
        "fixed_epochs": FIXED_EPOCHS,
        "selection_manifest_sha256": EXPECTED_HASHES[SELECTION_PATH],
        "protocol_amendment_sha256": EXPECTED_HASHES[AMENDMENT_PATH],
        "commands_executed": 0,
        "audit_loader_instantiated": False,
        "audit_targets_loaded": False,
        "audit_predictions_generated": False,
        "audit_metrics_computed": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    summary = generate(args.manifest.resolve(), args.summary.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
