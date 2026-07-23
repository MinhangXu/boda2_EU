#!/usr/bin/env python3
"""Fail-closed validation for the Lib1 dedup Stage 2 development manifests.

This verifier is intentionally independent of the launcher.  It validates the
660-cell analysis design, the 50 immutable Stage 1 reuse cells, and the 610 new
training commands without constructing any data loader or touching the frozen
audit partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boda.common import constants


DEFAULT_PREFIX = HERE / "outputs/hpo_manifests/lib1_dedup_stage2_july2026"
EXPECTED_ENTITY = "minhangxu1998-baylor-college-of-medicine"
EXPECTED_LANES = {
    "core_scratch": (500, 450, 50),
    "enhancer_transfer_challenger": (60, 60, 0),
    "utr3_utrbasset_challenger": (100, 100, 0),
}
EXPECTED_PROJECTS = {
    part: f"{part}__bashor_in_house__dedup_exact_v1__stage2_development"
    for part in ("enhancer", "promoter", "intron", "utr3", "utr5")
}
TRANSFER_INPUT_POLICY = "malinois_mpra_flank600_v1"
TRANSFER_ARTIFACT_SHA256 = (
    "06e926e42304b8207138f1fb871ec19e0654dcdb6b26a62ed23fe1e9ac8cc592"
)
EXPECTED_UTR_SELECTION_RUN_IDS = (
    "utc3cqzn",
    "r8gx494e",
    "dx4cw1l9",
    "11g559xo",
    "v0xdcm0y",
    "h5hkkd86",
    "okhto5as",
    "9kneglhi",
    "jfzrac53",
    "zwf5cj86",
)
EXPECTED_UTR_SELECTION_DIGEST = (
    "b5f3e773496a72759d9df4b6c9010f8fbc0e6bac712126843135915b8e6996ef"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_command(command: str) -> dict[str, list[str]]:
    tokens = shlex.split(command)
    if tokens[:2] != ["python", "train_wandb_log.py"]:
        raise ValueError("train_command must start with `python train_wandb_log.py`")
    options: dict[str, list[str]] = {}
    index = 2
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise ValueError(f"Unexpected command token {token!r}")
        name = token[2:]
        if name in options:
            raise ValueError(f"Duplicate command option --{name}")
        index += 1
        values = []
        while index < len(tokens) and not tokens[index].startswith("--"):
            values.append(tokens[index])
            index += 1
        options[name] = values
    return options


def one(options: dict[str, list[str]], name: str) -> str:
    values = options.get(name)
    if values is None or len(values) != 1:
        raise ValueError(f"Expected one value for --{name}; found {values!r}")
    return values[0]


def expected_row_fingerprint(row: dict) -> str:
    fields = (
        "analysis_lane",
        "part_slug",
        "base_config_id",
        "development_fold",
        "rc_mode",
        "execution_disposition",
        "dataset_sha256",
        "split_manifest_sha256",
        "planned_run_name",
        "train_command",
        "reuse_source_run_id",
        "reuse_prediction_sha256",
    )
    return sha256_json({field: row.get(field) for field in fields})


def require_file_hash(path_value: str, expected: str, cache: dict[str, str]) -> Path:
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"Required file is missing: {path}")
    key = str(path.resolve())
    if key not in cache:
        cache[key] = sha256_file(path)
    if cache[key] != expected:
        raise ValueError(
            f"File hash mismatch for {path}: expected {expected}, observed {cache[key]}"
        )
    return path


def validate_transfer_view(
    transfer_path: Path, canonical_path: Path, expected_transfer_sha: str
) -> None:
    if sha256_file(transfer_path) != expected_transfer_sha:
        raise ValueError("Enhancer transfer split SHA does not match manifest rows")
    transfer = json.loads(transfer_path.read_text())
    canonical = json.loads(canonical_path.read_text())
    if transfer["assignments"] != canonical["assignments"]:
        raise ValueError("Enhancer transfer view changed frozen split assignments")
    if transfer["folds"] != canonical["folds"]:
        raise ValueError("Enhancer transfer view changed frozen fold definitions")
    for field in (
        "assignment_sha256",
        "audit_ids_sha256",
        "development_ids_sha256",
        "train_only_ids_sha256",
    ):
        if transfer["expected"][field] != canonical["expected"][field]:
            raise ValueError(f"Enhancer transfer view changed expected.{field}")
    dataset = transfer["dataset"]
    expected_dataset = {
        "padded_seq_len": 600,
        "padding_mode": "mpra_flank",
        "input_policy_id": TRANSFER_INPUT_POLICY,
        "left_flank_sha256": hashlib.sha256(
            constants.MPRA_UPSTREAM.encode("utf-8")
        ).hexdigest(),
        "right_flank_sha256": hashlib.sha256(
            constants.MPRA_DOWNSTREAM.encode("utf-8")
        ).hexdigest(),
    }
    for field, expected in expected_dataset.items():
        if dataset.get(field) != expected:
            raise ValueError(
                f"Enhancer transfer split dataset.{field} mismatch: "
                f"expected {expected!r}, observed {dataset.get(field)!r}"
            )
    if transfer.get("source_manifest_sha256") != sha256_file(canonical_path):
        raise ValueError("Enhancer transfer view source-manifest SHA mismatch")


def validate_command(row: dict, file_hashes: dict[str, str]) -> None:
    options = parse_command(row["train_command"])
    expected_single = {
        "campaign_id": "lib1_dedup_phase1_rerun_july2026",
        "campaign_stage": "stage2_paired_rc",
        "part_slug": row["part_slug"],
        "analysis_lane": row["analysis_lane"],
        "challenger_family": row["challenger_family"],
        "policy_id": row["policy_id"],
        "config_origin": row["config_origin"],
        "training_regime": row["training_regime"],
        "cell_id": row["cell_id"],
        "rc_pair_id": row["rc_pair_id"],
        "rc_mode": row["rc_mode"],
        "execution_disposition": "launch",
        "initialization": row["initialization"],
        "input_policy": row["input_policy"],
        "data_generation_id": row["data_generation_id"],
        "dataset_sha256": row["dataset_sha256"],
        "split_manifest_id": row["split_manifest_id"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "development_fold": str(row["development_fold"]),
        "split_fold": str(row["development_fold"]),
        "base_config_id": row["base_config_id"],
        "architecture": row["architecture"],
        "loss_mode": "unweighted_mse",
        "target_definition": "log2(RNA_bc_counts_sum / DNA_bc_counts_sum)",
        "artifact_retention": "none",
        "evaluate_test_after_fit": "false",
        "logger_type": "wandb",
        "wandb_entity": EXPECTED_ENTITY,
        "logger_project": row["logger_project"],
        "wandb_group": row["wandb_group"],
        "wandb_job_type": "stage2_cell",
        "run_name": row["planned_run_name"],
        "exact_run_name": "true",
        "model_seed": "1701",
        "datafile_path": row["dataset_path"],
        "split_manifest_path": row["split_manifest_path"],
        "expected_data_sha256": row["dataset_sha256"],
        "expected_split_sha256": row["split_manifest_sha256"],
        "target_column": "log2_RNA_DNA",
        "normalize": "true",
        "test_min_barcodes": "8",
        "train_min_barcodes": "1",
        "use_reverse_complements": "true" if row["rc_mode"] == "on" else "false",
        "barcode_weighting": "false",
        "loss_criterion": "MSELoss",
        "reduction": "mean",
        "precision": "32",
        "default_root_dir": row["default_root_dir"],
        "enable_progress_bar": "false",
    }
    for name, expected in expected_single.items():
        observed = one(options, name)
        if observed != str(expected):
            raise ValueError(
                f"Cell {row['cell_id']} --{name} mismatch: expected "
                f"{expected!r}, observed {observed!r}"
            )
    if options.get("epoch_eval_splits") != ["train", "val"]:
        raise ValueError(f"Cell {row['cell_id']} epoch evaluation is not train/val only")
    if options.get("prediction_splits") != ["val"]:
        raise ValueError(f"Cell {row['cell_id']} must export validation predictions only")
    if any(value.lower() == "test" for values in options.values() for value in values):
        raise ValueError(f"Cell {row['cell_id']} command exposes the frozen audit/test split")
    if row["analysis_lane"] == "enhancer_transfer_challenger":
        transfer_expected = {
            "data_module": "Lib1EnhancerDataModule",
            "model_module": "BassetBranched",
            "graph_module": "CNNBassetBranchedScopedTransfer",
            "parent_artifact": row["pretrained_artifact_path"],
            "pretrained_artifact_sha256": TRANSFER_ARTIFACT_SHA256,
            "source_head": row["source_head"],
            "unfreeze_scope": row["unfreeze_scope"],
            "padded_seq_len": "600",
            "padding_mode": "mpra_flank",
            "head_lr": "0.0005",
            "backbone_lr": "0.0001",
            "transfer_weight_decay": "0.0001",
            "frozen_epochs": "2",
        }
        for name, expected in transfer_expected.items():
            if one(options, name) != str(expected):
                raise ValueError(f"Transfer cell {row['cell_id']} --{name} mismatch")
        require_file_hash(
            row["pretrained_artifact_path"], TRANSFER_ARTIFACT_SHA256, file_hashes
        )
    else:
        if one(options, "graph_module") != "CNNBasicTraining":
            raise ValueError(f"Scratch cell {row['cell_id']} uses a non-scratch graph")
        forbidden = {
            "parent_artifact",
            "source_head",
            "unfreeze_scope",
            "head_lr",
            "backbone_lr",
            "transfer_weight_decay",
            "frozen_epochs",
        }
        present = sorted(forbidden & options.keys())
        if present:
            raise ValueError(f"Scratch cell {row['cell_id']} has transfer flags: {present}")


def validate(args: argparse.Namespace) -> dict:
    analysis_rows = read_jsonl(args.analysis_manifest)
    launch_rows = read_jsonl(args.run_manifest)
    reuse_rows = read_jsonl(args.reuse_manifest)
    if (len(analysis_rows), len(reuse_rows), len(launch_rows)) != (660, 50, 610):
        raise ValueError(
            "Stage 2 accounting mismatch: expected analysis/reuse/launch=660/50/610; "
            f"observed {len(analysis_rows)}/{len(reuse_rows)}/{len(launch_rows)}"
        )

    analysis_counts = Counter(row["analysis_lane"] for row in analysis_rows)
    launch_counts = Counter(row["analysis_lane"] for row in launch_rows)
    reuse_counts = Counter(row["analysis_lane"] for row in reuse_rows)
    expected_analysis = Counter({lane: counts[0] for lane, counts in EXPECTED_LANES.items()})
    expected_launch = Counter({lane: counts[1] for lane, counts in EXPECTED_LANES.items()})
    expected_reuse = Counter(
        {lane: counts[2] for lane, counts in EXPECTED_LANES.items() if counts[2]}
    )
    if analysis_counts != expected_analysis:
        raise ValueError(f"Analysis lane counts changed: {analysis_counts}")
    if launch_counts != expected_launch:
        raise ValueError(f"Launch lane counts changed: {launch_counts}")
    if reuse_counts != expected_reuse:
        raise ValueError(f"Reuse lane counts changed: {reuse_counts}")

    if [int(row["analysis_cell"]) for row in analysis_rows] != list(range(1, 661)):
        raise ValueError("analysis_cell must be contiguous from 1 to 660")
    if [int(row["manifest_row"]) for row in launch_rows] != list(range(1, 611)):
        raise ValueError("launch manifest_row must be contiguous from 1 to 610")

    file_hashes: dict[str, str] = {}
    key_fields = (
        "analysis_lane",
        "part_slug",
        "base_config_id",
        "development_fold",
        "rc_mode",
    )
    keys = [tuple(row[field] for field in key_fields) for row in analysis_rows]
    if len(set(keys)) != 660:
        raise ValueError("Analysis lane/config/fold/RC keys are not unique")
    if len({row["cell_id"] for row in analysis_rows}) != 660:
        raise ValueError("Stage 2 cell_id values are not unique")

    config_counts = Counter(
        (row["analysis_lane"], row["part_slug"], row["base_config_id"])
        for row in analysis_rows
    )
    if len(config_counts) != 66 or set(config_counts.values()) != {10}:
        raise ValueError("Every one of 66 configs must have five folds x two RC modes")
    expected_configs = Counter({"core_scratch": 50, "enhancer_transfer_challenger": 6,
                                "utr3_utrbasset_challenger": 10})
    observed_configs = Counter(key[0] for key in config_counts)
    if observed_configs != expected_configs:
        raise ValueError(f"Stage 2 config counts changed: {observed_configs}")

    pair_groups: dict[str, list[dict]] = defaultdict(list)
    launch_by_cell = {row["cell_id"]: row for row in launch_rows}
    reuse_by_cell = {row["cell_id"]: row for row in reuse_rows}
    if len(launch_by_cell) != 610 or len(reuse_by_cell) != 50:
        raise ValueError("Launch/reuse cell IDs are not unique")

    transfer_split_paths = set()
    transfer_split_shas = set()

    def same_analysis_cell(left: dict, right: dict) -> bool:
        # The launch-only projection adds its own contiguous queue row number.
        left = {key: value for key, value in left.items() if key != "manifest_row"}
        right = {key: value for key, value in right.items() if key != "manifest_row"}
        return left == right

    for row in analysis_rows:
        if row["row_fingerprint"] != expected_row_fingerprint(row):
            raise ValueError(f"Cell {row['cell_id']} row fingerprint mismatch")
        if row["campaign_id"] != "lib1_dedup_phase1_rerun_july2026":
            raise ValueError(f"Cell {row['cell_id']} campaign ID mismatch")
        if row["campaign_stage"] != "stage2_paired_rc":
            raise ValueError(f"Cell {row['cell_id']} campaign stage mismatch")
        if row["wandb_entity"] != EXPECTED_ENTITY:
            raise ValueError(f"Cell {row['cell_id']} W&B entity mismatch")
        if row["logger_project"] != EXPECTED_PROJECTS[row["part_slug"]]:
            raise ValueError(f"Cell {row['cell_id']} W&B project mismatch")
        if row["model_seed"] != 1701 or row["loss_mode"] != "unweighted_mse":
            raise ValueError(f"Cell {row['cell_id']} fixed seed/loss contract changed")
        if row["artifact_retention"] != "none" or row["evaluate_test_after_fit"] is not False:
            raise ValueError(f"Cell {row['cell_id']} enables artifact/audit behavior")
        if (row["epoch_eval_splits"], row["prediction_splits"]) != (
            ["train", "val"],
            ["val"],
        ):
            raise ValueError(f"Cell {row['cell_id']} train/val-only split contract changed")
        if row["rc_mode"] not in {"off", "on"}:
            raise ValueError(f"Cell {row['cell_id']} has invalid RC mode")
        if row["use_reverse_complements"] is not (row["rc_mode"] == "on"):
            raise ValueError(f"Cell {row['cell_id']} RC metadata mismatch")
        require_file_hash(row["dataset_path"], row["dataset_sha256"], file_hashes)
        require_file_hash(
            row["split_manifest_path"], row["split_manifest_sha256"], file_hashes
        )
        pair_groups[row["rc_pair_id"]].append(row)
        if row["analysis_lane"] == "enhancer_transfer_challenger":
            transfer_split_paths.add(row["split_manifest_path"])
            transfer_split_shas.add(row["split_manifest_sha256"])

        disposition = row["execution_disposition"]
        if disposition == "launch":
            if row["cell_id"] not in launch_by_cell or row["train_command"] == "":
                raise ValueError(f"Launch cell {row['cell_id']} missing from run manifest")
            if not same_analysis_cell(launch_by_cell[row["cell_id"]], row):
                raise ValueError(f"Launch cell {row['cell_id']} differs across manifests")
            validate_command(row, file_hashes)
        elif disposition == "reuse_stage1":
            if row["cell_id"] not in reuse_by_cell or row["train_command"] != "":
                raise ValueError(f"Reuse cell {row['cell_id']} manifest mismatch")
            if reuse_by_cell[row["cell_id"]] != row:
                raise ValueError(f"Reuse cell {row['cell_id']} differs across manifests")
            if row["analysis_lane"] != "core_scratch" or row["development_fold"] != 0:
                raise ValueError(f"Reuse cell {row['cell_id']} is not core fold 0")
            if row["rc_mode"] != "off" or not row["reuse_source_run_id"]:
                raise ValueError(f"Reuse cell {row['cell_id']} is not RC-off Stage 1 output")
            require_file_hash(
                row["reuse_prediction_path"], row["reuse_prediction_sha256"], file_hashes
            )
        else:
            raise ValueError(f"Cell {row['cell_id']} has unknown disposition {disposition!r}")

    if len(pair_groups) != 330:
        raise ValueError(f"Expected 330 RC pairs; found {len(pair_groups)}")
    invariant_fields = (
        "analysis_lane",
        "part_slug",
        "base_config_id",
        "development_fold",
        "model_seed",
        "loss_mode",
        "dataset_sha256",
        "split_manifest_sha256",
    )
    for pair_id, rows in pair_groups.items():
        if len(rows) != 2 or {row["rc_mode"] for row in rows} != {"off", "on"}:
            raise ValueError(f"RC pair {pair_id} does not contain exactly off/on")
        for field in invariant_fields:
            if rows[0][field] != rows[1][field]:
                raise ValueError(f"RC pair {pair_id} changed invariant field {field}")

    transfer_grid = Counter(
        (row["source_head"], row["unfreeze_scope"])
        for row in analysis_rows
        if row["analysis_lane"] == "enhancer_transfer_challenger"
    )
    expected_transfer_grid = Counter(
        {
            (head, scope): 10
            for head in ("HepG2", "K562")
            for scope in ("branched_only", "conv3_plus", "full")
        }
    )
    if transfer_grid != expected_transfer_grid:
        raise ValueError(f"Enhancer N=2 x scope transfer grid changed: {transfer_grid}")

    selected_utr = read_jsonl(args.utr_selection)
    selection_payload = [
        {
            "selection_reason": row["selection_reason"],
            "source_run_id": row["source_run_id"],
            "base_config_id": row["base_config_id"],
        }
        for row in selected_utr
    ]
    if len(selection_payload) != 10:
        raise ValueError(f"UTRBasset K=10 artifact has {len(selection_payload)} rows")
    if tuple(row["source_run_id"] for row in selection_payload) != EXPECTED_UTR_SELECTION_RUN_IDS:
        raise ValueError("UTRBasset K=10 source run IDs changed")
    if sha256_json(selection_payload) != EXPECTED_UTR_SELECTION_DIGEST:
        raise ValueError("UTRBasset K=10 selection digest changed")
    selected_utr_pairs = {
        (row["source_run_id"], row["base_config_id"]) for row in selection_payload
    }
    manifest_utr_pairs = Counter(
        (row["source_run_ids"][0], row["base_config_id"])
        for row in analysis_rows
        if row["analysis_lane"] == "utr3_utrbasset_challenger"
    )
    if set(manifest_utr_pairs) != selected_utr_pairs or set(manifest_utr_pairs.values()) != {10}:
        raise ValueError("UTRBasset challenger cells do not match the frozen K=10 artifact")

    if len(transfer_split_paths) != 1 or len(transfer_split_shas) != 1:
        raise ValueError("Transfer cells do not share exactly one frozen split view")
    canonical_index = json.loads(args.split_index.read_text())
    canonical_path = Path(canonical_index["parts"]["enhancer"]["manifest_path"])
    validate_transfer_view(
        Path(next(iter(transfer_split_paths))),
        canonical_path,
        next(iter(transfer_split_shas)),
    )

    return {
        "analysis_cells": 660,
        "stage1_reuse_cells": 50,
        "launch_cells": 610,
        "rc_pairs": 330,
        "configs": 66,
        "analysis_counts_by_lane": dict(sorted(analysis_counts.items())),
        "launch_counts_by_lane": dict(sorted(launch_counts.items())),
        "run_manifest_sha256": sha256_file(args.run_manifest),
        "analysis_manifest_sha256": sha256_file(args.analysis_manifest),
        "audit_loader_instantiated": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-manifest",
        type=Path,
        default=Path(str(DEFAULT_PREFIX) + "__analysis_manifest.jsonl"),
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=Path(str(DEFAULT_PREFIX) + "__run_manifest.jsonl"),
    )
    parser.add_argument(
        "--reuse-manifest",
        type=Path,
        default=Path(str(DEFAULT_PREFIX) + "__stage1_reuse_cells.jsonl"),
    )
    parser.add_argument(
        "--split-index",
        type=Path,
        default=HERE / "data_manifests/lib1_dedup_exact_v1_split_manifests.json",
    )
    parser.add_argument(
        "--utr-selection",
        type=Path,
        default=Path(
            str(DEFAULT_PREFIX) + "__utr3_utrbassetvl_selected_configs.jsonl"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(validate(parse_args()), indent=2, sort_keys=True))
