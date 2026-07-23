#!/usr/bin/env python3
"""Generate the frozen Lib1 dedup Stage 2 paired-RC development manifests.

The analysis manifest contains all 660 cells.  Fifty fold-0/RC-off core cells
reuse Stage 1 predictions; the launch manifest contains only the 610 new jobs.
No W&B API calls or audit predictions are used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shlex
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generate_lib1_dedup_exact_replay_manifest as stage1
from boda.common import constants


CAMPAIGN_ID = "lib1_dedup_phase1_rerun_july2026"
CAMPAIGN_STAGE = "stage2_paired_rc"
MANIFEST_TAG = "lib1_dedup_stage2_july2026"
EXPECTED_ENTITY = "minhangxu1998-baylor-college-of-medicine"
MODEL_SEED = 1701
FOLDS = tuple(range(5))
RC_MODES = (False, True)
EXPECTED_ANALYSIS_CELLS = 660
EXPECTED_LAUNCH_CELLS = 610
EXPECTED_REUSE_CELLS = 50

ANALYSIS_LANES = (
    "core_scratch",
    "enhancer_transfer_challenger",
    "utr3_utrbasset_challenger",
)
INPUT_POLICIES = {
    "enhancer": "neutral_pad216_v1",
    "promoter": "neutral_pad51_v1",
    "intron": "exact80_v1",
    "utr3": "exact100_v1",
    "utr5": "exact50_v1",
}
TRANSFER_INPUT_POLICY = "malinois_mpra_flank600_v1"
TRANSFER_ARTIFACT_SHA256 = "06e926e42304b8207138f1fb871ec19e0654dcdb6b26a62ed23fe1e9ac8cc592"
TRANSFER_ADAPTER_VERSION = "malinois_single_head_scoped_v1"
TRANSFER_HEADS = ("HepG2", "K562")
TRANSFER_SCOPES = ("branched_only", "conv3_plus", "full")
UTR_SWEEP_ID = "nhoh1zuw"
UTR_PROJECT = "utr3__bashor_in_house__threeprime_modal100__scratch__utr_bassetvl_fp32"
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
EXPECTED_UTR_SELECTION_DIGEST = "b5f3e773496a72759d9df4b6c9010f8fbc0e6bac712126843135915b8e6996ef"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def cli_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def shell_command(tokens):
    return " ".join(shlex.quote(str(token)) for token in tokens)


def parse_command(command):
    tokens = shlex.split(command)
    if tokens[:2] != ["python", "train_wandb_log.py"]:
        raise ValueError(f"Unsupported source command prefix: {tokens[:2]}")
    options = OrderedDict()
    index = 2
    while index < len(tokens):
        flag = tokens[index]
        if not flag.startswith("--"):
            raise ValueError(f"Unexpected command token {flag!r}")
        name = flag[2:]
        index += 1
        values = []
        while index < len(tokens) and not tokens[index].startswith("--"):
            values.append(tokens[index])
            index += 1
        options[name] = values
    return options


def command_from_options(options):
    tokens = ["python", "train_wandb_log.py"]
    for name, values in options.items():
        tokens.append("--" + name)
        tokens.extend(str(value) for value in values)
    return shell_command(tokens)


def put(options, name, value):
    if value is None or value == "":
        options.pop(name, None)
        return
    values = value if isinstance(value, (list, tuple)) else [value]
    options[name] = [cli_value(item) for item in values]


def add_arg(tokens, name, value):
    if value is None or value == "":
        return
    tokens.extend(["--" + name, cli_value(value)])


def add_many(tokens, name, values):
    values = list(values or [])
    if not values:
        return
    tokens.append("--" + name)
    tokens.extend(cli_value(value) for value in values)


def lane_project(part_slug):
    return f"{part_slug}__bashor_in_house__dedup_exact_v1__stage2_development"


def pair_and_cell_ids(analysis_lane, part_slug, base_config_id, fold, rc_on):
    pair_payload = {
        "campaign_id": CAMPAIGN_ID,
        "analysis_lane": analysis_lane,
        "part_slug": part_slug,
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


def row_fingerprint(row):
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


def transfer_split_view(canonical_path, output_path):
    source = json.loads(Path(canonical_path).read_text())
    view = copy.deepcopy(source)
    view["manifest_id"] = "lib1_enhancer_dedup_exact_v1_transfer_mpra600_split_seed20260709"
    view["source_manifest_id"] = source["manifest_id"]
    view["source_manifest_sha256"] = sha256_file(canonical_path)
    dataset = view["dataset"]
    dataset["padded_seq_len"] = 600
    dataset["padding_mode"] = "mpra_flank"
    dataset["neutral_pad_char"] = "N"
    dataset["input_policy_id"] = TRANSFER_INPUT_POLICY
    dataset["left_flank_sha256"] = hashlib.sha256(
        constants.MPRA_UPSTREAM.encode("utf-8")
    ).hexdigest()
    dataset["right_flank_sha256"] = hashlib.sha256(
        constants.MPRA_DOWNSTREAM.encode("utf-8")
    ).hexdigest()
    write_json(output_path, view)

    reread = json.loads(Path(output_path).read_text())
    if reread["assignments"] != source["assignments"]:
        raise AssertionError("Transfer split view changed canonical assignments")
    for key in (
        "assignment_sha256",
        "audit_ids_sha256",
        "development_ids_sha256",
        "train_only_ids_sha256",
    ):
        if reread["expected"][key] != source["expected"][key]:
            raise AssertionError(f"Transfer split view changed expected.{key}")
    if reread["folds"] != source["folds"]:
        raise AssertionError("Transfer split view changed canonical fold hashes")
    return reread, sha256_file(output_path)


def _numeric(value):
    # Match the Stage 1 pandas transform: booleans are numeric 0/1 columns,
    # then z-scored, rather than one-hot categorical columns.
    return isinstance(value, (int, float, bool))


def diversity_vectors(candidates):
    fields = sorted(
        field
        for field in stage1.BASE_IDENTITY_FIELDS
        if len({canonical_json(row["base_identity"].get(field)) for row in candidates}) > 1
    )
    vectors = {row["source_run_ids"][0]: [] for row in candidates}
    for field in fields:
        values = [row["base_identity"].get(field) for row in candidates]
        nonnull = [value for value in values if value is not None]
        if nonnull and all(_numeric(value) for value in nonnull):
            transformed = []
            for value in values:
                numeric = np.nan if value is None else float(value)
                if (
                    field in {"lr", "weight_decay", "eps"}
                    and np.isfinite(numeric)
                    and numeric > 0
                ):
                    numeric = math.log10(numeric)
                transformed.append(numeric)
            array = np.asarray(transformed, dtype=float)
            fill = float(np.nanmedian(array))
            array = np.where(np.isnan(array), fill, array)
            scale = float(array.std())
            if scale <= 0:
                continue
            array = (array - float(array.mean())) / scale
            for row, value in zip(candidates, array):
                vectors[row["source_run_ids"][0]].append(float(value))
        else:
            categories = sorted({canonical_json(value) for value in values})
            for category in categories:
                for row, value in zip(candidates, values):
                    vectors[row["source_run_ids"][0]].append(
                        1.0 if canonical_json(value) == category else 0.0
                    )
    return {key: np.asarray(value, dtype=float) for key, value in vectors.items()}


def rms_distance(left, right):
    if left.shape != right.shape or left.size == 0:
        raise ValueError("Invalid diversity vector shapes")
    return float(np.sqrt(np.mean((left - right) ** 2)))


def resolve_utr3_utrbasset_candidates(learn_dir, runs_csv):
    _, by_run_id = stage1.read_runs_csv(runs_csv)
    config_index = stage1.build_local_run_config_index(learn_dir)
    sweep_dir = learn_dir / "wandb" / f"sweep-{UTR_SWEEP_ID}"
    candidates = []
    for sweep_path in sorted(sweep_dir.glob("config-*.yaml")):
        run_id = sweep_path.stem[len("config-"):]
        registry = by_run_id.get(run_id)
        if registry is None or registry.get("wandb_sweep_id") != UTR_SWEEP_ID:
            continue
        if registry.get("wandb_project") != UTR_PROJECT:
            raise ValueError(f"Unexpected UTRBasset project for {run_id}")
        full_path = config_index.get(run_id)
        if full_path is None:
            raise ValueError(f"Missing authoritative run config for {run_id}")
        full_snapshot = stage1.unwrap_wandb_yaml(full_path)
        snapshot = stage1.resolve_historical_snapshot(
            stage1.unwrap_wandb_yaml(sweep_path), full_snapshot
        )
        if snapshot.get("model_module") != "UTR_BassetVL":
            raise ValueError(f"Unexpected UTRBasset architecture for {run_id}")
        identity = stage1.base_identity(snapshot)
        digest = stage1.sha256_json(identity)
        val = stage1.historical_val(registry)
        if val is None:
            raise ValueError(f"Missing historical validation Pearson for {run_id}")
        source_reference = {
            "candidate_kind": "completed_june_hpo",
            "lane_id": "utr3__utr_bassetvl",
            "source_project": UTR_PROJECT,
            "source_sweep_id": UTR_SWEEP_ID,
            "source_run_ids": [run_id],
            "source_config_paths": [str(sweep_path.resolve()), str(full_path.resolve())],
            "source_sweep_config_sha256": sha256_file(sweep_path),
            "source_run_config_sha256": sha256_file(full_path),
        }
        candidates.append(
            {
                "candidate_kind": "completed_june_hpo",
                "lane_id": "utr3__utr_bassetvl",
                "part_slug": "utr3",
                "architecture": "UTR_BassetVL",
                "architecture_slug": "utr_bassetvl",
                "source_project": UTR_PROJECT,
                "source_sweep_id": UTR_SWEEP_ID,
                "source_run_ids": [run_id],
                "source_config_paths": source_reference["source_config_paths"],
                "source_run_config_sha256": source_reference["source_run_config_sha256"],
                "source_config_snapshot": snapshot,
                "resolved_historical_config": snapshot,
                "base_identity": identity,
                "base_config_id": "basecfg_" + digest,
                "base_config_sha256": digest,
                "excluded_from_base_config_id": stage1.excluded_audit(snapshot),
                "source_candidates": [source_reference],
                "historical_validation_values": [float(val)],
                "historical_validation_mean": float(val),
            }
        )
    if len(candidates) != 128:
        raise ValueError(f"Expected 128 completed nhoh1zuw configs; found {len(candidates)}")
    if len({row["base_config_id"] for row in candidates}) != 128:
        raise ValueError("nhoh1zuw did not resolve to 128 unique base identities")

    ranked = sorted(
        candidates,
        key=lambda row: (
            -row["historical_validation_mean"],
            row["base_config_id"],
            row["source_run_ids"][0],
        ),
    )
    for rank, row in enumerate(ranked, 1):
        row["historical_rank"] = rank
    leaders = ranked[:5]
    top_quartile = ranked[:32]
    vectors = diversity_vectors(candidates)
    selected = list(leaders)
    diversity_rows = []
    while len(selected) < 10:
        choices = []
        for candidate in top_quartile:
            if candidate in selected:
                continue
            run_id = candidate["source_run_ids"][0]
            distance = min(
                rms_distance(vectors[run_id], vectors[row["source_run_ids"][0]])
                for row in selected
            )
            choices.append(
                (
                    -distance,
                    -candidate["historical_validation_mean"],
                    candidate["base_config_id"],
                    run_id,
                    candidate,
                    distance,
                )
            )
        chosen_tuple = sorted(choices, key=lambda item: item[:4])[0]
        chosen = chosen_tuple[4]
        diversity_rows.append((chosen, chosen_tuple[5]))
        selected.append(chosen)

    for index, row in enumerate(leaders, 1):
        row["selection_reason"] = f"historical_validation_leader_{index}"
        row["maximin_distance"] = None
    for index, (row, distance) in enumerate(diversity_rows, 1):
        row["selection_reason"] = f"top_quartile_maximin_{index}"
        row["maximin_distance"] = float(distance)

    observed_ids = tuple(row["source_run_ids"][0] for row in selected)
    if observed_ids != EXPECTED_UTR_SELECTION_RUN_IDS:
        raise AssertionError(
            f"UTRBasset K=10 selection changed: expected {EXPECTED_UTR_SELECTION_RUN_IDS}, "
            f"observed {observed_ids}"
        )
    selection_digest = sha256_json(
        [
            {
                "selection_reason": row["selection_reason"],
                "source_run_id": row["source_run_ids"][0],
                "base_config_id": row["base_config_id"],
            }
            for row in selected
        ]
    )
    if selection_digest != EXPECTED_UTR_SELECTION_DIGEST:
        raise AssertionError(
            f"UTRBasset selection digest changed: {selection_digest}"
        )
    return selected, ranked, selection_digest


def scratch_launch_row(
    source_row,
    *,
    manifest_tag,
    analysis_lane,
    config_origin,
    challenger_family,
    fold,
    rc_on,
    output_root,
):
    part_slug = source_row["part_slug"]
    base_config_id = source_row["base_config_id"]
    arch_slug = source_row["architecture_slug"]
    pair_id, cell_id = pair_and_cell_ids(
        analysis_lane, part_slug, base_config_id, fold, rc_on
    )
    rc_mode = "on" if rc_on else "off"
    root = (
        output_root
        / analysis_lane
        / part_slug
        / base_config_id
        / f"fold_{fold}"
        / f"rc_{rc_mode}"
    )
    project = lane_project(part_slug)
    group = f"{CAMPAIGN_ID}__stage2__{part_slug}__{analysis_lane}"
    run_name = (
        f"{manifest_tag}__{analysis_lane}__{part_slug}__{arch_slug}__"
        f"{source_row['base_config_sha256'][:16]}__fold{fold}__rc_{rc_mode}"
    )

    options = parse_command(source_row["train_command"])
    replacements = {
        "artifact_path": str(root / "artifacts"),
        "best_checkpoint_dir": str(root / "published_checkpoint_disabled"),
        "prediction_output_dir": str(root / "predictions"),
        "provenance_output_dir": str(root / "provenance"),
        "logger_project": project,
        "wandb_entity": EXPECTED_ENTITY,
        "wandb_group": group,
        "wandb_job_type": "stage2_cell",
        "run_name": run_name,
        "exact_run_name": True,
        "campaign_stage": CAMPAIGN_STAGE,
        "part_slug": part_slug,
        "development_fold": int(fold),
        "split_fold": int(fold),
        "use_reverse_complements": bool(rc_on),
        "default_root_dir": str(root),
        "analysis_lane": analysis_lane,
        "challenger_family": challenger_family,
        "policy_id": base_config_id,
        "config_origin": config_origin,
        "training_regime": "scratch",
        "cell_id": cell_id,
        "rc_pair_id": pair_id,
        "rc_mode": rc_mode,
        "execution_disposition": "launch",
        "initialization": "scratch",
        "input_policy": INPUT_POLICIES[part_slug],
        "artifact_retention": "none",
        "evaluate_test_after_fit": False,
        "checkpoint_monitor": "val_pearson",
        "stopping_mode": "max",
    }
    for name, value in replacements.items():
        put(options, name, value)
    put(options, "source_head", None)
    put(options, "unfreeze_scope", None)
    put(options, "pretrained_artifact_sha256", None)
    put(
        options,
        "wandb_tags",
        [
            CAMPAIGN_ID,
            CAMPAIGN_STAGE,
            part_slug,
            analysis_lane,
            arch_slug,
            f"fold{fold}",
            f"rc_{rc_mode}",
            "seed1701",
            "unweighted_mse",
        ],
    )
    put(options, "epoch_eval_splits", ["train", "val"])
    put(options, "prediction_splits", ["val"])
    command = command_from_options(options)

    row = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "manifest_tag": manifest_tag,
        "analysis_lane": analysis_lane,
        "challenger_family": challenger_family,
        "config_origin": config_origin,
        "training_regime": "scratch",
        "part_slug": part_slug,
        "lane_id": (
            "core__" + source_row["lane_id"]
            if analysis_lane == "core_scratch"
            else "utr3__utr_bassetvl_challenger"
        ),
        "architecture": source_row["architecture"],
        "architecture_slug": arch_slug,
        "base_config_id": base_config_id,
        "base_config_sha256": source_row["base_config_sha256"],
        "base_identity": source_row["base_identity"],
        "policy_id": base_config_id,
        "initialization": "scratch",
        "source_head": "",
        "source_head_index": None,
        "unfreeze_scope": "",
        "input_policy": INPUT_POLICIES[part_slug],
        "pretrained_artifact_sha256": "",
        "source_run_ids": source_row.get("source_run_ids", []),
        "source_candidates": source_row.get("source_candidates", []),
        "historical_validation_mean": source_row.get("historical_validation_mean"),
        "data_generation_id": source_row["data_generation_id"],
        "dataset_path": source_row["dataset_path"],
        "dataset_sha256": source_row["dataset_sha256"],
        "split_manifest_id": source_row["split_manifest_id"],
        "split_manifest_path": source_row["split_manifest_path"],
        "split_manifest_sha256": source_row["split_manifest_sha256"],
        "development_fold": int(fold),
        "model_seed": MODEL_SEED,
        "use_reverse_complements": bool(rc_on),
        "rc_mode": rc_mode,
        "rc_pair_id": pair_id,
        "cell_id": cell_id,
        "loss_mode": "unweighted_mse",
        "target_column": "log2_RNA_DNA",
        "target_definition": "log2(RNA_bc_counts_sum / DNA_bc_counts_sum)",
        "length_policy": source_row["length_policy"],
        "artifact_retention": "none",
        "evaluate_test_after_fit": False,
        "epoch_eval_splits": ["train", "val"],
        "prediction_splits": ["val"],
        "wandb_entity": EXPECTED_ENTITY,
        "logger_project": project,
        "wandb_group": group,
        "wandb_job_type": "stage2_cell",
        "planned_run_name": run_name,
        "default_root_dir": str(root),
        "execution_disposition": "launch",
        "reuse_source_run_id": "",
        "reuse_prediction_path": "",
        "reuse_prediction_sha256": "",
        "train_command": command,
    }
    row["row_fingerprint"] = row_fingerprint(row)
    return row


def transfer_base_identity(source_head, scope):
    identity = {
        "model_module": "BassetBranched",
        "graph_module": "CNNBassetBranchedScopedTransfer",
        "transfer_adapter_version": TRANSFER_ADAPTER_VERSION,
        "pretrained_artifact_sha256": TRANSFER_ARTIFACT_SHA256,
        "source_head": source_head,
        "unfreeze_scope": scope,
        "input_policy": TRANSFER_INPUT_POLICY,
        "input_len": 600,
        "conv1_channels": 300,
        "conv1_kernel_size": 19,
        "conv2_channels": 200,
        "conv2_kernel_size": 11,
        "conv3_channels": 200,
        "conv3_kernel_size": 7,
        "n_linear_layers": 1,
        "linear_channels": 1000,
        "linear_activation": "ReLU",
        "linear_dropout_p": 0.11625456877954289,
        "n_branched_layers": 3,
        "branched_channels": 140,
        "branched_activation": "ReLU",
        "branched_dropout_p": 0.5757068086404574,
        "n_outputs": 1,
        "use_batch_norm": True,
        "use_weight_norm": False,
        "optimizer": "AdamW",
        "head_lr": 5e-4,
        "backbone_lr": 1e-4,
        "weight_decay": 1e-4,
        "scheduler": None,
        "batch_size": 256,
        "frozen_epochs": 2,
        "max_epochs": 250,
        "stopping_patience": 40,
        "checkpoint_monitor": "val_pearson",
        "stopping_mode": "max",
        "precision": 32,
    }
    return identity


def transfer_launch_row(
    *,
    manifest_tag,
    source_head,
    scope,
    fold,
    rc_on,
    enhancer_data,
    transfer_split_path,
    transfer_split,
    transfer_split_sha,
    artifact_path,
    output_root,
):
    identity = transfer_base_identity(source_head, scope)
    digest = sha256_json(identity)
    base_config_id = "basecfg_" + digest
    analysis_lane = "enhancer_transfer_challenger"
    part_slug = "enhancer"
    pair_id, cell_id = pair_and_cell_ids(
        analysis_lane, part_slug, base_config_id, fold, rc_on
    )
    rc_mode = "on" if rc_on else "off"
    head_index = {"K562": 0, "HepG2": 1}[source_head]
    policy_id = f"enhancer_transfer_{source_head.lower()}_v1"
    root = (
        output_root
        / analysis_lane
        / part_slug
        / base_config_id
        / f"fold_{fold}"
        / f"rc_{rc_mode}"
    )
    project = lane_project(part_slug)
    group = f"{CAMPAIGN_ID}__stage2__{part_slug}__{analysis_lane}"
    run_name = (
        f"{manifest_tag}__{analysis_lane}__{source_head.lower()}__{scope}__"
        f"{digest[:16]}__fold{fold}__rc_{rc_mode}"
    )

    tokens = ["python", "train_wandb_log.py"]
    main_args = (
        ("data_module", "Lib1EnhancerDataModule"),
        ("model_module", "BassetBranched"),
        ("graph_module", "CNNBassetBranchedScopedTransfer"),
        ("artifact_path", str(root / "artifacts")),
        ("best_checkpoint_dir", str(root / "published_checkpoint_disabled")),
        ("artifact_retention", "none"),
        ("evaluate_test_after_fit", False),
        ("prediction_output_dir", str(root / "predictions")),
        ("provenance_output_dir", str(root / "provenance")),
        ("checkpoint_monitor", "val_pearson"),
        ("stopping_mode", "max"),
        ("stopping_patience", 40),
        ("logger_type", "wandb"),
        ("logger_project", project),
        ("wandb_entity", EXPECTED_ENTITY),
        ("wandb_group", group),
        ("wandb_job_type", "stage2_cell"),
        ("run_name", run_name),
        ("exact_run_name", True),
        ("model_seed", MODEL_SEED),
        ("campaign_id", CAMPAIGN_ID),
        ("campaign_stage", CAMPAIGN_STAGE),
        ("part_slug", part_slug),
        ("analysis_lane", analysis_lane),
        ("challenger_family", "enhancer_transfer"),
        ("policy_id", policy_id),
        ("config_origin", "historical_transfer_policy"),
        ("training_regime", "transfer"),
        ("cell_id", cell_id),
        ("rc_pair_id", pair_id),
        ("rc_mode", rc_mode),
        ("execution_disposition", "launch"),
        ("initialization", "malinois_pretrained"),
        ("source_head", source_head),
        ("unfreeze_scope", scope),
        ("input_policy", TRANSFER_INPUT_POLICY),
        ("pretrained_artifact_sha256", TRANSFER_ARTIFACT_SHA256),
        ("data_generation_id", enhancer_data["data_generation_id"]),
        ("dataset_sha256", enhancer_data["output_sha256"]),
        ("split_manifest_id", transfer_split["manifest_id"]),
        ("split_manifest_sha256", transfer_split_sha),
        ("development_fold", int(fold)),
        ("base_config_id", base_config_id),
        ("architecture", "BassetBranched"),
        ("loss_mode", "unweighted_mse"),
        ("target_definition", "log2(RNA_bc_counts_sum / DNA_bc_counts_sum)"),
        ("length_policy", TRANSFER_INPUT_POLICY),
    )
    for name, value in main_args:
        add_arg(tokens, name, value)
    add_many(tokens, "prediction_splits", ["val"])
    add_many(
        tokens,
        "wandb_tags",
        [
            CAMPAIGN_ID,
            CAMPAIGN_STAGE,
            part_slug,
            analysis_lane,
            source_head,
            scope,
            f"fold{fold}",
            f"rc_{rc_mode}",
            "seed1701",
            "unweighted_mse",
        ],
    )
    add_many(tokens, "epoch_eval_splits", ["train", "val"])

    data_args = (
        ("datafile_path", enhancer_data["output_path"]),
        ("sep", "tab"),
        ("sequence_column", enhancer_data["sequence_column"]),
        ("target_column", "log2_RNA_DNA"),
        ("barcode_column", enhancer_data["barcode_column"]),
        ("batch_size", 256),
        ("padded_seq_len", 600),
        ("left_flank", constants.MPRA_UPSTREAM),
        ("right_flank", constants.MPRA_DOWNSTREAM),
        ("padding_mode", "mpra_flank"),
        ("neutral_pad_char", "N"),
        ("num_workers", 8),
        ("normalize", True),
        ("split_manifest_path", str(Path(transfer_split_path).resolve())),
        ("split_fold", int(fold)),
        ("split_id_column", enhancer_data["split_id_column"]),
        ("expected_data_sha256", enhancer_data["output_sha256"]),
        ("expected_split_sha256", transfer_split_sha),
        ("test_min_barcodes", 8),
        ("train_min_barcodes", 1),
        ("train_size_frac", 1.0),
        ("train_sampling_mode", "random"),
        ("use_reverse_complements", bool(rc_on)),
        ("barcode_weighting", False),
    )
    for name, value in data_args:
        add_arg(tokens, name, value)

    model_args = (
        ("input_len", 600),
        ("conv1_channels", 300),
        ("conv1_kernel_size", 19),
        ("conv2_channels", 200),
        ("conv2_kernel_size", 11),
        ("conv3_channels", 200),
        ("conv3_kernel_size", 7),
        ("n_linear_layers", 1),
        ("linear_channels", 1000),
        ("linear_activation", "ReLU"),
        ("linear_dropout_p", 0.11625456877954289),
        ("n_branched_layers", 3),
        ("branched_channels", 140),
        ("branched_activation", "ReLU"),
        ("branched_dropout_p", 0.5757068086404574),
        ("n_outputs", 1),
        ("use_batch_norm", True),
        ("use_weight_norm", False),
        ("loss_criterion", "MSELoss"),
        ("reduction", "mean"),
    )
    for name, value in model_args:
        add_arg(tokens, name, value)

    graph_args = (
        ("parent_artifact", str(Path(artifact_path).resolve())),
        ("head_lr", 5e-4),
        ("backbone_lr", 1e-4),
        ("transfer_weight_decay", 1e-4),
        ("frozen_epochs", 2),
        ("transfer_adapter_version", TRANSFER_ADAPTER_VERSION),
        ("output_names", "log2_RNA_DNA"),
        ("log_per_output_metric_details", False),
        ("log_legacy_metric_aliases", False),
    )
    for name, value in graph_args:
        add_arg(tokens, name, value)

    trainer_args = (
        ("max_epochs", 250),
        ("min_epochs", 0),
        ("max_steps", -1),
        ("check_val_every_n_epoch", 1),
        ("overfit_batches", 0.0),
        ("fast_dev_run", False),
        ("num_sanity_val_steps", 2),
        ("reload_dataloaders_every_n_epochs", 0),
        ("detect_anomaly", False),
        ("enable_checkpointing", True),
        ("multiple_trainloader_mode", "max_size_cycle"),
        ("accelerator", "gpu"),
        ("devices", 1),
        ("precision", 32),
        ("default_root_dir", str(root)),
        ("enable_progress_bar", False),
    )
    for name, value in trainer_args:
        add_arg(tokens, name, value)

    command = shell_command(tokens)
    row = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "manifest_tag": manifest_tag,
        "analysis_lane": analysis_lane,
        "challenger_family": "enhancer_transfer",
        "config_origin": "historical_transfer_policy",
        "training_regime": "transfer",
        "part_slug": part_slug,
        "lane_id": f"enhancer__basset_branched_transfer__{source_head.lower()}__{scope}",
        "architecture": "BassetBranched",
        "architecture_slug": "basset_branched_transfer",
        "base_config_id": base_config_id,
        "base_config_sha256": digest,
        "base_identity": identity,
        "policy_id": policy_id,
        "initialization": "malinois_pretrained",
        "source_head": source_head,
        "source_head_index": head_index,
        "unfreeze_scope": scope,
        "input_policy": TRANSFER_INPUT_POLICY,
        "pretrained_artifact_path": str(Path(artifact_path).resolve()),
        "pretrained_artifact_sha256": TRANSFER_ARTIFACT_SHA256,
        "source_run_ids": [],
        "source_candidates": [
            {
                "candidate_kind": "historical_multiseed_transfer_policy",
                "analysis_notebook": str(
                    (
                        REPO_ROOT
                        / "tutorials/lib1_tasks/fine_tuning/enhancer_finetune_w_boda_pretrain"
                        / "may15_2026_hq8_multiseed_hpo_analysis.ipynb"
                    ).resolve()
                ),
            }
        ],
        "historical_validation_mean": None,
        "data_generation_id": enhancer_data["data_generation_id"],
        "dataset_path": enhancer_data["output_path"],
        "dataset_sha256": enhancer_data["output_sha256"],
        "split_manifest_id": transfer_split["manifest_id"],
        "split_manifest_path": str(Path(transfer_split_path).resolve()),
        "split_manifest_sha256": transfer_split_sha,
        "development_fold": int(fold),
        "model_seed": MODEL_SEED,
        "use_reverse_complements": bool(rc_on),
        "rc_mode": rc_mode,
        "rc_pair_id": pair_id,
        "cell_id": cell_id,
        "loss_mode": "unweighted_mse",
        "target_column": "log2_RNA_DNA",
        "target_definition": "log2(RNA_bc_counts_sum / DNA_bc_counts_sum)",
        "length_policy": TRANSFER_INPUT_POLICY,
        "artifact_retention": "none",
        "evaluate_test_after_fit": False,
        "epoch_eval_splits": ["train", "val"],
        "prediction_splits": ["val"],
        "wandb_entity": EXPECTED_ENTITY,
        "logger_project": project,
        "wandb_group": group,
        "wandb_job_type": "stage2_cell",
        "planned_run_name": run_name,
        "default_root_dir": str(root),
        "execution_disposition": "launch",
        "reuse_source_run_id": "",
        "reuse_prediction_path": "",
        "reuse_prediction_sha256": "",
        "train_command": command,
    }
    row["row_fingerprint"] = row_fingerprint(row)
    return row


def core_reuse_row(launch_template, selection_row, stage1_source):
    row = copy.deepcopy(launch_template)
    row["execution_disposition"] = "reuse_stage1"
    row["planned_run_name"] = stage1_source["planned_run_name"]
    row["reuse_source_run_id"] = selection_row["run_id"]
    row["reuse_prediction_path"] = selection_row["prediction_path"]
    row["reuse_prediction_sha256"] = sha256_file(selection_row["prediction_path"])
    row["reuse_source_row_fingerprint"] = selection_row["row_fingerprint"]
    row["train_command"] = ""
    row["default_root_dir"] = stage1_source["default_root_dir"]
    row["row_fingerprint"] = row_fingerprint(row)
    return row


def validate_manifests(analysis_rows, launch_rows):
    if len(analysis_rows) != EXPECTED_ANALYSIS_CELLS:
        raise AssertionError(f"Expected 660 analysis cells; found {len(analysis_rows)}")
    if len(launch_rows) != EXPECTED_LAUNCH_CELLS:
        raise AssertionError(f"Expected 610 launch cells; found {len(launch_rows)}")
    reuse = [row for row in analysis_rows if row["execution_disposition"] == "reuse_stage1"]
    if len(reuse) != EXPECTED_REUSE_CELLS:
        raise AssertionError(f"Expected 50 reuse cells; found {len(reuse)}")

    key_fields = (
        "analysis_lane",
        "part_slug",
        "base_config_id",
        "development_fold",
        "rc_mode",
    )
    keys = [tuple(row[field] for field in key_fields) for row in analysis_rows]
    if len(set(keys)) != len(keys):
        raise AssertionError("Analysis-cell key is not unique")
    if len({row["cell_id"] for row in analysis_rows}) != len(analysis_rows):
        raise AssertionError("cell_id is not unique")
    if len({row["planned_run_name"] for row in launch_rows}) != len(launch_rows):
        raise AssertionError("Launch run names are not unique")
    if len({row["row_fingerprint"] for row in launch_rows}) != len(launch_rows):
        raise AssertionError("Launch row fingerprints are not unique")

    lane_counts = Counter(row["analysis_lane"] for row in analysis_rows)
    if lane_counts != Counter(
        {
            "core_scratch": 500,
            "enhancer_transfer_challenger": 60,
            "utr3_utrbasset_challenger": 100,
        }
    ):
        raise AssertionError(f"Unexpected analysis lane counts: {lane_counts}")
    core_part_counts = Counter(
        row["part_slug"] for row in analysis_rows if row["analysis_lane"] == "core_scratch"
    )
    if set(core_part_counts.values()) != {100} or len(core_part_counts) != 5:
        raise AssertionError(f"Unexpected core part counts: {core_part_counts}")

    config_cells = Counter(
        (row["analysis_lane"], row["part_slug"], row["base_config_id"])
        for row in analysis_rows
    )
    if set(config_cells.values()) != {10}:
        raise AssertionError("Every analysis config must have exactly five folds x two RC cells")

    pair_groups = {}
    for row in analysis_rows:
        pair_groups.setdefault(row["rc_pair_id"], []).append(row)
        if row["loss_mode"] != "unweighted_mse" or row["model_seed"] != MODEL_SEED:
            raise AssertionError("Stage 2 fixed seed/loss contract changed")
        if row["evaluate_test_after_fit"] is not False:
            raise AssertionError("Audit/test evaluation must be disabled")
        if row["epoch_eval_splits"] != ["train", "val"] or row["prediction_splits"] != ["val"]:
            raise AssertionError("Stage 2 train/val history or val-prediction contract changed")
    for pair_id, pair in pair_groups.items():
        if len(pair) != 2 or {row["rc_mode"] for row in pair} != {"off", "on"}:
            raise AssertionError(f"Invalid RC pair {pair_id}")
        invariant = (
            "analysis_lane",
            "part_slug",
            "base_config_id",
            "development_fold",
            "model_seed",
            "loss_mode",
            "dataset_sha256",
            "split_manifest_sha256",
        )
        if any(pair[0][field] != pair[1][field] for field in invariant):
            raise AssertionError(f"RC pair invariant mismatch for {pair_id}")

    launch_ids = {row["cell_id"] for row in launch_rows}
    expected_launch_ids = {
        row["cell_id"]
        for row in analysis_rows
        if row["execution_disposition"] == "launch"
    }
    if launch_ids != expected_launch_ids:
        raise AssertionError("Launch manifest does not equal analysis launch subset")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-tag", default=MANIFEST_TAG)
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=HERE / "data_manifests/lib1_single_part_dedup_exact_v1.json",
    )
    parser.add_argument(
        "--split-index",
        type=Path,
        default=HERE / "data_manifests/lib1_dedup_exact_v1_split_manifests.json",
    )
    parser.add_argument(
        "--stage1-manifest",
        type=Path,
        default=HERE / "outputs/hpo_manifests/lib1_dedup_phase1_exact_replay_july2026__run_manifest.jsonl",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=(
            REPO_ROOT
            / "tutorials/lib1_tasks/pretrain_CRE_inhouse_data"
            / "dedup_phase1_rerun_july2026/outputs/stage2_candidate_selection_draft.csv"
        ),
    )
    parser.add_argument("--runs-csv", type=Path, default=HERE / "run_registry/runs.csv")
    parser.add_argument(
        "--transfer-artifact",
        type=Path,
        default=REPO_ROOT / "tutorials/malinois_artifacts__20211113_021200__287348.tar.gz",
    )
    parser.add_argument(
        "--transfer-split-manifest",
        type=Path,
        default=(
            HERE
            / "data_manifests/splits"
            / "lib1_enhancer_dedup_exact_v1_transfer_mpra600_split.json"
        ),
    )
    parser.add_argument("--outdir", type=Path, default=HERE / "outputs/hpo_manifests")
    return parser.parse_args()


def main():
    args = parse_args()
    if sha256_file(args.transfer_artifact) != TRANSFER_ARTIFACT_SHA256:
        raise ValueError("Canonical Malinois transfer artifact hash changed")

    data_manifest = json.loads(args.data_manifest.read_text())
    split_index = json.loads(args.split_index.read_text())
    stage1_rows = [
        row for row in read_jsonl(args.stage1_manifest)
        if row["run_kind"] == "exact_replay"
    ]
    stage1_by_key = {
        (row["part_slug"], row["base_config_id"]): row for row in stage1_rows
    }

    with args.selection.open(newline="") as handle:
        selection_rows = list(csv.DictReader(handle))
    if len(selection_rows) != 50:
        raise ValueError(f"Expected 50 selected core configs; found {len(selection_rows)}")
    if Counter(row["part_slug"] for row in selection_rows) != Counter(
        {"enhancer": 10, "promoter": 10, "intron": 10, "utr3": 10, "utr5": 10}
    ):
        raise ValueError("Core selection must contain 10 configs per part")

    canonical_enhancer_split = split_index["parts"]["enhancer"]["manifest_path"]
    transfer_split, transfer_split_sha = transfer_split_view(
        canonical_enhancer_split, args.transfer_split_manifest
    )

    utr_selected, utr_ranked, utr_selection_digest = resolve_utr3_utrbasset_candidates(
        HERE, args.runs_csv
    )
    selected_output_rows = []
    for row in utr_selected:
        selected_output_rows.append(
            {
                "selection_reason": row["selection_reason"],
                "historical_rank": row["historical_rank"],
                "source_run_id": row["source_run_ids"][0],
                "historical_validation_mean": row["historical_validation_mean"],
                "maximin_distance": row["maximin_distance"],
                "base_config_id": row["base_config_id"],
                "base_config_sha256": row["base_config_sha256"],
                "source_config_paths": row["source_config_paths"],
                "source_run_config_sha256": row["source_run_config_sha256"],
                "base_identity": row["base_identity"],
                "resolved_historical_config": row["resolved_historical_config"],
            }
        )

    prefix = args.outdir / args.manifest_tag
    write_jsonl(
        Path(str(prefix) + "__utr3_utrbassetvl_selected_configs.jsonl"),
        selected_output_rows,
    )
    write_csv(
        Path(str(prefix) + "__utr3_utrbassetvl_selected_configs.csv"),
        [
            {
                key: canonical_json(value) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            for row in selected_output_rows
        ],
    )

    output_root = HERE / "outputs/hpo_runs" / args.manifest_tag
    analysis_rows = []

    for selected in sorted(
        selection_rows,
        key=lambda row: (stage1.PART_ORDER[row["part_slug"]], int(row["selection_order"])),
    ):
        key = (selected["part_slug"], selected["base_config_id"])
        source = stage1_by_key.get(key)
        if source is None:
            raise ValueError(f"Selected core config is absent from Stage 1 manifest: {key}")
        if source["development_fold"] != 0 or source["use_reverse_complements"] is not False:
            raise ValueError(f"Stage 1 reuse source is not fold0/RC-off: {key}")
        if source["dataset_sha256"] != selected["dataset_sha256"]:
            raise ValueError(f"Selected core dataset SHA mismatch: {key}")
        if source["split_manifest_sha256"] != selected["split_manifest_sha256"]:
            raise ValueError(f"Selected core split SHA mismatch: {key}")
        if source["row_fingerprint"] != selected["row_fingerprint"]:
            raise ValueError(f"Selected core row fingerprint mismatch: {key}")
        if not Path(selected["prediction_path"]).is_file():
            raise FileNotFoundError(selected["prediction_path"])

        for fold in FOLDS:
            for rc_on in RC_MODES:
                launch_template = scratch_launch_row(
                    source,
                    manifest_tag=args.manifest_tag,
                    analysis_lane="core_scratch",
                    config_origin="stage1_selected",
                    challenger_family="none",
                    fold=fold,
                    rc_on=rc_on,
                    output_root=output_root,
                )
                if fold == 0 and not rc_on:
                    row = core_reuse_row(launch_template, selected, source)
                else:
                    row = launch_template
                analysis_rows.append(row)

    for base in utr_selected:
        pseudo = stage1.build_run_row(
            base,
            data_manifest,
            split_index,
            HERE,
            args.manifest_tag + "__utr3_source_template",
            "dedup",
        )
        for fold in FOLDS:
            for rc_on in RC_MODES:
                analysis_rows.append(
                    scratch_launch_row(
                        pseudo,
                        manifest_tag=args.manifest_tag,
                        analysis_lane="utr3_utrbasset_challenger",
                        config_origin="completed_june_hpo",
                        challenger_family="utr3_utrbasset",
                        fold=fold,
                        rc_on=rc_on,
                        output_root=output_root,
                    )
                )

    enhancer_data = data_manifest["datasets"]["enhancer"]["dedup"]
    for source_head in TRANSFER_HEADS:
        for scope in TRANSFER_SCOPES:
            for fold in FOLDS:
                for rc_on in RC_MODES:
                    analysis_rows.append(
                        transfer_launch_row(
                            manifest_tag=args.manifest_tag,
                            source_head=source_head,
                            scope=scope,
                            fold=fold,
                            rc_on=rc_on,
                            enhancer_data=enhancer_data,
                            transfer_split_path=args.transfer_split_manifest,
                            transfer_split=transfer_split,
                            transfer_split_sha=transfer_split_sha,
                            artifact_path=args.transfer_artifact,
                            output_root=output_root,
                        )
                    )

    analysis_rows.sort(
        key=lambda row: (
            ANALYSIS_LANES.index(row["analysis_lane"]),
            stage1.PART_ORDER[row["part_slug"]],
            row["base_config_id"],
            row["development_fold"],
            row["use_reverse_complements"],
        )
    )
    for index, row in enumerate(analysis_rows, 1):
        row["analysis_cell"] = index

    launch_rows = [
        copy.deepcopy(row)
        for row in analysis_rows
        if row["execution_disposition"] == "launch"
    ]
    for index, row in enumerate(launch_rows, 1):
        row["manifest_row"] = index

    validate_manifests(analysis_rows, launch_rows)

    analysis_path = Path(str(prefix) + "__analysis_manifest.jsonl")
    launch_path = Path(str(prefix) + "__run_manifest.jsonl")
    reuse_path = Path(str(prefix) + "__stage1_reuse_cells.jsonl")
    write_jsonl(analysis_path, analysis_rows)
    write_jsonl(launch_path, launch_rows)
    write_jsonl(
        reuse_path,
        [row for row in analysis_rows if row["execution_disposition"] == "reuse_stage1"],
    )
    write_csv(
        Path(str(prefix) + "__run_manifest.csv"),
        launch_rows,
        fieldnames=(
            "manifest_row",
            "analysis_cell",
            "cell_id",
            "rc_pair_id",
            "analysis_lane",
            "part_slug",
            "lane_id",
            "base_config_id",
            "policy_id",
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
        "schema_version": "lib1_dedup_stage2_manifest_v1",
        "manifest_tag": args.manifest_tag,
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "approved_N_enhancer_transfer_policies": 2,
        "approved_K_utr3_utrbasset_configs": 10,
        "analysis_cells": len(analysis_rows),
        "launch_cells": len(launch_rows),
        "stage1_reuse_cells": len(analysis_rows) - len(launch_rows),
        "analysis_counts_by_lane": dict(Counter(row["analysis_lane"] for row in analysis_rows)),
        "launch_counts_by_lane": dict(Counter(row["analysis_lane"] for row in launch_rows)),
        "core_counts_by_part": dict(
            Counter(
                row["part_slug"]
                for row in analysis_rows
                if row["analysis_lane"] == "core_scratch"
            )
        ),
        "utr3_utrbasset_selection_digest": utr_selection_digest,
        "utr3_utrbasset_selection_run_ids": list(EXPECTED_UTR_SELECTION_RUN_IDS),
        "transfer_heads": list(TRANSFER_HEADS),
        "transfer_scopes": list(TRANSFER_SCOPES),
        "transfer_adapter_version": TRANSFER_ADAPTER_VERSION,
        "transfer_artifact_path": str(args.transfer_artifact.resolve()),
        "transfer_artifact_sha256": TRANSFER_ARTIFACT_SHA256,
        "transfer_split_manifest_path": str(args.transfer_split_manifest.resolve()),
        "transfer_split_manifest_sha256": transfer_split_sha,
        "canonical_enhancer_split_manifest_path": str(Path(canonical_enhancer_split).resolve()),
        "canonical_enhancer_split_manifest_sha256": sha256_file(canonical_enhancer_split),
        "fixed_policy": {
            "development_folds": list(FOLDS),
            "rc_modes": ["off", "on"],
            "model_seed": MODEL_SEED,
            "loss_mode": "unweighted_mse",
            "checkpoint_monitor": "val_pearson",
            "audit_loader": False,
            "evaluate_test_after_fit": False,
            "epoch_eval_splits": ["train", "val"],
            "prediction_splits": ["val"],
            "artifact_retention": "none",
        },
        "analysis_manifest_path": str(analysis_path.resolve()),
        "analysis_manifest_sha256": sha256_file(analysis_path),
        "run_manifest_path": str(launch_path.resolve()),
        "run_manifest_sha256": sha256_file(launch_path),
        "stage1_reuse_manifest_path": str(reuse_path.resolve()),
        "stage1_reuse_manifest_sha256": sha256_file(reuse_path),
    }
    write_json(Path(str(prefix) + "__summary.json"), summary)

    print("Generated frozen Lib1 dedup Stage 2 manifests")
    print(f"  analysis cells: {len(analysis_rows)}")
    print(f"  Stage 1 reused cells: {EXPECTED_REUSE_CELLS}")
    print(f"  new launch cells: {len(launch_rows)}")
    print(f"  lanes: {json.dumps(summary['analysis_counts_by_lane'], sort_keys=True)}")
    print(f"  UTRBasset selection digest: {utr_selection_digest}")
    print(f"  analysis manifest: {analysis_path}")
    print(f"  launch manifest: {launch_path}")
    print(f"  launch SHA256: {summary['run_manifest_sha256']}")


if __name__ == "__main__":
    main()
