#!/usr/bin/env python3
"""
Enhanced training script with Weights & Biases integration for BODA2.
Supports hyperparameter sweeps across different data modules, model architectures,
and training configurations.
"""
import os
import sys
import re
import csv
import json
import time
import shutil
import socket
import argparse
import tarfile
import tempfile
import random
import subprocess
import ast
import fcntl
import hashlib
import shlex
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

import torch
from lightning.pytorch import Trainer, LightningDataModule
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import boda
from boda.common import utils
from boda.common.utils import unpack_artifact, model_fn

import hypertune
import wandb
from lightning.pytorch.loggers import WandbLogger

#####################
#  Provenance defs  #
#####################

WANDB_HISTORY_CONTRACT_VERSION = "canonical_metrics_v2"
WANDB_HISTORY_CANARY_KEY = "wandb_history_canary"
LIB1_DEDUP_CAMPAIGN_ID = "lib1_dedup_phase1_rerun_july2026"
LIB1_DEDUP_WANDB_ENTITY = "minhangxu1998-baylor-college-of-medicine"
LIB1_DEDUP_TARGETED_UTR3_PROJECT = (
    "utr3__bashor_in_house__dedup_exact_v1__targeted_hpo_development"
)
LIB1_DEDUP_FINAL_REFIT_MANIFEST_SHA256 = (
    "83ec532cf84e83d3477f2e6e8c716a04284fcc43b7d7c4426338a8b0f093582c"
)
LIB1_DEDUP_STAGE4_MANIFEST_SHA256 = (
    "dd6abda4726846f482536a235093b2ed9aa5a36b12591613c400601dcb27a84a"
)
LIB1_DEDUP_STAGE4_MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "outputs/hpo_manifests/"
    "lib1_dedup_stage4_downsampling_july2026__dry_run_manifest.jsonl"
)
LIB1_DEDUP_STAGE4_REGISTRY_PATH = (
    Path(__file__).resolve().parent
    / "outputs/hpo_runs/status/lib1_dedup_stage4_downsampling_july2026/"
    "stage4_runs.csv"
)
CANONICAL_WANDB_HISTORY_METRICS = tuple(
    f"{split}_{metric}"
    for split in ("train", "val", "test")
    for metric in ("loss", "mse", "pearson", "pearson_r2", "spearman", "cod_r2")
)

# Canonical column order for the per-run manifest written to
# `src/learn/run_registry/runs.csv`. Extend intentionally; never reorder.
RUNS_CSV_COLUMNS = [
    "timestamp",
    "run_id",
    "run_name",
    "run_url",
    "wandb_entity",
    "wandb_project",
    "wandb_sweep_id",
    "wandb_sweep_path",
    "logger_project",
    "task_family",
    "target_family",
    "comparison_group",
    "launch_script",
    "config_path",
    "data_module",
    "model_module",
    "graph_module",
    "checkpoint_monitor",
    "best_epoch",
    "best_metric_name",
    "best_metric_value",
    "val_loss",
    "val_r2",
    "val_pearson",
    "val_spearman",
    "test_loss",
    "test_r2",
    "test_pearson",
    "test_spearman",
    "train_loss",
    "train_r2",
    "train_pearson",
    "train_spearman",
    "artifact_path",
    "status",
    "hostname",
    "git_commit",
    "notes",
    "val_pearson_r2",
    "val_cod_r2",
    "val_mse",
    "test_pearson_r2",
    "test_cod_r2",
    "test_mse",
    "train_pearson_r2",
    "train_cod_r2",
    "train_mse",
    "campaign_id",
    "campaign_stage",
    "part_slug",
    "analysis_lane",
    "challenger_family",
    "policy_id",
    "config_origin",
    "training_regime",
    "cell_id",
    "rc_pair_id",
    "loss_pair_id",
    "source_unweighted_cell_id",
    "rc_mode",
    "execution_disposition",
    "initialization",
    "source_head",
    "unfreeze_scope",
    "input_policy",
    "pretrained_artifact_sha256",
    "data_generation_id",
    "dataset_sha256",
    "split_manifest_id",
    "split_manifest_sha256",
    "development_fold",
    "base_config_id",
    "source_run_ids",
    "architecture",
    "model_seed",
    "loss_mode",
    "target_definition",
    "length_policy",
    "artifact_retention",
    "prediction_path",
    "train_row_id_hash",
    "val_row_id_hash",
    "audit_row_id_hash",
    "normalization_row_id_hash",
    "selected_row_hash",
    # Appended Stage 4 launch/reconciliation evidence. Keep all prior columns
    # in their historical order so generic registries remain migration-safe.
    "config_manifest_sha256",
    "manifest_row",
    "manifest_row_fingerprint",
    "runtime_argv_sha256",
    "resolved_arguments_sha256",
    "run_registry_path",
    "optimizer_steps",
]


def _safe_wandb_summary_get(key: str) -> Any:
    """Read a scalar from the active W&B run's summary, returning None on failure."""
    try:
        if wandb.run is None:
            return None
        value = wandb.run.summary.get(key)
        return value
    except Exception:
        return None


def _coerce_scalar(value: Any) -> Any:
    """Convert tensors / numpy scalars / W&B summary objects to plain Python scalars."""
    if value is None:
        return None
    try:
        if hasattr(value, "item") and callable(value.item):
            return value.item()
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return str(value)
        except Exception:
            return None


def _configure_wandb_history_contract(splits=None) -> None:
    """Define canonical W&B history metrics and emit a cloud-history canary row."""
    if wandb.run is None:
        raise RuntimeError(
            "W&B logger_type=wandb was requested, but wandb.run is not initialized. "
            "Check wandb login/API key, WANDB_MODE, entity/project access, and network connectivity."
        )

    wandb.define_metric("trainer/global_step")
    wandb.define_metric("epoch", step_metric="trainer/global_step")
    wandb.define_metric(WANDB_HISTORY_CANARY_KEY, step_metric="trainer/global_step")
    requested_splits = tuple(dict.fromkeys(splits or ("train", "val", "test")))
    unknown = set(requested_splits) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"Unknown W&B history contract splits: {sorted(unknown)}")
    history_metrics = (
        f"{split}_{metric}"
        for split in requested_splits
        for metric in ("loss", "mse", "pearson", "pearson_r2", "spearman", "cod_r2")
    )
    for key in history_metrics:
        wandb.define_metric(key, step_metric="trainer/global_step")

    wandb.run.summary["wandb_history_contract"] = WANDB_HISTORY_CONTRACT_VERSION
    wandb.log(
        {
            "trainer/global_step": 0,
            "epoch": 0,
            WANDB_HISTORY_CANARY_KEY: 1.0,
        },
        commit=True,
    )


def _resolve_git_commit() -> Optional[str]:
    """Best-effort git commit for the boda2_EU checkout. Silent on failure."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(
            ["git", "-C", here, "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        sha = result.stdout.strip()
        return sha or None
    except Exception:
        return None


def _extract_wandb_identity() -> Dict[str, Optional[str]]:
    """
    Collect authoritative W&B identifiers as seen at runtime.

    When running inside a `wandb agent`, the agent injects `WANDB_RUN_ID`,
    `WANDB_SWEEP_ID`, and `WANDB_PROJECT`. We also consult `wandb.run` directly
    because the logger may have upgraded to a live run by the time we ask.
    """
    identity: Dict[str, Optional[str]] = {
        "run_id": None,
        "run_name": None,
        "run_url": None,
        "entity": None,
        "project": None,
        "sweep_id": None,
        "sweep_path": None,
    }

    if wandb.run is not None:
        try:
            identity["run_id"] = wandb.run.id
            identity["run_name"] = wandb.run.name
            identity["entity"] = wandb.run.entity
            identity["project"] = wandb.run.project
            identity["run_url"] = wandb.run.get_url()
            sweep_id_attr = getattr(wandb.run, "sweep_id", None)
            if sweep_id_attr:
                identity["sweep_id"] = sweep_id_attr
        except Exception:
            pass

    # Fall back to environment for anything still missing.
    identity["run_id"] = identity["run_id"] or os.environ.get("WANDB_RUN_ID")
    identity["entity"] = identity["entity"] or os.environ.get("WANDB_ENTITY") or os.environ.get("BODA_WANDB_ENTITY")
    identity["project"] = identity["project"] or os.environ.get("WANDB_PROJECT") or os.environ.get("BODA_WANDB_PROJECT")
    identity["sweep_id"] = identity["sweep_id"] or os.environ.get("WANDB_SWEEP_ID") or os.environ.get("BODA_SWEEP_ID")
    if identity["entity"] and identity["project"] and identity["sweep_id"]:
        identity["sweep_path"] = f"{identity['entity']}/{identity['project']}/{identity['sweep_id']}"
    else:
        identity["sweep_path"] = os.environ.get("BODA_SWEEP_PATH")

    return identity


def build_provenance_record(
    args: Dict[str, Any],
    use_callbacks: Dict[str, Any],
    artifact_path: Optional[str] = None,
    status: str = "completed",
) -> Dict[str, Any]:
    """
    Build a single flat dict of provenance fields for both provenance.json
    inside the tarball and a runs.csv row. Always returns every column in
    `RUNS_CSV_COLUMNS`, using empty strings for missing values.
    """
    identity = _extract_wandb_identity()
    main_args = args.get("Main args")
    main = vars(main_args) if isinstance(main_args, argparse.Namespace) else {}

    best_epoch = None
    best_metric_name = None
    best_metric_value = None
    mc = use_callbacks.get("model_checkpoint") if use_callbacks else None
    if mc is not None:
        try:
            best_metric_name = getattr(mc, "monitor", None)
            raw_metric = getattr(mc, "best_model_score", None)
            best_metric_value = _coerce_scalar(raw_metric)
            best_path = getattr(mc, "best_model_path", "") or ""
            m = re.search(r"epoch=(\d+)", best_path)
            if m:
                best_epoch = int(m.group(1))
        except Exception:
            pass

    def _get(key: str) -> Any:
        return _coerce_scalar(_safe_wandb_summary_get(key))

    def _get_first(*keys: str) -> Any:
        for key in keys:
            value = _get(key)
            if value is not None:
                return value
        return None

    def _squared_pearson_fallback(prefix: str) -> Any:
        explicit_keys = [
            f"{prefix}_pearson_r2",
            f"epoch_end_{prefix}_pearson_r2",
            f"{prefix}_r2",
            f"epoch_end_{prefix}_r2",
        ]
        if prefix == "val":
            explicit_keys.append("val_r2_score")
        explicit = _get_first(*explicit_keys)
        if explicit is not None:
            return explicit
        pearson = _get(f"{prefix}_pearson")
        if pearson is None:
            pearson = _get(f"epoch_end_{prefix}_pearson")
        if pearson is None:
            return None
        try:
            pearson = float(pearson)
            return pearson * pearson
        except Exception:
            return None

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": identity.get("run_id") or "",
        "run_name": identity.get("run_name") or main.get("run_name", "") or "",
        "run_url": identity.get("run_url") or "",
        "wandb_entity": identity.get("entity") or "",
        "wandb_project": identity.get("project") or "",
        "wandb_sweep_id": identity.get("sweep_id") or "",
        "wandb_sweep_path": identity.get("sweep_path") or "",
        "logger_project": main.get("logger_project", "") or "",
        "task_family": os.environ.get("BODA_TASK_FAMILY", ""),
        "target_family": os.environ.get("BODA_TARGET_FAMILY", ""),
        "comparison_group": os.environ.get("BODA_COMPARISON_GROUP", ""),
        "launch_script": os.environ.get("BODA_LAUNCH_SCRIPT", ""),
        "config_path": os.environ.get("BODA_CONFIG_PATH", ""),
        "config_manifest_sha256": os.environ.get("BODA_CONFIG_MANIFEST_SHA256", ""),
        "manifest_row": os.environ.get("BODA_MANIFEST_ROW", ""),
        "manifest_row_fingerprint": os.environ.get("BODA_MANIFEST_ROW_FINGERPRINT", ""),
        "runtime_argv_sha256": os.environ.get("BODA_RUNTIME_ARGV_SHA256", ""),
        "resolved_arguments_sha256": "",
        "run_registry_path": os.environ.get("BODA_RUNS_CSV", ""),
        "data_module": main.get("data_module", "") or "",
        "model_module": main.get("model_module", "") or "",
        "graph_module": main.get("graph_module", "") or "",
        "checkpoint_monitor": main.get("checkpoint_monitor", "") or "",
        "best_epoch": best_epoch if best_epoch is not None else "",
        "best_metric_name": best_metric_name or "",
        "best_metric_value": best_metric_value if best_metric_value is not None else "",
        "optimizer_steps": "",
        "val_loss": _get("val_loss"),
        "val_r2": _squared_pearson_fallback("val"),
        "val_pearson": _get("val_pearson") if _get("val_pearson") is not None else _get("epoch_end_val_pearson"),
        "val_spearman": _get("val_spearman") if _get("val_spearman") is not None else _get("epoch_end_val_spearman"),
        "test_loss": _get("test_loss"),
        "test_r2": _squared_pearson_fallback("test"),
        "test_pearson": _get("test_pearson") if _get("test_pearson") is not None else _get("epoch_end_test_pearson"),
        "test_spearman": _get("test_spearman") if _get("test_spearman") is not None else _get("epoch_end_test_spearman"),
        "train_loss": _get("train_loss"),
        "train_r2": _squared_pearson_fallback("train"),
        "train_pearson": _get("train_pearson") if _get("train_pearson") is not None else _get("epoch_end_train_pearson"),
        "train_spearman": _get("train_spearman") if _get("train_spearman") is not None else _get("epoch_end_train_spearman"),
        "artifact_path": artifact_path or "",
        "status": status,
        "hostname": socket.gethostname(),
        "git_commit": _resolve_git_commit() or "",
        "notes": os.environ.get("BODA_LAUNCH_NOTES", os.environ.get("LAUNCH_NOTES", "")),
        "val_pearson_r2": _squared_pearson_fallback("val"),
        "val_cod_r2": _get_first("val_cod_r2", "epoch_end_val_cod_r2"),
        "val_mse": _get_first("val_mse", "epoch_end_val_mse"),
        "test_pearson_r2": _squared_pearson_fallback("test"),
        "test_cod_r2": _get_first("test_cod_r2", "epoch_end_test_cod_r2"),
        "test_mse": _get_first("test_mse", "epoch_end_test_mse"),
        "train_pearson_r2": _squared_pearson_fallback("train"),
        "train_cod_r2": _get_first("train_cod_r2", "epoch_end_train_cod_r2"),
        "train_mse": _get_first("train_mse", "epoch_end_train_mse"),
        "campaign_id": main.get("campaign_id", "") or "",
        "campaign_stage": main.get("campaign_stage", "") or "",
        "part_slug": main.get("part_slug", "") or "",
        "analysis_lane": main.get("analysis_lane", "") or "",
        "challenger_family": main.get("challenger_family", "") or "",
        "policy_id": main.get("policy_id", "") or "",
        "config_origin": main.get("config_origin", "") or "",
        "training_regime": main.get("training_regime", "") or "",
        "cell_id": main.get("cell_id", "") or "",
        "rc_pair_id": main.get("rc_pair_id", "") or "",
        "loss_pair_id": main.get("loss_pair_id", "") or "",
        "source_unweighted_cell_id": main.get("source_unweighted_cell_id", "") or "",
        "rc_mode": main.get("rc_mode", "") or "",
        "execution_disposition": main.get("execution_disposition", "") or "",
        "initialization": main.get("initialization", "") or "",
        "source_head": main.get("source_head", "") or "",
        "unfreeze_scope": main.get("unfreeze_scope", "") or "",
        "input_policy": main.get("input_policy", "") or "",
        "pretrained_artifact_sha256": main.get("pretrained_artifact_sha256", "") or "",
        "data_generation_id": main.get("data_generation_id", "") or "",
        "dataset_sha256": main.get("dataset_sha256", "") or "",
        "split_manifest_id": main.get("split_manifest_id", "") or "",
        "split_manifest_sha256": main.get("split_manifest_sha256", "") or "",
        "development_fold": main.get("development_fold", "") if main.get("development_fold") is not None else "",
        "base_config_id": main.get("base_config_id", "") or "",
        "source_run_ids": json.dumps(main.get("source_run_ids", []) or [], sort_keys=True),
        "architecture": main.get("architecture", "") or "",
        "model_seed": main.get("model_seed", "") if main.get("model_seed") is not None else "",
        "loss_mode": main.get("loss_mode", "") or "",
        "target_definition": main.get("target_definition", "") or "",
        "length_policy": main.get("length_policy", "") or "",
        "artifact_retention": main.get("artifact_retention", "") or "",
        "prediction_path": "",
        "train_row_id_hash": "",
        "val_row_id_hash": "",
        "audit_row_id_hash": "",
        "normalization_row_id_hash": "",
        "selected_row_hash": "",
    }
    # Replace None with "" for CSV-friendliness.
    for k, v in list(record.items()):
        if v is None:
            record[k] = ""
    return record


def _ensure_runs_csv_columns(target_path: str) -> List[str]:
    """
    Return the fieldnames to use when appending a runs.csv row.

    If an existing manifest was written by an older script, rewrite only the
    header/rows needed to add newly introduced columns before appending.
    """
    if not os.path.isfile(target_path):
        return list(RUNS_CSV_COLUMNS)

    with open(target_path, newline="") as fh:
        reader = csv.DictReader(fh)
        existing_columns = list(reader.fieldnames or [])
        existing_rows = list(reader)

    if not existing_columns:
        return list(RUNS_CSV_COLUMNS)

    missing_columns = [col for col in RUNS_CSV_COLUMNS if col not in existing_columns]
    if not missing_columns:
        return existing_columns

    extra_columns = [col for col in existing_columns if col not in RUNS_CSV_COLUMNS]
    migrated_columns = list(RUNS_CSV_COLUMNS) + extra_columns
    tmp_path = target_path + ".tmp"
    with open(tmp_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=migrated_columns,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({col: row.get(col, "") for col in migrated_columns})
    os.replace(tmp_path, target_path)
    return migrated_columns


def append_runs_csv_row(record: Dict[str, Any]) -> Optional[str]:
    """
    Append a per-run row to `run_registry/runs.csv`. The destination can be
    overridden via `BODA_RUNS_CSV`; otherwise we write to a sibling of this
    script at `../run_registry/runs.csv`.

    Returns the path written to, or None on failure.
    """
    target_env = os.environ.get("BODA_RUNS_CSV")
    if target_env:
        target_path = target_env
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        target_path = os.path.join(here, "run_registry", "runs.csv")

    target_dir = os.path.dirname(target_path)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
    try:
        # Multiple manifest workers can finish together. Hold one lock across
        # both schema migration and append so no worker can overwrite another
        # worker's row (or race on the shared migration temporary file).
        lock_path = target_path + ".lock"
        with open(lock_path, "a+") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            is_new = (not os.path.isfile(target_path)) or os.path.getsize(target_path) == 0
            fieldnames = _ensure_runs_csv_columns(target_path)
            with open(target_path, "a", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=fieldnames,
                    extrasaction="ignore",
                    lineterminator="\n",
                )
                if is_new:
                    writer.writeheader()
                writer.writerow({col: record.get(col, "") for col in fieldnames})
                fh.flush()
                os.fsync(fh.fileno())
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        return target_path
    except Exception as exc:
        print(f"WARN: failed to append runs.csv row: {exc}", file=sys.stderr)
        return None

#####################
# Helper Functions  #
#####################

def convert_to_list(param):
    """
    Convert a parameter into a list.
    
    This function handles various input formats:
    - Lists of strings that might be representations of lists
    - Space-separated strings
    - Actual list objects
    
    Args:
        param: The parameter to convert (string, list, or other)
        
    Returns:
        list: The converted parameter as a list
    """
    if isinstance(param, list):
        flattened = []
        for item in param:
            if isinstance(item, str):
                item = item.strip()
                if item.startswith('[') and item.endswith(']'):
                    try:
                        parsed = ast.literal_eval(item)
                        if isinstance(parsed, list):
                            flattened.extend(parsed)
                        else:
                            flattened.append(item)
                    except Exception:
                        flattened.append(item)
                else:
                    flattened.append(item)
            else:
                flattened.append(item)
        return flattened
    elif isinstance(param, str):
        param = param.strip()
        if param.startswith('[') and param.endswith(']'):
            try:
                return ast.literal_eval(param)
            except Exception:
                return param.split()
        else:
            return param.split()
    else:
        return param

def _coerce_split_list(value):
    """Normalize CLI/W&B split lists such as `train val test` or `[train,val]`."""
    parsed = convert_to_list(value)
    if parsed is None:
        return ["val"]
    if isinstance(parsed, str):
        parsed = parsed.replace(",", " ").split()
    if not isinstance(parsed, list):
        parsed = [parsed]
    splits = [str(item).strip() for item in parsed if str(item).strip()]
    return splits or ["val"]

def configure_epoch_eval_dataloaders(data, graph, split_names):
    """
    Optionally evaluate named data splits every validation epoch.

    Lightning treats extra validation dataloaders as diagnostic loaders. The
    graph maps them back to canonical prefixes (`train_*`, `val_*`, `test_*`)
    and checkpointing remains controlled by whatever `checkpoint_monitor`
    names, typically a validation metric such as `val_pearson`.
    """
    split_names = _coerce_split_list(split_names)
    valid_splits = {"train", "val", "test"}
    unknown = [split for split in split_names if split not in valid_splits]
    if unknown:
        raise ValueError(f"Unknown epoch_eval_splits values: {unknown}. Use train, val, and/or test.")

    if split_names == ["val"]:
        return split_names

    loader_fns = {}
    for split in split_names:
        method_name = "train_eval_dataloader" if split == "train" and hasattr(data, "train_eval_dataloader") else f"{split}_dataloader"
        if not hasattr(data, method_name):
            raise ValueError(f"{data.__class__.__name__} has no {method_name} for epoch diagnostics.")
        loader_fns[split] = getattr(data, method_name)

    def diagnostic_val_dataloader():
        return [loader_fns[split]() for split in split_names]

    # Preserve the canonical split loaders for best-checkpoint prediction
    # export after val_dataloader is replaced by the diagnostic multi-loader.
    data._boda_epoch_eval_loader_fns = loader_fns
    data.val_dataloader = diagnostic_val_dataloader
    graph.validation_loader_names = split_names
    return split_names

def _normalize_optional_name(value):
    """Treat string sentinels like 'None' as Python None."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"none", "null"}:
            return None
    return value

def _leftovers_are_ignorable_scheduler_args(leftover_args, scheduler_name):
    """
    W&B can still pass scheduler-specific params even when a sweep samples
    `scheduler="None"`. Accept those leftovers so argparse doesn't fail before
    the training script can intentionally ignore them.
    """
    if not leftover_args or _normalize_optional_name(scheduler_name) is not None:
        return False

    scheduler_flags = {
        "--step_size", "--gamma", "--last_epoch",
        "--T_max", "--scheduler_mode", "--factor", "--patience",
        "--threshold", "--threshold_mode", "--cooldown", "--min_lr",
        "--base_lr", "--max_lr", "--step_size_up", "--step_size_down",
        "--scale_mode", "--cycle_momentum", "--base_momentum",
        "--max_momentum", "--total_steps", "--epochs", "--steps_per_epoch",
        "--pct_start", "--anneal_strategy", "--div_factor",
        "--final_div_factor", "--three_phase", "--T_0", "--T_mult",
        "--eta_min",
    }

    idx = 0
    while idx < len(leftover_args):
        token = leftover_args[idx]
        if not token.startswith("--"):
            return False

        flag = token.split("=", 1)[0]
        if flag not in scheduler_flags:
            return False

        if "=" not in token and (idx + 1) < len(leftover_args) and not leftover_args[idx + 1].startswith("--"):
            idx += 2
        else:
            idx += 1

    return True

def set_best(my_model, callbacks):
    """
    Set the model to the best checkpoint based on the monitored metric.
    
    Args:
        my_model: The model to update
        callbacks: Dictionary of callbacks including 'model_checkpoint'
        
    Returns:
        The updated model with weights from the best checkpoint
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        try:
            best_path = callbacks['model_checkpoint'].best_model_path
            get_epoch = re.search('epoch=(\d*)', best_path).group(1)
            if 'gs://' in best_path:
                subprocess.call(['gsutil','cp',best_path,tmpdirname])
                best_path = os.path.join(tmpdirname, os.path.basename(best_path))
            print(f'Best model stashed at: {best_path}', file=sys.stderr)
            print(f'Exists: {os.path.isfile(best_path)}', file=sys.stderr)
            ckpt = torch.load(best_path)
            my_model.load_state_dict(ckpt['state_dict'])
            print(f'Setting model from epoch: {get_epoch}', file=sys.stderr)
        except KeyError:
            print('Setting most recent model', file=sys.stderr)
    return my_model


def publish_best_checkpoint_model(local_dir, final_artifact_path, provenance_record,
                                  use_callbacks=None, args=None):
    """
    Optionally publish each run's best model bundle to a cleaner per-project
    directory for human browsing and downstream handoff.

    The full archive in `local_artifacts` remains the canonical portable
    artifact. This publisher writes a small pointer/metadata layer into:

        <best_checkpoint_dir>/<run_id>/

    so the noisy Lightning/W&B run directories do not need to be the first
    place humans look, without duplicating large model payloads.
    """
    if args is None:
        return None

    main_args = args.get('Main args')
    publish_root = getattr(main_args, 'best_checkpoint_dir', None)
    if not publish_root:
        return None

    publish_root = os.path.abspath(os.path.expanduser(publish_root))
    run_id = provenance_record.get('run_id') or f"norun_{time.strftime('%Y%m%d_%H%M%S')}"
    publish_dir = os.path.join(publish_root, run_id)
    os.makedirs(publish_dir, exist_ok=True)

    linked_files = {}

    def _link_if_exists(src, dest_name):
        if not src or not os.path.isfile(src):
            return
        dest = os.path.join(publish_dir, dest_name)
        try:
            if os.path.lexists(dest):
                os.unlink(dest)
            os.symlink(os.path.abspath(src), dest)
            linked_files[dest_name] = dest
        except Exception as exc:
            print(f"WARN: failed to create symlink {dest} -> {src}: {exc}", file=sys.stderr)

    # Keep the human-facing mirror lightweight. The canonical tarball contains
    # artifacts/torch_checkpoint.pt and artifacts/provenance.json.
    if final_artifact_path and not str(final_artifact_path).startswith('gs://'):
        _link_if_exists(final_artifact_path, 'model_artifacts.tar.gz')

    best_model_path = ""
    mc = use_callbacks.get('model_checkpoint') if use_callbacks else None
    if mc is not None:
        best_model_path = getattr(mc, 'best_model_path', "") or ""
        if best_model_path.startswith('gs://'):
            best_model_path = ""

    published_provenance = dict(provenance_record)
    if final_artifact_path:
        published_provenance['artifact_path'] = final_artifact_path
    with open(os.path.join(publish_dir, 'provenance.json'), 'w') as fh:
        json.dump(published_provenance, fh, indent=2, default=str)

    selection = {
        'run_id': run_id,
        'wandb_project': provenance_record.get('wandb_project', ''),
        'task_family': provenance_record.get('task_family', ''),
        'target_family': provenance_record.get('target_family', ''),
        'metric_name': provenance_record.get('best_metric_name', ''),
        'metric_value': provenance_record.get('best_metric_value', ''),
        'best_epoch': provenance_record.get('best_epoch', ''),
        'source_lightning_checkpoint': best_model_path,
        'source_artifact_path': final_artifact_path or '',
        'linked_files': linked_files,
        'artifact_contents': ['artifacts/torch_checkpoint.pt', 'artifacts/provenance.json'],
    }
    with open(os.path.join(publish_dir, 'selection.json'), 'w') as fh:
        json.dump(selection, fh, indent=2, default=str)

    latest_link = os.path.join(publish_root, 'latest')
    try:
        if os.path.islink(latest_link) or os.path.isfile(latest_link):
            os.unlink(latest_link)
        if not os.path.exists(latest_link):
            os.symlink(run_id, latest_link)
    except Exception:
        with open(os.path.join(publish_root, 'latest_run.txt'), 'w') as fh:
            fh.write(f"{run_id}\n")

    print(f"Published best checkpoint model to {publish_dir}")
    if wandb.run is not None:
        wandb.run.summary["best_checkpoint_publish_dir"] = publish_dir

    return publish_dir


def prune_lightning_checkpoints(use_callbacks=None, keep=False, extra_checkpoint_dirs=None):
    """Remove transient Lightning .ckpt files after the portable artifact is saved."""
    if keep or not use_callbacks:
        return []

    removed = []
    mc = use_callbacks.get('model_checkpoint')
    best_model_path = getattr(mc, 'best_model_path', "") if mc is not None else ""
    checkpoint_dirs = {
        os.path.abspath(str(path))
        for path in (extra_checkpoint_dirs or [])
        if path and not str(path).startswith('gs://')
    }
    if best_model_path and not best_model_path.startswith('gs://'):
        checkpoint_dirs.add(os.path.dirname(best_model_path))
        if os.path.isfile(best_model_path):
            try:
                os.remove(best_model_path)
                removed.append(best_model_path)
            except OSError as exc:
                print(f"WARN: failed to remove transient checkpoint {best_model_path}: {exc}", file=sys.stderr)

    for checkpoint_dir in checkpoint_dirs:
        # A retried manifest row can leave an older versioned checkpoint in
        # the same transient directory. Retention=none must remove every ckpt,
        # not only the callback's current best path.
        try:
            for filename in os.listdir(checkpoint_dir):
                candidate = os.path.join(checkpoint_dir, filename)
                if filename.endswith('.ckpt') and os.path.isfile(candidate):
                    os.remove(candidate)
                    if candidate not in removed:
                        removed.append(candidate)
        except OSError as exc:
            print(f"WARN: failed to sweep transient checkpoints in {checkpoint_dir}: {exc}", file=sys.stderr)
        try:
            if os.path.isdir(checkpoint_dir) and not os.listdir(checkpoint_dir):
                os.rmdir(checkpoint_dir)
        except OSError:
            pass
    return removed


def save_model(data_module, model_module, graph_module, model, trainer, args,
               use_callbacks=None, provenance_record=None):
    """
    Save the model and associated artifacts.

    Persists:
        - `torch_checkpoint.pt` with hparams + state_dict (unchanged).
        - `provenance.json` with full W&B/launch/run identifiers and scalar
          metrics so the tarball is self-describing.
        - `.tar.gz` whose filename encodes the W&B project + run_id, making
          the on-disk artifact independently resolvable back to a W&B run.

    Args:
        data_module: The data module class
        model_module: The model module class
        graph_module: The graph module class
        model: The trained model
        trainer: The PyTorch Lightning trainer
        args: Dictionary of input arguments
        use_callbacks: Callback dict (for best-checkpoint metadata). Optional.
        provenance_record: Pre-built provenance dict. If None, a minimal one is built here.

    Returns:
        The absolute path to the saved .tar.gz artifact (local path), or None on failure.
    """
    local_dir = args['pl.Trainer'].default_root_dir
    if local_dir is None:
        local_dir = getattr(trainer, 'default_root_dir', None)
    if local_dir is None:
        local_dir = '/tmp/output/artifacts'
    os.makedirs(local_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    random_tag = random.randint(100000, 999999)

    save_dict = {
        'data_module': data_module.__name__,
        'data_hparams': data_module.process_args(args),
        'model_module': model_module.__name__,
        'model_hparams': model_module.process_args(args),
        'graph_module': graph_module.__name__,
        'graph_hparams': graph_module.process_args(args),
        'model_state_dict': model.state_dict(),
        'timestamp': timestamp,
        'random_tag': random_tag,
    }
    torch.save(save_dict, os.path.join(local_dir, 'torch_checkpoint.pt'))

    # Embed provenance so the tarball is self-describing even without wandb API access.
    if provenance_record is None:
        provenance_record = build_provenance_record(args, use_callbacks or {}, artifact_path=None)
    provenance_path = os.path.join(local_dir, 'provenance.json')
    try:
        with open(provenance_path, 'w') as fh:
            json.dump(provenance_record, fh, indent=2, default=str)
    except Exception as exc:
        print(f"WARN: failed to write provenance.json: {exc}", file=sys.stderr)

    # Compose a filename that always contains the W&B run_id when available so
    # the artifact is trivially discoverable via filesystem search.
    project_tag = (provenance_record.get('wandb_project') or 'local').replace('/', '_')
    run_id = provenance_record.get('run_id') or f"norun{random_tag}"
    filename = f'model_artifacts__{project_tag}__{run_id}__{timestamp}.tar.gz'

    artifact_dir = args['Main args'].artifact_path
    final_artifact_path = None
    with tempfile.TemporaryDirectory() as tmpdirname:
        tar_src = os.path.join(tmpdirname, filename)
        with tarfile.open(tar_src, 'w:gz') as tar:
            tar.add(os.path.join(local_dir, 'torch_checkpoint.pt'), arcname='artifacts/torch_checkpoint.pt')
            if os.path.isfile(provenance_path):
                tar.add(provenance_path, arcname='artifacts/provenance.json')

        if 'gs://' in artifact_dir:
            final_artifact_path = os.path.join(artifact_dir, filename)
            subprocess.check_call(['gsutil', 'cp', tar_src, final_artifact_path])
        else:
            os.makedirs(artifact_dir, exist_ok=True)
            final_artifact_path = os.path.join(artifact_dir, filename)
            shutil.copy(tar_src, final_artifact_path)

    print(f"Model saved to {final_artifact_path}")

    publish_best_checkpoint_model(
        local_dir,
        final_artifact_path,
        provenance_record,
        use_callbacks=use_callbacks or {},
        args=args,
    )

    removed_checkpoints = prune_lightning_checkpoints(
        use_callbacks=use_callbacks or {},
        keep=bool(getattr(args['Main args'], 'keep_lightning_checkpoints', False)),
    )
    if removed_checkpoints:
        print(f"Pruned {len(removed_checkpoints)} transient Lightning checkpoint(s).")

    if wandb.run is not None:
        wandb.run.summary["model_saved_path"] = final_artifact_path
        wandb.run.summary["model_artifact_filename"] = filename
        if removed_checkpoints:
            wandb.run.summary["pruned_lightning_checkpoint_count"] = len(removed_checkpoints)

    return final_artifact_path


#######################
# Main Training Logic #
#######################

def _log_train_eval_metrics(graph, data):
    """
    Run a single forward pass over the training dataloader using the best
    checkpoint and log canonical train metrics to the active W&B run summary so
    `runs.csv` gets populated.  These are post-fit best-checkpoint metrics, not
    another chronological epoch, so they must not be appended to W&B history.

    This is intentionally separate from `training_step` / `validation_step`:
    those run on mini-batches with dropout/BN-training semantics (for loss)
    and are not appropriate for a clean "how well does the trained model fit
    the training set" readout.
    """
    if wandb.run is None:
        return

    try:
        from boda.graph.utils import pearson_correlation, spearman_correlation, pearson_r2_score, coefficient_of_determination
    except Exception as exc:
        print(f"WARN: cannot import metrics for train eval: {exc}", file=sys.stderr)
        return

    if not hasattr(data, "train_dataloader"):
        return
    try:
        loader_fn = (
            data.train_eval_dataloader
            if hasattr(data, "train_eval_dataloader")
            else data.train_dataloader
        )
        loader = loader_fn()
    except Exception as exc:
        print(f"WARN: cannot build train_dataloader for eval: {exc}", file=sys.stderr)
        return

    device = next(graph.parameters()).device
    graph.eval()
    losses: List[torch.Tensor] = []
    preds: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            try:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x, y = batch[:2]
                else:
                    x, y = batch
            except Exception:
                continue
            x = x.to(device)
            y = y.to(device)
            y_hat = graph(x)
            if y_hat.dim() == 2 and y_hat.shape[1] == 1 and y.dim() == 1:
                y_hat = y_hat.squeeze(1)
            if y.dim() == 2 and y.shape[1] == 1 and y_hat.dim() == 1:
                y = y.squeeze(1)
            try:
                losses.append(graph.criterion(y_hat, y).detach().cpu())
            except Exception:
                pass
            preds.append(y_hat.detach().cpu())
            labels.append(y.detach().cpu())

    if not preds:
        return

    all_preds = torch.cat(preds, dim=0)
    all_labels = torch.cat(labels, dim=0)

    try:
        train_cod_r2 = float(coefficient_of_determination(all_labels, all_preds))
    except Exception:
        train_cod_r2 = None
    try:
        train_pearson_vals, train_pearson = pearson_correlation(all_preds, all_labels)
        train_pearson = float(train_pearson)
    except Exception:
        train_pearson_vals = None
        train_pearson = None
    try:
        train_spearman_vals, train_spearman = spearman_correlation(all_preds, all_labels)
        train_spearman = float(train_spearman)
    except Exception:
        train_spearman_vals = None
        train_spearman = None
    train_loss = float(torch.stack(losses).mean()) if losses else None
    try:
        train_mse = float((all_preds - all_labels).pow(2).mean())
    except Exception:
        train_mse = None

    summary_updates = {
        "train_loss": train_loss,
        "train_mse": train_mse,
        "train_cod_r2": train_cod_r2,
        "train_pearson": train_pearson,
        "train_spearman": train_spearman,
    }

    try:
        log_per_output = bool(getattr(graph, "log_per_output_metric_details", True))
        metric_source = train_pearson_vals if train_pearson_vals is not None else train_spearman_vals
        if log_per_output and metric_source is not None:
            if metric_source.dim() == 0:
                metric_source = metric_source.unsqueeze(0)
            n_outputs = int(metric_source.numel())
            if hasattr(graph, "output_names_for"):
                output_names = graph.output_names_for(n_outputs)
            elif n_outputs == 1:
                output_names = ["SingleOutput"]
            else:
                output_names = [f"output_{idx}" for idx in range(n_outputs)]
            if train_pearson_vals is not None and train_pearson_vals.dim() == 0:
                train_pearson_vals = train_pearson_vals.unsqueeze(0)
            if train_spearman_vals is not None and train_spearman_vals.dim() == 0:
                train_spearman_vals = train_spearman_vals.unsqueeze(0)
            train_mse_vals = (all_preds - all_labels).pow(2).mean(dim=0)
            if train_mse_vals.dim() == 0:
                train_mse_vals = train_mse_vals.unsqueeze(0)
            for idx, name in enumerate(output_names):
                if train_pearson_vals is not None and idx < train_pearson_vals.numel():
                    coeff = float(train_pearson_vals[idx])
                    summary_updates[f"train_pearson_{name}"] = coeff
                    if bool(getattr(graph, "log_legacy_metric_aliases", True)):
                        summary_updates[f"train_pearson_squared_{name}"] = coeff ** 2
                if train_spearman_vals is not None and idx < train_spearman_vals.numel():
                    summary_updates[f"train_spearman_{name}"] = float(train_spearman_vals[idx])
                if idx < train_mse_vals.numel():
                    summary_updates[f"train_mse_{name}"] = float(train_mse_vals[idx])
    except Exception as exc:
        print(f"WARN: train-set per-output metrics failed: {exc}", file=sys.stderr)

    for k, v in summary_updates.items():
        if v is not None:
            try:
                wandb.run.summary[k] = v
                # Make the provenance of this clean, post-fit readout explicit
                # without adding a discontinuous final point to epoch charts.
                wandb.run.summary[f"best_checkpoint_{k}"] = v
            except Exception:
                pass
    print("Train-set eval summary: " + ", ".join(
        f"{k}={v}" for k, v in summary_updates.items() if v is not None
    ))


def _sanitize_metric_fragment(value: Any) -> str:
    """Make a short value safe for use inside a W&B summary key."""
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")
    return text or "unknown"


def _log_library_split_eval_metrics(graph, data):
    """
    For combined Hani Lib1+Lib2 tables, log held-out metrics split by library.

    The canonical datamodule trains on one CSV and reports combined validation
    and test metrics. Phase 3 also needs Lib1-only and Lib2-only readouts, so
    this best-checkpoint post-fit hook reuses the datamodule's target
    normalization stats and evaluates each `fold` x `library` slice from the
    source table. It is intentionally best-effort and silently skips non-Hani
    tables or CSVs without a `library` column.
    """
    if wandb.run is None:
        return
    if not all(hasattr(data, attr) for attr in ("datafile_path", "activity_columns", "sequence_column")):
        return

    datafile_path = getattr(data, "datafile_path", None)
    if not datafile_path:
        return

    try:
        import numpy as np
        import pandas as pd
        from boda.graph.utils import pearson_correlation, spearman_correlation, pearson_r2_score, coefficient_of_determination
    except Exception as exc:
        print(f"WARN: cannot import libraries for split/library eval: {exc}", file=sys.stderr)
        return

    try:
        df = pd.read_csv(datafile_path)
    except Exception as exc:
        print(f"WARN: cannot read datafile for split/library eval: {exc}", file=sys.stderr)
        return

    fold_column = getattr(data, "fold_column", "fold")
    library_column = "library"
    sequence_column = getattr(data, "sequence_column", "seq")
    activity_columns = list(getattr(data, "activity_columns", []) or [])
    required = [sequence_column, fold_column, library_column, *activity_columns]
    if not activity_columns or any(column not in df.columns for column in required):
        return

    df = df.dropna(subset=required).copy()
    if df.empty:
        return
    for column in activity_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=activity_columns).copy()
    if df.empty:
        return

    means = None
    stds = None
    if getattr(data, "normalize_activity", False):
        raw_means = getattr(data, "activity_means", None)
        raw_stds = getattr(data, "activity_stds", None)
        try:
            means = np.asarray([float(raw_means[column]) for column in activity_columns], dtype=np.float32)
            stds = np.asarray([float(raw_stds[column]) for column in activity_columns], dtype=np.float32)
            stds[~np.isfinite(stds) | (np.abs(stds) < 1e-8)] = 1.0
        except Exception:
            means = None
            stds = None

    device = next(graph.parameters()).device
    graph.eval()

    def predict_scaled(sequences: List[str], batch_size: int) -> np.ndarray:
        tensors = torch.stack([utils.dna2tensor(str(seq).strip().upper()) for seq in sequences])
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(tensors),
            batch_size=batch_size,
            shuffle=False,
        )
        preds = []
        with torch.no_grad():
            for (x_batch,) in loader:
                preds.append(graph(x_batch.to(device)).detach().cpu())
        return torch.cat(preds, dim=0).numpy()

    def safe_float(value: Any) -> Optional[float]:
        try:
            value = float(value)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        return value

    batch_size = int(getattr(data, "batch_size", 512) or 512)
    summary_updates: Dict[str, Any] = {}
    slice_specs = []
    for fold_value in sorted(df[fold_column].dropna().unique()):
        fold_df = df[df[fold_column].eq(fold_value)].copy()
        if fold_df.empty:
            continue
        slice_specs.append((fold_value, "combined", fold_df))
        for library_value in sorted(fold_df[library_column].dropna().unique()):
            lib_df = fold_df[fold_df[library_column].eq(library_value)].copy()
            if not lib_df.empty:
                slice_specs.append((fold_value, library_value, lib_df))

    for fold_value, library_value, sub in slice_specs:
        if len(sub) < 2:
            continue
        true_raw_np = sub[activity_columns].to_numpy(dtype=np.float32)
        if means is not None and stds is not None:
            labels_scaled_np = (true_raw_np - means) / stds
        else:
            labels_scaled_np = true_raw_np

        try:
            pred_scaled_np = predict_scaled(sub[sequence_column].tolist(), batch_size=batch_size)
        except Exception as exc:
            print(
                f"WARN: split/library eval prediction failed for {fold_value}/{library_value}: {exc}",
                file=sys.stderr,
            )
            continue
        if pred_scaled_np.ndim == 1:
            pred_scaled_np = pred_scaled_np.reshape(-1, 1)
        if pred_scaled_np.shape != labels_scaled_np.shape:
            print(
                f"WARN: split/library eval shape mismatch for {fold_value}/{library_value}: "
                f"pred={pred_scaled_np.shape}, true={labels_scaled_np.shape}",
                file=sys.stderr,
            )
            continue

        pred_raw_np = pred_scaled_np * stds + means if means is not None and stds is not None else pred_scaled_np
        pred_scaled = torch.as_tensor(pred_scaled_np, dtype=torch.float32)
        labels_scaled = torch.as_tensor(labels_scaled_np, dtype=torch.float32)
        pred_raw = torch.as_tensor(pred_raw_np, dtype=torch.float32)
        true_raw = torch.as_tensor(true_raw_np, dtype=torch.float32)

        fold_key = _sanitize_metric_fragment(fold_value)
        library_key = _sanitize_metric_fragment(library_value)
        prefix = f"eval_{fold_key}_{library_key}"

        try:
            loss = graph.criterion(pred_scaled, labels_scaled)
            summary_updates[f"{prefix}_loss"] = safe_float(loss)
        except Exception:
            pass
        try:
            summary_updates[f"{prefix}_pearson_r2"] = safe_float(pearson_r2_score(labels_scaled, pred_scaled))
        except Exception:
            pass
        try:
            summary_updates[f"{prefix}_cod_r2"] = safe_float(coefficient_of_determination(labels_scaled, pred_scaled))
        except Exception:
            pass
        try:
            pearson_vals, mean_pearson = pearson_correlation(pred_scaled, labels_scaled)
            summary_updates[f"{prefix}_mean_per_head_pearson"] = safe_float(mean_pearson)
            for idx, name in enumerate(graph.output_names_for(int(pearson_vals.numel())) if hasattr(graph, "output_names_for") else activity_columns):
                if idx < pearson_vals.numel():
                    coeff = safe_float(pearson_vals[idx])
                    summary_updates[f"{prefix}_pearson_{_sanitize_metric_fragment(name)}"] = coeff
                    summary_updates[f"{prefix}_pearson_squared_{_sanitize_metric_fragment(name)}"] = (
                        coeff * coeff if coeff is not None else None
                    )
        except Exception:
            pass
        try:
            spearman_vals, mean_spearman = spearman_correlation(pred_scaled, labels_scaled)
            summary_updates[f"{prefix}_spearman"] = safe_float(mean_spearman)
            names = graph.output_names_for(int(spearman_vals.numel())) if hasattr(graph, "output_names_for") else activity_columns
            for idx, name in enumerate(names):
                if idx < spearman_vals.numel():
                    summary_updates[f"{prefix}_spearman_{_sanitize_metric_fragment(name)}"] = safe_float(spearman_vals[idx])
        except Exception:
            pass
        try:
            avg_pred = pred_raw.mean(dim=1)
            avg_true = true_raw.mean(dim=1)
            _, avg_pearson = pearson_correlation(avg_pred, avg_true)
            _, avg_spearman = spearman_correlation(avg_pred, avg_true)
            _, flat_pearson = pearson_correlation(pred_raw.reshape(-1), true_raw.reshape(-1))
            _, flat_spearman = spearman_correlation(pred_raw.reshape(-1), true_raw.reshape(-1))
            summary_updates[f"{prefix}_average_activity_pearson"] = safe_float(avg_pearson)
            summary_updates[f"{prefix}_average_activity_spearman"] = safe_float(avg_spearman)
            summary_updates[f"{prefix}_flattened_activity_pearson"] = safe_float(flat_pearson)
            summary_updates[f"{prefix}_flattened_activity_spearman"] = safe_float(flat_spearman)
            summary_updates[f"{prefix}_n_sequences"] = int(len(sub))
        except Exception:
            pass

    clean_updates = {key: value for key, value in summary_updates.items() if value is not None}
    for key, value in clean_updates.items():
        try:
            wandb.run.summary[key] = value
        except Exception:
            pass
    if clean_updates:
        print(
            "Split/library eval summary: "
            + ", ".join(f"{key}={value}" for key, value in sorted(clean_updates.items())[:24])
            + (" ..." if len(clean_updates) > 24 else "")
        )


def _has_overridden_dataloader(data_module: Any, loader_name: str) -> bool:
    method = getattr(type(data_module), loader_name, None)
    base_method = getattr(LightningDataModule, loader_name, None)
    return method is not None and method is not base_method


def _run_optional_postfit_test(trainer, graph, data, enabled: bool) -> bool:
    """Run post-fit test only when explicitly authorized; return whether called."""
    if not enabled:
        print("Post-fit test evaluation disabled by evaluate_test_after_fit=false.")
        return False
    try:
        test_loader = None
        if _has_overridden_dataloader(data, "test_dataloader"):
            try:
                test_loader = data.test_dataloader()
            except Exception as exc:
                print(f"WARN: could not build test_dataloader: {exc}", file=sys.stderr)
        if test_loader is not None:
            trainer.test(graph, dataloaders=test_loader)
            return True
    except Exception as exc:
        print(f"WARN: trainer.test failed: {exc}", file=sys.stderr)
    return False


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    """Hash JSON-like launch evidence without depending on dict insertion order."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime_argv_sha256(argv: Optional[List[str]] = None) -> str:
    return _canonical_json_sha256(list(sys.argv if argv is None else argv))


def _resolved_argument_groups(args: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        group: vars(namespace)
        for group, namespace in args.items()
        if isinstance(namespace, argparse.Namespace)
    }


def _campaign_wandb_fields(main_args: argparse.Namespace) -> Dict[str, Any]:
    """Return flat campaign fields that are easy to filter in the W&B UI."""
    keys = (
        "campaign_id",
        "campaign_stage",
        "part_slug",
        "analysis_lane",
        "challenger_family",
        "policy_id",
        "config_origin",
        "training_regime",
        "cell_id",
        "rc_pair_id",
        "loss_pair_id",
        "source_unweighted_cell_id",
        "rc_mode",
        "execution_disposition",
        "initialization",
        "source_head",
        "unfreeze_scope",
        "input_policy",
        "pretrained_artifact_sha256",
        "data_generation_id",
        "dataset_sha256",
        "split_manifest_id",
        "split_manifest_sha256",
        "development_fold",
        "base_config_id",
        "source_run_ids",
        "architecture",
        "model_seed",
        "loss_mode",
        "target_definition",
        "length_policy",
        "artifact_retention",
        "evaluate_test_after_fit",
    )
    fields = {key: getattr(main_args, key, None) for key in keys}
    fields['hostname'] = socket.gethostname()
    fields['git_commit'] = _resolve_git_commit()
    return fields


def _validate_campaign_wandb_contract(main_args: argparse.Namespace) -> None:
    """Give the dedup campaign an identity independent of caller/env input."""
    if getattr(main_args, 'campaign_id', '') != LIB1_DEDUP_CAMPAIGN_ID:
        return
    if str(getattr(main_args, 'logger_type', '')).lower() != 'wandb':
        raise ValueError('The Lib1 dedup campaign requires logger_type=wandb.')
    if getattr(main_args, 'wandb_entity', '') != LIB1_DEDUP_WANDB_ENTITY:
        raise ValueError(
            f'The Lib1 dedup campaign requires --wandb_entity '
            f'{LIB1_DEDUP_WANDB_ENTITY!r}; received '
            f'{getattr(main_args, "wandb_entity", "")!r}.'
        )
    expected_project_fragment = '__bashor_in_house__'
    project = str(getattr(main_args, 'logger_project', '') or '')
    stage = str(getattr(main_args, 'campaign_stage', '') or '')
    if stage.startswith('stage1_'):
        valid_suffix = project.endswith('__exact_replay')
    elif stage.startswith('stage2_'):
        valid_suffix = project.endswith('__stage2_development')
    elif stage == 'targeted_utr3_hpo':
        # This post-Stage-2 development-only route is intentionally exact,
        # rather than a general "targeted_*" wildcard.  Adding another
        # targeted route requires its own explicit campaign contract.
        valid_suffix = project == LIB1_DEDUP_TARGETED_UTR3_PROJECT
    elif stage == 'stage3_weighted_loss':
        valid_suffix = project.endswith('__stage3_weighted_development')
    elif stage == 'stage4_downsampling':
        valid_suffix = project.endswith('__stage4_downsampling_development')
    elif stage == 'final_refit':
        valid_suffix = project.endswith('__final_refit_development')
    else:
        valid_suffix = False
    if expected_project_fragment not in project or not valid_suffix:
        raise ValueError(
            f'Unexpected Lib1 dedup campaign W&B project {project!r} for stage {stage!r}.'
        )


def _validate_stage3_weighted_contract(
    main_args: argparse.Namespace, data_args: argparse.Namespace
) -> None:
    """Fail before model allocation if a Stage 3 row is only nominally weighted."""
    if str(getattr(main_args, 'campaign_stage', '') or '') != 'stage3_weighted_loss':
        return
    graph = str(getattr(main_args, 'graph_module', '') or '')
    training_regime = str(getattr(main_args, 'training_regime', '') or '')
    if training_regime not in {'scratch', 'transfer'}:
        raise ValueError(
            "Stage 3 training_regime must be exactly 'scratch' or 'transfer'."
        )
    expected_graph = (
        'CNNBassetBranchedScopedWeightedTransfer'
        if training_regime == 'transfer'
        else 'CNNWeightedRegressionTraining'
    )
    if graph != expected_graph:
        raise ValueError(
            f'Stage 3 {training_regime!r} rows require graph_module='
            f'{expected_graph!r}; received {graph!r}.'
        )
    if str(getattr(main_args, 'loss_mode', '') or '') != 'barcode_weighted_mse':
        raise ValueError('Stage 3 launch rows require loss_mode=barcode_weighted_mse.')
    part = str(getattr(main_args, 'part_slug', '') or '')
    if part not in {'enhancer', 'promoter', 'intron', 'utr3', 'utr5'}:
        raise ValueError(f'Unexpected Stage 3 part_slug={part!r}.')
    expected_project = (
        f'{part}__bashor_in_house__dedup_exact_v1__stage3_weighted_development'
    )
    if str(getattr(main_args, 'logger_project', '') or '') != expected_project:
        raise ValueError(
            f'Stage 3 part/project mismatch: expected {expected_project!r}.'
        )
    for field in ('cell_id', 'loss_pair_id', 'source_unweighted_cell_id'):
        if not str(getattr(main_args, field, '') or ''):
            raise ValueError(f'Stage 3 launch rows require nonempty {field}.')
    rc_mode = str(getattr(main_args, 'rc_mode', '') or '')
    if rc_mode not in {'off', 'on'}:
        raise ValueError("Stage 3 rc_mode must be exactly 'off' or 'on'.")
    if part == 'utr3' and rc_mode != 'off':
        raise ValueError("Stage 3 3'UTR rows require rc_mode=off.")
    rc_pair_id = str(getattr(main_args, 'rc_pair_id', '') or '')
    if part == 'utr3' and rc_pair_id:
        raise ValueError("Stage 3 3'UTR rows cannot carry an rc_pair_id.")
    if part != 'utr3' and not rc_pair_id:
        raise ValueError('Stage 3 non-3\'UTR rows require a nonempty rc_pair_id.')
    if not bool(getattr(data_args, 'barcode_weighting', False)):
        raise ValueError('Stage 3 weighted rows require barcode_weighting=true.')
    if bool(getattr(data_args, 'use_reverse_complements', False)) != (
        rc_mode == 'on'
    ):
        raise ValueError('Stage 3 rc_mode disagrees with use_reverse_complements.')
    if float(getattr(data_args, 'barcode_weight_cap', float('nan'))) != 8.0:
        raise ValueError('Stage 3 weighted rows require barcode_weight_cap=8.0.')
    if float(getattr(data_args, 'barcode_weight_min', float('nan'))) != 0.1:
        raise ValueError('Stage 3 weighted rows require barcode_weight_min=0.1.')
    if bool(getattr(main_args, 'evaluate_test_after_fit', True)):
        raise ValueError('Stage 3 development rows cannot evaluate test/audit data.')


def _validate_stage4_manifest_launch_contract(
    main_args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Bind a Stage 4 process to one row of the exact frozen manifest.

    This check happens before model allocation.  It intentionally requires the
    runner-provided row and registry environment instead of allowing a manifest
    command to be copied into an unrelated launch context.
    """
    if str(getattr(main_args, 'campaign_stage', '') or '') != 'stage4_downsampling':
        return None

    config_path_value = str(os.environ.get('BODA_CONFIG_PATH', '') or '')
    if not config_path_value:
        raise ValueError('Stage 4 requires BODA_CONFIG_PATH to the frozen manifest.')
    config_path = Path(config_path_value).expanduser().resolve()
    if config_path != LIB1_DEDUP_STAGE4_MANIFEST_PATH.resolve():
        raise ValueError(
            f'Stage 4 BODA_CONFIG_PATH must resolve to '
            f'{str(LIB1_DEDUP_STAGE4_MANIFEST_PATH.resolve())!r}.'
        )
    if not config_path.is_file():
        raise ValueError('The frozen Stage 4 manifest is missing.')
    observed_manifest_sha = _sha256_file(str(config_path))
    if observed_manifest_sha != LIB1_DEDUP_STAGE4_MANIFEST_SHA256:
        raise ValueError('Stage 4 manifest SHA256 does not match the frozen protocol.')
    if os.environ.get('BODA_CONFIG_MANIFEST_SHA256', '') != observed_manifest_sha:
        raise ValueError('Stage 4 BODA_CONFIG_MANIFEST_SHA256 is absent or mismatched.')

    row_text = str(os.environ.get('BODA_MANIFEST_ROW', '') or '')
    try:
        row_number = int(row_text)
    except ValueError as exc:
        raise ValueError('Stage 4 requires an integer BODA_MANIFEST_ROW.') from exc
    rows = [
        json.loads(line)
        for line in config_path.read_text().splitlines()
        if line.strip()
    ]
    bound = [row for row in rows if int(row.get('row', -1)) == row_number]
    if len(bound) != 1:
        raise ValueError(f'Stage 4 manifest row={row_number!r} is not unique.')
    row = bound[0]
    cell_id = str(getattr(main_args, 'cell_id', '') or '')
    if str(row.get('cell_id', '')) != cell_id:
        raise ValueError('Stage 4 manifest row and runtime cell_id disagree.')
    row_fingerprint = str(row.get('row_fingerprint', '') or '')
    if os.environ.get('BODA_MANIFEST_ROW_FINGERPRINT', '') != row_fingerprint:
        raise ValueError('Stage 4 manifest row fingerprint is absent or mismatched.')

    expected_argv = shlex.split(str(row['train_command']))[1:]
    if expected_argv != sys.argv:
        raise ValueError(
            'Stage 4 runtime arguments differ from the SHA-bound manifest row command.'
        )
    observed_argv_sha = _runtime_argv_sha256()
    if os.environ.get('BODA_RUNTIME_ARGV_SHA256', '') != observed_argv_sha:
        raise ValueError('Stage 4 runtime argv SHA256 is absent or mismatched.')

    registry_value = str(os.environ.get('BODA_RUNS_CSV', '') or '')
    if not registry_value or Path(registry_value).expanduser().resolve() != (
        LIB1_DEDUP_STAGE4_REGISTRY_PATH.resolve()
    ):
        raise ValueError('Stage 4 requires its dedicated Stage4-only run registry.')
    if os.environ.get('BODA_LAUNCH_SCRIPT', '') != (
        'run_lib1_dedup_stage4_downsampling_campaign.py'
    ):
        raise ValueError('Stage 4 requires the frozen campaign runner identity.')
    return row


def _validate_stage4_resolved_row_contract(
    row: Optional[Dict[str, Any]],
    main_args: argparse.Namespace,
    data_args: argparse.Namespace,
    trainer_args: argparse.Namespace,
) -> None:
    """Check high-risk parsed values against the row after argparse/YAML resolution."""
    if row is None:
        return

    main_expected = {
        'campaign_id': row['campaign_id'],
        'campaign_stage': row['campaign_stage'],
        'part_slug': row['part_slug'],
        'analysis_lane': row['analysis_lane'],
        'training_regime': row['training_regime'],
        'cell_id': row['cell_id'],
        'rc_mode': row['rc_mode'],
        'execution_disposition': 'launch',
        'data_generation_id': row['data_generation_id'],
        'dataset_sha256': row['dataset_sha256'],
        'split_manifest_id': row['split_manifest_id'],
        'split_manifest_sha256': row['split_manifest_sha256'],
        'development_fold': row['outer_oof_fold'],
        'base_config_id': row['base_config_id'],
        'architecture': row['architecture'],
        'model_seed': row['model_seed'],
        'loss_mode': row['loss_mode'],
        'target_definition': row['target_definition'],
        'length_policy': row['length_policy'],
        'run_name': row['planned_run_name'],
        'logger_project': row['logger_project'],
        'wandb_entity': row['wandb_entity'],
        'artifact_retention': 'none',
        'evaluate_test_after_fit': False,
    }
    for field, expected in main_expected.items():
        if getattr(main_args, field, None) != expected:
            raise ValueError(
                f'Stage 4 resolved Main args.{field} differs from manifest row: '
                f'{getattr(main_args, field, None)!r} != {expected!r}.'
            )

    data_expected = {
        'manifest_mode': 'development_inner_oof',
        'split_manifest_path': row['split_manifest_path'],
        'split_fold': row['outer_oof_fold'],
        'train_size_n': row['train_size_n'],
        'train_subsample_seed': row['train_subsample_seed'],
        'expected_data_sha256': row['dataset_sha256'],
        'expected_split_sha256': row['split_manifest_sha256'],
        'use_reverse_complements': row['rc_mode'] == 'on',
        'barcode_weighting': row['loss_mode'] == 'barcode_weighted_mse',
    }
    for field, expected in data_expected.items():
        observed = getattr(data_args, field, None)
        if field == 'split_manifest_path':
            observed = str(Path(str(observed)).expanduser().resolve())
            expected = str(Path(str(expected)).expanduser().resolve())
        if observed != expected:
            raise ValueError(
                f'Stage 4 resolved data argument {field} differs from manifest row: '
                f'{observed!r} != {expected!r}.'
            )

    observed_root = str(Path(str(getattr(trainer_args, 'default_root_dir', ''))).expanduser().resolve())
    expected_root = str(Path(str(row['default_root_dir'])).expanduser().resolve())
    if observed_root != expected_root:
        raise ValueError(
            f'Stage 4 resolved trainer default_root_dir differs from manifest row: '
            f'{observed_root!r} != {expected_root!r}.'
        )


def _validate_stage4_downsampling_contract(
    main_args: argparse.Namespace, data_args: argparse.Namespace
) -> None:
    """Fail closed for Stage 4 inner-validation/outer-OOF learning curves."""
    if str(getattr(main_args, 'campaign_stage', '') or '') != 'stage4_downsampling':
        return

    part = str(getattr(main_args, 'part_slug', '') or '')
    if part not in {'enhancer', 'promoter', 'intron', 'utr3', 'utr5'}:
        raise ValueError(f'Unexpected Stage 4 part_slug={part!r}.')
    expected_project = (
        f'{part}__bashor_in_house__dedup_exact_v1__'
        'stage4_downsampling_development'
    )
    if str(getattr(main_args, 'logger_project', '') or '') != expected_project:
        raise ValueError(
            f'Stage 4 part/project mismatch: expected {expected_project!r}.'
        )
    if bool(getattr(main_args, 'evaluate_test_after_fit', True)):
        raise ValueError('Stage 4 cannot evaluate final-test data.')
    if _coerce_split_list(getattr(main_args, 'prediction_splits', [])) != ['oof']:
        raise ValueError('Stage 4 prediction_splits must be exactly ["oof"].')
    if _coerce_split_list(getattr(main_args, 'epoch_eval_splits', [])) != [
        'train', 'val'
    ]:
        raise ValueError(
            'Stage 4 epoch_eval_splits must be exactly ["train", "val"].'
        )
    if str(getattr(data_args, 'manifest_mode', '') or '') != 'development_inner_oof':
        raise ValueError(
            'Stage 4 requires manifest_mode=development_inner_oof.'
        )
    if not str(getattr(data_args, 'split_manifest_path', '') or ''):
        raise ValueError('Stage 4 requires a frozen split_manifest_path.')
    if str(getattr(data_args, 'train_sampling_mode', '') or '') != 'random':
        raise ValueError('Stage 4 requires train_sampling_mode=random.')
    if int(getattr(data_args, 'train_min_barcodes', -1)) != 1:
        raise ValueError('Stage 4 requires train_min_barcodes=1.')
    if getattr(data_args, 'train_max_barcodes', None) is not None:
        raise ValueError('Stage 4 cannot set train_max_barcodes.')
    if float(getattr(data_args, 'train_size_frac', float('nan'))) != 1.0:
        raise ValueError('Stage 4 requires train_size_frac=1.0.')

    rc_mode = str(getattr(main_args, 'rc_mode', '') or '')
    if rc_mode not in {'off', 'on'}:
        raise ValueError("Stage 4 rc_mode must be exactly 'off' or 'on'.")
    if bool(getattr(data_args, 'use_reverse_complements', False)) != (
        rc_mode == 'on'
    ):
        raise ValueError('Stage 4 rc_mode disagrees with use_reverse_complements.')

    training_regime = str(getattr(main_args, 'training_regime', '') or '')
    if training_regime not in {'scratch', 'transfer'}:
        raise ValueError(
            "Stage 4 training_regime must be exactly 'scratch' or 'transfer'."
        )
    loss_mode = str(getattr(main_args, 'loss_mode', '') or '')
    if loss_mode not in {'unweighted_mse', 'barcode_weighted_mse'}:
        raise ValueError(f'Unexpected Stage 4 loss_mode={loss_mode!r}.')
    weighted = loss_mode == 'barcode_weighted_mse'
    if bool(getattr(data_args, 'barcode_weighting', False)) != weighted:
        raise ValueError('Stage 4 loss_mode disagrees with barcode_weighting.')
    expected_graphs = {
        ('scratch', False): 'CNNBasicTraining',
        ('scratch', True): 'CNNWeightedRegressionTraining',
        ('transfer', False): 'CNNBassetBranchedScopedTransfer',
        ('transfer', True): 'CNNBassetBranchedScopedWeightedTransfer',
    }
    expected_graph = expected_graphs[(training_regime, weighted)]
    graph = str(getattr(main_args, 'graph_module', '') or '')
    if graph != expected_graph:
        raise ValueError(
            f'Stage 4 {training_regime}/{loss_mode} requires '
            f'graph_module={expected_graph!r}; received {graph!r}.'
        )
    if weighted:
        if float(getattr(data_args, 'barcode_weight_cap', float('nan'))) != 8.0:
            raise ValueError('Stage 4 weighted rows require barcode_weight_cap=8.0.')
        if float(getattr(data_args, 'barcode_weight_min', float('nan'))) != 0.1:
            raise ValueError('Stage 4 weighted rows require barcode_weight_min=0.1.')


FINAL_REFIT_POLICIES = {
    'enhancer': {
        'base_config_id': 'basecfg_6e6b2b979116f3e9cd83a8747792d89a97918ce57e72949f810c309afa068036',
        'architecture': 'BassetBranched',
        'graph_module': 'CNNBassetBranchedScopedTransfer',
        'rc_mode': 'on',
        'loss_mode': 'unweighted_mse',
        'fixed_epochs': 6,
    },
    'promoter': {
        'base_config_id': 'basecfg_bff24362f7f5a2013947c22336ec779dc986c42124230dae5ff4fcc9904a5d0d',
        'architecture': 'PromoterBassetVL',
        'graph_module': 'CNNWeightedRegressionTraining',
        'rc_mode': 'off',
        'loss_mode': 'barcode_weighted_mse',
        'fixed_epochs': 44,
    },
    'intron': {
        'base_config_id': 'basecfg_58481a479285bf26af4a9813d37abecc1e6a548795eb3f606fe4d5758ecc4a86',
        'architecture': 'ResNet1DRegressor',
        'graph_module': 'CNNWeightedRegressionTraining',
        'rc_mode': 'off',
        'loss_mode': 'barcode_weighted_mse',
        'fixed_epochs': 21,
    },
    'utr3': {
        'base_config_id': 'basecfg_7b1f881265b0fc0aee9e2b601352b93e064e37bee808c65b6b038e6a260e2062',
        'architecture': 'UTR_BassetVL',
        'graph_module': 'CNNWeightedRegressionTraining',
        'rc_mode': 'off',
        'loss_mode': 'barcode_weighted_mse',
        'fixed_epochs': 36,
    },
    'utr5': {
        'base_config_id': 'basecfg_9dd728c0df617152551b366c304a265d52be567ad04fb35dbdcecd406235d315',
        'architecture': 'UTR_BassetVL',
        'graph_module': 'CNNWeightedRegressionTraining',
        'rc_mode': 'off',
        'loss_mode': 'barcode_weighted_mse',
        'fixed_epochs': 83,
    },
}


def _validate_final_refit_contract(
    main_args: argparse.Namespace,
    data_args: argparse.Namespace,
    trainer_args: argparse.Namespace,
) -> None:
    """Fail closed for the locked all-development, pre-audit refits."""
    if str(getattr(main_args, 'campaign_stage', '') or '') != 'final_refit':
        return
    manifest_path = Path(os.environ.get('BODA_CONFIG_PATH', '')).expanduser()
    if not manifest_path.is_file():
        raise ValueError('Final refits require BODA_CONFIG_PATH to the frozen manifest.')
    if _sha256_file(str(manifest_path)) != LIB1_DEDUP_FINAL_REFIT_MANIFEST_SHA256:
        raise ValueError('Final-refit manifest SHA256 does not match the frozen protocol.')
    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text().splitlines()
        if line.strip()
    ]
    cell_id = str(getattr(main_args, 'cell_id', '') or '')
    bound_rows = [row for row in manifest_rows if row.get('cell_id') == cell_id]
    if len(bound_rows) != 1:
        raise ValueError(f'Final-refit cell_id={cell_id!r} is not unique in the manifest.')
    expected_argv = shlex.split(bound_rows[0]['train_command'])[1:]
    if expected_argv != sys.argv:
        raise ValueError(
            'Final-refit runtime arguments differ from the SHA-bound manifest command.'
        )
    part = str(getattr(main_args, 'part_slug', '') or '')
    if part not in FINAL_REFIT_POLICIES:
        raise ValueError(f'Unexpected final-refit part_slug={part!r}.')
    policy = FINAL_REFIT_POLICIES[part]
    expected_project = (
        f'{part}__bashor_in_house__dedup_exact_v1__final_refit_development'
    )
    checks = {
        'logger_project': expected_project,
        'base_config_id': policy['base_config_id'],
        'architecture': policy['architecture'],
        'graph_module': policy['graph_module'],
        'rc_mode': policy['rc_mode'],
        'loss_mode': policy['loss_mode'],
    }
    for field, expected in checks.items():
        observed = str(getattr(main_args, field, '') or '')
        if observed != str(expected):
            raise ValueError(
                f'Final-refit {part} requires {field}={expected!r}; '
                f'received {observed!r}.'
            )
    if int(getattr(main_args, 'model_seed', -1)) not in {1701, 1702, 1703}:
        raise ValueError('Final refits require model_seed in {1701,1702,1703}.')
    if not str(getattr(main_args, 'cell_id', '') or '').startswith('refitcell_'):
        raise ValueError('Final refits require a nonempty refitcell_* cell_id.')
    if str(getattr(main_args, 'artifact_retention', '') or '') != 'selected':
        raise ValueError('Final refits require artifact_retention=selected.')
    if bool(getattr(main_args, 'evaluate_test_after_fit', True)):
        raise ValueError('Final refit training cannot evaluate audit/test data.')
    if getattr(main_args, 'checkpoint_monitor', None) is not None:
        raise ValueError('Final refits cannot checkpoint or stop on validation metrics.')
    if getattr(main_args, 'prediction_splits', []):
        raise ValueError('Final refits cannot export prediction splits.')
    if str(getattr(data_args, 'manifest_mode', '')) != 'final_refit':
        raise ValueError('Final refits require manifest_mode=final_refit.')
    if int(getattr(data_args, 'train_min_barcodes', -1)) != 1:
        raise ValueError('Final refits require train_min_barcodes=1.')
    if float(getattr(data_args, 'train_size_frac', float('nan'))) != 1.0:
        raise ValueError('Final refits require train_size_frac=1.0.')
    if getattr(data_args, 'train_size_n', None) is not None:
        raise ValueError('Final refits cannot set train_size_n.')
    if getattr(data_args, 'train_max_barcodes', None) is not None:
        raise ValueError('Final refits cannot set train_max_barcodes.')
    weighted = policy['loss_mode'] == 'barcode_weighted_mse'
    if bool(getattr(data_args, 'barcode_weighting', False)) != weighted:
        raise ValueError('Final-refit loss_mode disagrees with barcode_weighting.')
    if bool(getattr(data_args, 'use_reverse_complements', False)) != (
        policy['rc_mode'] == 'on'
    ):
        raise ValueError('Final-refit rc_mode disagrees with augmentation.')
    if int(getattr(trainer_args, 'max_epochs', -1)) != int(policy['fixed_epochs']):
        raise ValueError(
            f"Final-refit {part} requires max_epochs={policy['fixed_epochs']}."
        )
    if int(getattr(trainer_args, 'limit_val_batches', -1)) != 0:
        raise ValueError('Final refits require limit_val_batches=0.')
    if bool(getattr(trainer_args, 'enable_checkpointing', True)):
        raise ValueError('Final refits require enable_checkpointing=false.')
    if int(getattr(trainer_args, 'max_steps', -1)) != -1:
        raise ValueError('Final refits require max_steps=-1.')
    if bool(getattr(trainer_args, 'fast_dev_run', False)):
        raise ValueError('Final refits forbid fast_dev_run.')
    if float(getattr(trainer_args, 'overfit_batches', 0.0)) != 0.0:
        raise ValueError('Final refits require overfit_batches=0.')
    limit_train = getattr(trainer_args, 'limit_train_batches', None)
    if limit_train not in (None, 1, 1.0):
        raise ValueError('Final refits require the complete training dataloader.')


def _assert_wandb_identity(expected_entity: str, expected_project: str) -> Dict[str, str]:
    """Force W&B initialization and fail before fit if it resolved elsewhere."""
    if wandb.run is None:
        raise RuntimeError("W&B did not initialize a run before identity validation.")
    resolved_entity = str(getattr(wandb.run, "entity", "") or "")
    resolved_project = str(getattr(wandb.run, "project", "") or "")
    run_url = str(wandb.run.get_url() or "")
    print(f"Resolved W&B entity: {resolved_entity}")
    print(f"Resolved W&B project: {resolved_project}")
    print(f"Resolved W&B run URL: {run_url}")
    if expected_entity and resolved_entity != expected_entity:
        try:
            wandb.finish(exit_code=1)
        finally:
            raise RuntimeError(
                f"W&B entity mismatch: expected {expected_entity!r}, resolved {resolved_entity!r}. "
                "Training was aborted before trainer.fit."
            )
    if expected_project and resolved_project != expected_project:
        try:
            wandb.finish(exit_code=1)
        finally:
            raise RuntimeError(
                f"W&B project mismatch: expected {expected_project!r}, resolved {resolved_project!r}. "
                "Training was aborted before trainer.fit."
            )
    return {"entity": resolved_entity, "project": resolved_project, "run_url": run_url}


def _export_prediction_tables(graph, data, output_dir: str, split_names: List[str]) -> Dict[str, str]:
    """Export compact, stable-ID predictions from the loaded best checkpoint."""
    if not output_dir or not split_names:
        return {}
    os.makedirs(output_dir, exist_ok=True)
    device = next(graph.parameters()).device
    graph.eval()
    exported: Dict[str, str] = {}

    for split in split_names:
        frame = getattr(data, f"df_{split}", None)
        if frame is None or len(frame) == 0:
            continue
        preserved_loaders = getattr(data, "_boda_epoch_eval_loader_fns", {})
        if split in preserved_loaders:
            loader = preserved_loaders[split]()
        else:
            loader_name = "train_eval_dataloader" if split == "train" and hasattr(data, "train_eval_dataloader") else f"{split}_dataloader"
            if not hasattr(data, loader_name):
                raise ValueError(f"Cannot export {split} predictions: {type(data).__name__} has no {loader_name}.")
            loader = getattr(data, loader_name)()
        if loader is None:
            continue

        predictions: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                predictions.append(graph(x.to(device)).detach().cpu())
        if not predictions:
            continue
        pred = torch.cat(predictions, dim=0)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        if len(pred) != len(frame):
            raise RuntimeError(
                f"Prediction row mismatch for {split}: predictions={len(pred)}, metadata={len(frame)}"
            )

        id_column = getattr(data, "split_id_column", None)
        keep_columns = [
            column
            for column in (id_column, "row_id", getattr(data, "barcode_column", None), getattr(data, "target_column", None), "target_processed")
            if column and column in frame.columns
        ]
        output = frame[keep_columns].copy().reset_index(drop=True)
        target_mean = getattr(data, "target_mean", None)
        target_std = getattr(data, "target_std", None)
        for index in range(pred.shape[1]):
            suffix = "" if pred.shape[1] == 1 else f"_{index}"
            processed = pred[:, index].numpy()
            output[f"prediction_processed{suffix}"] = processed
            if isinstance(target_mean, (int, float)) and isinstance(target_std, (int, float)):
                output[f"prediction_raw{suffix}"] = processed * float(target_std) + float(target_mean)

        run_id = str(getattr(wandb.run, "id", "local") or "local")
        path = os.path.join(output_dir, f"{run_id}__{split}_predictions.tsv")
        output.to_csv(path, sep="\t", index=False)
        exported[split] = path
        if wandb.run is not None:
            wandb.run.save(path, base_path=output_dir, policy="end")
            wandb.run.summary[f"{split}_predictions_path"] = path
            wandb.run.summary[f"{split}_predictions_sha256"] = _sha256_file(path)
            wandb.run.summary[f"{split}_prediction_rows"] = int(len(output))
            if "target_processed" in frame.columns and pred.shape[1] == 1:
                labels = torch.as_tensor(
                    frame["target_processed"].to_numpy(), dtype=pred.dtype
                ).reshape(-1, 1)
                try:
                    from boda.graph.utils import (
                        coefficient_of_determination,
                        pearson_correlation,
                        spearman_correlation,
                    )

                    mse = float((pred - labels).pow(2).mean())
                    pearson = float(pearson_correlation(pred, labels)[1])
                    spearman = float(spearman_correlation(pred, labels)[1])
                    cod_r2 = float(coefficient_of_determination(labels, pred))
                    metrics = {
                        "mse": mse,
                        "loss": mse,
                        "pearson": pearson,
                        "pearson_r2": pearson * pearson,
                        "spearman": spearman,
                        "cod_r2": cod_r2,
                    }
                    for metric_name, value in metrics.items():
                        wandb.run.summary[f"best_checkpoint_{split}_{metric_name}"] = value
                        # The registry's canonical final split fields should
                        # describe the loaded best checkpoint, not the last
                        # pre-early-stop epoch.
                        wandb.run.summary[f"{split}_{metric_name}"] = value
                except Exception as exc:
                    print(
                        f"WARN: failed to compute best-checkpoint {split} metrics: {exc}",
                        file=sys.stderr,
                    )
        print(f"Exported {len(output)} {split} predictions to {path}")
    return exported


def _write_compact_provenance(
    provenance: Dict[str, Any],
    args: Dict[str, Any],
    split_summary: Optional[Dict[str, Any]],
    output_dir: str,
) -> str:
    """Keep a small run record even when model retention is disabled."""
    os.makedirs(output_dir, exist_ok=True)
    run_id = provenance.get("run_id") or "local"
    path = os.path.join(output_dir, f"{run_id}__run_provenance.json")
    grouped_args = _resolved_argument_groups(args)
    resolved_hash = _canonical_json_sha256(grouped_args)
    if provenance.get("resolved_arguments_sha256") != resolved_hash:
        raise RuntimeError(
            "Compact provenance resolved-argument hash was not captured from "
            "the exact runtime namespaces."
        )
    payload = dict(provenance)
    payload["data_split_summary"] = split_summary or {}
    payload["resolved_arguments"] = grouped_args
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    if wandb.run is not None:
        wandb.run.save(path, base_path=output_dir, policy="end")
        wandb.run.summary["compact_provenance_path"] = path
        wandb.run.summary["compact_provenance_sha256"] = _sha256_file(path)
    print(f"Wrote compact provenance to {path}")
    return path


def main(args):
    """
    Main function for training a model using Pytorch Lightning with W&B integration.
    
    Args:
        args: Dictionary containing all input arguments organized by module
        
    Returns:
        None
    """
    # Validate cross-cutting policies before allocating a model or GPU.
    main_args = args['Main args']
    _validate_campaign_wandb_contract(main_args)
    stage4_manifest_row = _validate_stage4_manifest_launch_contract(main_args)
    artifact_retention = str(getattr(main_args, 'artifact_retention', 'all')).lower()
    if artifact_retention not in {'none', 'selected', 'all'}:
        raise ValueError(
            f"Unknown artifact_retention={artifact_retention!r}; use none, selected, or all."
        )
    epoch_eval_splits_requested = _coerce_split_list(main_args.epoch_eval_splits)
    if not bool(main_args.evaluate_test_after_fit) and 'test' in epoch_eval_splits_requested:
        raise ValueError(
            "evaluate_test_after_fit=false cannot be combined with epoch_eval_splits containing test; "
            "that policy is intended to make test/audit loaders unavailable during model selection."
        )

    # Get the module classes
    data_module = getattr(boda.data, args['Main args'].data_module)
    model_module = getattr(boda.model, args['Main args'].model_module)
    graph_module = getattr(boda.graph, args['Main args'].graph_module)

    model_seed = getattr(args['Main args'], 'model_seed', None)
    if model_seed is not None:
        utils.set_all_seeds(int(model_seed))
        print(f"Set model/training random seed: {model_seed}")

    # Initialize data module
    data_args = data_module.process_args(args)
    if str(getattr(data_args, 'manifest_mode', 'development')) == 'audit_eval':
        raise ValueError(
            'audit_eval is forbidden in train_wandb_log.py; use the separate '
            'checkpoint-allowlist-bound one-time scorer.'
        )
    _validate_stage3_weighted_contract(main_args, data_args)
    _validate_stage4_downsampling_contract(main_args, data_args)
    _validate_stage4_resolved_row_contract(
        stage4_manifest_row, main_args, data_args, args['pl.Trainer']
    )
    _validate_final_refit_contract(main_args, data_args, args['pl.Trainer'])
    
    # Special handling for list-type arguments that might come from YAML config
    if hasattr(data_args, 'activity_columns'):
        data_args.activity_columns = convert_to_list(data_args.activity_columns)
    if hasattr(data_args, 'stderr_columns'):
        data_args.stderr_columns = convert_to_list(data_args.stderr_columns)
    if hasattr(data_args, 'val_chrs'):
        data_args.val_chrs = convert_to_list(data_args.val_chrs)
    if hasattr(data_args, 'test_chrs'):
        data_args.test_chrs = convert_to_list(data_args.test_chrs)
    
    data = data_module(**vars(data_args))
    
    # Initialize model
    model = model_module(**vars(model_module.process_args(args)))
    model_parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    model_trainable_parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    
    # Initialize graph module with model
    graph = graph_module(model=model, **vars(graph_module.process_args(args)))
    # Transfer adapters can change trainability and attach weights during graph
    # construction. Report the post-adapter state, not the raw model default.
    model_parameter_count = int(sum(parameter.numel() for parameter in graph.model.parameters()))
    model_trainable_parameter_count = int(
        sum(parameter.numel() for parameter in graph.model.parameters() if parameter.requires_grad)
    )

    epoch_eval_splits = configure_epoch_eval_dataloaders(
        data,
        graph,
        epoch_eval_splits_requested,
    )
    print(f"Epoch diagnostic eval splits: {epoch_eval_splits}")

    # Set up logger based on command-line input
    run_id = ""
    run_name = args['Main args'].run_name
    os.environ["BODA_REQUIRE_WANDB_HISTORY"] = "0"
    if args['Main args'].logger_type.lower() == 'wandb':
        wandb_mode = os.environ.get("WANDB_MODE", "").strip().lower()
        if wandb_mode in {"disabled", "dryrun", "offline"}:
            raise RuntimeError(
                "logger_type=wandb requires online W&B history for standardized HPO. "
                "Unset WANDB_MODE or set WANDB_MODE=online before launching."
            )
        try:
            # Generate a unique run ID and process the run name
            run_id = wandb.util.generate_id()
            if args['Main args'].exact_run_name:
                run_name = args['Main args'].run_name
            elif "{runid}" in args['Main args'].run_name:
                run_name = args['Main args'].run_name.replace("{runid}", run_id)
            else:
                run_name = f"{args['Main args'].run_name}_{run_id}"
            print(f"Original run_name: {args['Main args'].run_name}")
            print(f"Generated run_id: {run_id}")
            print(f"Processed run_name: {run_name}")

            # Initialize W&B logger
            wandb_tags = convert_to_list(getattr(args['Main args'], 'wandb_tags', [])) or []
            logger = WandbLogger(
                id=run_id,
                entity=args['Main args'].wandb_entity or None,
                project=args['Main args'].logger_project,
                name=run_name,
                group=args['Main args'].wandb_group or None,
                job_type=args['Main args'].wandb_job_type or None,
                tags=[str(tag) for tag in wandb_tags],
                log_model=(artifact_retention == 'all'),
            )
            
            # Log more comprehensive hyperparameters
            all_hparams = {}
            for group_name, group_args in args.items():
                if isinstance(group_args, argparse.Namespace):
                    group_dict = {f"{group_name}.{k}": v for k, v in vars(group_args).items()}
                    all_hparams.update(group_dict)
            
            logger.log_hyperparams(all_hparams)
            # Accessing experiment forces wandb.init; identity validation must
            # happen now, before Trainer and trainer.fit are constructed.
            _ = logger.experiment
            identity = _assert_wandb_identity(
                args['Main args'].wandb_entity,
                args['Main args'].logger_project,
            )
            campaign_fields = _campaign_wandb_fields(args['Main args'])
            wandb.run.config.update(campaign_fields, allow_val_change=True)
            for key, value in campaign_fields.items():
                if value is not None:
                    wandb.run.summary[key] = value
            wandb.run.summary['resolved_wandb_entity'] = identity['entity']
            wandb.run.summary['resolved_wandb_project'] = identity['project']
            wandb.run.summary['resolved_wandb_run_url'] = identity['run_url']
            wandb.run.summary['wandb_model_logging_enabled'] = artifact_retention == 'all'
            wandb.run.summary['model_parameter_count'] = model_parameter_count
            wandb.run.summary['model_trainable_parameter_count'] = model_trainable_parameter_count
            history_contract_splits = list(epoch_eval_splits_requested)
            if 'train' not in history_contract_splits:
                history_contract_splits.append('train')
            if bool(main_args.evaluate_test_after_fit) and 'test' not in history_contract_splits:
                history_contract_splits.append('test')
            _configure_wandb_history_contract(history_contract_splits)
            os.environ["BODA_REQUIRE_WANDB_HISTORY"] = "1"
            
            print(f"Initialized Wandb logging with run ID: {run_id}, name: {run_name}")
        except Exception as e:
            raise RuntimeError(
                "W&B initialization failed for logger_type=wandb. "
                "The run was not started because standardized Lib1 HPO requires "
                "cloud history rows for requested train/validation metrics. Check `wandb login`, "
                "WANDB_API_KEY, WANDB_MODE, entity/project permissions, and network connectivity."
            ) from e
    elif args['Main args'].logger_type.lower() == 'tensorboard':
        logger = pl_loggers.TensorBoardLogger(
            save_dir='./logs',
            name=args['Main args'].logger_project
        )
    else:
        logger = True  # Default Lightning logger

    print(f"Original run_name: {args['Main args'].run_name}")
    print(f"Generated run_id: {run_id}")
    print(f"Processed run_name: {run_name}")

    trainer_root_dir = args['pl.Trainer'].default_root_dir
    if trainer_root_dir is None:
        trainer_root_dir = '/tmp/output/artifacts'
    if not str(trainer_root_dir).startswith('gs://'):
        trainer_root_dir = os.path.abspath(os.path.expanduser(str(trainer_root_dir)))
    args['pl.Trainer'].default_root_dir = trainer_root_dir

    # Set up callbacks
    use_callbacks = {
        'learning_rate_monitor': LearningRateMonitor()
    }
    
    if args['Main args'].checkpoint_monitor is not None:
        checkpoint_dir = (
            os.path.join(trainer_root_dir, 'checkpoints')
            if not str(trainer_root_dir).startswith('gs://')
            else os.path.join(trainer_root_dir, 'checkpoints')
        )
        use_callbacks['model_checkpoint'] = ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=1,
            monitor=args['Main args'].checkpoint_monitor,
            mode=args['Main args'].stopping_mode
        )
        use_callbacks['early_stopping'] = EarlyStopping(
            monitor=args['Main args'].checkpoint_monitor,
            patience=args['Main args'].stopping_patience,
            mode=args['Main args'].stopping_mode
        )

    # Ensure output directory exists
    os.makedirs('/tmp/output/artifacts', exist_ok=True)
    
    # Create trainer
    trainer = Trainer.from_argparse_args(
        args['pl.Trainer'],
        callbacks=list(use_callbacks.values()),
        logger=logger
    )

    # Train the model
    fit_started_at = time.monotonic()
    trainer.fit(graph, data)
    # Capture the total completed optimizer updates before set_best() can load
    # an earlier checkpoint and change graph-side step state.
    optimizer_steps_after_fit = int(trainer.global_step)
    if str(getattr(main_args, 'campaign_stage', '') or '') == 'stage4_downsampling':
        if optimizer_steps_after_fit <= 0:
            raise RuntimeError('Stage 4 completed no optimizer updates; refusing success.')
    fit_wall_time_seconds = float(time.monotonic() - fit_started_at)
    if str(getattr(main_args, 'campaign_stage', '') or '') == 'final_refit':
        expected_epochs = int(FINAL_REFIT_POLICIES[main_args.part_slug]['fixed_epochs'])
        if int(trainer.current_epoch) != expected_epochs:
            raise RuntimeError(
                f'Final refit ended at trainer.current_epoch={trainer.current_epoch}; '
                f'expected exactly {expected_epochs} completed epochs.'
            )
    print(f"Trainer fit wall time: {fit_wall_time_seconds:.3f} seconds")
    if wandb.run is not None:
        wandb.run.summary['fit_wall_time_seconds'] = fit_wall_time_seconds
        wandb.run.summary['optimizer_steps'] = optimizer_steps_after_fit

    # Load the best model
    graph = set_best(graph, use_callbacks)

    # End-of-fit evaluation on explicitly allowed test and clean train splits
    # using the best checkpoint.
    # trainer.test handles device/precision and calls `test_epoch_end` which
    # logs R2/Pearson/Spearman/loss to wandb summary. We then do a lightweight
    # inference pass over the training loader to obtain train metrics.
    _run_optional_postfit_test(
        trainer,
        graph,
        data,
        enabled=bool(args['Main args'].evaluate_test_after_fit),
    )

    try:
        _log_train_eval_metrics(graph, data)
    except Exception as exc:
        print(f"WARN: train-set evaluation failed: {exc}", file=sys.stderr)

    try:
        _log_library_split_eval_metrics(graph, data)
    except Exception as exc:
        print(f"WARN: split/library evaluation failed: {exc}", file=sys.stderr)

    prediction_dir = args['Main args'].prediction_output_dir or os.path.join(
        str(args['pl.Trainer'].default_root_dir), 'predictions'
    )
    exported_predictions = _export_prediction_tables(
        graph,
        data,
        prediction_dir,
        _coerce_split_list(args['Main args'].prediction_splits)
        if args['Main args'].prediction_splits
        else [],
    )

    # Report metrics and save the model
    try:
        if 'model_checkpoint' in use_callbacks:
            mc_dict = vars(use_callbacks['model_checkpoint'])
            keys = ['monitor', 'best_model_score']
            tag, metric = [mc_dict[key] for key in keys]
            
            # Report to hypertune if available
            try:
                graph.hpt.report_hyperparameter_tuning_metric(
                    hyperparameter_metric_tag=tag,
                    metric_value=metric.item(),
                    global_step=graph.global_step + 1
                )
            except (AttributeError, NameError):
                pass
                
            print(f'{tag} at {graph.global_step}: {metric}', file=sys.stderr)
    except (KeyError, AttributeError):
        print("Couldn't report best metric, using final model state", file=sys.stderr)

    # Build provenance once so it lands in both the tarball and runs.csv.
    # The W&B run is still live here; identifiers are available from wandb.run.
    provenance = build_provenance_record(args, use_callbacks, artifact_path=None, status='completed')
    provenance['optimizer_steps'] = optimizer_steps_after_fit
    provenance['resolved_arguments_sha256'] = _canonical_json_sha256(
        _resolved_argument_groups(args)
    )
    if wandb.run is not None:
        for key in (
            'best_epoch',
            'best_metric_name',
            'best_metric_value',
            'campaign_id',
            'campaign_stage',
            'part_slug',
            'analysis_lane',
            'challenger_family',
            'policy_id',
            'config_origin',
            'training_regime',
            'cell_id',
            'rc_pair_id',
            'loss_pair_id',
            'source_unweighted_cell_id',
            'rc_mode',
            'execution_disposition',
            'initialization',
            'source_head',
            'unfreeze_scope',
            'input_policy',
            'pretrained_artifact_sha256',
            'data_generation_id',
            'dataset_sha256',
            'split_manifest_id',
            'split_manifest_sha256',
            'development_fold',
            'base_config_id',
            'architecture',
            'model_seed',
            'loss_mode',
            'artifact_retention',
            'optimizer_steps',
            'config_manifest_sha256',
            'manifest_row',
            'manifest_row_fingerprint',
            'runtime_argv_sha256',
            'resolved_arguments_sha256',
            'run_registry_path',
        ):
            value = provenance.get(key)
            if value not in (None, ''):
                wandb.run.summary[key] = value
    split_summary = getattr(data, 'split_summary', None)
    if isinstance(split_summary, dict):
        provenance['data_split_summary'] = split_summary
        provenance['train_row_id_hash'] = split_summary.get('train_row_id_hash', split_summary.get('train_final_row_id_hash', ''))
        provenance['val_row_id_hash'] = split_summary.get('val_row_id_hash', '')
        provenance['audit_row_id_hash'] = split_summary.get('audit_row_id_hash', '')
        provenance['normalization_row_id_hash'] = split_summary.get(
            'normalization_row_id_hash', provenance['train_row_id_hash']
        )
        provenance['selected_row_hash'] = split_summary.get(
            'selected_row_hash', provenance['train_row_id_hash']
        )
        if wandb.run is not None:
            try:
                wandb.run.summary['data_split_summary_json'] = json.dumps(split_summary, sort_keys=True, default=str)
                for key in (
                    'train_subsample_seed',
                    'train_pool_row_id_hash',
                    'train_final_row_id_hash',
                    'train_row_id_hash',
                    'val_row_id_hash',
                    'audit_row_id_hash',
                    'normalization_row_id_hash',
                    'selected_row_hash',
                    'dataset_sha256',
                    'split_manifest_sha256',
                ):
                    if key in split_summary:
                        wandb.run.summary[f'data_split_{key}'] = split_summary[key]
            except Exception:
                pass

    if exported_predictions:
        provenance['prediction_path'] = exported_predictions.get('val') or next(iter(exported_predictions.values()))

    artifact_file = None
    if artifact_retention in {'selected', 'all'}:
        artifact_file = save_model(
            data_module, model_module, graph_module, graph.model, trainer, args,
            use_callbacks=use_callbacks,
            provenance_record=provenance,
        )
    else:
        removed_checkpoints = prune_lightning_checkpoints(
            use_callbacks=use_callbacks,
            keep=False,
            extra_checkpoint_dirs=[os.path.join(trainer_root_dir, 'checkpoints')],
        )
        if removed_checkpoints:
            print(f"Pruned {len(removed_checkpoints)} transient Lightning checkpoint(s).")
        if wandb.run is not None:
            wandb.run.summary['model_artifact_retained'] = False
            wandb.run.summary['pruned_lightning_checkpoint_count'] = len(removed_checkpoints)
    if artifact_file is not None:
        provenance['artifact_path'] = artifact_file

    provenance_dir = args['Main args'].provenance_output_dir or os.path.join(
        str(args['pl.Trainer'].default_root_dir), 'provenance'
    )
    _write_compact_provenance(provenance, args, split_summary, provenance_dir)

    csv_path = append_runs_csv_row(provenance)
    if csv_path is None and getattr(args['Main args'], 'campaign_id', ''):
        raise RuntimeError(
            'Campaign training completed but the required structured run-registry '
            'row could not be written; refusing to report success.'
        )
    if csv_path is not None:
        print(f"Appended run row to: {csv_path}")
    if wandb.run is not None:
        try:
            wandb.run.summary['provenance_csv_path'] = csv_path or ''
            wandb.run.summary['run_id_recorded'] = provenance.get('run_id', '')
        except Exception:
            pass

    if args['Main args'].logger_type.lower() == 'wandb' and wandb.run is not None:
        wandb.finish()

if __name__ == '__main__':
    print("============Starting BODA training script with W&B integration==============")
    # Build the base parser
    parser = argparse.ArgumentParser(description="BODA trainer with W&B integration", add_help=False)
    group = parser.add_argument_group('Main args')
    group.add_argument('--data_module', type=str, required=True,
                       help='BODA data module to process dataset.')
    group.add_argument('--model_module', type=str, required=True,
                       help='BODA model module to fit dataset.')
    group.add_argument('--graph_module', type=str, required=True,
                       help='BODA graph module to define computations.')
    group.add_argument('--artifact_path', type=str, default='/opt/ml/checkpoints/',
                       help='Path where model artifacts are deposited.')
    group.add_argument('--best_checkpoint_dir', type=str, default='',
                       help='Optional clean directory where each run publishes its best model bundle under <run_id>/.')
    group.add_argument('--keep_lightning_checkpoints', type=utils.str2bool, default=False,
                       help='Keep transient Lightning .ckpt files after exporting the portable artifact.')
    group.add_argument('--artifact_retention', choices=['none', 'selected', 'all'], default='all',
                       help='none keeps only metrics/predictions/provenance; selected keeps a local model; all also enables W&B model logging.')
    group.add_argument('--evaluate_test_after_fit', type=utils.str2bool, default=True,
                       help='Whether to call trainer.test after fit. Use false for selection/replay runs without audit access.')
    group.add_argument('--prediction_output_dir', type=str, default='',
                       help='Directory for compact best-checkpoint prediction tables; defaults under default_root_dir.')
    group.add_argument('--prediction_splits', type=str, nargs='+', default=[],
                       help='Optional splits to export after loading the best checkpoint, e.g. val.')
    group.add_argument('--provenance_output_dir', type=str, default='',
                       help='Directory for compact run provenance; defaults under default_root_dir.')
    group.add_argument('--pretrained_weights', type=str, help='Pretrained weights.')
    group.add_argument('--checkpoint_monitor', type=str,
                       help='String to monitor PTL logs if saving best.')
    group.add_argument('--stopping_mode', type=str, default='min',
                       help='Goal for monitored metric e.g. (max or min).')
    group.add_argument('--stopping_patience', type=int, default=100,
                       help='Number of epochs of non-improvement tolerated before early stopping.')
    group.add_argument('--tolerate_unknown_args', type=utils.str2bool, default=False,
                       help='Skips unknown command line args without exceptions. Useful for HPO, but high risk of silent errors.')

    # New logger control arguments (renamed to avoid conflict with Trainer's logger)
    group.add_argument('--logger_type', type=str, default='wandb',
                       help='Which logger to use (wandb, tensorboard, none)')
    group.add_argument('--logger_project', type=str, default='boda_train',
                       help='Project name for the logger.')
    group.add_argument('--wandb_entity', type=str, default='',
                       help='Explicit W&B entity. When set, the resolved live run must match before fit.')
    group.add_argument('--wandb_group', type=str, default='',
                       help='Optional W&B run group.')
    group.add_argument('--wandb_job_type', type=str, default='',
                       help='Optional W&B job type.')
    group.add_argument('--wandb_tags', type=str, nargs='*', default=[],
                       help='Optional W&B tags.')
    group.add_argument('--run_name', type=str, default='default_run',
                       help='Run name for the logger.')
    group.add_argument('--exact_run_name', type=utils.str2bool, default=False,
                       help='Use run_name exactly instead of appending/replacing a generated run id.')
    group.add_argument('--epoch_eval_splits', type=str, nargs='+', default=['val'],
                       help='Splits to evaluate every validation epoch, e.g. val or train val test.')
    group.add_argument('--model_seed', type=int, default=None,
                       help='Optional seed for model initialization and trainer-side randomness.')
    group.add_argument('--campaign_id', type=str, default='')
    group.add_argument('--campaign_stage', type=str, default='')
    group.add_argument('--part_slug', type=str, default='')
    group.add_argument('--analysis_lane', type=str, default='')
    group.add_argument('--challenger_family', type=str, default='')
    group.add_argument('--policy_id', type=str, default='')
    group.add_argument('--config_origin', type=str, default='')
    group.add_argument('--training_regime', type=str, default='')
    group.add_argument('--cell_id', type=str, default='')
    group.add_argument('--rc_pair_id', type=str, default='')
    group.add_argument('--loss_pair_id', type=str, default='')
    group.add_argument('--source_unweighted_cell_id', type=str, default='')
    group.add_argument('--rc_mode', type=str, default='')
    group.add_argument('--execution_disposition', type=str, default='')
    group.add_argument('--initialization', type=str, default='')
    group.add_argument('--source_head', type=str, default='')
    group.add_argument('--unfreeze_scope', type=str, default='')
    group.add_argument('--input_policy', type=str, default='')
    group.add_argument('--pretrained_artifact_sha256', type=str, default='')
    group.add_argument('--data_generation_id', type=str, default='')
    group.add_argument('--dataset_sha256', type=str, default='')
    group.add_argument('--split_manifest_id', type=str, default='')
    group.add_argument('--split_manifest_sha256', type=str, default='')
    group.add_argument('--development_fold', type=int, default=None)
    group.add_argument('--base_config_id', type=str, default='')
    group.add_argument('--source_run_ids', type=str, nargs='*', default=[])
    group.add_argument('--architecture', type=str, default='')
    group.add_argument('--loss_mode', type=str, default='')
    group.add_argument('--target_definition', type=str, default='')
    group.add_argument('--length_policy', type=str, default='')

    # Parse initial arguments to get module classes
    known_args, leftover_args = parser.parse_known_args()
    
    # Import the respective modules
    try:
        Data = getattr(boda.data, known_args.data_module)
        Model = getattr(boda.model, known_args.model_module)
        Graph = getattr(boda.graph, known_args.graph_module)
    except AttributeError as e:
        print(f"Error: {str(e)}")
        print(f"Available data modules: {dir(boda.data)}")
        print(f"Available model modules: {dir(boda.model)}")
        print(f"Available graph modules: {dir(boda.graph)}")
        sys.exit(1)

    # Add module-specific arguments
    parser = Data.add_data_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    parser = Graph.add_graph_specific_args(parser)
    
    # Get updated known arguments
    known_args, leftover_args = parser.parse_known_args()
    
    # Add conditional arguments based on the known arguments
    parser = Data.add_conditional_args(parser, known_args)
    parser = Model.add_conditional_args(parser, known_args)
    parser = Graph.add_conditional_args(parser, known_args)
    
    # Add Trainer-specific arguments
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--help', '-h', action='help')
    
    # Parse all arguments
    args, leftover_args = parser.parse_known_args()
    if args.tolerate_unknown_args:
        print("Skipping unexpected args. Check leftovers for typos:", file=sys.stderr)
        print(leftover_args, file=sys.stderr)
    else:
        if _leftovers_are_ignorable_scheduler_args(leftover_args, getattr(args, 'scheduler', None)):
            print(
                "Ignoring scheduler-specific args because scheduler is None:",
                leftover_args,
                file=sys.stderr,
            )
        elif leftover_args:
            parser.error(f"unrecognized arguments: {' '.join(leftover_args)}")
    
    # Organize arguments into groups
    args = utils.organize_args(parser, args)
    
    # Print argument summary
    print('-' * 80)
    print("Starting training with configuration:")
    for group_title, namespace_obj in args.items():
        print(f"\n{group_title}:")
        for key, value in sorted(vars(namespace_obj).items()):
            print(f"  {key}: {value}")
    print('-' * 80)
    
    # Start the training
    main(args)
