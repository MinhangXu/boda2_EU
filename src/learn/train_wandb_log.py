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
        "data_module": main.get("data_module", "") or "",
        "model_module": main.get("model_module", "") or "",
        "graph_module": main.get("graph_module", "") or "",
        "checkpoint_monitor": main.get("checkpoint_monitor", "") or "",
        "best_epoch": best_epoch if best_epoch is not None else "",
        "best_metric_name": best_metric_name or "",
        "best_metric_value": best_metric_value if best_metric_value is not None else "",
        "val_loss": _get("val_loss"),
        "val_r2": _get_first("val_pearson_r2", "epoch_end_val_pearson_r2", "epoch_end_val_r2", "val_r2_score"),
        "val_pearson": _get("val_pearson") if _get("val_pearson") is not None else _get("epoch_end_val_pearson"),
        "val_spearman": _get("val_spearman") if _get("val_spearman") is not None else _get("epoch_end_val_spearman"),
        "test_loss": _get("test_loss"),
        "test_r2": _get_first("test_pearson_r2", "epoch_end_test_pearson_r2", "test_r2", "epoch_end_test_r2"),
        "test_pearson": _get("test_pearson") if _get("test_pearson") is not None else _get("epoch_end_test_pearson"),
        "test_spearman": _get("test_spearman") if _get("test_spearman") is not None else _get("epoch_end_test_spearman"),
        "train_loss": _get("train_loss"),
        "train_r2": _get_first("train_pearson_r2", "epoch_end_train_pearson_r2", "train_r2", "epoch_end_train_r2"),
        "train_pearson": _get("train_pearson") if _get("train_pearson") is not None else _get("epoch_end_train_pearson"),
        "train_spearman": _get("train_spearman") if _get("train_spearman") is not None else _get("epoch_end_train_spearman"),
        "artifact_path": artifact_path or "",
        "status": status,
        "hostname": socket.gethostname(),
        "git_commit": _resolve_git_commit() or "",
        "notes": os.environ.get("BODA_LAUNCH_NOTES", os.environ.get("LAUNCH_NOTES", "")),
    }
    # Replace None with "" for CSV-friendliness.
    for k, v in list(record.items()):
        if v is None:
            record[k] = ""
    return record


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

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    is_new = not os.path.isfile(target_path)
    try:
        with open(target_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=RUNS_CSV_COLUMNS, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            writer.writerow({col: record.get(col, "") for col in RUNS_CSV_COLUMNS})
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
            tar.add(local_dir, arcname='artifacts')

        if 'gs://' in artifact_dir:
            final_artifact_path = os.path.join(artifact_dir, filename)
            subprocess.check_call(['gsutil', 'cp', tar_src, final_artifact_path])
        else:
            os.makedirs(artifact_dir, exist_ok=True)
            final_artifact_path = os.path.join(artifact_dir, filename)
            shutil.copy(tar_src, final_artifact_path)

    print(f"Model saved to {final_artifact_path}")

    if wandb.run is not None:
        wandb.run.summary["model_saved_path"] = final_artifact_path
        wandb.run.summary["model_artifact_filename"] = filename

    return final_artifact_path


#######################
# Main Training Logic #
#######################

def _log_train_eval_metrics(graph, data):
    """
    Run a single forward pass over the training dataloader using the best
    checkpoint and log (train_loss, train_pearson_r2, train_cod_r2,
    train_pearson, train_spearman) to the active W&B run summary so
    `runs.csv` gets populated.

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
        loader = data.train_dataloader()
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
        train_pearson_r2 = float(pearson_r2_score(all_labels, all_preds))
    except Exception:
        train_pearson_r2 = None
    try:
        train_cod_r2 = float(coefficient_of_determination(all_labels, all_preds))
    except Exception:
        train_cod_r2 = None
    try:
        _, train_pearson = pearson_correlation(all_preds, all_labels)
        train_pearson = float(train_pearson)
    except Exception:
        train_pearson = None
    try:
        _, train_spearman = spearman_correlation(all_preds, all_labels)
        train_spearman = float(train_spearman)
    except Exception:
        train_spearman = None
    train_loss = float(torch.stack(losses).mean()) if losses else None

    summary_updates = {
        "train_loss": train_loss,
        "train_pearson_r2": train_pearson_r2,
        "train_cod_r2": train_cod_r2,
        "train_pearson": train_pearson,
        "train_spearman": train_spearman,
    }
    for k, v in summary_updates.items():
        if v is not None:
            try:
                wandb.run.summary[k] = v
            except Exception:
                pass
    print("Train-set eval summary: " + ", ".join(
        f"{k}={v}" for k, v in summary_updates.items() if v is not None
    ))


def _has_overridden_dataloader(data_module: Any, loader_name: str) -> bool:
    method = getattr(type(data_module), loader_name, None)
    base_method = getattr(LightningDataModule, loader_name, None)
    return method is not None and method is not base_method


def main(args):
    """
    Main function for training a model using Pytorch Lightning with W&B integration.
    
    Args:
        args: Dictionary containing all input arguments organized by module
        
    Returns:
        None
    """
    # Get the module classes
    data_module = getattr(boda.data, args['Main args'].data_module)
    model_module = getattr(boda.model, args['Main args'].model_module)
    graph_module = getattr(boda.graph, args['Main args'].graph_module)

    # Initialize data module
    data_args = data_module.process_args(args)
    
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
    
    # Initialize graph module with model
    graph = graph_module(model=model, **vars(graph_module.process_args(args)))

    # Set up logger based on command-line input
    if args['Main args'].logger_type.lower() == 'wandb':
        try:
            # Generate a unique run ID and process the run name
            run_id = wandb.util.generate_id()
            if "{runid}" in args['Main args'].run_name:
                run_name = args['Main args'].run_name.replace("{runid}", run_id)
            else:
                run_name = f"{args['Main args'].run_name}_{run_id}"
            print(f"Original run_name: {args['Main args'].run_name}")
            print(f"Generated run_id: {run_id}")
            print(f"Processed run_name: {run_name}")

            # Initialize W&B logger
            logger = WandbLogger(
                project=args['Main args'].logger_project,
                name=run_name,
                log_model=True
            )
            
            # Log more comprehensive hyperparameters
            all_hparams = {}
            for group_name, group_args in args.items():
                if isinstance(group_args, argparse.Namespace):
                    group_dict = {f"{group_name}.{k}": v for k, v in vars(group_args).items()}
                    all_hparams.update(group_dict)
            
            logger.log_hyperparams(all_hparams)
            
            print(f"Initialized Wandb logging with run ID: {run_id}, name: {run_name}")
        except Exception as e:
            print(f"Wandb initialization failed: {str(e)}")
            print("Falling back to default logger")
            logger = True
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

    # Set up callbacks
    use_callbacks = {
        'learning_rate_monitor': LearningRateMonitor()
    }
    
    if args['Main args'].checkpoint_monitor is not None:
        use_callbacks['model_checkpoint'] = ModelCheckpoint(
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
    trainer.fit(graph, data)

    # Load the best model
    graph = set_best(graph, use_callbacks)

    # End-of-fit evaluation on test and train splits using the best checkpoint.
    # trainer.test handles device/precision and calls `test_epoch_end` which
    # logs R2/Pearson/Spearman/loss to wandb summary. We then do a lightweight
    # inference pass over the training loader to obtain train metrics.
    try:
        test_loader = None
        if _has_overridden_dataloader(data, "test_dataloader"):
            try:
                test_loader = data.test_dataloader()
            except Exception as exc:
                print(f"WARN: could not build test_dataloader: {exc}", file=sys.stderr)
        if test_loader is not None:
            trainer.test(graph, dataloaders=test_loader)
    except Exception as exc:
        print(f"WARN: trainer.test failed: {exc}", file=sys.stderr)

    try:
        _log_train_eval_metrics(graph, data)
    except Exception as exc:
        print(f"WARN: train-set evaluation failed: {exc}", file=sys.stderr)

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

    artifact_file = save_model(
        data_module, model_module, graph_module, graph.model, trainer, args,
        use_callbacks=use_callbacks,
        provenance_record=provenance,
    )
    if artifact_file is not None:
        provenance['artifact_path'] = artifact_file

    csv_path = append_runs_csv_row(provenance)
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
    group.add_argument('--run_name', type=str, default='default_run',
                       help='Run name for the logger.')

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
