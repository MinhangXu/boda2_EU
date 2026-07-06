#!/usr/bin/env python3
"""Generate the Lib1 no-RC outer-split-seed prior-informed HPO manifest.

This lifts the manifest prototype from
``tutorials/lib1_tasks/pretrain_CRE_inhouse_data/
lib1_inhouse_scratch_hpo_seed_split_diagnostics_june2026.ipynb`` into a
reusable script. It uses only validation metrics to rank prior runs; test
metrics are copied as diagnostic source metadata.
"""

import argparse
import json
import math
import shlex
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover - environment guard
    yaml = None


LEARN_DIR = Path(__file__).resolve().parent
REPO_DIR = LEARN_DIR.parent.parent
RUN_REGISTRY = LEARN_DIR / "run_registry"
WANDB_CACHE = LEARN_DIR / "wandb"
DEFAULT_OUTDIR = LEARN_DIR / "outputs" / "hpo_manifests"

MANIFEST_TAG = "lib1_outer_seed_prior_no_rc_june2026"
DEFAULT_WANDB_ENTITY = "minhangxu1998-baylor-college-of-medicine"
OUTER_SEED_SPLIT_SEEDS = [101, 202, 303, 404, 505]
OUTER_SEED_MODEL_SEED = 1701
N_EXACT_ELITE = 8
N_LOCAL_VARIANT = 12
N_NARROW_PRIOR = 10


PROJECT_ROWS = [
    ("promoter__bashor_in_house__lib1_allvalid__scratch__resnet1d", "Promoter", "ResNet1D", "promoter"),
    ("promoter__bashor_in_house__lib1_allvalid__scratch__promoter_bassetvl", "Promoter", "BassetVL", "promoter"),
    ("introns__bashor_in_house__lib1_intron_modal80__scratch__resnet1d", "Intron", "ResNet1D", "introns"),
    ("utr3__bashor_in_house__threeprime_modal100__scratch__resnet1d_fp32", "3UTR", "ResNet1D", "utr3"),
    ("utr3__bashor_in_house__threeprime_modal100__scratch__utr_bassetvl_fp32", "3UTR", "BassetVL", "utr3"),
    ("utr5__bashor_in_house__fiveprime_modal50__scratch__resnet1d_fp32", "5UTR", "ResNet1D", "utr5"),
    ("utr5__bashor_in_house__fiveprime_modal50__scratch__utr_bassetvl_fp32", "5UTR", "BassetVL", "utr5"),
    ("enhancer__bashor_in_house__no_flank_hq8__scratch__resnet1d_fp32", "Enhancer", "ResNet1D", "enhancer"),
    ("enhancer__bashor_in_house__no_flank_hq8__scratch__bassetvl_fp32", "Enhancer", "BassetVL", "enhancer"),
]


PART_PRIORS = {
    "Promoter": {
        "part_slug": "promoter",
        "architecture": "PromoterBassetVL",
        "source_architecture": "BassetVL",
        "model_module": "PromoterBassetVL",
        "task_family": "promoter",
        "target_family": "bashor_in_house_lib1_promoter_allvalid_fastqs1_5",
        "source_logger_project": "promoter__bashor_in_house__lib1_allvalid__scratch__promoter_bassetvl",
        "logger_project": "promoter__bashor_in_house__lib1_allvalid__outer_seed_prior_no_rc__promoter_bassetvl",
        "comparison_group": "promoter__bashor_in_house__lib1_allvalid__outer_seed_prior_no_rc__promoter_bassetvl",
        "prior_sweep_id": "vi17zxcm",
        "source_config_path": "configs/promoter/bashor_in_house/promoter_bassetvl/lib1_promoter__scratch_promoter_bassetvl__bayes.yml",
        "trainer_overrides": {"max_epochs": 220, "stopping_patience": 35},
    },
    "Intron": {
        "part_slug": "intron",
        "architecture": "ResNet1DRegressor",
        "source_architecture": "ResNet1D",
        "model_module": "ResNet1DRegressor",
        "task_family": "introns",
        "target_family": "bashor_in_house_lib1_intron_modal80_fastqs1_5",
        "source_logger_project": "introns__bashor_in_house__lib1_intron_modal80__scratch__resnet1d",
        "logger_project": "introns__bashor_in_house__lib1_intron_modal80__outer_seed_prior_no_rc__resnet1d",
        "comparison_group": "introns__bashor_in_house__lib1_intron_modal80__outer_seed_prior_no_rc__resnet1d",
        "prior_sweep_id": "5b0njbjz",
        "source_config_path": "configs/introns/bashor_in_house/resnet1d/lib1_intron_modal80__scratch_resnet1d__bayes.yml",
        "trainer_overrides": {"max_epochs": 180, "stopping_patience": 35},
    },
    "3UTR": {
        "part_slug": "utr3",
        "architecture": "ResNet1DRegressor",
        "source_architecture": "ResNet1D",
        "model_module": "ResNet1DRegressor",
        "task_family": "utr3",
        "target_family": "bashor_in_house_threeprime_modal100_fastqs1_5",
        "source_logger_project": "utr3__bashor_in_house__threeprime_modal100__scratch__resnet1d_fp32",
        "logger_project": "utr3__bashor_in_house__threeprime_modal100__outer_seed_prior_no_rc__resnet1d_fp32",
        "comparison_group": "utr3__bashor_in_house__threeprime_modal100__outer_seed_prior_no_rc__resnet1d_fp32",
        "prior_sweep_id": "bnyvegba",
        "source_config_path": "configs/utr3/bashor_in_house/resnet1d/lib1_threeprime__scratch_resnet1d__bayes.yml",
        "trainer_overrides": {"max_epochs": 180, "stopping_patience": 30},
    },
    "5UTR": {
        "part_slug": "utr5",
        "architecture": "ResNet1DRegressor",
        "source_architecture": "ResNet1D",
        "model_module": "ResNet1DRegressor",
        "task_family": "utr5",
        "target_family": "bashor_in_house_lib1_fiveprime_modal50_fastqs1_5",
        "source_logger_project": "utr5__bashor_in_house__fiveprime_modal50__scratch__resnet1d_fp32",
        "logger_project": "utr5__bashor_in_house__fiveprime_modal50__outer_seed_prior_no_rc__resnet1d_fp32",
        "comparison_group": "utr5__bashor_in_house__fiveprime_modal50__outer_seed_prior_no_rc__resnet1d_fp32",
        "prior_sweep_id": "87uud4bc",
        "source_config_path": "configs/utr5/bashor_in_house/resnet1d/lib1_fiveprime_modal50__scratch_resnet1d__bayes.yml",
        "trainer_overrides": {"max_epochs": 220, "stopping_patience": 35},
    },
}


RESNET_MANIFEST_KEYS = [
    "optimizer",
    "lr",
    "weight_decay",
    "amsgrad",
    "beta1",
    "beta2",
    "scheduler",
    "T_0",
    "batch_size",
    "use_batch_norm",
    "stem_channels",
    "stem_kernel_size",
    "block_kernel_size",
    "dropout_p",
    "head_hidden_channels",
]

BASSET_MANIFEST_KEYS = [
    "optimizer",
    "lr",
    "weight_decay",
    "amsgrad",
    "beta1",
    "beta2",
    "scheduler",
    "batch_size",
    "use_batch_norm",
    "conv1_channels",
    "conv1_kernel_size",
    "conv2_channels",
    "conv2_kernel_size",
    "conv3_channels",
    "conv3_kernel_size",
    "adaptive_pool_output_size",
    "n_linear_layers",
    "linear_channels",
    "linear_activation",
    "linear_dropout_p",
]

OUTER_SEED_SPECS = {
    "Promoter": {
        "keys": BASSET_MANIFEST_KEYS,
        "log": {"lr": (1e-4, 1e-3), "weight_decay": (1e-5, 2e-3)},
        "uniform": {"linear_dropout_p": (0.38, 0.65), "beta1": (0.84, 0.96), "beta2": (0.988, 0.999)},
        "int": {"conv1_channels": (48, 128), "conv2_channels": (40, 128), "conv3_channels": (24, 96), "linear_channels": (50, 192)},
        "categorical": {
            "optimizer": ["AdamW"],
            "scheduler": ["None"],
            "amsgrad": [False, True],
            "batch_size": [64, 128, 256],
            "use_batch_norm": [True, False],
            "conv1_kernel_size": [5, 7, 9, 11],
            "conv2_kernel_size": [7, 9, 3, 5],
            "conv3_kernel_size": [7, 5, 3],
            "adaptive_pool_output_size": [12, 8, 6],
            "n_linear_layers": [2, 1],
            "linear_activation": ["LeakyReLU", "ELU", "ReLU"],
        },
    },
    "Intron": {
        "keys": RESNET_MANIFEST_KEYS,
        "log": {"lr": (3e-5, 4e-4), "weight_decay": (2e-6, 2e-3)},
        "uniform": {"dropout_p": (0.07, 0.40), "beta1": (0.88, 0.95), "beta2": (0.985, 0.996)},
        "int": {"stem_channels": (36, 96), "head_hidden_channels": (32, 160)},
        "categorical": {
            "optimizer": ["Adam", "AdamW"],
            "scheduler": ["CosineAnnealingWarmRestarts", "None"],
            "T_0": [500, 1000, 2000],
            "amsgrad": [False, True],
            "batch_size": [256, 128],
            "use_batch_norm": [False, True],
            "stem_kernel_size": [7, 9, 5, 3],
            "block_kernel_size": [9, 7, 5, 3],
        },
    },
    "3UTR": {
        "keys": RESNET_MANIFEST_KEYS,
        "log": {"lr": (3e-5, 4e-4), "weight_decay": (1e-6, 3e-5)},
        "uniform": {"dropout_p": (0.08, 0.40), "beta1": (0.84, 0.95), "beta2": (0.985, 0.999)},
        "int": {"stem_channels": (54, 128), "head_hidden_channels": (76, 256)},
        "categorical": {
            "optimizer": ["Adam", "AdamW"],
            "scheduler": ["None", "CosineAnnealingWarmRestarts"],
            "T_0": [500, 1000, 2000],
            "amsgrad": [True, False],
            "batch_size": [64, 128, 256],
            "use_batch_norm": [False, True],
            "stem_kernel_size": [5, 7],
            "block_kernel_size": [3, 5],
        },
    },
    "5UTR": {
        "keys": RESNET_MANIFEST_KEYS,
        "log": {"lr": (3e-5, 1.5e-4), "weight_decay": (1e-6, 5e-4)},
        "uniform": {"dropout_p": (0.16, 0.40), "beta1": (0.84, 0.92), "beta2": (0.985, 0.998)},
        "int": {"stem_channels": (50, 160), "head_hidden_channels": (100, 256)},
        "categorical": {
            "optimizer": ["AdamW", "Adam"],
            "scheduler": ["CosineAnnealingWarmRestarts", "None"],
            "T_0": [500, 1000, 2000],
            "amsgrad": [True, False],
            "batch_size": [256, 128, 64],
            "use_batch_norm": [False, True],
            "stem_kernel_size": [5, 7, 11],
            "block_kernel_size": [3, 5],
        },
    },
}

FIXED_TRAIN_KEYS = [
    "epoch_eval_splits",
    "data_module",
    "datafile_path",
    "sep",
    "sequence_column",
    "target_column",
    "barcode_column",
    "padded_seq_len",
    "padding_mode",
    "neutral_pad_char",
    "normalize",
    "test_min_barcodes",
    "train_min_barcodes",
    "train_max_barcodes",
    "val_frac_within_hq",
    "test_frac_within_hq",
    "val_size_within_hq",
    "test_size_within_hq",
    "train_size_frac",
    "train_size_n",
    "train_sampling_mode",
    "train_subsample_seed",
    "barcode_weighting",
    "barcode_weight_cap",
    "barcode_weight_min",
    "graph_module",
    "model_module",
    "input_len",
    "n_outputs",
    "output_names",
    "log_per_output_metric_details",
    "log_legacy_metric_aliases",
    "weighted_loss_reduction",
    "use_weight_norm",
    "loss_criterion",
    "reduction",
    "T_mult",
    "eta_min",
    "scheduler_interval",
    "num_workers",
    "max_epochs",
    "min_epochs",
    "stopping_patience",
    "stopping_mode",
    "checkpoint_monitor",
    "accelerator",
    "devices",
    "precision",
    "logger_type",
    "logger_project",
    "artifact_path",
    "best_checkpoint_dir",
    "default_root_dir",
]

TRAIN_COMMAND_KEYS = [
    "data_module",
    "model_module",
    "graph_module",
    "artifact_path",
    "best_checkpoint_dir",
    "checkpoint_monitor",
    "stopping_mode",
    "stopping_patience",
    "logger_type",
    "logger_project",
    "run_name",
    "exact_run_name",
    "epoch_eval_splits",
    "model_seed",
    "datafile_path",
    "sep",
    "sequence_column",
    "target_column",
    "barcode_column",
    "batch_size",
    "padded_seq_len",
    "padding_mode",
    "neutral_pad_char",
    "num_workers",
    "normalize",
    "split_seed",
    "test_min_barcodes",
    "train_min_barcodes",
    "train_max_barcodes",
    "val_frac_within_hq",
    "test_frac_within_hq",
    "val_size_within_hq",
    "test_size_within_hq",
    "train_size_frac",
    "train_size_n",
    "train_sampling_mode",
    "train_subsample_seed",
    "use_reverse_complements",
    "barcode_weighting",
    "barcode_weight_cap",
    "barcode_weight_min",
    "input_len",
    "n_outputs",
    "output_names",
    "log_per_output_metric_details",
    "log_legacy_metric_aliases",
    "weighted_loss_reduction",
    "conv1_channels",
    "conv1_kernel_size",
    "conv2_channels",
    "conv2_kernel_size",
    "conv3_channels",
    "conv3_kernel_size",
    "adaptive_pool_output_size",
    "n_linear_layers",
    "linear_channels",
    "linear_activation",
    "linear_dropout_p",
    "stem_channels",
    "stem_kernel_size",
    "block_kernel_size",
    "dropout_p",
    "head_hidden_channels",
    "use_batch_norm",
    "use_weight_norm",
    "loss_criterion",
    "reduction",
    "optimizer",
    "lr",
    "weight_decay",
    "amsgrad",
    "beta1",
    "beta2",
    "scheduler",
    "T_0",
    "T_mult",
    "eta_min",
    "scheduler_interval",
    "max_epochs",
    "min_epochs",
    "accelerator",
    "devices",
    "precision",
    "default_root_dir",
]

SCHEDULER_ONLY_KEYS = {"T_0", "T_mult", "eta_min"}
INT_VALUE_KEYS = {
    "manifest_row",
    "split_seed",
    "model_seed",
    "source_split_seed",
    "source_model_seed",
    "test_min_barcodes",
    "train_min_barcodes",
    "train_max_barcodes",
    "val_size_within_hq",
    "test_size_within_hq",
    "train_size_n",
    "train_subsample_seed",
    "batch_size",
    "padded_seq_len",
    "num_workers",
    "input_len",
    "n_outputs",
    "conv1_channels",
    "conv1_kernel_size",
    "conv2_channels",
    "conv2_kernel_size",
    "conv3_channels",
    "conv3_kernel_size",
    "adaptive_pool_output_size",
    "n_linear_layers",
    "linear_channels",
    "stem_channels",
    "stem_kernel_size",
    "block_kernel_size",
    "head_hidden_channels",
    "T_0",
    "T_mult",
    "max_epochs",
    "min_epochs",
    "stopping_patience",
    "devices",
    "precision",
}
BOOL_VALUE_KEYS = {
    "amsgrad",
    "use_batch_norm",
    "use_weight_norm",
    "normalize",
    "use_reverse_complements",
    "barcode_weighting",
    "log_per_output_metric_details",
    "log_legacy_metric_aliases",
    "exact_run_name",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _normalize_scheduler(value: Any) -> Optional[str]:
    if _is_missing(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null"}:
            return None
        return text
    return str(value)


def _safe_bool(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes", "y"}:
            return True
        if lower in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _jsonable(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def normalize_record_types(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in record.items():
        if _is_missing(value):
            normalized[key] = None
        elif key in INT_VALUE_KEYS:
            normalized[key] = int(float(value))
        elif key in BOOL_VALUE_KEYS:
            normalized[key] = _safe_bool(value)
        else:
            normalized[key] = _jsonable(value)
    return normalized


def _freeze_value(value: Any) -> Any:
    if _is_missing(value):
        return "<NA>"
    scheduler = _normalize_scheduler(value)
    if scheduler is None and isinstance(value, str):
        return "None"
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, np.generic):
        return _freeze_value(value.item())
    if isinstance(value, list):
        return tuple(_freeze_value(v) for v in value)
    return value


def _config_fingerprint(cfg: Dict[str, Any], keys: Iterable[str]) -> Tuple[Tuple[str, Any], ...]:
    return tuple((key, _freeze_value(cfg.get(key))) for key in keys)


def _config_subset(cfg: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    subset = {}
    for key in keys:
        if key in cfg and not _is_missing(cfg.get(key)):
            subset[key] = cfg.get(key)
    return normalize_config(subset)


def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(cfg)
    if "scheduler" in result:
        scheduler = _normalize_scheduler(result.get("scheduler"))
        result["scheduler"] = scheduler if scheduler is not None else "None"
    if _normalize_scheduler(result.get("scheduler")) is None:
        result["T_0"] = None
    for key in ("amsgrad", "use_batch_norm", "use_reverse_complements", "barcode_weighting", "normalize"):
        if key in result and not _is_missing(result[key]):
            result[key] = _safe_bool(result[key])
    for key in (
        "batch_size",
        "conv1_channels",
        "conv1_kernel_size",
        "conv2_channels",
        "conv2_kernel_size",
        "conv3_channels",
        "conv3_kernel_size",
        "adaptive_pool_output_size",
        "n_linear_layers",
        "linear_channels",
        "stem_channels",
        "stem_kernel_size",
        "block_kernel_size",
        "head_hidden_channels",
        "T_0",
    ):
        if key in result and not _is_missing(result[key]):
            result[key] = int(result[key])
    return result


def flatten_wandb_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    flat = {}
    for key, value in (raw or {}).items():
        if str(key).startswith("_"):
            continue
        flat[key] = value.get("value") if isinstance(value, dict) and "value" in value else value
    return flat


def build_config_path_index() -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    # Prefer per-run config files when available; they reflect the final run.
    for path in sorted(WANDB_CACHE.glob("run-*/files/config.yaml")):
        run_id = path.parents[1].name.split("-")[-1]
        index.setdefault(run_id, []).append(path)
    for path in sorted(WANDB_CACHE.glob("sweep-*/config-*.yaml")):
        run_id = path.stem.replace("config-", "")
        index.setdefault(run_id, []).append(path)
    return index


CONFIG_PATHS_BY_RUN = build_config_path_index()


@lru_cache(maxsize=None)
def load_run_config(run_id: str) -> Tuple[Dict[str, Any], Optional[Path]]:
    if yaml is None:
        raise SystemExit(
            "PyYAML is required to read local W&B config cache files. "
            "Run this from the BODA training environment."
        )
    paths = CONFIG_PATHS_BY_RUN.get(str(run_id), [])
    if not paths:
        return {}, None
    path = paths[0]
    with path.open() as fh:
        return flatten_wandb_config(yaml.safe_load(fh)), path


@lru_cache(maxsize=None)
def load_fixed_config_values(relative_config_path: str) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit(
            "PyYAML is required to read local sweep config files. "
            "Run this from the BODA training environment."
        )
    path = LEARN_DIR / relative_config_path
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    params = raw.get("parameters", {})
    fixed = {}
    for key, value in params.items():
        if isinstance(value, dict) and "value" in value:
            fixed[key] = value["value"]
    return fixed


def part_fixed_values(part: str) -> Dict[str, Any]:
    info = PART_PRIORS[part]
    fixed = load_fixed_config_values(info["source_config_path"])
    fixed = {key: fixed[key] for key in FIXED_TRAIN_KEYS if key in fixed}
    for stale_path_key in ("artifact_path", "best_checkpoint_dir", "default_root_dir"):
        fixed.pop(stale_path_key, None)
    fixed.update(info["trainer_overrides"])
    fixed.update(
        {
            "epoch_eval_splits": ["train", "val", "test"],
            "model_module": info["model_module"],
            "logger_project": info["logger_project"],
            "logger_type": "wandb",
            "checkpoint_monitor": "val_pearson",
            "stopping_mode": "max",
            "accelerator": "gpu",
            "devices": 1,
            "precision": 32,
        }
    )
    return normalize_config(fixed)


def load_hpo_registry() -> pd.DataFrame:
    runs_path = RUN_REGISTRY / "runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(runs_path)
    runs = pd.read_csv(runs_path, low_memory=False)
    project_table = pd.DataFrame(
        PROJECT_ROWS,
        columns=["logger_project", "cre_region", "source_architecture", "task_family_expected"],
    )
    hpo = runs.merge(project_table, on="logger_project", how="inner")
    hpo["wandb_sweep_id"] = hpo["wandb_sweep_id"].astype(str)
    selected_sweeps = {info["prior_sweep_id"] for info in PART_PRIORS.values()}
    hpo = hpo[hpo["wandb_sweep_id"].isin(selected_sweeps)].copy()
    hpo["run_id"] = hpo["run_id"].astype(str)

    config_rows = []
    for run_id in hpo["run_id"].unique():
        cfg, config_path = load_run_config(run_id)
        config_rows.append(
            {
                "run_id": run_id,
                "config_found": config_path is not None,
                "config_path_local": str(config_path) if config_path else None,
                "split_seed": cfg.get("split_seed"),
                "model_seed": cfg.get("model_seed"),
                "use_reverse_complements": cfg.get("use_reverse_complements"),
            }
        )
    config_df = pd.DataFrame(config_rows)
    if not config_df.empty:
        for col in ["split_seed", "model_seed"]:
            config_df[col] = pd.to_numeric(config_df[col], errors="coerce")
        config_df["use_reverse_complements"] = config_df["use_reverse_complements"].map(_safe_bool)
        hpo = hpo.merge(config_df, on="run_id", how="left")

    hpo["completed"] = hpo["status"].astype(str).eq("completed")
    for col in [
        "best_metric_value",
        "best_epoch",
        "train_pearson",
        "val_pearson",
        "test_pearson",
        "train_spearman",
        "val_spearman",
        "test_spearman",
        "train_mse",
        "val_mse",
        "test_mse",
        "train_cod_r2",
        "val_cod_r2",
        "test_cod_r2",
        "train_loss",
        "val_loss",
        "test_loss",
    ]:
        if col in hpo.columns:
            hpo[col] = pd.to_numeric(hpo[col], errors="coerce")
    pearson_cols = [col for col in ["train_pearson", "val_pearson", "test_pearson"] if col in hpo.columns]
    invalid_available = pd.Series(False, index=hpo.index)
    for col in pearson_cols:
        invalid_available = invalid_available | hpo[col].notna() & ~hpo[col].between(-1, 1)
    hpo["selection_ready"] = hpo["completed"] & ~invalid_available & hpo["val_pearson"].between(-1, 1)
    return hpo


def prior_runs_for_outer_seed(hpo: pd.DataFrame, part: str) -> pd.DataFrame:
    info = PART_PRIORS[part]
    df = hpo[
        hpo["logger_project"].eq(info["source_logger_project"])
        & hpo["wandb_sweep_id"].astype(str).eq(info["prior_sweep_id"])
        & hpo["selection_ready"]
    ].copy()
    if df.empty:
        raise ValueError("No prior runs found for {}".format(part))
    for col in ["split_seed", "model_seed", "use_reverse_complements"]:
        if col not in df.columns:
            df[col] = np.nan
    df["within_seed_val_pct"] = df.groupby("split_seed", dropna=False)["val_pearson"].rank(
        pct=True, method="average"
    )
    if df["split_seed"].nunique(dropna=True) > 1:
        df["prior_selection_score"] = df["within_seed_val_pct"]
    else:
        df["prior_selection_score"] = df["val_pearson"].rank(pct=True, method="average")
    return df.sort_values(["prior_selection_score", "val_pearson"], ascending=False)


def dedup_prior_runs(hpo: pd.DataFrame, part: str) -> pd.DataFrame:
    spec = OUTER_SEED_SPECS[part]
    keys = spec["keys"]
    rows = []
    seen = set()
    missing_configs = []
    for _, row in prior_runs_for_outer_seed(hpo, part).iterrows():
        cfg, config_path = load_run_config(str(row["run_id"]))
        if not cfg:
            missing_configs.append(str(row["run_id"]))
            continue
        cfg_subset = _config_subset(cfg, keys)
        fp = _config_fingerprint(cfg_subset, keys)
        if fp in seen:
            continue
        seen.add(fp)
        rows.append(
            {
                "part": part,
                "architecture": PART_PRIORS[part]["architecture"],
                "source_architecture": PART_PRIORS[part]["source_architecture"],
                "source_logger_project": PART_PRIORS[part]["source_logger_project"],
                "source_prior_sweep_id": PART_PRIORS[part]["prior_sweep_id"],
                "source_run_id": row["run_id"],
                "source_val_pearson": row.get("val_pearson"),
                "source_test_pearson": row.get("test_pearson"),
                "source_split_seed": row.get("split_seed"),
                "source_model_seed": row.get("model_seed"),
                "source_use_reverse_complements": _safe_bool(row.get("use_reverse_complements")),
                "prior_selection_score": row["prior_selection_score"],
                "source_config_path_local": str(config_path) if config_path else None,
                **cfg_subset,
            }
        )
    if missing_configs:
        print(
            "WARN: skipped {} {} prior runs with missing local W&B configs: {}".format(
                len(missing_configs), part, ", ".join(missing_configs[:8])
            ),
            file=sys.stderr,
        )
    return pd.DataFrame(rows)


def sample_log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_from_spec(part: str, rng: np.random.Generator) -> Dict[str, Any]:
    spec = OUTER_SEED_SPECS[part]
    cfg: Dict[str, Any] = {}
    for key, (lo, hi) in spec.get("log", {}).items():
        cfg[key] = sample_log_uniform(rng, lo, hi)
    for key, (lo, hi) in spec.get("uniform", {}).items():
        cfg[key] = float(rng.uniform(lo, hi))
    for key, (lo, hi) in spec.get("int", {}).items():
        cfg[key] = int(rng.integers(int(lo), int(hi) + 1))
    for key, values in spec.get("categorical", {}).items():
        cfg[key] = values[int(rng.integers(0, len(values)))]
    return normalize_config(cfg)


def jitter_from_template(part: str, template: Dict[str, Any], rng: np.random.Generator) -> Dict[str, Any]:
    spec = OUTER_SEED_SPECS[part]
    cfg = {
        key: template.get(key)
        for key in spec["keys"]
        if key in template and not _is_missing(template.get(key))
    }
    for key, (lo, hi) in spec.get("log", {}).items():
        center = float(cfg.get(key, sample_log_uniform(rng, lo, hi)))
        cfg[key] = float(np.clip(center * np.exp(rng.normal(0, 0.35)), lo, hi))
    for key, (lo, hi) in spec.get("uniform", {}).items():
        center = float(cfg.get(key, rng.uniform(lo, hi)))
        cfg[key] = float(np.clip(center + rng.normal(0, 0.12 * (hi - lo)), lo, hi))
    for key, (lo, hi) in spec.get("int", {}).items():
        center = int(cfg.get(key, rng.integers(lo, hi + 1)))
        width = max(2, int(round(0.20 * (hi - lo))))
        cfg[key] = int(np.clip(center + rng.integers(-width, width + 1), lo, hi))
    for key, values in spec.get("categorical", {}).items():
        if key not in cfg or rng.random() < 0.25:
            cfg[key] = values[int(rng.integers(0, len(values)))]
    scheduler = _normalize_scheduler(cfg.get("scheduler"))
    if scheduler is None:
        cfg["scheduler"] = "None"
        cfg["T_0"] = None
    elif cfg.get("T_0") is None and "T_0" in spec.get("categorical", {}):
        values = spec["categorical"]["T_0"]
        cfg["T_0"] = values[int(rng.integers(0, len(values)))]
    return normalize_config(cfg)


def config_id_for(part: str, number: int) -> str:
    return "{}_cfg{:03d}".format(PART_PRIORS[part]["part_slug"], number)


def build_base_configs_for_part(hpo: pd.DataFrame, part: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + sum(ord(ch) for ch in part))
    spec = OUTER_SEED_SPECS[part]
    keys = spec["keys"]
    fixed = part_fixed_values(part)
    prior_dedup = dedup_prior_runs(hpo, part)
    if len(prior_dedup) < N_EXACT_ELITE:
        raise RuntimeError("{}: expected at least {} deduplicated prior configs, got {}".format(part, N_EXACT_ELITE, len(prior_dedup)))
    exact = prior_dedup.head(N_EXACT_ELITE).copy()
    rows: List[Dict[str, Any]] = []
    seen = set()

    def add_row(config_source: str, cfg: Dict[str, Any], source: Optional[Dict[str, Any]] = None) -> None:
        cfg = normalize_config(cfg)
        fp = _config_fingerprint(cfg, keys)
        tries = 0
        while fp in seen and tries < 50:
            cfg = jitter_from_template(part, cfg, rng)
            fp = _config_fingerprint(cfg, keys)
            tries += 1
        if fp in seen:
            return
        seen.add(fp)
        row = {
            "part": part,
            "part_slug": PART_PRIORS[part]["part_slug"],
            "architecture": PART_PRIORS[part]["architecture"],
            "source_architecture": PART_PRIORS[part]["source_architecture"],
            "model_module": PART_PRIORS[part]["model_module"],
            "wandb_entity": DEFAULT_WANDB_ENTITY,
            "logger_project": PART_PRIORS[part]["logger_project"],
            "task_family": PART_PRIORS[part]["task_family"],
            "target_family": PART_PRIORS[part]["target_family"],
            "comparison_group": PART_PRIORS[part]["comparison_group"],
            "source_logger_project": PART_PRIORS[part]["source_logger_project"],
            "source_prior_sweep_id": PART_PRIORS[part]["prior_sweep_id"],
            "source_sweep_config_path": str(LEARN_DIR / PART_PRIORS[part]["source_config_path"]),
            "config_id": config_id_for(part, len(rows) + 1),
            "config_source": config_source,
            "source_run_id": None if source is None else source.get("source_run_id"),
            "source_val_pearson": None if source is None else source.get("source_val_pearson"),
            "source_test_pearson": None if source is None else source.get("source_test_pearson"),
            "source_split_seed": None if source is None else source.get("source_split_seed"),
            "source_model_seed": None if source is None else source.get("source_model_seed"),
            "source_use_reverse_complements": None if source is None else source.get("source_use_reverse_complements"),
            "prior_selection_score": None if source is None else source.get("prior_selection_score"),
            "source_config_path_local": None if source is None else source.get("source_config_path_local"),
        }
        row.update(fixed)
        row.update({key: cfg.get(key) for key in keys})
        rows.append(normalize_record_types(row))

    for _, source_row in exact.iterrows():
        source = source_row.to_dict()
        cfg = {key: source.get(key) for key in keys if key in source and not _is_missing(source.get(key))}
        add_row("exact_elite", cfg, source=source)

    exact_templates = [dict(row) for row in rows]
    for i in range(N_LOCAL_VARIANT):
        template = exact_templates[i % len(exact_templates)]
        cfg = jitter_from_template(part, template, rng)
        add_row("local_variant", cfg, source=template)

    for _ in range(N_NARROW_PRIOR):
        cfg = sample_from_spec(part, rng)
        add_row("narrow_prior", cfg)

    expected = N_EXACT_ELITE + N_LOCAL_VARIANT + N_NARROW_PRIOR
    if len(rows) != expected:
        raise RuntimeError("{}: expected {} base configs, got {}".format(part, expected, len(rows)))
    return pd.DataFrame(rows)


def value_to_cli_tokens(key: str, value: Any) -> List[str]:
    if _is_missing(value):
        return []
    if isinstance(value, str) and value == "":
        return []
    if isinstance(value, bool):
        rendered = "true" if value else "false"
        return ["--" + key, rendered]
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        return ["--" + key] + [str(item) for item in value]
    return ["--" + key, str(value)]


def build_train_command(row: Dict[str, Any]) -> str:
    row = normalize_record_types(row)
    scheduler_on = _normalize_scheduler(row.get("scheduler")) is not None
    tokens = ["python", "train_wandb_log.py"]
    for key in TRAIN_COMMAND_KEYS:
        if key in SCHEDULER_ONLY_KEYS and not scheduler_on:
            continue
        if key not in row:
            continue
        tokens.extend(value_to_cli_tokens(key, row.get(key)))
    return " ".join(shlex.quote(token) for token in tokens)


def expand_manifest(base: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = []
    manifest_row = 1
    for _, base_row in base.iterrows():
        for split_seed in OUTER_SEED_SPLIT_SEEDS:
            rec = base_row.to_dict()
            part_slug = rec["part_slug"]
            config_id = rec["config_id"]
            seed_label = "split_seed_{}".format(split_seed)
            planned_run_name = "{}__{}__{}__seed{}".format(tag, part_slug, config_id, split_seed)
            rec.update(
                {
                    "manifest_tag": tag,
                    "manifest_row": manifest_row,
                    "split_seed": int(split_seed),
                    "model_seed": OUTER_SEED_MODEL_SEED,
                    "use_reverse_complements": False,
                    "planned_run_name": planned_run_name,
                    "run_name": planned_run_name,
                    "exact_run_name": True,
                    "artifact_path": str(LEARN_DIR / "local_artifacts" / tag / part_slug / config_id / seed_label),
                    "default_root_dir": str(LEARN_DIR / "outputs" / "hpo_runs" / tag / part_slug / config_id / seed_label),
                    "best_checkpoint_dir": str(
                        LEARN_DIR
                        / "outputs"
                        / "hpo_runs"
                        / "by_project"
                        / str(rec["logger_project"])
                        / "best_checkpoint_model"
                    ),
                    "launcher_status_dir": str(LEARN_DIR / "outputs" / "hpo_runs" / "status" / tag),
                }
            )
            rec = normalize_record_types(rec)
            rec["train_command"] = build_train_command(rec)
            rows.append(rec)
            manifest_row += 1
    return pd.DataFrame(rows)


def build_manifest(seed: int, tag: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    hpo = load_hpo_registry()
    base = pd.concat(
        [build_base_configs_for_part(hpo, part, seed) for part in PART_PRIORS],
        ignore_index=True,
    )
    manifest = expand_manifest(base, tag)
    validate_manifest(base, manifest)
    return base, manifest


def validate_manifest(base: pd.DataFrame, manifest: pd.DataFrame) -> None:
    expected_base = len(PART_PRIORS) * (N_EXACT_ELITE + N_LOCAL_VARIANT + N_NARROW_PRIOR)
    expected_runs = expected_base * len(OUTER_SEED_SPLIT_SEEDS)
    if len(base) != expected_base:
        raise RuntimeError("Expected {} base configs, got {}".format(expected_base, len(base)))
    if len(manifest) != expected_runs:
        raise RuntimeError("Expected {} manifest rows, got {}".format(expected_runs, len(manifest)))
    source_counts = base.groupby(["part", "config_source"]).size().unstack(fill_value=0)
    for part in PART_PRIORS:
        got = source_counts.loc[part].to_dict()
        expected = {
            "exact_elite": N_EXACT_ELITE,
            "local_variant": N_LOCAL_VARIANT,
            "narrow_prior": N_NARROW_PRIOR,
        }
        if any(int(got.get(k, 0)) != v for k, v in expected.items()):
            raise RuntimeError("{} source counts mismatch: {}".format(part, got))
    if set(manifest["model_seed"].astype(int)) != {OUTER_SEED_MODEL_SEED}:
        raise RuntimeError("model_seed is not fixed to {}".format(OUTER_SEED_MODEL_SEED))
    if set(manifest["use_reverse_complements"].map(_safe_bool)) != {False}:
        raise RuntimeError("use_reverse_complements must be false for all rows")
    expected_seeds = set(OUTER_SEED_SPLIT_SEEDS)
    grouped = manifest.groupby(["part", "config_id"], dropna=False)
    for (part, config_id), group in grouped:
        seeds = set(int(x) for x in group["split_seed"])
        if seeds != expected_seeds:
            raise RuntimeError("{} {} has split seeds {}, expected {}".format(part, config_id, sorted(seeds), OUTER_SEED_SPLIT_SEEDS))
        keys = OUTER_SEED_SPECS[part]["keys"]
        fingerprints = {_config_fingerprint(row.to_dict(), keys) for _, row in group.iterrows()}
        if len(fingerprints) != 1:
            raise RuntimeError("{} {} changed hyperparameters across split seeds".format(part, config_id))


def records_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = []
    for record in df.to_dict(orient="records"):
        records.append(normalize_record_types(record))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def write_outputs(base: pd.DataFrame, manifest: pd.DataFrame, outdir: Path, tag: str) -> Dict[str, str]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "base_csv": outdir / "{}__base_configs.csv".format(tag),
        "base_json": outdir / "{}__base_configs.json".format(tag),
        "base_jsonl": outdir / "{}__base_configs.jsonl".format(tag),
        "manifest_csv": outdir / "{}__run_manifest.csv".format(tag),
        "manifest_json": outdir / "{}__run_manifest.json".format(tag),
        "manifest_jsonl": outdir / "{}__run_manifest.jsonl".format(tag),
        "summary_json": outdir / "{}__summary.json".format(tag),
    }
    base_records = records_from_df(base)
    manifest_records = records_from_df(manifest)
    base.to_csv(paths["base_csv"], index=False)
    manifest.to_csv(paths["manifest_csv"], index=False)
    paths["base_json"].write_text(json.dumps(base_records, indent=2, sort_keys=True) + "\n")
    paths["manifest_json"].write_text(json.dumps(manifest_records, indent=2, sort_keys=True) + "\n")
    write_jsonl(paths["base_jsonl"], base_records)
    write_jsonl(paths["manifest_jsonl"], manifest_records)

    summary = {
        "manifest_tag": tag,
        "base_config_rows": int(len(base)),
        "run_manifest_rows": int(len(manifest)),
        "split_seeds": OUTER_SEED_SPLIT_SEEDS,
        "model_seed": OUTER_SEED_MODEL_SEED,
        "use_reverse_complements": False,
        "parts": list(PART_PRIORS.keys()),
        "config_sources_per_part": (
            base.groupby(["part", "config_source"]).size().unstack(fill_value=0).astype(int).to_dict(orient="index")
        ),
        "runs_per_part_split_seed": (
            manifest.groupby(["part", "split_seed"]).size().unstack(fill_value=0).astype(int).to_dict(orient="index")
        ),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {key: str(value) for key, value in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--manifest-tag", default=MANIFEST_TAG)
    parser.add_argument("--seed", type=int, default=OUTER_SEED_MODEL_SEED)
    parser.add_argument("--no-write", action="store_true", help="Build and validate tables but do not write output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base, manifest = build_manifest(seed=args.seed, tag=args.manifest_tag)
    print("Base configs: {} rows".format(len(base)))
    print("Expanded manifest: {} rows".format(len(manifest)))
    print(base.groupby(["part", "architecture", "config_source"]).size().unstack(fill_value=0))
    print(manifest.groupby(["part", "split_seed"]).size().unstack(fill_value=0))
    if not args.no_write:
        paths = write_outputs(base, manifest, args.outdir, args.manifest_tag)
        for label, path in paths.items():
            print("{}: {}".format(label, path))


if __name__ == "__main__":
    main()
