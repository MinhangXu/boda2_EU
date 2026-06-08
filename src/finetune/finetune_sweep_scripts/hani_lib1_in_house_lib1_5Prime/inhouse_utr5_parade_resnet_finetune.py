#!/usr/bin/env python3
"""Fine-tune BODA ResNet1D and PARADE UTR5 checkpoints on in-house FivePrime data.

The split policy is intentionally conservative for the first in-house 5'UTR HPO:

* exact 50 nt FivePrime rows with finite positive RNA/DNA are eligible;
* a configurable fraction of rows with number_of_barcodes >= heldout_min_barcodes
  is split into validation and untouched test sets;
* the remaining high-barcode rows are supplemented back into the training pool,
  then train_thresholds such as 1+, 2+, and 3+ are applied.

This mirrors the enhancer learning-curve idea while keeping a high-barcode proxy
set out of training.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hani_utr5_lib2_finetune as hani_ft  # noqa: E402


REPO_ROOT = hani_ft.REPO_ROOT
WORK_ROOT = hani_ft.WORK_ROOT
HEADS = hani_ft.HEADS
CELL_TYPE_NAMES = hani_ft.CELL_TYPE_NAMES
DEFAULT_BODA_ARTIFACT_PATH = hani_ft.DEFAULT_ARTIFACT_PATH
DEFAULT_LIB1_PATH = hani_ft.DEFAULT_LIB1_PATH
DEFAULT_INHOUSE_PATH = hani_ft.DEFAULT_INHOUSE_PATH
DEFAULT_PARADE_CHECKPOINT_PATH = (
    WORK_ROOT
    / "external_models"
    / "parade"
    / "parade"
    / "predictor"
    / "regression_multiple"
    / "saved_models"
    / "model-utr5-deltas-epoch=9-step=840.ckpt"
)
DEFAULT_PARADE_MODEL_DIR = (
    WORK_ROOT
    / "external_models"
    / "parade"
    / "parade"
    / "predictor"
    / "model"
)
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "src"
    / "finetune"
    / "learning_curve"
    / "inhouse_utr5_parade_resnet_small_hpo_jun2026"
)

MODEL_FAMILIES = ["boda_resnet1d", "parade"]
TARGET_COL = "log2_RNA_DNA"
VALID_CELL_HEADS = [*HEADS, "average"]


@dataclass(frozen=True)
class ScalarTargetScaler:
    mean: float
    std: float
    source: str

    @classmethod
    def from_frame(cls, df: pd.DataFrame, target_column: str, source: str) -> "ScalarTargetScaler":
        values = pd.to_numeric(df[target_column], errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError(f"No finite values for target column {target_column!r}.")
        std = float(np.std(values, ddof=1))
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        return cls(mean=float(np.mean(values)), std=std, source=source)

    def transform(self, values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return (arr - np.float32(self.mean)) / np.float32(self.std)

    def inverse(self, values: Any) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        return arr * np.float32(self.std) + np.float32(self.mean)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "mean": self.mean, "std": self.std}


@dataclass(frozen=True)
class ExperimentSpec:
    model_family: str
    seed: int
    train_threshold: int
    train_size: str
    cell_head: str
    unfreeze_scope: str
    head_lr: float
    backbone_lr: float
    freeze_backbone_epochs: int
    weight_decay: float

    def tag(self) -> str:
        parts = [
            self.model_family,
            f"seed{self.seed}",
            f"thr{self.train_threshold}",
            f"n{sanitize_tag(self.train_size)}",
            f"head{self.cell_head}",
            self.unfreeze_scope,
            f"hlr{hani_ft.lr_tag(self.head_lr)}",
            f"blr{hani_ft.lr_tag(self.backbone_lr)}",
            f"freeze{self.freeze_backbone_epochs}",
            f"wd{hani_ft.lr_tag(self.weight_decay)}",
        ]
        return "__".join(parts)


class InhouseUTRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        model_family: str,
        cell_head: str,
        scaler: ScalarTargetScaler,
        target_column: str,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.model_family = model_family
        self.cell_head = cell_head
        self.scaler = scaler
        self.target_column = target_column
        self.x = torch.stack(
            [
                encode_sequence(seq, model_family=model_family, cell_head=cell_head)
                for seq in self.df["candidate_seq"].tolist()
            ]
        )
        self.y = torch.tensor(self.scaler.transform(self.df[self.target_column]), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def sanitize_tag(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def encode_sequence(seq: str, model_family: str, cell_head: str) -> torch.Tensor:
    if model_family == "boda_resnet1d":
        return hani_ft.utils.dna2tensor(seq)
    if model_family == "parade":
        if cell_head == "average":
            raise ValueError("PARADE requires a concrete cell_head condition, not 'average'.")
        return encode_parade_sequence(seq, cell_head=cell_head)
    raise ValueError(f"Unknown model_family: {model_family}")


def encode_parade_sequence(seq: str, cell_head: str) -> torch.Tensor:
    base_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    seq = str(seq).upper()
    one_hot = torch.zeros(4, len(seq), dtype=torch.float32)
    for pos, base in enumerate(seq):
        idx = base_to_idx.get(base)
        if idx is None:
            one_hot[:, pos] = 0.25
        else:
            one_hot[idx, pos] = 1.0
    positional = ((torch.arange(len(seq)) % 3) == 0).float().unsqueeze(0)
    condition = torch.zeros(len(HEADS), len(seq), dtype=torch.float32)
    condition[HEADS.index(cell_head), :] = 1.0
    return torch.cat([one_hot, positional, condition], dim=0)


def finite_split_metrics(frame: pd.DataFrame, value_column: str) -> dict[str, float]:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    return {
        "n": int(len(frame)),
        "target_mean": float(values.mean()),
        "target_std": float(values.std()),
        "barcode_mean": float(frame["number_of_barcodes"].mean()),
        "barcode_median": float(frame["number_of_barcodes"].median()),
    }


def load_inhouse_frame(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    df, audit = hani_ft.load_inhouse_fiveprime(path)
    df = df.loc[np.isfinite(df[TARGET_COL])].copy()
    df["candidate_seq"] = df["candidate_seq"].astype(str).str.upper()
    df = df.drop_duplicates("candidate_seq", keep="first").reset_index(drop=True)
    audit = {
        **audit,
        "usable_exact_finite_log2_rows_after_sequence_dedup": int(len(df)),
        "dedup_policy": "drop duplicate candidate_seq after exact 50 nt / finite target filtering",
    }
    return df, audit


def add_inhouse_strata(df: pd.DataFrame, quantile_bins: int = 5) -> pd.DataFrame:
    out = df.copy()
    out["seq_upper"] = out["candidate_seq"]
    out["activity_quantile"] = hani_ft.quantile_codes(out[TARGET_COL], quantile_bins)
    out["barcode_quantile"] = hani_ft.quantile_codes(out["number_of_barcodes"], quantile_bins)
    out["stratum"] = (
        "activity"
        + out["activity_quantile"].astype(str)
        + "__barcode"
        + out["barcode_quantile"].astype(str)
    )
    return out


def assign_heldout_splits(
    df: pd.DataFrame,
    heldout_min_barcodes: int,
    heldout_frac_within_hq: float,
    heldout_val_frac: float,
    split_seed: int,
) -> pd.DataFrame:
    if not (0.0 < heldout_frac_within_hq < 1.0):
        raise ValueError(f"heldout_frac_within_hq must be in (0, 1), got {heldout_frac_within_hq}")
    if not (0.0 < heldout_val_frac < 1.0):
        raise ValueError(f"heldout_val_frac must be in (0, 1), got {heldout_val_frac}")

    out = add_inhouse_strata(df)
    out["split_hash"] = out["candidate_seq"].map(
        lambda seq: hani_ft.hash_float(seq, seed=split_seed, salt="inhouse_utr5_heldout")
    )
    out["split"] = "train_pool"
    out["barcode_pool"] = np.where(
        out["number_of_barcodes"] >= heldout_min_barcodes,
        "high_quality_pool",
        "lower_barcode_pool",
    )
    high_quality_idx = out.index[out["number_of_barcodes"] >= heldout_min_barcodes].tolist()
    if len(high_quality_idx) < 20:
        raise ValueError(
            f"Only {len(high_quality_idx)} rows have number_of_barcodes >= {heldout_min_barcodes}; "
            "choose a lower heldout threshold."
        )
    high_quality = out.loc[high_quality_idx].sort_values(["split_hash", "candidate_seq"])
    holdout_mask = hani_ft.stratified_holdout_mask(
        high_quality,
        holdout_frac=heldout_frac_within_hq,
        seed=split_seed,
        hash_salt="inhouse_utr5_hq_val_test",
    )
    heldout = high_quality.loc[holdout_mask].copy()
    if len(heldout) < 2:
        raise ValueError(
            f"Heldout fraction selected only {len(heldout)} rows from the >= {heldout_min_barcodes} pool."
        )
    val_mask = hani_ft.stratified_holdout_mask(
        heldout,
        holdout_frac=heldout_val_frac,
        seed=split_seed,
        hash_salt="inhouse_utr5_val_from_hq_holdout",
    )
    val_idx = heldout.loc[val_mask].index
    test_idx = heldout.loc[~val_mask].index
    out.loc[val_idx, "split"] = "val"
    out.loc[test_idx, "split"] = "test"
    return out.sort_values(["split", "split_hash", "candidate_seq"]).reset_index(drop=True)


def split_summary(
    split_df: pd.DataFrame,
    train_thresholds: list[int],
    heldout_min_barcodes: int,
    heldout_frac_within_hq: float,
    target_column: str,
) -> pd.DataFrame:
    rows = []
    for split_name, sub in split_df.groupby("split", observed=True):
        rows.append(
            {
                "split": split_name,
                "train_threshold": np.nan,
                "heldout_min_barcodes": heldout_min_barcodes,
                "heldout_frac_within_hq": heldout_frac_within_hq,
                "n_high_quality_rows": int((sub["number_of_barcodes"] >= heldout_min_barcodes).sum()),
                **finite_split_metrics(sub, target_column),
            }
        )
    for threshold in train_thresholds:
        train_pool = get_train_pool(split_df, threshold, heldout_min_barcodes)
        rows.append(
            {
                "split": "eligible_train_pool",
                "train_threshold": int(threshold),
                "heldout_min_barcodes": heldout_min_barcodes,
                "heldout_frac_within_hq": heldout_frac_within_hq,
                "n_high_quality_rows": int((train_pool["number_of_barcodes"] >= heldout_min_barcodes).sum()),
                **finite_split_metrics(train_pool, target_column),
            }
        )
    return pd.DataFrame(rows)


def get_train_pool(
    split_df: pd.DataFrame,
    train_threshold: int,
    heldout_min_barcodes: int,
) -> pd.DataFrame:
    train = split_df.loc[
        (split_df["number_of_barcodes"] >= train_threshold)
        & split_df["split"].eq("train_pool")
    ].copy()
    return train.reset_index(drop=True)


def resolve_train_subset(
    train_pool: pd.DataFrame,
    train_size: str,
    seed: int,
    split_seed: int,
    train_threshold: int,
) -> pd.DataFrame:
    if train_pool.empty:
        raise ValueError("Training pool is empty.")
    if str(train_size).lower() == "full":
        return train_pool.copy().reset_index(drop=True)
    requested = int(train_size)
    if requested <= 0:
        raise ValueError(f"train_size must be positive or 'full', got {train_size!r}")
    n = min(requested, len(train_pool))
    salt = f"inhouse_train_subset_thr{train_threshold}_n{train_size}_seed{seed}"
    sampled = train_pool.assign(
        train_subset_hash=train_pool["candidate_seq"].map(
            lambda seq: hani_ft.hash_float(seq, seed=split_seed, salt=salt)
        )
    ).sort_values(["train_subset_hash", "candidate_seq"])
    return sampled.head(n).drop(columns=["train_subset_hash"]).reset_index(drop=True)


def make_loader(
    df: pd.DataFrame,
    model_family: str,
    cell_head: str,
    scaler: ScalarTargetScaler,
    target_column: str,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    dataset = InhouseUTRDataset(
        df=df,
        model_family=model_family,
        cell_head=cell_head,
        scaler=scaler,
        target_column=target_column,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_boda_checkpoint(path: Path) -> dict[str, Any]:
    return hani_ft.load_checkpoint_from_tar(path, map_location="cpu")


def load_boda_model(checkpoint: dict[str, Any], device: str) -> torch.nn.Module:
    return hani_ft.build_model_from_checkpoint(checkpoint, device=device)


def load_boda_pretrained_scaler(lib1_path: Path) -> hani_ft.TargetScaler:
    lib1_df = hani_ft.load_lib1_wide(lib1_path, HEADS)
    lib1_train = lib1_df.loc[lib1_df["fold"] == "train"].copy()
    if lib1_train.empty:
        raise ValueError(f"No Lib1 train rows in {lib1_path}")
    return hani_ft.TargetScaler.from_frame(lib1_train, HEADS, source="boda_pretrained_lib1_train")


def load_parade_model(checkpoint_path: Path, model_dir: Path, device: str) -> torch.nn.Module:
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    from pl_regressor import RNARegressor  # noqa: WPS433

    lightning_model = RNARegressor.load_from_checkpoint(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    model = lightning_model.model
    model.to(device)
    return model


def build_model_for_spec(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    boda_checkpoint: dict[str, Any] | None,
    device: str,
) -> torch.nn.Module:
    if spec.model_family == "boda_resnet1d":
        if boda_checkpoint is None:
            raise ValueError("BODA checkpoint is required for model_family=boda_resnet1d")
        return load_boda_model(boda_checkpoint, device=device)
    if spec.model_family == "parade":
        return load_parade_model(
            checkpoint_path=args.parade_checkpoint_path,
            model_dir=args.parade_model_dir,
            device=device,
        )
    raise ValueError(f"Unknown model_family: {spec.model_family}")


def boda_last_stage_hparams(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return hani_ft.namespace_to_dict(checkpoint["model_hparams"])


def parade_last_block_prefixes(model: torch.nn.Module) -> list[str]:
    indices = []
    for name, _ in model.named_parameters():
        match = re.match(r"seqextractor\.(?:inv_res_blc|resize_blc)(\d+)\.", name)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        return []
    last_idx = max(indices)
    return [
        f"seqextractor.inv_res_blc{last_idx}.",
        f"seqextractor.resize_blc{last_idx}.",
    ]


def set_trainable_scope(
    model: torch.nn.Module,
    spec: ExperimentSpec,
    boda_model_hparams: dict[str, Any] | None,
    head_only_warmup: bool,
) -> None:
    active_scope = "head_only" if head_only_warmup else spec.unfreeze_scope
    if spec.model_family == "boda_resnet1d":
        if boda_model_hparams is None:
            raise ValueError("BODA model hparams are required for BODA unfreeze scopes.")
        hani_ft.set_trainable_scope(
            model,
            scope=spec.unfreeze_scope,
            model_hparams=boda_model_hparams,
            head_only_warmup=head_only_warmup,
        )
        return

    last_prefixes = parade_last_block_prefixes(model)
    for name, parameter in model.named_parameters():
        if active_scope == "full":
            parameter.requires_grad = True
        elif active_scope == "head_only":
            parameter.requires_grad = name.startswith("linear.")
        elif active_scope == "last_stage_plus_head":
            parameter.requires_grad = (
                name.startswith("linear.")
                or name.startswith("mapper.")
                or any(name.startswith(prefix) for prefix in last_prefixes)
            )
        else:
            raise ValueError(f"Unknown unfreeze_scope: {spec.unfreeze_scope}")


def build_optimizer(
    model: torch.nn.Module,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if name.startswith("head.") or name.startswith("linear."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def scalar_output(
    model: torch.nn.Module,
    x: torch.Tensor,
    model_family: str,
    cell_head: str,
    parade_output_index: int,
) -> torch.Tensor:
    pred = model(x)
    if model_family == "boda_resnet1d":
        if cell_head == "average":
            return pred.mean(dim=1)
        return pred[:, HEADS.index(cell_head)]
    if model_family == "parade":
        return pred[:, int(parade_output_index)]
    raise ValueError(f"Unknown model_family: {model_family}")


@torch.no_grad()
def predict_finetuned_raw(
    model: torch.nn.Module,
    df: pd.DataFrame,
    model_family: str,
    cell_head: str,
    scaler: ScalarTargetScaler,
    batch_size: int,
    device: str,
    parade_output_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(
        InhouseUTRDataset(
            df=df,
            model_family=model_family,
            cell_head=cell_head,
            scaler=scaler,
            target_column=TARGET_COL,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    pred_scaled = []
    true_scaled = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        out = scalar_output(
            model,
            x_batch,
            model_family=model_family,
            cell_head=cell_head,
            parade_output_index=parade_output_index,
        )
        pred_scaled.append(out.detach().cpu().numpy())
        true_scaled.append(y_batch.detach().cpu().numpy())
    pred_scaled_arr = np.concatenate(pred_scaled)
    true_scaled_arr = np.concatenate(true_scaled)
    return scaler.inverse(pred_scaled_arr), true_scaled_arr


@torch.no_grad()
def predict_baseline_raw(
    model: torch.nn.Module,
    df: pd.DataFrame,
    model_family: str,
    cell_head: str,
    batch_size: int,
    device: str,
    parade_output_index: int,
    boda_pretrained_scaler: hani_ft.TargetScaler | None,
) -> np.ndarray:
    model.eval()
    x = torch.stack(
        [
            encode_sequence(seq, model_family=model_family, cell_head=cell_head)
            for seq in df["candidate_seq"].tolist()
        ]
    )
    loader = DataLoader(x, batch_size=batch_size, shuffle=False)
    preds = []
    for x_batch in loader:
        x_batch = x_batch.to(device)
        out = model(x_batch).detach().cpu().numpy()
        if model_family == "boda_resnet1d":
            if boda_pretrained_scaler is None:
                raise ValueError("BODA baseline needs the pretrained Lib1 scaler.")
            raw = boda_pretrained_scaler.inverse_array(out, HEADS)
            if cell_head == "average":
                preds.append(raw.mean(axis=1))
            else:
                preds.append(raw[:, HEADS.index(cell_head)])
        elif model_family == "parade":
            preds.append(out[:, int(parade_output_index)])
        else:
            raise ValueError(f"Unknown model_family: {model_family}")
    return np.concatenate(preds)


def metrics_record(
    y_true: Any,
    y_pred: Any,
    split_name: str,
    model_label: str,
    spec: ExperimentSpec | None,
    model_family: str,
    cell_head: str,
    train_threshold: int | None,
    train_size: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = hani_ft.regression_metrics(y_true, y_pred)
    context = extra or {}
    record = {
        "model_label": model_label,
        "model_family": model_family,
        "split": split_name,
        "cell_head": cell_head,
        "train_threshold": train_threshold,
        "train_size": train_size,
        "run_seed": None if spec is None else spec.seed,
        "unfreeze_scope": None if spec is None else spec.unfreeze_scope,
        "head_lr": None if spec is None else spec.head_lr,
        "backbone_lr": None if spec is None else spec.backbone_lr,
        "freeze_backbone_epochs": None if spec is None else spec.freeze_backbone_epochs,
        "weight_decay": None if spec is None else spec.weight_decay,
        "target": TARGET_COL,
        **context,
        **metrics,
    }
    return record


def evaluate_finetuned_split(
    model: torch.nn.Module,
    df: pd.DataFrame,
    split_name: str,
    spec: ExperimentSpec,
    scaler: ScalarTargetScaler,
    batch_size: int,
    device: str,
    parade_output_index: int,
) -> tuple[dict[str, Any], pd.DataFrame, float]:
    pred_raw, true_scaled = predict_finetuned_raw(
        model=model,
        df=df,
        model_family=spec.model_family,
        cell_head=spec.cell_head,
        scaler=scaler,
        batch_size=batch_size,
        device=device,
        parade_output_index=parade_output_index,
    )
    true_raw = df[TARGET_COL].to_numpy(dtype=np.float32)
    pred_scaled = scaler.transform(pred_raw)
    standardized_mse = float(np.mean((pred_scaled - true_scaled) ** 2))
    record = metrics_record(
        y_true=true_raw,
        y_pred=pred_raw,
        split_name=split_name,
        model_label=f"finetuned__{spec.tag()}",
        spec=spec,
        model_family=spec.model_family,
        cell_head=spec.cell_head,
        train_threshold=spec.train_threshold,
        train_size=spec.train_size,
        extra={
            "loss_standardized": standardized_mse,
            "target_scaler_source": scaler.source,
        },
    )
    pred_df = df[
        [
            "candidate_id",
            "candidate_seq",
            "number_of_barcodes",
            "DNA_bc_counts_sum",
            "RNA_bc_counts_sum",
            "RNA/DNA",
            TARGET_COL,
        ]
    ].copy()
    pred_df["split"] = split_name
    pred_df["model_label"] = record["model_label"]
    pred_df["prediction"] = pred_raw
    pred_df["prediction_scaled"] = pred_scaled
    pred_df["target_scaled"] = true_scaled
    return record, pred_df, standardized_mse


def evaluate_baseline_splits(
    model: torch.nn.Module,
    split_frames: dict[str, pd.DataFrame],
    model_family: str,
    cell_head: str,
    train_threshold: int,
    train_size: str,
    batch_size: int,
    device: str,
    parade_output_index: int,
    boda_pretrained_scaler: hani_ft.TargetScaler | None,
) -> pd.DataFrame:
    records = []
    model_label = f"pretrained__{model_family}__head{cell_head}"
    for split_name, split_df in split_frames.items():
        pred = predict_baseline_raw(
            model=model,
            df=split_df,
            model_family=model_family,
            cell_head=cell_head,
            batch_size=batch_size,
            device=device,
            parade_output_index=parade_output_index,
            boda_pretrained_scaler=boda_pretrained_scaler,
        )
        records.append(
            metrics_record(
                y_true=split_df[TARGET_COL],
                y_pred=pred,
                split_name=split_name,
                model_label=model_label,
                spec=None,
                model_family=model_family,
                cell_head=cell_head,
                train_threshold=train_threshold,
                train_size=train_size,
                extra={"target_scaler_source": "pretrained_native_output"},
            )
        )
    return pd.DataFrame(records)


def monitor_improved(value: float, best_value: float, mode: str) -> bool:
    if not np.isfinite(value):
        return False
    if not np.isfinite(best_value):
        return True
    if mode == "min":
        return value < best_value
    if mode == "max":
        return value > best_value
    raise ValueError(f"Unknown monitor mode: {mode}")


def train_one_spec(
    spec: ExperimentSpec,
    args: argparse.Namespace,
    split_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    boda_checkpoint: dict[str, Any] | None,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    hani_ft.set_global_seed(spec.seed)
    scaler = ScalarTargetScaler.from_frame(
        train_df,
        target_column=TARGET_COL,
        source=f"inhouse_train_thr{spec.train_threshold}_n{spec.train_size}",
    )
    model = build_model_for_spec(spec, args=args, boda_checkpoint=boda_checkpoint, device=device)
    boda_hparams = boda_last_stage_hparams(boda_checkpoint) if spec.model_family == "boda_resnet1d" else None
    optimizer = build_optimizer(
        model,
        head_lr=spec.head_lr,
        backbone_lr=spec.backbone_lr,
        weight_decay=spec.weight_decay,
    )
    train_loader = make_loader(
        train_df,
        model_family=spec.model_family,
        cell_head=spec.cell_head,
        scaler=scaler,
        target_column=TARGET_COL,
        batch_size=args.train_batch_size,
        shuffle=True,
        seed=spec.seed,
    )
    criterion = torch.nn.MSELoss()
    monitor_mode = "min" if args.monitor_metric == "val_loss_standardized" else "max"
    best_value = math.inf if monitor_mode == "min" else -math.inf
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = []
    epoch_records = []

    for epoch in tqdm(range(args.max_epochs), desc=spec.tag()):
        head_only_warmup = epoch < spec.freeze_backbone_epochs
        set_trainable_scope(
            model,
            spec=spec,
            boda_model_hparams=boda_hparams,
            head_only_warmup=head_only_warmup,
        )
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = scalar_output(
                model,
                x_batch,
                model_family=spec.model_family,
                cell_head=spec.cell_head,
                parade_output_index=args.parade_output_index,
            )
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(x_batch)
            train_items += len(x_batch)

        val_record, _, val_loss = evaluate_finetuned_split(
            model=model,
            df=val_df,
            split_name="val",
            spec=spec,
            scaler=scaler,
            batch_size=args.pred_batch_size,
            device=device,
            parade_output_index=args.parade_output_index,
        )
        epoch_record = {
            "epoch": int(epoch),
            "train_loss_batch_standardized": float(train_loss_sum / max(train_items, 1)),
            "head_only_warmup": bool(head_only_warmup),
            "val_loss_standardized": val_loss,
            "val_pearson": val_record["pearson"],
            "val_spearman": val_record["spearman"],
            "val_cod_r2": val_record["cod_r2"],
            "val_rmse": val_record["rmse"],
            **asdict(spec),
        }
        history.append(epoch_record)
        epoch_records.append({**epoch_record, "split": "val", **val_record})

        monitor_value = float(epoch_record[args.monitor_metric])
        improved = monitor_improved(monitor_value, best_value, monitor_mode)
        if improved or best_epoch < 0:
            best_value = monitor_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch + 1 >= args.min_epochs and patience_counter >= args.patience:
            break

    model.load_state_dict(best_state)
    final_records = []
    prediction_frames = []
    for split_name, eval_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        record, pred_df, _ = evaluate_finetuned_split(
            model=model,
            df=eval_df,
            split_name=split_name,
            spec=spec,
            scaler=scaler,
            batch_size=args.pred_batch_size,
            device=device,
            parade_output_index=args.parade_output_index,
        )
        record["best_epoch"] = int(best_epoch)
        record["best_monitor_metric"] = args.monitor_metric
        record["best_monitor_value"] = float(best_value)
        record["train_pool_eligible_size"] = int(len(get_train_pool(split_df, spec.train_threshold, args.heldout_min_barcodes)))
        record["actual_train_size"] = int(len(train_df))
        final_records.append(record)
        if split_name in {"val", "test"}:
            prediction_frames.append(pred_df)

    fit_info = {
        **asdict(spec),
        "tag": spec.tag(),
        "best_epoch": int(best_epoch),
        "best_monitor_metric": args.monitor_metric,
        "best_monitor_value": float(best_value),
        "target_scaler": scaler.to_dict(),
        "actual_train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
    }
    run_dir = args.outdir / "runs" / spec.tag()
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(run_dir / "val_test_predictions.csv", index=False)
    hani_ft.write_json(run_dir / "fit_info.json", fit_info)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "fit_info": fit_info,
            "target_scaler": scaler.to_dict(),
            "source_boda_artifact_path": str(args.boda_artifact_path) if spec.model_family == "boda_resnet1d" else None,
            "source_parade_checkpoint_path": str(args.parade_checkpoint_path) if spec.model_family == "parade" else None,
        },
        run_dir / "finetuned_model.pt",
    )
    return pd.DataFrame(final_records), pd.DataFrame(epoch_records), fit_info


def build_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    specs = []
    for model_family in args.model_families:
        for cell_head in args.cell_heads:
            if model_family == "parade" and cell_head == "average":
                continue
            for seed in args.seeds:
                for threshold in args.train_thresholds:
                    for train_size in args.train_sizes:
                        for scope in args.unfreeze_scopes:
                            for head_lr in args.head_lrs:
                                scope_backbone_lrs = args.backbone_lrs
                                if scope == "head_only":
                                    scope_backbone_lrs = [args.backbone_lrs[0]]
                                for backbone_lr in scope_backbone_lrs:
                                    for freeze_epochs in args.freeze_backbone_epochs_list:
                                        for weight_decay in args.weight_decays:
                                            specs.append(
                                                ExperimentSpec(
                                                    model_family=model_family,
                                                    seed=int(seed),
                                                    train_threshold=int(threshold),
                                                    train_size=str(train_size),
                                                    cell_head=cell_head,
                                                    unfreeze_scope=scope,
                                                    head_lr=float(head_lr),
                                                    backbone_lr=float(backbone_lr),
                                                    freeze_backbone_epochs=int(freeze_epochs),
                                                    weight_decay=float(weight_decay),
                                                )
                                            )
    return specs


def validate_args(args: argparse.Namespace) -> None:
    if args.heldout_min_barcodes <= max(args.train_thresholds):
        raise ValueError("--heldout_min_barcodes should be higher than all training thresholds.")
    for path in [args.inhouse_path, args.lib1_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    if "boda_resnet1d" in args.model_families and not args.boda_artifact_path.exists():
        raise FileNotFoundError(args.boda_artifact_path)
    if "parade" in args.model_families:
        if not args.parade_checkpoint_path.exists():
            raise FileNotFoundError(args.parade_checkpoint_path)
        if not args.parade_model_dir.exists():
            raise FileNotFoundError(args.parade_model_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boda_artifact_path", type=Path, default=DEFAULT_BODA_ARTIFACT_PATH)
    parser.add_argument("--parade_checkpoint_path", type=Path, default=DEFAULT_PARADE_CHECKPOINT_PATH)
    parser.add_argument("--parade_model_dir", type=Path, default=DEFAULT_PARADE_MODEL_DIR)
    parser.add_argument("--lib1_path", type=Path, default=DEFAULT_LIB1_PATH)
    parser.add_argument("--inhouse_path", type=Path, default=DEFAULT_INHOUSE_PATH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--model_families", nargs="+", choices=MODEL_FAMILIES, default=MODEL_FAMILIES)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17])
    parser.add_argument("--train_thresholds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--train_sizes", nargs="+", default=["full"])
    parser.add_argument("--cell_heads", nargs="+", choices=VALID_CELL_HEADS, default=HEADS)
    parser.add_argument(
        "--unfreeze_scopes",
        nargs="+",
        choices=["head_only", "last_stage_plus_head", "full"],
        default=["head_only", "last_stage_plus_head", "full"],
    )
    parser.add_argument("--head_lrs", nargs="+", type=float, default=[1e-4])
    parser.add_argument("--backbone_lrs", nargs="+", type=float, default=[1e-5])
    parser.add_argument("--freeze_backbone_epochs_list", nargs="+", type=int, default=[2])
    parser.add_argument("--weight_decays", nargs="+", type=float, default=[1e-4])

    parser.add_argument("--heldout_min_barcodes", type=int, default=8)
    parser.add_argument(
        "--heldout_frac_within_hq",
        type=float,
        default=0.15,
        help="Fraction of rows with barcodes >= heldout_min_barcodes reserved for val/test. The rest enter train_pool.",
    )
    parser.add_argument("--heldout_val_frac", type=float, default=0.50)
    parser.add_argument("--split_seed", type=int, default=20260603)
    parser.add_argument("--target_column", choices=[TARGET_COL], default=TARGET_COL)

    parser.add_argument("--parade_output_index", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument("--min_epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--pred_batch_size", type=int, default=512)
    parser.add_argument(
        "--monitor_metric",
        choices=["val_spearman", "val_pearson", "val_cod_r2", "val_loss_standardized"],
        default="val_spearman",
    )
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--preview_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_ranking(outdir: Path, summary_df: pd.DataFrame) -> None:
    if summary_df.empty or "split" not in summary_df.columns:
        return
    ranking = summary_df.loc[
        summary_df["split"].eq("val") & summary_df["model_label"].astype(str).str.startswith("finetuned__")
    ].copy()
    if ranking.empty:
        return
    ranking = ranking.sort_values(["spearman", "pearson", "cod_r2"], ascending=[False, False, False])
    ranking.to_csv(outdir / "validation_model_ranking.csv", index=False)


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "runs").mkdir(parents=True, exist_ok=True)

    inhouse_df, inhouse_audit = load_inhouse_frame(args.inhouse_path)
    split_df = assign_heldout_splits(
        inhouse_df,
        heldout_min_barcodes=args.heldout_min_barcodes,
        heldout_frac_within_hq=args.heldout_frac_within_hq,
        heldout_val_frac=args.heldout_val_frac,
        split_seed=args.split_seed,
    )
    val_df = split_df.loc[split_df["split"].eq("val")].copy().reset_index(drop=True)
    test_df = split_df.loc[split_df["split"].eq("test")].copy().reset_index(drop=True)
    split_summary_df = split_summary(
        split_df,
        train_thresholds=args.train_thresholds,
        heldout_min_barcodes=args.heldout_min_barcodes,
        heldout_frac_within_hq=args.heldout_frac_within_hq,
        target_column=args.target_column,
    )
    split_df.to_csv(args.outdir / "split_membership_rows.csv", index=False)
    split_summary_df.to_csv(args.outdir / "split_membership_summary.csv", index=False)

    specs = build_specs(args)
    planned_df = pd.DataFrame([{**asdict(spec), "tag": spec.tag()} for spec in specs])
    planned_df.to_csv(args.outdir / "planned_finetune_specs.csv", index=False)
    hani_ft.write_json(
        args.outdir / "run_manifest.json",
        {
            "created_unix": time.time(),
            "script": str(Path(__file__).resolve()),
            "repo_root": str(REPO_ROOT),
            "work_root": str(WORK_ROOT),
            "device": device,
            "heads": HEADS,
            "cell_type_names": CELL_TYPE_NAMES,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "planned_specs": planned_df.to_dict(orient="records"),
            "split_policy": {
                "unit": "unique exact 50 nt FivePrime sequence",
                "high_quality_pool": f"number_of_barcodes >= {args.heldout_min_barcodes}",
                "heldout_pool": (
                    f"{args.heldout_frac_within_hq:.3g} fraction of high_quality_pool, "
                    "split into val/test"
                ),
                "train_pool": (
                    "all non-heldout rows with number_of_barcodes >= train_threshold, "
                    "including high-quality rows not selected for val/test"
                ),
                "heldout_frac_within_hq": float(args.heldout_frac_within_hq),
                "heldout_val_frac": float(args.heldout_val_frac),
                "split_seed": int(args.split_seed),
            },
        },
    )
    hani_ft.write_json(
        args.outdir / "data_audit.json",
        {
            "inhouse": inhouse_audit,
            "split_summary": split_summary_df.to_dict(orient="records"),
        },
    )

    print(f"Repo root: {REPO_ROOT}")
    print(f"Using device: {device}")
    print(f"Output dir: {args.outdir}")
    print(f"Usable in-house rows: {len(inhouse_df)}")
    print(f"Heldout fraction within >= {args.heldout_min_barcodes} barcode rows: {args.heldout_frac_within_hq}")
    print(f"Validation / test rows: {len(val_df)} / {len(test_df)}")
    print(f"Planned fine-tune runs: {len(specs)}")

    if args.preview_only:
        print("Preview only: wrote split manifests and planned specs without loading checkpoints.")
        return

    boda_checkpoint = load_boda_checkpoint(args.boda_artifact_path) if "boda_resnet1d" in args.model_families else None
    boda_pretrained_scaler = (
        load_boda_pretrained_scaler(args.lib1_path) if "boda_resnet1d" in args.model_families else None
    )

    all_summary_frames = []
    all_epoch_frames = []
    baseline_cache: dict[tuple[str, str], torch.nn.Module] = {}
    try:
        for spec in specs:
            train_pool = get_train_pool(split_df, spec.train_threshold, args.heldout_min_barcodes)
            train_df = resolve_train_subset(
                train_pool,
                train_size=spec.train_size,
                seed=spec.seed,
                split_seed=args.split_seed,
                train_threshold=spec.train_threshold,
            )
            if len(train_df) < 20:
                raise ValueError(f"{spec.tag()} has only {len(train_df)} train rows.")

            if not args.skip_baseline:
                cache_key = (spec.model_family, spec.cell_head)
                if cache_key not in baseline_cache:
                    baseline_cache[cache_key] = build_model_for_spec(
                        spec,
                        args=args,
                        boda_checkpoint=boda_checkpoint,
                        device=device,
                    )
                baseline_summary = evaluate_baseline_splits(
                    model=baseline_cache[cache_key],
                    split_frames={"val": val_df, "test": test_df},
                    model_family=spec.model_family,
                    cell_head=spec.cell_head,
                    train_threshold=spec.train_threshold,
                    train_size=spec.train_size,
                    batch_size=args.pred_batch_size,
                    device=device,
                    parade_output_index=args.parade_output_index,
                    boda_pretrained_scaler=boda_pretrained_scaler,
                )
                all_summary_frames.append(baseline_summary)

            run_dir = args.outdir / "runs" / spec.tag()
            summary_path = run_dir / "model_comparison_summary.csv"
            epoch_path = run_dir / "per_epoch_diagnostics.csv"
            if summary_path.exists() and epoch_path.exists() and not args.force:
                print(f"Skipping existing run {spec.tag()}")
                all_summary_frames.append(pd.read_csv(summary_path))
                all_epoch_frames.append(pd.read_csv(epoch_path))
                continue

            print(f"Training {spec.tag()} with {len(train_df)} rows")
            summary_df, epoch_df, _ = train_one_spec(
                spec=spec,
                args=args,
                split_df=split_df,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                boda_checkpoint=boda_checkpoint,
                device=device,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(summary_path, index=False)
            epoch_df.to_csv(epoch_path, index=False)
            all_summary_frames.append(summary_df)
            all_epoch_frames.append(epoch_df)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        baseline_cache.clear()

    summary = hani_ft.combine_frames(all_summary_frames)
    epochs = hani_ft.combine_frames(all_epoch_frames)
    hani_ft.write_if_not_empty(args.outdir / "model_comparison_summary.csv", summary)
    hani_ft.write_if_not_empty(args.outdir / "per_epoch_diagnostics.csv", epochs)
    write_ranking(args.outdir, summary)

    print("Wrote:")
    print(f"  {args.outdir / 'split_membership_summary.csv'}")
    print(f"  {args.outdir / 'planned_finetune_specs.csv'}")
    print(f"  {args.outdir / 'model_comparison_summary.csv'}")
    print(f"  {args.outdir / 'per_epoch_diagnostics.csv'}")
    if (args.outdir / "validation_model_ranking.csv").exists():
        print(f"  {args.outdir / 'validation_model_ranking.csv'}")


if __name__ == "__main__":
    main()
