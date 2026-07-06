#!/usr/bin/env python3
"""Run fixed-test learning-curve fine-tuning experiments on lib1 enhancer data.

Designed for transfer learning from a pretrained BassetBranched model to a single-output
regression task on in-house enhancer data.

Key features
------------
1. Keep a fixed high-quality validation/test split across all experiments.
2. Vary train barcode threshold independently from test quality threshold.
3. Vary train size in approximately log-spaced fashion.
4. Compare fine-tuning scopes (which parameter blocks are unfrozen).
5. Optionally use RC augmentation and barcode-weighted MSE.
6. Save tidy CSV outputs for plotting in the notebook.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
import re
import sys
import tarfile
import tempfile
import warnings
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


# find the repo root containing the boda package
def locate_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "boda").is_dir():
            return d
    for candidate in (
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
        Path.cwd().resolve().parent.parent,
    ):
        if (candidate / "boda").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root containing `boda`.")


REPO_ROOT = locate_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from boda.common import utils  # noqa: E402
from boda.model.basset import BassetBranched  # noqa: E402


DEFAULT_DATA_PATH = Path(
    "/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/single_part_variant_level/enhancers/"
    "20251218_np_fastq1_500000NPreads_enh_variants_bc_sum_avg_expression.txt"
)
DEFAULT_MODEL_PATH = REPO_ROOT / "tutorials" / "malinois_artifacts__20211113_021200__287348.tar.gz"
DEFAULT_OUTDIR = REPO_ROOT / "src" / "finetune" / "learning_curve" / "lib1_enhancer_mar25"

SEQUENCE_COLUMN = "Enhancers"
BARCODE_COLUMN = "n_barcodes"
TARGET_COLUMN = "RNA_DNA_Ratio_log10_scaled"
PRETRAINED_HEADS = ["K562", "HepG2", "SKNSH"]
INPUT_LEN = 600

PRED_BATCH_SIZE = 256
TRAIN_BATCH_SIZE = 128
MAX_EPOCHS = 70
EARLY_STOPPING_PATIENCE = 10
DEFAULT_FROZEN_EPOCHS = 2  # number of epochs to warm up the selected head/branch before deeper unfreezing
DEFAULT_SEEDS = [7, 8, 9]


@dataclass(frozen=True)
class TargetScaler:
    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def from_series(cls, series: pd.Series) -> "TargetScaler":
        mean = float(series.mean())
        std = float(series.std())
        if (not np.isfinite(std)) or std < 1e-8:
            std = 1.0
        return cls(mean=mean, std=std)

    def transform(self, values: Any) -> np.ndarray:
        return (np.asarray(values, dtype=np.float32) - self.mean) / self.std

    def inverse(self, values: Any) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.std + self.mean


@dataclass(frozen=True)
class FineTuneSetting:
    name: str
    use_rc_augmentation: bool = False
    use_barcode_weighting: bool = False
    b_cap: float | None = None
    min_weight: float = 0.1


@dataclass(frozen=True)
class ExperimentSpec:
    seed: int
    head_idx: int
    init_head: str
    setting_name: str
    train_threshold: int
    train_size: int
    train_fraction: float
    unfreeze_scope: str
    train_sampling_mode: str
    head_lr: float
    backbone_lr: float

    def tag(self) -> str:
        frac = f"{self.train_fraction:.4f}".replace(".", "p")
        head_lr_tag = re.sub(r"[^a-zA-Z0-9]+", "", f"{self.head_lr:.2e}".replace(".", "p"))
        backbone_lr_tag = re.sub(r"[^a-zA-Z0-9]+", "", f"{self.backbone_lr:.2e}".replace(".", "p"))
        return (
            f"seed{self.seed}__head{self.init_head}__{self.setting_name}"
            f"__thr{self.train_threshold}__n{self.train_size}__frac{frac}"
            f"__{self.unfreeze_scope}__{self.train_sampling_mode}"
            f"__hlr{head_lr_tag}__blr{backbone_lr_tag}"
        )


class PaddedSequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str = "target_standardized",
        sequence_column: str = "padded_seq",
        weight_column: str | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.target_column = target_column
        self.sequence_column = sequence_column
        self.weight_column = weight_column

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq_tensor = utils.row_dna2tensor(row, in_column_name=self.sequence_column)
        y = torch.tensor(row[self.target_column], dtype=torch.float32).view(-1)
        if self.weight_column is None:
            return seq_tensor, y
        w = torch.tensor(float(row[self.weight_column]), dtype=torch.float32)
        return seq_tensor, y, w


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_barcode_weight(barcode_count: float, b_cap: float = 10.0, min_weight: float = 0.1) -> float:
    raw = np.log1p(float(barcode_count)) / np.log1p(float(b_cap))
    weight = min(1.0, raw)
    weight = max(min_weight, weight)
    return float(weight)


def add_barcode_weights(
    df: pd.DataFrame,
    barcode_column: str = BARCODE_COLUMN,
    b_cap: float = 10.0,
    min_weight: float = 0.1,
) -> pd.DataFrame:
    out = df.copy()
    out["sample_weight"] = out[barcode_column].map(
        lambda x: compute_barcode_weight(x, b_cap=b_cap, min_weight=min_weight)
    )
    return out


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    weight = weight.reshape(-1)
    se = (pred - target) ** 2
    return (weight * se).sum() / weight.sum().clamp_min(1e-8)


def reverse_complement_seq(seq: str) -> str:
    trans = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(trans)[::-1]


def augment_train_df_with_rc(train_df_padded: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    rc_df = train_df_padded.copy()
    rc_df["padded_seq"] = rc_df["padded_seq"].map(reverse_complement_seq)
    rc_df["is_rc_aug"] = True

    out_df = train_df_padded.copy()
    out_df["is_rc_aug"] = False

    aug_df = pd.concat([out_df, rc_df], axis=0, ignore_index=True)
    aug_df = aug_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return aug_df


def add_padded_sequences(df: pd.DataFrame, sequence_column: str = SEQUENCE_COLUMN, padded_seq_len: int = INPUT_LEN) -> pd.DataFrame:
    out = df.copy()
    out["padded_seq"] = out.apply(
        lambda row: utils.row_pad_sequence(row, in_column_name=sequence_column, padded_seq_len=padded_seq_len),
        axis=1,
    )
    return out


def make_loader(
    df: pd.DataFrame,
    target_column: str = "target_standardized",
    batch_size: int = TRAIN_BATCH_SIZE,
    shuffle: bool = False,
    weight_column: str | None = None,
    seed: int | None = None,
) -> torch.utils.data.DataLoader:
    ds = PaddedSequenceDataset(df=df, target_column=target_column, sequence_column="padded_seq", weight_column=weight_column)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def load_checkpoint_from_tar(artifact_path: Path, map_location: str = "cpu") -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(artifact_path, "r:gz") as tar_handle:
            tar_handle.extractall(tmpdir)
        checkpoint_path = Path(tmpdir) / "artifacts" / "torch_checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    return checkpoint


def build_multitask_model(checkpoint: dict[str, Any], device: str) -> BassetBranched:
    model_hparams = vars(checkpoint["model_hparams"]).copy()
    model = BassetBranched(**model_hparams)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model.to(device)


def slice_state_dict_for_head(state_dict: dict[str, Any], head_idx: int, n_heads: int) -> dict[str, Any]:
    single_head_state: dict[str, Any] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            single_head_state[key] = value
            continue
        if (
            (key.startswith("branched.branched_layer_") or key.startswith("output."))
            and value.ndim >= 1
            and value.shape[0] == n_heads
        ):
            single_head_state[key] = value[head_idx : head_idx + 1].clone()
        else:
            single_head_state[key] = value.clone()
    return single_head_state


def build_single_head_model(checkpoint: dict[str, Any], head_idx: int, device: str) -> BassetBranched:
    model_hparams = vars(checkpoint["model_hparams"]).copy()
    n_heads = int(model_hparams["n_outputs"])
    model_hparams["n_outputs"] = 1
    model_hparams["loss_criterion"] = "MSELoss"
    model_hparams["loss_args"] = {"reduction": "mean"}

    model = BassetBranched(**model_hparams)
    single_head_state = slice_state_dict_for_head(checkpoint["model_state_dict"], head_idx=head_idx, n_heads=n_heads)
    model.load_state_dict(single_head_state, strict=True)
    return model.to(device)


@torch.no_grad()
def predict_model(model: BassetBranched, loader: torch.utils.data.DataLoader, device: str) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[:2]
            targets.append(y.cpu().numpy())
        else:
            x = batch
        x = x.to(device)
        pred = model(x).detach().cpu().numpy()
        preds.append(pred)
    pred_array = np.concatenate(preds, axis=0)
    target_array = np.concatenate(targets, axis=0) if targets else None
    return pred_array, target_array


def compute_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    out: dict[str, float] = {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    out["r2"] = np.nan if ss_tot < 1e-8 else float(1.0 - (ss_res / ss_tot))
    out["r2_cod"] = out["r2"]
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        out["pearson"] = np.nan
        out["spearman"] = np.nan
        out["pearson_sq"] = np.nan
    else:
        p = float(pearsonr(y_true, y_pred)[0])
        out["pearson"] = p
        out["spearman"] = float(spearmanr(y_true, y_pred)[0])
        out["pearson_sq"] = p * p
    return out


def load_clean_df(data_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(data_path, sep="\t").copy()
    raw_df[TARGET_COLUMN] = pd.to_numeric(raw_df[TARGET_COLUMN], errors="coerce")
    raw_df[BARCODE_COLUMN] = pd.to_numeric(raw_df[BARCODE_COLUMN], errors="coerce")
    clean_df = raw_df.loc[
        raw_df[SEQUENCE_COLUMN].notna()
        & raw_df[BARCODE_COLUMN].notna()
        & np.isfinite(raw_df[TARGET_COLUMN])
    ].copy().reset_index(drop=True)
    clean_df["sequence_len"] = clean_df[SEQUENCE_COLUMN].str.len()
    clean_df["row_id"] = np.arange(len(clean_df))
    return clean_df


def split_fixed_val_test(
    df: pd.DataFrame,
    test_min_barcodes: int,
    val_frac_within_hq: float,
    test_frac_within_hq: float,
    split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    high_quality_df = df.loc[df[BARCODE_COLUMN] >= test_min_barcodes].copy()
    if len(high_quality_df) < 10:
        raise ValueError("Not enough high-quality rows to create fixed val/test splits.")

    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(high_quality_df.index.to_numpy())
    n_hq = len(idx)
    n_test = max(1, int(round(n_hq * test_frac_within_hq)))
    n_val = max(1, int(round(n_hq * val_frac_within_hq)))
    if n_test + n_val >= n_hq:
        n_test = max(1, n_hq // 5)
        n_val = max(1, n_hq // 5)

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    test_df = high_quality_df.loc[test_idx].copy()
    val_df = high_quality_df.loc[val_idx].copy()
    used = set(test_idx.tolist()) | set(val_idx.tolist())
    train_rest_df = df.loc[~df.index.isin(used)].copy()
    return train_rest_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def prepare_global_payload(
    clean_df: pd.DataFrame,
    split_seed: int,
    test_min_barcodes: int,
    val_frac_within_hq: float,
    test_frac_within_hq: float,
) -> dict[str, Any]:
    train_rest_df, val_df, test_df = split_fixed_val_test(
        clean_df,
        test_min_barcodes=test_min_barcodes,
        val_frac_within_hq=val_frac_within_hq,
        test_frac_within_hq=test_frac_within_hq,
        split_seed=split_seed,
    )
    train_rest_padded = add_padded_sequences(train_rest_df)
    val_padded = add_padded_sequences(val_df)
    test_padded = add_padded_sequences(test_df)
    return {
        "split_seed": split_seed,
        "test_min_barcodes": test_min_barcodes,
        "train_rest_df": train_rest_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_rest_padded": train_rest_padded,
        "val_padded": val_padded,
        "test_padded": test_padded,
    }


def build_train_pool_components(
    train_rest_df: pd.DataFrame,
    train_threshold: int,
    test_min_barcodes: int,
) -> dict[str, Any]:
    eligible = train_rest_df.loc[train_rest_df[BARCODE_COLUMN] >= train_threshold].copy().reset_index(drop=True)
    if len(eligible) == 0:
        raise ValueError(f"No train rows available at threshold >= {train_threshold}.")

    leftover_hq = eligible.loc[eligible[BARCODE_COLUMN] >= test_min_barcodes].copy().reset_index(drop=True)
    lower_quality = eligible.loc[eligible[BARCODE_COLUMN] < test_min_barcodes].copy().reset_index(drop=True)
    return {
        "eligible": eligible,
        "leftover_hq": leftover_hq,
        "lower_quality": lower_quality,
        "test_min_barcodes": int(test_min_barcodes),
    }


def build_train_pool(
    train_pool_components: dict[str, pd.DataFrame],
    train_size: int | None,
    subsample_seed: int,
    sampling_mode: str,
) -> pd.DataFrame:
    eligible = train_pool_components["eligible"]
    leftover_hq = train_pool_components["leftover_hq"]
    lower_quality = train_pool_components["lower_quality"]

    if len(eligible) == 0:
        raise ValueError("Eligible training pool is empty.")

    if train_size is None or train_size >= len(eligible):
        sampled = eligible.copy()
    elif sampling_mode == "random":
        sampled = eligible.sample(n=int(train_size), random_state=subsample_seed, replace=False).reset_index(drop=True)
    elif sampling_mode == "hq_first":
        n_target = int(train_size)
        hq_take = min(n_target, len(leftover_hq))
        lq_take = max(0, n_target - hq_take)

        sampled_parts = []
        if hq_take > 0:
            sampled_parts.append(
                leftover_hq.sample(n=hq_take, random_state=subsample_seed, replace=False).reset_index(drop=True)
            )
        if lq_take > 0:
            sampled_parts.append(
                lower_quality.sample(n=lq_take, random_state=subsample_seed + 100_003, replace=False).reset_index(drop=True)
            )
        sampled = pd.concat(sampled_parts, axis=0, ignore_index=True)
        sampled = sampled.sample(frac=1.0, random_state=subsample_seed + 7_919).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

    sampled["train_sampling_mode"] = sampling_mode
    sampled["is_leftover_hq_train"] = sampled[BARCODE_COLUMN] >= int(train_pool_components["test_min_barcodes"])
    return sampled


def make_train_size_grid(n_available: int, min_train_size: int, train_size_fracs: list[float] | None) -> list[int]:
    if train_size_fracs:
        sizes = sorted({max(min_train_size, min(n_available, int(round(n_available * frac)))) for frac in train_size_fracs})
    else:
        exponents = np.linspace(np.log10(min_train_size), np.log10(n_available), num=min(7, max(2, n_available)))
        sizes = sorted({int(round(10 ** x)) for x in exponents})
        sizes = [min(n_available, max(min_train_size, s)) for s in sizes]
        sizes = sorted(set(sizes))
    if sizes[-1] != n_available:
        sizes.append(n_available)
    return sorted(set(sizes))


def summarize_trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def scope_match(name: str, scope: str) -> bool:
    if scope == "head_only":
        return name.startswith("output.")
    if scope == "branched_only":
        return name.startswith("branched.") or name.startswith("output.")
    if scope == "linear_all_head":
        return name.startswith("linear") or name.startswith("branched.") or name.startswith("output.")
    if scope == "conv3_plus":
        return (
            name.startswith("conv3.")
            or name.startswith("linear")
            or name.startswith("branched.")
            or name.startswith("output.")
        )
    if scope == "full":
        return True
    raise ValueError(f"Unknown unfreeze scope: {scope}")


def apply_unfreeze_scope(model: torch.nn.Module, scope: str) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = scope_match(name, scope)



def apply_stagewise_unfreeze(model: torch.nn.Module, scope: str, train_backbone_now: bool) -> None:
    """Apply scope-aware warmup.

    During warmup:
      - head_only stays output-only
      - branched_only stays branch+output
      - deeper scopes warm up with branch+output before opening shared layers
    After warmup:
      - apply the requested scope exactly
    """
    if not train_backbone_now:
        if scope == "head_only":
            apply_unfreeze_scope(model, "head_only")
            return
        if scope == "branched_only":
            apply_unfreeze_scope(model, "branched_only")
            return
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("branched.") or name.startswith("output.")
        return
    apply_unfreeze_scope(model, scope)



def get_trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]



def summarize_trainable_modules(model: torch.nn.Module) -> dict[str, int]:
    module_counts = {
        "conv1": 0,
        "conv2": 0,
        "conv3": 0,
        "linear": 0,
        "branched": 0,
        "output": 0,
        "other": 0,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("conv1."):
            module_counts["conv1"] += parameter.numel()
        elif name.startswith("conv2."):
            module_counts["conv2"] += parameter.numel()
        elif name.startswith("conv3."):
            module_counts["conv3"] += parameter.numel()
        elif name.startswith("linear"):
            module_counts["linear"] += parameter.numel()
        elif name.startswith("branched."):
            module_counts["branched"] += parameter.numel()
        elif name.startswith("output."):
            module_counts["output"] += parameter.numel()
        else:
            module_counts["other"] += parameter.numel()
    return module_counts


def make_optimizer(
    model: torch.nn.Module,
    unfreeze_scope: str,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(named_params) == 0:
        raise RuntimeError("No trainable parameters found for optimizer.")

    if unfreeze_scope in {"head_only", "branched_only"}:
        return torch.optim.AdamW([p for _, p in named_params], lr=head_lr, weight_decay=weight_decay)

    head_params = [p for n, p in named_params if n.startswith("branched.") or n.startswith("output.")]
    backbone_like_params = [p for n, p in named_params if not (n.startswith("branched.") or n.startswith("output."))]
    param_groups = []
    if backbone_like_params:
        param_groups.append({"params": backbone_like_params, "lr": backbone_lr, "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return torch.optim.AdamW(param_groups)


def to_scaler_from_train(train_df: pd.DataFrame) -> TargetScaler:
    return TargetScaler.from_series(train_df[TARGET_COLUMN])


def prepare_train_val_test_for_run(
    train_df_raw: pd.DataFrame,
    val_df_raw: pd.DataFrame,
    test_df_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TargetScaler]:
    scaler = to_scaler_from_train(train_df_raw)
    train_df = add_padded_sequences(train_df_raw)
    val_df = add_padded_sequences(val_df_raw)
    test_df = add_padded_sequences(test_df_raw)
    for frame in (train_df, val_df, test_df):
        frame["target_standardized"] = scaler.transform(frame[TARGET_COLUMN])
    return train_df, val_df, test_df, scaler


def evaluate_single_head_model(
    model: BassetBranched,
    df: pd.DataFrame,
    scaler: TargetScaler,
    device: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    loader = make_loader(df, target_column="target_standardized", batch_size=PRED_BATCH_SIZE, shuffle=False)
    pred_std, true_std = predict_model(model, loader, device=device)
    pred_std = pred_std.reshape(-1)
    true_std = true_std.reshape(-1)
    pred_raw = scaler.inverse(pred_std)
    true_raw = scaler.inverse(true_std)
    metrics = compute_regression_metrics(true_raw, pred_raw)
    metrics["loss_standardized"] = float(np.mean((pred_std - true_std) ** 2))
    pred_df = df[[SEQUENCE_COLUMN, BARCODE_COLUMN, TARGET_COLUMN, "row_id"]].copy()
    pred_df["pred"] = pred_raw
    return metrics, pred_df


def train_single_head_model(
    checkpoint: dict[str, Any],
    head_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scaler: TargetScaler,
    training_seed: int,
    device: str,
    setting: FineTuneSetting,
    unfreeze_scope: str,
    frozen_epochs: int,
    max_epochs: int,
    patience: int,
    train_batch_size: int,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
    test_df: pd.DataFrame | None = None,
    log_test_metrics_per_epoch: bool = False,
    log_train_metrics_per_epoch: bool = False,
) -> tuple[BassetBranched, pd.DataFrame, dict[str, Any]]:
    set_global_seed(training_seed)
    model = build_single_head_model(checkpoint, head_idx=head_idx, device=device)

    train_df_used = train_df.copy()
    if setting.use_rc_augmentation:
        train_df_used = augment_train_df_with_rc(train_df_used, random_seed=training_seed)
    if setting.use_barcode_weighting:
        train_df_used = add_barcode_weights(
            train_df_used,
            barcode_column=BARCODE_COLUMN,
            b_cap=float(setting.b_cap if setting.b_cap is not None else 10.0),
            min_weight=setting.min_weight,
        )

    train_loader = make_loader(
        train_df_used,
        target_column="target_standardized",
        batch_size=train_batch_size,
        shuffle=True,
        weight_column="sample_weight" if setting.use_barcode_weighting else None,
        seed=training_seed,
    )
    train_eval_loader = None
    if log_train_metrics_per_epoch:
        train_eval_loader = make_loader(
            train_df,
            target_column="target_standardized",
            batch_size=PRED_BATCH_SIZE,
            shuffle=False,
        )
    val_loader = make_loader(val_df, target_column="target_standardized", batch_size=PRED_BATCH_SIZE, shuffle=False)
    test_loader = None
    if log_test_metrics_per_epoch:
        if test_df is None:
            raise ValueError("test_df must be provided when log_test_metrics_per_epoch=True")
        test_loader = make_loader(test_df, target_column="target_standardized", batch_size=PRED_BATCH_SIZE, shuffle=False)

    # apply warmup-aware unfreezing at the start of training
    apply_stagewise_unfreeze(model, scope=unfreeze_scope, train_backbone_now=(frozen_epochs <= 0))
    optimizer = make_optimizer(
        model,
        unfreeze_scope=unfreeze_scope,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        weight_decay=weight_decay,
    )
    criterion = torch.nn.MSELoss()

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = math.inf
    best_epoch = -1
    patience_counter = 0
    history: list[dict[str, Any]] = []
    initial_trainable, total_params = summarize_trainable_params(model)
    initial_trainable_names = get_trainable_parameter_names(model)
    initial_module_counts = summarize_trainable_modules(model)

    desc = f"seed{training_seed} {PRETRAINED_HEADS[head_idx]} {setting.name} {unfreeze_scope}"
    for epoch in tqdm(range(max_epochs), desc=desc, leave=False):
        # if the epoch is the frozen epochs and the unfreeze scope is not head_only or branched_only, unfreeze the model
        if epoch == frozen_epochs and frozen_epochs > 0 and unfreeze_scope not in {"head_only", "branched_only"}:
            apply_stagewise_unfreeze(model, scope=unfreeze_scope, train_backbone_now=True)   # train the backbone now
            optimizer = make_optimizer(
                model,
                unfreeze_scope=unfreeze_scope,
                head_lr=head_lr,
                backbone_lr=backbone_lr,
                weight_decay=weight_decay,
            )

        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for batch in train_loader:
            if setting.use_barcode_weighting:
                x_batch, y_batch, w_batch = batch
                w_batch = w_batch.to(device)
            else:
                x_batch, y_batch = batch
                w_batch = None
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred_batch = model(x_batch)
            if setting.use_barcode_weighting:
                loss = weighted_mse_loss(pred_batch, y_batch, w_batch)
            else:
                loss = criterion(pred_batch, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(x_batch)
            train_items += len(x_batch)

        train_loss = train_loss_sum / max(train_items, 1)
        train_history_metrics: dict[str, float] = {}
        if train_eval_loader is not None:
            train_pred_std, train_true_std = predict_model(model, train_eval_loader, device=device)
            train_pred_std = train_pred_std.reshape(-1)
            train_true_std = train_true_std.reshape(-1)
            train_metrics = compute_regression_metrics(scaler.inverse(train_true_std), scaler.inverse(train_pred_std))
            train_history_metrics = {
                "train_eval_loss_standardized": float(np.mean((train_pred_std - train_true_std) ** 2)),
                **{f"train_{k}": v for k, v in train_metrics.items()},
            }
        val_pred_std, val_true_std = predict_model(model, val_loader, device=device)
        val_pred_std = val_pred_std.reshape(-1)
        val_true_std = val_true_std.reshape(-1)
        val_loss = float(np.mean((val_pred_std - val_true_std) ** 2))
        val_metrics = compute_regression_metrics(scaler.inverse(val_true_std), scaler.inverse(val_pred_std))
        test_history_metrics: dict[str, float] = {}
        if test_loader is not None:
            test_pred_std, test_true_std = predict_model(model, test_loader, device=device)
            test_pred_std = test_pred_std.reshape(-1)
            test_true_std = test_true_std.reshape(-1)
            test_metrics = compute_regression_metrics(scaler.inverse(test_true_std), scaler.inverse(test_pred_std))
            test_history_metrics = {
                "test_loss_standardized": float(np.mean((test_pred_std - test_true_std) ** 2)),
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        trainable_now, _ = summarize_trainable_params(model)

        history_row = {
            "epoch": epoch,
            "train_loss_standardized": train_loss,
            **train_history_metrics,
            "val_loss_standardized": val_loss,
            "trainable_params": trainable_now,
            "total_params": total_params,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **test_history_metrics,
        }
        history.append(history_row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    final_trainable, _ = summarize_trainable_params(model)
    final_trainable_names = get_trainable_parameter_names(model)
    final_module_counts = summarize_trainable_modules(model)

    fit_info = {
        "best_epoch": int(best_epoch),
        "best_val_loss_standardized": float(best_val_loss),
        "initial_trainable_params": int(initial_trainable),
        "total_params": int(total_params),
        "final_trainable_params": int(final_trainable),
        "frozen_epochs": int(frozen_epochs),
        "initial_trainable_names": json.dumps(initial_trainable_names),
        "final_trainable_names": json.dumps(final_trainable_names),
        **{f"initial_trainable_{k}_params": int(v) for k, v in initial_module_counts.items()},
        **{f"final_trainable_{k}_params": int(v) for k, v in final_module_counts.items()},
    }
    return model, pd.DataFrame(history), fit_info


def run_zero_shot_eval_on_fixed_test(checkpoint: dict[str, Any], test_df_padded: pd.DataFrame, device: str) -> pd.DataFrame:
    multitask_model = build_multitask_model(checkpoint, device=device)
    loader = make_loader(test_df_padded, target_column=TARGET_COLUMN, batch_size=PRED_BATCH_SIZE, shuffle=False)
    preds, truth = predict_model(multitask_model, loader, device=device)
    truth = truth.reshape(-1)
    rows = []
    for head_idx, head_name in enumerate(PRETRAINED_HEADS):
        metrics = compute_regression_metrics(truth, preds[:, head_idx])
        metrics["init_head"] = head_name
        rows.append(metrics)
    return pd.DataFrame(rows)


def maybe_load_or_build(path: Path, force: bool, builder) -> Any:
    if path.exists() and not force:
        with path.open("rb") as handle:
            return pickle.load(handle)
    payload = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return payload


def build_settings(
    include_b1: bool,
    include_b2: bool,
    include_b3: bool,
    b3_bcaps: list[float],
    min_weight: float,
) -> list[FineTuneSetting]:
    settings: list[FineTuneSetting] = []
    if include_b1:
        settings.append(FineTuneSetting(name="B1_no_RC", use_rc_augmentation=False, use_barcode_weighting=False))
    if include_b2:
        settings.append(FineTuneSetting(name="B2_with_RC", use_rc_augmentation=True, use_barcode_weighting=False))
    if include_b3:
        seen_bcaps: set[float] = set()
        for b3_bcap in b3_bcaps:
            b3_bcap = float(b3_bcap)
            if b3_bcap in seen_bcaps:
                continue
            seen_bcaps.add(b3_bcap)
            settings.append(
                FineTuneSetting(
                    name=f"B3_with_RC_weighted_bcap_{b3_bcap:g}",
                    use_rc_augmentation=True,
                    use_barcode_weighting=True,
                    b_cap=b3_bcap,
                    min_weight=min_weight,
                )
            )
    return settings


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cols = []
    for col in out.columns:
        if isinstance(col, tuple):
            cols.append("_".join(str(x) for x in col if x).rstrip("_"))
        else:
            cols.append(str(col))
    out.columns = cols
    return out


def aggregate_metric_summary(frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    out = frame.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"]).reset_index()
    out = flatten_columns(out)
    return out.sort_values(group_cols).reset_index(drop=True)


def parse_int_list(values: Iterable[str]) -> list[int]:
    return [int(v) for v in values]


def parse_float_list(values: Iterable[str]) -> list[float]:
    return [float(v) for v in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-test learning-curve fine-tuning for lib1 enhancer transfer learning.")
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split_seed", type=int, default=7)
    parser.add_argument("--train_thresholds", nargs="+", type=int, default=[1, 2, 3])
    # thresholds for min barcodes that get selected for testing
    parser.add_argument("--test_min_barcodes", type=int, default=4)
    # fractions of high-quality rows that get selected for validation
    parser.add_argument("--val_frac_within_hq", type=float, default=0.20)
    # fractions of high-quality rows that get selected for testing
    parser.add_argument("--test_frac_within_hq", type=float, default=0.20)

    # vary training size for learning-curve experiment (how effective increasing training size is) Matt's downsampling idea
    # are we still strongly data-limited?
    parser.add_argument("--train_size_fracs", nargs="*", type=float, default=[0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    parser.add_argument(
        "--train_sampling_mode",
        type=str,
        default="hq_first",
        choices=["hq_first", "random"],
        help="How to grow the training set as train_size increases: prioritize leftover HQ rows first, or sample uniformly at random from the eligible pool.",
    )
    parser.add_argument("--min_train_size", type=int, default=32)
    parser.add_argument("--unfreeze_scopes", nargs="+", type=str,
                        default=["head_only", "branched_only", "linear_all_head", "conv3_plus", "full"],
                        choices=["head_only", "branched_only", "linear_all_head", "conv3_plus", "full"])
    parser.add_argument("--include_b1", action="store_true")
    parser.add_argument("--include_b2", action="store_true")
    parser.add_argument("--include_b3", action="store_true")
    parser.add_argument(
        "--b3_bcaps",
        nargs="+",
        type=float,
        default=None,
        help="Grid of b_cap values for B3 weighted-loss runs.",
    )
    parser.add_argument("--b3_bcap", type=float, default=8.0)
    parser.add_argument("--min_weight", type=float, default=0.1)
    parser.add_argument(
        "--head_lrs",
        nargs="+",
        type=float,
        default=None,
        help="Grid of head learning rates (branch/output parameter group).",
    )
    parser.add_argument(
        "--backbone_lrs",
        nargs="+",
        type=float,
        default=None,
        help="Grid of backbone learning rates (shared/deeper layers).",
    )
    parser.add_argument("--head_lr", type=float, default=2e-4)
    parser.add_argument("--backbone_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--frozen_epochs", type=int, default=DEFAULT_FROZEN_EPOCHS)
    parser.add_argument("--train_batch_size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    overall_start_time = time.time()
    args = parse_args()
    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)

    if not any([args.include_b1, args.include_b2, args.include_b3]):
        args.include_b2 = True

    b3_bcap_grid = list(args.b3_bcaps) if args.b3_bcaps else [args.b3_bcap]
    head_lr_grid = list(args.head_lrs) if args.head_lrs else [args.head_lr]
    backbone_lr_grid = list(args.backbone_lrs) if args.backbone_lrs else [args.backbone_lr]

    device = resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    settings = build_settings(
        args.include_b1,
        args.include_b2,
        args.include_b3,
        b3_bcaps=b3_bcap_grid,
        min_weight=args.min_weight,
    )
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    manifest = {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    manifest["repo_root"] = str(REPO_ROOT)
    manifest["device_resolved"] = device
    manifest["run_id"] = run_id
    manifest["run_started_at"] = run_started_at
    manifest["settings"] = [asdict(x) for x in settings]
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Repo root: {REPO_ROOT}")
    print(f"Using device: {device}")
    print(f"Output dir: {args.outdir}")
    print(f"Head LR grid: {head_lr_grid}")
    print(f"Backbone LR grid: {backbone_lr_grid}")
    print(f"B3 b_cap grid: {b3_bcap_grid}")

    clean_df = load_clean_df(args.data_path)
    checkpoint = load_checkpoint_from_tar(args.model_path, map_location="cpu")

    global_payload = maybe_load_or_build(
        cache_dir / "global_split.pkl",
        force=args.force,
        builder=lambda: prepare_global_payload(
            clean_df,
            split_seed=args.split_seed,
            test_min_barcodes=args.test_min_barcodes,
            val_frac_within_hq=args.val_frac_within_hq,
            test_frac_within_hq=args.test_frac_within_hq,
        ),
    )

    zero_shot_df = run_zero_shot_eval_on_fixed_test(checkpoint, global_payload["test_padded"], device=device)
    zero_shot_df.to_csv(args.outdir / "zero_shot_fixed_test.csv", index=False)

    run_records: list[dict[str, Any]] = []
    history_records: list[pd.DataFrame] = []

    train_rest_df = global_payload["train_rest_df"]
    val_df_raw = global_payload["val_df"]
    test_df_raw = global_payload["test_df"]

    for train_threshold in args.train_thresholds:
        # first prep the data for training
        pool_components = build_train_pool_components(
            train_rest_df=train_rest_df,
            train_threshold=train_threshold,
            test_min_barcodes=args.test_min_barcodes,
        )
        pool = pool_components["eligible"]
        leftover_hq_pool = pool_components["leftover_hq"]
        lower_quality_pool = pool_components["lower_quality"]
        if len(pool) < args.min_train_size:
            print(f"Skipping threshold {train_threshold}: only {len(pool)} rows available.")
            continue
        # then make a grid of training sizes to train on
        size_grid = make_train_size_grid(
            n_available=len(pool),
            min_train_size=args.min_train_size,
            train_size_fracs=list(args.train_size_fracs) if args.train_size_fracs else None,
        )
        print(
            f"Threshold >= {train_threshold}: eligible={len(pool)} "
            f"(leftover_HQ={len(leftover_hq_pool)}, lower_quality={len(lower_quality_pool)}), "
            f"sampling={args.train_sampling_mode}, sizes={size_grid}"
        )
        
        for seed in args.seeds:
            for train_size in size_grid:
                train_raw = build_train_pool(
                    train_pool_components=pool_components,
                    train_size=train_size,
                    subsample_seed=seed,
                    sampling_mode=args.train_sampling_mode,
                )
                train_df, val_df, test_df, scaler = prepare_train_val_test_for_run(train_raw, val_df_raw, test_df_raw)
                train_hq_count = int((train_raw[BARCODE_COLUMN] >= args.test_min_barcodes).sum())
                train_lq_count = int(len(train_raw) - train_hq_count)
                # then loop through the settings and train the model
                for setting in settings:
                    # then loop through the heads and train the model
                    for head_idx, head_name in enumerate(PRETRAINED_HEADS):
                        # then loop through the unfreeze scopes and train the model
                        for unfreeze_scope in args.unfreeze_scopes:
                            for head_lr in head_lr_grid:
                                for backbone_lr in backbone_lr_grid:
                                    # use the experiment spec to define the experiment
                                    spec = ExperimentSpec(
                                        seed=seed,
                                        head_idx=head_idx,
                                        init_head=head_name,
                                        setting_name=setting.name,
                                        train_threshold=train_threshold,
                                        train_size=len(train_df),
                                        train_fraction=len(train_df) / len(pool),
                                        unfreeze_scope=unfreeze_scope,
                                        train_sampling_mode=args.train_sampling_mode,
                                        head_lr=float(head_lr),
                                        backbone_lr=float(backbone_lr),
                                    )
                                    cache_path = cache_dir / "runs" / f"{spec.tag()}.pkl"

                                    # this function is used to train the model and return the fit information, history dataframe, and evaluation metrics
                                    def _builder(spec=spec, setting=setting, train_df=train_df, val_df=val_df, test_df=test_df, scaler=scaler):
                                        # define the training seed
                                        training_seed = spec.seed * 1000 + spec.head_idx * 100 + spec.train_threshold * 10
                                        # call the train_single_head_model function to train the model
                                        model, history_df, fit_info = train_single_head_model(
                                            checkpoint=checkpoint,
                                            head_idx=spec.head_idx,
                                            train_df=train_df,
                                            val_df=val_df,
                                            scaler=scaler,
                                            training_seed=training_seed,
                                            device=device,
                                            setting=setting,
                                            unfreeze_scope=spec.unfreeze_scope,
                                            frozen_epochs=args.frozen_epochs,
                                            max_epochs=args.max_epochs,
                                            patience=args.patience,
                                            train_batch_size=args.train_batch_size,
                                            head_lr=spec.head_lr,
                                            backbone_lr=spec.backbone_lr,
                                            weight_decay=args.weight_decay,
                                        )
                                        val_metrics, _ = evaluate_single_head_model(model, val_df, scaler, device=device)
                                        test_metrics, pred_df = evaluate_single_head_model(model, test_df, scaler, device=device)
                                        # Train R² on the original (non–RC-augmented) training rows; comparable to val/test on raw targets.
                                        train_metrics, _ = evaluate_single_head_model(
                                            model, train_df, scaler, device=device
                                        )
                                        return {
                                            "fit_info": fit_info,
                                            "history_df": history_df,
                                            "train_metrics": train_metrics,
                                            "val_metrics": val_metrics,
                                            "test_metrics": test_metrics,
                                            "pred_df": pred_df,
                                        }

                                    payload = maybe_load_or_build(cache_path, force=args.force, builder=_builder)
                                    fit_info = payload["fit_info"]
                                    train_m = payload.get("train_metrics")
                                    row = {
                                        "run_id": run_id,
                                        "seed": spec.seed,
                                        "init_head": spec.init_head,
                                        "head_idx": spec.head_idx,
                                        "setting": spec.setting_name,
                                        "use_rc_augmentation": setting.use_rc_augmentation,
                                        "use_barcode_weighting": setting.use_barcode_weighting,
                                        "b_cap": float(setting.b_cap) if setting.use_barcode_weighting and setting.b_cap is not None else np.nan,
                                        "min_weight": float(setting.min_weight) if setting.use_barcode_weighting else np.nan,
                                        "train_threshold": spec.train_threshold,
                                        "train_size": spec.train_size,
                                        "train_fraction": spec.train_fraction,
                                        "unfreeze_scope": spec.unfreeze_scope,
                                        "train_sampling_mode": spec.train_sampling_mode,
                                        "train_pool_eligible_size": len(pool),
                                        "train_pool_leftover_hq_size": len(leftover_hq_pool),
                                        "train_pool_lower_quality_size": len(lower_quality_pool),
                                        "train_hq_count": train_hq_count,
                                        "train_lower_quality_count": train_lq_count,
                                        "train_hq_fraction": train_hq_count / max(len(train_df), 1),
                                        "val_size": len(val_df),
                                        "test_size": len(test_df),
                                        "head_lr": spec.head_lr,
                                        "backbone_lr": spec.backbone_lr,
                                        "weight_decay": args.weight_decay,
                                        **(
                                            {f"train_{k}": v for k, v in train_m.items()}
                                            if train_m is not None
                                            else {
                                                f"train_{k}": np.nan
                                                for k in ("n", "mae", "rmse", "pearson", "spearman", "r2", "pearson_sq", "loss_standardized")
                                            }
                                        ),
                                        **{f"val_{k}": v for k, v in payload["val_metrics"].items()},
                                        **{f"test_{k}": v for k, v in payload["test_metrics"].items()},
                                        **fit_info,
                                    }
                                    run_records.append(row)

                                    hist = payload["history_df"].copy()
                                    hist["run_id"] = run_id
                                    hist["seed"] = spec.seed
                                    hist["init_head"] = spec.init_head
                                    hist["head_idx"] = spec.head_idx
                                    hist["setting"] = spec.setting_name
                                    hist["use_rc_augmentation"] = setting.use_rc_augmentation
                                    hist["use_barcode_weighting"] = setting.use_barcode_weighting
                                    hist["b_cap"] = float(setting.b_cap) if setting.use_barcode_weighting and setting.b_cap is not None else np.nan
                                    hist["min_weight"] = float(setting.min_weight) if setting.use_barcode_weighting else np.nan
                                    hist["train_threshold"] = spec.train_threshold
                                    hist["train_size"] = spec.train_size
                                    hist["train_fraction"] = spec.train_fraction
                                    hist["unfreeze_scope"] = spec.unfreeze_scope
                                    hist["train_sampling_mode"] = spec.train_sampling_mode
                                    hist["head_lr"] = spec.head_lr
                                    hist["backbone_lr"] = spec.backbone_lr
                                    history_records.append(hist)

    if len(run_records) == 0:
        raise RuntimeError("No runs were executed. Check thresholds/min_train_size/settings.")

    runs_df = pd.DataFrame(run_records).sort_values(
        [
            "setting",
            "b_cap",
            "head_lr",
            "backbone_lr",
            "train_sampling_mode",
            "unfreeze_scope",
            "train_threshold",
            "train_size",
            "init_head",
            "seed",
        ]
    ).reset_index(drop=True)
    history_df = pd.concat(history_records, ignore_index=True)

    aggregate_df = aggregate_metric_summary(
        runs_df,
        group_cols=[
            "setting",
            "b_cap",
            "head_lr",
            "backbone_lr",
            "train_sampling_mode",
            "unfreeze_scope",
            "train_threshold",
            "train_size",
            "init_head",
        ],
        metrics=[
            "train_mae", "train_rmse", "train_pearson", "train_spearman", "train_r2", "train_pearson_sq", "train_loss_standardized",
            "val_mae", "val_rmse", "val_pearson", "val_spearman", "val_r2", "val_pearson_sq", "val_loss_standardized",
            "test_mae", "test_rmse", "test_pearson", "test_spearman", "test_r2", "test_pearson_sq", "test_loss_standardized",
            "best_epoch", "best_val_loss_standardized", "initial_trainable_params", "final_trainable_params",
        ],
    )

    scope_summary_df = aggregate_metric_summary(
        runs_df,
        group_cols=["setting", "b_cap", "head_lr", "backbone_lr", "train_sampling_mode", "unfreeze_scope"],
        metrics=[
            "test_mae", "test_rmse", "test_pearson", "test_spearman", "test_r2", "test_pearson_sq", "test_loss_standardized",
            "initial_trainable_params", "final_trainable_params",
        ],
    )

    runs_df.to_csv(args.outdir / "learning_curve_runs.csv", index=False)
    history_df.to_csv(args.outdir / "learning_curve_histories.csv", index=False)
    aggregate_df.to_csv(args.outdir / "learning_curve_summary_mean_std.csv", index=False)
    scope_summary_df.to_csv(args.outdir / "unfreeze_scope_summary_mean_std.csv", index=False)

    print("\nWrote outputs:")
    for path in [
        args.outdir / "zero_shot_fixed_test.csv",
        args.outdir / "learning_curve_runs.csv",
        args.outdir / "learning_curve_histories.csv",
        args.outdir / "learning_curve_summary_mean_std.csv",
        args.outdir / "unfreeze_scope_summary_mean_std.csv",
        args.outdir / "run_manifest.json",
    ]:
        print(f"  {path}")
    elapsed_seconds = time.time() - overall_start_time
    elapsed_minutes = elapsed_seconds / 60.0
    print(f"\nTotal runtime: {elapsed_seconds:.1f} seconds ({elapsed_minutes:.2f} minutes)")


if __name__ == "__main__":
    main()
