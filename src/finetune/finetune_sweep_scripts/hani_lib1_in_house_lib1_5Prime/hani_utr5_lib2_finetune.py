#!/usr/bin/env python3
"""Phase 2 BODA-first 5'UTR Lib2 fine-tuning from the 1mmy39ku artifact.

This runner mirrors the standalone enhancer fine-tuning scripts, but adapts the
data path to Hani/Goodarzi 5'UTR Lib2:

1. Aggregate Lib2 replicate rows to one row per sequence and cell head.
2. Create or consume deterministic sequence-level split manifests.
3. Fine-tune the current BODA 5'UTR ResNet1D winner on Lib2.
4. Evaluate pretrained and fine-tuned models on allowed Lib2 splits, Lib1 retention, and
   exact-length in-house FivePrime candidates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import tarfile
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


def locate_repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "boda").is_dir():
            return candidate
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "boda").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root containing `boda`.")


REPO_ROOT = locate_repo_root()
WORK_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import boda  # noqa: E402
from boda.common import utils  # noqa: E402


HEADS = ["c1", "c2", "c4", "c6", "c17"]
CELL_TYPE_NAMES = {
    "c1": "MDA-MB-231",
    "c2": "HepG2",
    "c4": "Jurkat",
    "c6": "SW480",
    "c17": "NALM6",
}
INPUT_LEN = 50
VALID_DNA = set("ACGT")
DEFAULT_RUN_ID = "1mmy39ku"

DEFAULT_ARTIFACT_PATH = (
    REPO_ROOT
    / "src"
    / "learn"
    / "local_artifacts"
    / "utr5"
    / "hani_rna_activity"
    / "resnet1d"
    / "bayes"
    / "model_artifacts__utr5__hani_rna_activity__resnet1d_challenger__1mmy39ku__20260508_153215.tar.gz"
)
DEFAULT_LIB1_PATH = (
    WORK_ROOT
    / "opt_EU_learn_n_design"
    / "utr_hani_2025"
    / "processed_utr_data"
    / "5UTR_lib1_branched_observed_heads.csv"
)
DEFAULT_LIB2_PATH = (
    WORK_ROOT
    / "opt_EU_learn_n_design"
    / "utr_hani_2025"
    / "processed_utr_data"
    / "5UTR_lib2_processed.csv"
)
DEFAULT_INHOUSE_PATH = (
    WORK_ROOT
    / "opt_EU_learn_n_design"
    / "MattLee_lib1"
    / "FivePrimes"
    / "L1_final_fastqs1-5_sublibrary_FivePrime_subset.csv"
)
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "src"
    / "finetune"
    / "learning_curve"
    / "hani_utr5_lib2_resnet1d_1mmy39ku_phase2_v2_may2026"
)


@dataclass(frozen=True)
class TargetScaler:
    means: dict[str, float]
    stds: dict[str, float]
    source: str

    @classmethod
    def from_frame(cls, df: pd.DataFrame, heads: list[str], source: str) -> "TargetScaler":
        means = {}
        stds = {}
        for head in heads:
            values = pd.to_numeric(df[head], errors="coerce")
            mean = float(values.mean())
            std = float(values.std())
            if not np.isfinite(std) or std < 1e-8:
                std = 1.0
            means[head] = mean
            stds[head] = std
        return cls(means=means, stds=stds, source=source)

    def transform_frame(self, df: pd.DataFrame, heads: list[str]) -> np.ndarray:
        values = df[heads].to_numpy(dtype=np.float32)
        means = np.asarray([self.means[head] for head in heads], dtype=np.float32)
        stds = np.asarray([self.stds[head] for head in heads], dtype=np.float32)
        return (values - means) / stds

    def inverse_array(self, values: Any, heads: list[str]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        means = np.asarray([self.means[head] for head in heads], dtype=np.float32)
        stds = np.asarray([self.stds[head] for head in heads], dtype=np.float32)
        return arr * stds + means

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "means": self.means, "stds": self.stds}


@dataclass(frozen=True)
class ExperimentSpec:
    seed: int
    unfreeze_scope: str
    head_lr: float
    backbone_lr: float
    target_scaler_source: str
    stage: str = "legacy_v1"
    outer_split_seed: int | None = None
    inner_split_seed: int | None = None
    split_id: str | None = None
    freeze_backbone_epochs: int = 3
    weight_decay: float = 1e-4

    def tag(self) -> str:
        head_lr_tag = lr_tag(self.head_lr)
        backbone_lr_tag = lr_tag(self.backbone_lr)
        weight_decay_tag = lr_tag(self.weight_decay)
        scaler_tag = re.sub(r"[^A-Za-z0-9]+", "_", self.target_scaler_source).strip("_")
        base = (
            f"seed{self.seed}__{self.unfreeze_scope}"
            f"__hlr{head_lr_tag}__blr{backbone_lr_tag}__scaler_{scaler_tag}"
            f"__freeze{self.freeze_backbone_epochs}__wd{weight_decay_tag}"
        )
        if self.stage == "legacy_v1":
            return base
        split_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.split_id or "no_split")).strip("_")
        outer_tag = "outerNA" if self.outer_split_seed is None else f"outer{self.outer_split_seed}"
        inner_tag = "innerNA" if self.inner_split_seed is None else f"inner{self.inner_split_seed}"
        return f"{self.stage}__{split_tag}__{outer_tag}__{inner_tag}__{base}"

    @property
    def training_seed(self) -> int:
        return int(self.seed)

    def context(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "outer_split_seed": self.outer_split_seed,
            "inner_split_seed": self.inner_split_seed,
            "training_seed": int(self.seed),
            "split_id": self.split_id,
            "freeze_backbone_epochs": int(self.freeze_backbone_epochs),
            "weight_decay": float(self.weight_decay),
        }


class UTR5WideDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        heads: list[str],
        scaler: TargetScaler,
        sequence_column: str = "seq_upper",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.heads = heads
        self.scaler = scaler
        self.sequence_column = sequence_column
        self.x = torch.stack([utils.dna2tensor(seq) for seq in self.df[self.sequence_column].tolist()])
        self.y = torch.tensor(self.scaler.transform_frame(self.df, self.heads), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def lr_tag(value: float) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", f"{float(value):.1e}".replace(".", "p"))


def clean_sequence(value: Any) -> str:
    return str(value).strip().upper()


def is_valid_exact_dna(seq: str, length: int = INPUT_LEN) -> bool:
    return len(seq) == length and set(seq).issubset(VALID_DNA)


def gc_fraction(seq: str) -> float:
    if not seq:
        return float("nan")
    return float((seq.count("G") + seq.count("C")) / len(seq))


def finite_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=np.float64).reshape(-1)
    b = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def pearson_np(y_true: Any, y_pred: Any) -> float:
    a, b = finite_pair(y_true, y_pred)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(pearsonr(a, b)[0])


def spearman_np(y_true: Any, y_pred: Any) -> float:
    a, b = finite_pair(y_true, y_pred)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b)[0])


def cod_r2_np(y_true: Any, y_pred: Any) -> float:
    a, b = finite_pair(y_true, y_pred)
    if len(a) < 2:
        return float("nan")
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    ss_res = float(np.sum((a - b) ** 2))
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    a, b = finite_pair(y_true, y_pred)
    if len(a) == 0:
        return {
            "n": 0,
            "pearson": float("nan"),
            "pearson_r2": float("nan"),
            "cod_r2": float("nan"),
            "spearman": float("nan"),
            "mse": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
        }
    r = pearson_np(a, b)
    mse = float(np.mean((a - b) ** 2))
    return {
        "n": int(len(a)),
        "pearson": r,
        "pearson_r2": float(r * r) if np.isfinite(r) else float("nan"),
        "cod_r2": cod_r2_np(a, b),
        "spearman": spearman_np(a, b),
        "mse": mse,
        "mae": float(np.mean(np.abs(a - b))),
        "rmse": float(np.sqrt(mse)),
    }


def hash_float(value: str, seed: int, salt: str = "") -> float:
    digest = hashlib.sha256(f"{seed}|{salt}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def safe_extract(tar: tarfile.TarFile, target_dir: Path) -> None:
    target_root = target_dir.resolve()
    for member in tar.getmembers():
        member_path = (target_dir / member.name).resolve()
        if os.path.commonpath([str(target_root), str(member_path)]) != str(target_root):
            raise RuntimeError(f"Unsafe path in tarball: {member.name}")
    tar.extractall(str(target_dir))


def load_checkpoint_from_tar(artifact_path: Path, map_location: str = "cpu") -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(str(artifact_path)) as tar:
            safe_extract(tar, tmp_path)
        return torch.load(tmp_path / "artifacts" / "torch_checkpoint.pt", map_location=map_location)


def namespace_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dict__"):
        return vars(value).copy()
    if isinstance(value, dict):
        return value.copy()
    raise TypeError(f"Cannot convert {type(value)} to dict")


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: str) -> torch.nn.Module:
    model_module = str(checkpoint["model_module"])
    model_cls = getattr(boda.model, model_module)
    model_hparams = namespace_to_dict(checkpoint["model_hparams"])
    model = model_cls(**model_hparams)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    return model


def load_lib1_wide(lib1_path: Path, heads: list[str]) -> pd.DataFrame:
    df = pd.read_csv(lib1_path)
    required = ["seq", "fold", *heads]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required Lib1 columns in {lib1_path}: {missing}")
    out = df.copy()
    out["seq_original"] = out["seq"].astype(str)
    out["seq_upper"] = out["seq_original"].map(clean_sequence)
    out["sequence_len"] = out["seq_upper"].str.len()
    out["is_valid_exact_50nt"] = out["seq_upper"].map(is_valid_exact_dna)
    out["gc_fraction"] = out["seq_upper"].map(gc_fraction)
    for head in heads:
        out[head] = pd.to_numeric(out[head], errors="coerce")
    out = out.loc[out["is_valid_exact_50nt"]].dropna(subset=heads).copy()
    return out.reset_index(drop=True)


def load_lib2_wide(
    lib2_path: Path,
    heads: list[str],
    require_all_heads: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(lib2_path)
    required = ["seq", "cell_type", "rna_activity"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required Lib2 columns in {lib2_path}: {missing}")

    df = raw.copy()
    df["seq_original"] = df["seq"].astype(str)
    df["seq_upper"] = df["seq_original"].map(clean_sequence)
    df["sequence_len"] = df["seq_upper"].str.len()
    df["is_valid_exact_50nt"] = df["seq_upper"].map(is_valid_exact_dna)
    df["rna_activity"] = pd.to_numeric(df["rna_activity"], errors="coerce")
    df["cell_type"] = df["cell_type"].astype(str)

    filtered = df.loc[
        df["is_valid_exact_50nt"]
        & df["cell_type"].isin(heads)
        & np.isfinite(df["rna_activity"])
    ].copy()
    if filtered.empty:
        raise ValueError(f"No usable exact-length Lib2 rows found in {lib2_path}")

    agg = (
        filtered.groupby(["seq_upper", "cell_type"], observed=True)
        .agg(
            rna_activity=("rna_activity", "mean"),
            n_observations=("rna_activity", "size"),
            seq_original_example=("seq_original", "first"),
        )
        .reset_index()
    )
    wide = agg.pivot(index="seq_upper", columns="cell_type", values="rna_activity")
    counts = agg.pivot(index="seq_upper", columns="cell_type", values="n_observations")

    meta = (
        filtered.groupby("seq_upper", observed=True)
        .agg(
            seq_original_example=("seq_original", "first"),
            n_raw_rows=("rna_activity", "size"),
        )
        .reset_index()
        .set_index("seq_upper")
    )
    wide = wide.join(meta, how="left")
    for head in heads:
        if head not in wide.columns:
            wide[head] = np.nan
        count_col = f"n_obs_{head}"
        wide[count_col] = counts[head] if head in counts.columns else 0
        wide[count_col] = wide[count_col].fillna(0).astype(int)

    wide = wide.reset_index()
    wide["sequence_len"] = wide["seq_upper"].str.len()
    wide["gc_fraction"] = wide["seq_upper"].map(gc_fraction)
    if require_all_heads:
        wide = wide.dropna(subset=heads).copy()

    audit = {
        "raw_rows": int(len(raw)),
        "usable_long_rows": int(len(filtered)),
        "unique_exact_sequences_before_head_filter": int(filtered["seq_upper"].nunique()),
        "unique_sequences_after_wide_filter": int(len(wide)),
        "require_all_heads": bool(require_all_heads),
        "heads": heads,
        "aggregation_policy": "mean rna_activity by uppercased sequence and cell_type",
        "raw_rows_by_head": {
            head: int((filtered["cell_type"] == head).sum())
            for head in heads
        },
    }
    return wide.reset_index(drop=True), audit


def split_by_sequence_hash(
    wide: pd.DataFrame,
    split_seed: int,
    val_frac: float,
    test_frac: float,
) -> pd.DataFrame:
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")
    if len(wide) < 10:
        raise ValueError("Need at least 10 Lib2 sequences for a stable train/val/test split.")

    out = wide.copy()
    out["split_hash"] = out["seq_upper"].map(lambda seq: hash_float(seq, seed=split_seed))
    out = out.sort_values(["split_hash", "seq_upper"]).reset_index(drop=True)

    n_total = len(out)
    n_test = max(1, int(round(n_total * test_frac)))
    n_val = max(1, int(round(n_total * val_frac)))
    if n_test + n_val >= n_total:
        raise ValueError("Requested split fractions leave no Lib2 training sequences.")

    split = np.full(n_total, "train", dtype=object)
    split[:n_test] = "test"
    split[n_test : n_test + n_val] = "val"
    out["split"] = split
    return out.sample(frac=1.0, random_state=split_seed).reset_index(drop=True)


def quantile_codes(values: pd.Series, bins: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if bins <= 1 or numeric.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(numeric), dtype=int), index=values.index)
    n_bins = min(int(bins), int(numeric.nunique(dropna=True)))
    ranked = numeric.rank(method="first")
    codes = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop")
    return pd.Series(codes, index=values.index).astype("Int64").fillna(-1).astype(int)


def add_stratification_columns(
    wide: pd.DataFrame,
    heads: list[str],
    activity_quantile_bins: int,
    gc_quantile_bins: int,
) -> pd.DataFrame:
    out = wide.copy()
    out["average_activity"] = out[heads].mean(axis=1)
    out["activity_quantile"] = quantile_codes(out["average_activity"], activity_quantile_bins)
    out["gc_quantile"] = quantile_codes(out["gc_fraction"], gc_quantile_bins)
    out["stratum"] = (
        out["activity_quantile"].astype(str)
        + "__gc"
        + out["gc_quantile"].astype(str)
    )
    return out


def stratified_holdout_mask(
    df: pd.DataFrame,
    holdout_frac: float,
    seed: int,
    hash_salt: str,
    stratum_col: str = "stratum",
) -> pd.Series:
    if not (0.0 < holdout_frac < 1.0):
        raise ValueError(f"holdout_frac must be in (0, 1), got {holdout_frac}")
    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    target_total = max(1, int(round(len(df) * holdout_frac)))
    groups = df.groupby(stratum_col, observed=True, sort=True)
    quotas = []
    for stratum, sub in groups:
        exact = len(sub) * holdout_frac
        base = int(math.floor(exact))
        quotas.append(
            {
                "stratum": stratum,
                "n": int(len(sub)),
                "base": base,
                "remainder": float(exact - base),
            }
        )

    quota_df = pd.DataFrame(quotas)
    quota_df["take"] = quota_df["base"]
    remaining = int(target_total - quota_df["take"].sum())
    if remaining > 0:
        order = quota_df.sort_values(["remainder", "n", "stratum"], ascending=[False, False, True]).index
        for idx in order[:remaining]:
            quota_df.loc[idx, "take"] += 1
    elif remaining < 0:
        order = quota_df.sort_values(["remainder", "n", "stratum"], ascending=[True, True, True]).index
        to_remove = -remaining
        for idx in order:
            if to_remove <= 0:
                break
            if quota_df.loc[idx, "take"] > 0:
                quota_df.loc[idx, "take"] -= 1
                to_remove -= 1

    take_by_stratum = dict(zip(quota_df["stratum"], quota_df["take"]))
    holdout_indices: list[int] = []
    for stratum, sub in groups:
        n_take = int(take_by_stratum.get(stratum, 0))
        if n_take <= 0:
            continue
        sorted_sub = sub.assign(
            split_hash=sub["seq_upper"].map(lambda seq: hash_float(seq, seed=seed, salt=hash_salt))
        ).sort_values(["split_hash", "seq_upper"])
        holdout_indices.extend(sorted_sub.index[:n_take].tolist())

    mask = pd.Series(False, index=df.index)
    mask.loc[holdout_indices] = True
    return mask


def manifest_columns(heads: list[str]) -> list[str]:
    preferred = [
        "seq_upper",
        "seq_original_example",
        "split",
        "outer_split",
        "split_id",
        "stage",
        "outer_split_seed",
        "inner_split_seed",
        "sequence_len",
        "gc_fraction",
        "average_activity",
        "activity_quantile",
        "gc_quantile",
        "stratum",
        "outer_split_hash",
        "inner_split_hash",
        "n_raw_rows",
        *[f"n_obs_{head}" for head in heads],
        *heads,
    ]
    return preferred


def write_manifest(path: Path, frame: pd.DataFrame, heads: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [column for column in manifest_columns(heads) if column in frame.columns]
    frame[columns].to_csv(path, index=False)


def make_split_id(inner_split_seed: int, index: int | None = None) -> str:
    if index is None:
        return f"inner_seed_{inner_split_seed}"
    return f"inner{index}_seed_{inner_split_seed}"


def build_phase2_v2_split_manifests(
    wide: pd.DataFrame,
    heads: list[str],
    split_manifest_dir: Path,
    outer_split_seed: int,
    final_test_frac: float,
    inner_split_seeds: list[int],
    inner_val_frac: float,
    activity_quantile_bins: int,
    gc_quantile_bins: int,
    force: bool = False,
) -> dict[str, Path]:
    split_manifest_dir.mkdir(parents=True, exist_ok=True)
    outer_path = split_manifest_dir / "outer_final_test_manifest.csv"
    audit_path = split_manifest_dir / "split_audit.csv"
    policy_path = split_manifest_dir / "split_policy.json"
    inner_paths = {
        make_split_id(seed, idx): split_manifest_dir / f"inner_split_manifest_{make_split_id(seed, idx)}.csv"
        for idx, seed in enumerate(inner_split_seeds)
    }
    expected_paths = [outer_path, audit_path, policy_path, *inner_paths.values()]
    if all(path.exists() for path in expected_paths) and not force:
        return {
            "outer": outer_path,
            "audit": audit_path,
            "policy": policy_path,
            **inner_paths,
        }

    stratified = add_stratification_columns(
        wide,
        heads=heads,
        activity_quantile_bins=activity_quantile_bins,
        gc_quantile_bins=gc_quantile_bins,
    )
    outer = stratified.copy()
    outer["outer_split_hash"] = outer["seq_upper"].map(
        lambda seq: hash_float(seq, seed=outer_split_seed, salt="outer_final_test")
    )
    final_mask = stratified_holdout_mask(
        outer,
        holdout_frac=final_test_frac,
        seed=outer_split_seed,
        hash_salt="outer_final_test",
    )
    outer["outer_split"] = np.where(final_mask, "final_test", "hpo_pool")
    outer["split"] = outer["outer_split"]
    outer["stage"] = "all"
    outer["split_id"] = "outer"
    outer["outer_split_seed"] = int(outer_split_seed)
    outer["inner_split_seed"] = np.nan
    write_manifest(outer_path, outer, heads)

    hpo_pool = outer.loc[outer["outer_split"] == "hpo_pool"].copy()
    outer_audit = outer.copy()
    outer_audit["split"] = outer_audit["outer_split"]
    audit_frames = [split_audit(outer_audit, heads).assign(level="outer", split_id="outer")]
    for idx, inner_seed in enumerate(inner_split_seeds):
        split_id = make_split_id(inner_seed, idx)
        inner = hpo_pool.copy()
        inner["inner_split_hash"] = inner["seq_upper"].map(
            lambda seq: hash_float(seq, seed=inner_seed, salt=f"{split_id}_inner_val")
        )
        val_mask = stratified_holdout_mask(
            inner,
            holdout_frac=inner_val_frac,
            seed=inner_seed,
            hash_salt=f"{split_id}_inner_val",
        )
        inner["split"] = np.where(val_mask, "val", "train")
        inner["stage"] = "hpo"
        inner["split_id"] = split_id
        inner["outer_split_seed"] = int(outer_split_seed)
        inner["inner_split_seed"] = int(inner_seed)
        write_manifest(inner_paths[split_id], inner, heads)
        audit_frames.append(
            split_audit(inner, heads).assign(
                level="inner",
                split_id=split_id,
                inner_split_seed=int(inner_seed),
            )
        )

    audit_df = pd.concat(audit_frames, ignore_index=True, sort=False)
    audit_df.to_csv(audit_path, index=False)
    write_json(
        policy_path,
        {
            "version": "phase2_v2",
            "split_unit": "unique uppercased exact 50-nt sequence",
            "outer_split_seed": int(outer_split_seed),
            "final_test_frac": float(final_test_frac),
            "inner_split_seeds": [int(seed) for seed in inner_split_seeds],
            "inner_val_frac": float(inner_val_frac),
            "activity_quantile_bins": int(activity_quantile_bins),
            "gc_quantile_bins": int(gc_quantile_bins),
            "stratification_variables": ["average_activity", "gc_fraction"],
            "final_test_usage_policy": (
                "Final-test rows are excluded from screening and confirmation ranking, "
                "early stopping, checkpoint selection, and plot-driven decisions."
            ),
        },
    )
    return {
        "outer": outer_path,
        "audit": audit_path,
        "policy": policy_path,
        **inner_paths,
    }


def split_audit(split_df: pd.DataFrame, heads: list[str]) -> pd.DataFrame:
    rows = []
    for split, sub in split_df.groupby("split", observed=True):
        row = {
            "split": split,
            "n_sequences": int(len(sub)),
            "gc_mean": float(sub["gc_fraction"].mean()),
            "gc_std": float(sub["gc_fraction"].std()),
            "avg_activity_mean": float(sub[heads].mean(axis=1).mean()),
            "avg_activity_std": float(sub[heads].mean(axis=1).std()),
        }
        for head in heads:
            row[f"{head}_mean"] = float(sub[head].mean())
            row[f"{head}_std"] = float(sub[head].std())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("split").reset_index(drop=True)


def load_inhouse_fiveprime(inhouse_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(inhouse_path)
    required = ["FivePrime", "RNA/DNA"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required in-house columns in {inhouse_path}: {missing}")

    df = raw.copy()
    df["candidate_id"] = np.arange(len(df))
    df["candidate_seq_original"] = df["FivePrime"].astype(str)
    df["candidate_seq"] = df["candidate_seq_original"].map(clean_sequence)
    df["sequence_len"] = df["candidate_seq"].str.len()
    df["is_valid_exact_50nt"] = df["candidate_seq"].map(is_valid_exact_dna)
    df["RNA/DNA"] = pd.to_numeric(df["RNA/DNA"], errors="coerce")
    if "number_of_barcodes" in df.columns:
        df["number_of_barcodes"] = pd.to_numeric(df["number_of_barcodes"], errors="coerce")
    else:
        df["number_of_barcodes"] = np.nan
    for column in ["DNA_bc_counts_sum", "RNA_bc_counts_sum"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    usable = df.loc[df["is_valid_exact_50nt"] & np.isfinite(df["RNA/DNA"])].copy()
    usable["log2_RNA_DNA"] = np.where(
        usable["RNA/DNA"] > 0,
        np.log2(usable["RNA/DNA"]),
        np.nan,
    )
    usable["gc_fraction"] = usable["candidate_seq"].map(gc_fraction)

    audit = {
        "raw_rows": int(len(raw)),
        "exact_50nt_rows": int(df["is_valid_exact_50nt"].sum()),
        "usable_exact_finite_rows": int(len(usable)),
        "unique_usable_sequences": int(usable["candidate_seq"].nunique()),
    }
    return usable.reset_index(drop=True), audit


def make_prediction_loader(seqs: Iterable[str], batch_size: int) -> DataLoader:
    x = torch.stack([utils.dna2tensor(seq) for seq in seqs])
    return DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)


@torch.no_grad()
def predict_scaled(
    model: torch.nn.Module,
    seqs: Iterable[str],
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    preds = []
    loader = make_prediction_loader(seqs, batch_size=batch_size)
    for (x_batch,) in loader:
        x_batch = x_batch.to(device)
        preds.append(model(x_batch).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_raw(
    model: torch.nn.Module,
    seqs: Iterable[str],
    scaler: TargetScaler,
    heads: list[str],
    batch_size: int,
    device: str,
) -> np.ndarray:
    pred_scaled = predict_scaled(model, seqs=seqs, batch_size=batch_size, device=device)
    return scaler.inverse_array(pred_scaled, heads=heads)


def evaluate_wide_predictions(
    pred_raw: np.ndarray,
    truth_df: pd.DataFrame,
    heads: list[str],
    model_label: str,
    split_name: str,
    run_seed: int | None,
    unfreeze_scope: str | None,
    head_lr: float | None,
    backbone_lr: float | None,
    scaler_source: str,
    run_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    context = run_context or {}
    true_raw = truth_df[heads].to_numpy(dtype=np.float32)
    pred_df = pd.DataFrame({
        "seq_upper": truth_df["seq_upper"].to_numpy(),
        "split": split_name,
        "model_label": model_label,
    })
    for key, value in context.items():
        pred_df[key] = value
    if "seq_original_example" in truth_df.columns:
        pred_df["seq_original_example"] = truth_df["seq_original_example"].to_numpy()
    elif "seq_original" in truth_df.columns:
        pred_df["seq_original_example"] = truth_df["seq_original"].to_numpy()

    for idx, head in enumerate(heads):
        pred_df[f"true_{head}"] = true_raw[:, idx]
        pred_df[f"pred_{head}"] = pred_raw[:, idx]
    pred_df["true_average_activity"] = true_raw.mean(axis=1)
    pred_df["pred_average_activity"] = pred_raw.mean(axis=1)

    per_head_records = []
    pearsons = []
    for idx, head in enumerate(heads):
        metrics = regression_metrics(true_raw[:, idx], pred_raw[:, idx])
        pearsons.append(metrics["pearson"])
        per_head_records.append({
            "model_label": model_label,
            "split": split_name,
            "head": head,
            "cell_type_name": CELL_TYPE_NAMES.get(head, ""),
            "run_seed": run_seed,
            "unfreeze_scope": unfreeze_scope,
            "head_lr": head_lr,
            "backbone_lr": backbone_lr,
            "target_scaler_source": scaler_source,
            **context,
            **metrics,
        })

    flat_metrics = regression_metrics(true_raw.ravel(), pred_raw.ravel())
    avg_metrics = regression_metrics(true_raw.mean(axis=1), pred_raw.mean(axis=1))
    mean_per_head = float(np.nanmean(pearsons)) if len(pearsons) else float("nan")
    summary_records = [{
        "model_label": model_label,
        "split": split_name,
        "run_seed": run_seed,
        "unfreeze_scope": unfreeze_scope,
        "head_lr": head_lr,
        "backbone_lr": backbone_lr,
        "target_scaler_source": scaler_source,
        **context,
        "n_sequences": int(len(truth_df)),
        "n_heads": int(len(heads)),
        "heads_used": ",".join(heads),
        "mean_per_head_pearson": mean_per_head,
        "mean_per_head_pearson_r2": float(mean_per_head * mean_per_head) if np.isfinite(mean_per_head) else float("nan"),
        "flattened_activity_pearson": flat_metrics["pearson"],
        "flattened_activity_pearson_r2": flat_metrics["pearson_r2"],
        "average_activity_pearson": avg_metrics["pearson"],
        "average_activity_pearson_r2": avg_metrics["pearson_r2"],
        "average_activity_spearman": avg_metrics["spearman"],
        "average_activity_mae": avg_metrics["mae"],
        "average_activity_rmse": avg_metrics["rmse"],
    }]
    return summary_records, per_head_records, pred_df


def evaluate_inhouse_predictions(
    pred_raw: np.ndarray,
    inhouse_df: pd.DataFrame,
    heads: list[str],
    model_label: str,
    run_seed: int | None,
    unfreeze_scope: str | None,
    head_lr: float | None,
    backbone_lr: float | None,
    scaler_source: str,
    min_barcodes: int,
    run_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    context = run_context or {}
    pred_df = inhouse_df[
        [
            "candidate_id",
            "candidate_seq",
            "RNA/DNA",
            "log2_RNA_DNA",
            "number_of_barcodes",
        ]
    ].copy()
    for key, value in context.items():
        pred_df[key] = value
    for column in ["DNA_bc_counts_sum", "RNA_bc_counts_sum"]:
        if column in inhouse_df.columns:
            pred_df[column] = inhouse_df[column].to_numpy()
    pred_df["model_label"] = model_label
    for idx, head in enumerate(heads):
        pred_df[f"pred_activity_{head}"] = pred_raw[:, idx]
    pred_df["pred_average_activity"] = pred_raw.mean(axis=1)

    target_columns = ["RNA/DNA", "log2_RNA_DNA"]
    predictor_columns = [f"pred_activity_{head}" for head in heads] + ["pred_average_activity"]
    subsets = {
        "all_exact_finite": pred_df,
        f"barcode_min_{min_barcodes}": pred_df.loc[pred_df["number_of_barcodes"] >= min_barcodes],
    }
    metric_records = []
    for subset_name, subset_df in subsets.items():
        for target_column in target_columns:
            for predictor_column in predictor_columns:
                metrics = regression_metrics(subset_df[target_column], subset_df[predictor_column])
                metric_records.append({
                    "model_label": model_label,
                    "subset": subset_name,
                    "target": target_column,
                    "predictor": predictor_column,
                    "run_seed": run_seed,
                    "unfreeze_scope": unfreeze_scope,
                    "head_lr": head_lr,
                    "backbone_lr": backbone_lr,
                    "target_scaler_source": scaler_source,
                    **context,
                    **metrics,
                })
    return metric_records, pred_df


def evaluate_model_everywhere(
    model: torch.nn.Module,
    scaler: TargetScaler,
    heads: list[str],
    model_label: str,
    lib2_split_df: pd.DataFrame | None,
    lib1_test_df: pd.DataFrame,
    inhouse_df: pd.DataFrame,
    pred_batch_size: int,
    device: str,
    prediction_dir: Path,
    run_seed: int | None = None,
    unfreeze_scope: str | None = None,
    head_lr: float | None = None,
    backbone_lr: float | None = None,
    inhouse_min_barcodes: int = 8,
    lib2_eval_sets: list[tuple[str, pd.DataFrame]] | None = None,
    run_context: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    prediction_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_label)
    all_summary: list[dict[str, Any]] = []
    all_per_head: list[dict[str, Any]] = []
    all_inhouse: list[dict[str, Any]] = []

    if lib2_eval_sets is None:
        if lib2_split_df is None:
            raise ValueError("Either lib2_split_df or lib2_eval_sets is required.")
        lib2_eval_sets = [
            ("lib2_val", lib2_split_df.loc[lib2_split_df["split"] == "val"].copy()),
            ("lib2_test", lib2_split_df.loc[lib2_split_df["split"] == "test"].copy()),
        ]

    eval_sets = [
        *[(name, frame.copy()) for name, frame in lib2_eval_sets if frame is not None and not frame.empty],
        ("lib1_test_retention", lib1_test_df.copy()),
    ]
    for split_name, truth_df in eval_sets:
        pred_raw = predict_raw(
            model,
            truth_df["seq_upper"].tolist(),
            scaler=scaler,
            heads=heads,
            batch_size=pred_batch_size,
            device=device,
        )
        summary, per_head, pred_df = evaluate_wide_predictions(
            pred_raw=pred_raw,
            truth_df=truth_df,
            heads=heads,
            model_label=model_label,
            split_name=split_name,
            run_seed=run_seed,
            unfreeze_scope=unfreeze_scope,
            head_lr=head_lr,
            backbone_lr=backbone_lr,
            scaler_source=scaler.source,
            run_context=run_context,
        )
        all_summary.extend(summary)
        all_per_head.extend(per_head)
        pred_df.to_csv(prediction_dir / f"{safe_label}__{split_name}_predictions.csv", index=False)

    inhouse_pred_raw = predict_raw(
        model,
        inhouse_df["candidate_seq"].tolist(),
        scaler=scaler,
        heads=heads,
        batch_size=pred_batch_size,
        device=device,
    )
    inhouse_metrics, inhouse_pred_df = evaluate_inhouse_predictions(
        pred_raw=inhouse_pred_raw,
        inhouse_df=inhouse_df,
        heads=heads,
        model_label=model_label,
        run_seed=run_seed,
        unfreeze_scope=unfreeze_scope,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        scaler_source=scaler.source,
        min_barcodes=inhouse_min_barcodes,
        run_context=run_context,
    )
    all_inhouse.extend(inhouse_metrics)
    inhouse_pred_df.to_csv(prediction_dir / f"{safe_label}__inhouse_fiveprime_predictions.csv", index=False)

    return {
        "summary": pd.DataFrame(all_summary),
        "per_head": pd.DataFrame(all_per_head),
        "inhouse": pd.DataFrame(all_inhouse),
    }


def make_loader(
    df: pd.DataFrame,
    heads: list[str],
    scaler: TargetScaler,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
) -> DataLoader:
    dataset = UTR5WideDataset(df=df, heads=heads, scaler=scaler)
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def last_stage_prefixes(model: torch.nn.Module, model_hparams: dict[str, Any]) -> list[str]:
    stage_blocks = list(model_hparams.get("stage_blocks", [2, 2, 2]))
    last_stage_blocks = int(stage_blocks[-1]) if stage_blocks else 0
    n_encoder_blocks = len(getattr(model, "encoder", []))
    start = max(0, n_encoder_blocks - last_stage_blocks)
    return [f"encoder.{idx}." for idx in range(start, n_encoder_blocks)]


def set_trainable_scope(
    model: torch.nn.Module,
    scope: str,
    model_hparams: dict[str, Any],
    head_only_warmup: bool,
) -> None:
    active_scope = "head_only" if head_only_warmup else scope
    last_prefixes = last_stage_prefixes(model, model_hparams)

    for name, parameter in model.named_parameters():
        if active_scope == "full":
            parameter.requires_grad = True
        elif active_scope == "head_only":
            parameter.requires_grad = name.startswith("head.")
        elif active_scope == "last_stage_plus_head":
            parameter.requires_grad = name.startswith("head.") or any(
                name.startswith(prefix) for prefix in last_prefixes
            )
        else:
            raise ValueError(f"Unknown unfreeze scope: {scope}")


def build_optimizer(
    model: torch.nn.Module,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if name.startswith("head."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)
    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr, "name": "head"})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": backbone_lr, "name": "backbone"})
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


@torch.no_grad()
def evaluate_hani_epoch_split(
    model: torch.nn.Module,
    split_name: str,
    df: pd.DataFrame,
    scaler: TargetScaler,
    heads: list[str],
    batch_size: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    pred_scaled = predict_scaled(
        model,
        seqs=df["seq_upper"].tolist(),
        batch_size=batch_size,
        device=device,
    )
    true_scaled = scaler.transform_frame(df, heads)
    pred_raw = scaler.inverse_array(pred_scaled, heads=heads)
    true_raw = df[heads].to_numpy(dtype=np.float32)

    records: list[dict[str, Any]] = []
    pearsons = []
    spearmans = []
    cod_r2_values = []
    per_head_losses = []
    for idx, head in enumerate(heads):
        raw_metrics = regression_metrics(true_raw[:, idx], pred_raw[:, idx])
        standardized_mse = float(np.mean((true_scaled[:, idx] - pred_scaled[:, idx]) ** 2))
        pearsons.append(raw_metrics["pearson"])
        spearmans.append(raw_metrics["spearman"])
        cod_r2_values.append(raw_metrics["cod_r2"])
        per_head_losses.append(standardized_mse)
        records.append(
            {
                "split": split_name,
                "scope": "per_head",
                "head": head,
                "cell_type_name": CELL_TYPE_NAMES.get(head, ""),
                "target": head,
                "predictor": f"pred_activity_{head}",
                "metric_target": "hani_rna_activity",
                "n": raw_metrics["n"],
                "pearson": raw_metrics["pearson"],
                "spearman": raw_metrics["spearman"],
                "pearson_r2": raw_metrics["pearson_r2"],
                "cod_r2": raw_metrics["cod_r2"],
                "loss": standardized_mse,
                "loss_kind": "standardized_mse",
                "mse_raw": raw_metrics["mse"],
                "mae_raw": raw_metrics["mae"],
                "rmse_raw": raw_metrics["rmse"],
            }
        )

    loss_standardized = float(np.mean((true_scaled - pred_scaled) ** 2))
    flattened = regression_metrics(true_raw.ravel(), pred_raw.ravel())
    average_activity = regression_metrics(true_raw.mean(axis=1), pred_raw.mean(axis=1))
    aggregate = {
        f"{split_name}_loss_standardized": loss_standardized,
        f"{split_name}_mean_per_head_pearson": float(np.nanmean(pearsons)),
        f"{split_name}_mean_per_head_spearman": float(np.nanmean(spearmans)),
        f"{split_name}_mean_per_head_cod_r2": float(np.nanmean(cod_r2_values)),
        f"{split_name}_average_activity_pearson": average_activity["pearson"],
        f"{split_name}_average_activity_spearman": average_activity["spearman"],
        f"{split_name}_average_activity_cod_r2": average_activity["cod_r2"],
        f"{split_name}_average_activity_rmse": average_activity["rmse"],
        f"{split_name}_flattened_activity_pearson": flattened["pearson"],
        f"{split_name}_flattened_activity_spearman": flattened["spearman"],
        f"{split_name}_flattened_activity_cod_r2": flattened["cod_r2"],
    }
    for record in records:
        head = record["head"]
        prefix = f"{split_name}_{head}"
        aggregate[f"{prefix}_pearson"] = record["pearson"]
        aggregate[f"{prefix}_spearman"] = record["spearman"]
        aggregate[f"{prefix}_pearson_r2"] = record["pearson_r2"]
        aggregate[f"{prefix}_cod_r2"] = record["cod_r2"]
        aggregate[f"{prefix}_loss"] = record["loss"]
        aggregate[f"{prefix}_loss_kind"] = record["loss_kind"]
        aggregate[f"{prefix}_mse_raw"] = record["mse_raw"]
    return records, aggregate


@torch.no_grad()
def evaluate_inhouse_epoch_split(
    model: torch.nn.Module,
    inhouse_df: pd.DataFrame,
    scaler: TargetScaler,
    heads: list[str],
    batch_size: int,
    device: str,
    min_barcodes: int,
    target_column: str = "log2_RNA_DNA",
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    model.eval()
    pred_raw = predict_raw(
        model,
        seqs=inhouse_df["candidate_seq"].tolist(),
        scaler=scaler,
        heads=heads,
        batch_size=batch_size,
        device=device,
    )
    pred_df = inhouse_df[[target_column, "number_of_barcodes"]].copy()
    for idx, head in enumerate(heads):
        pred_df[f"pred_activity_{head}"] = pred_raw[:, idx]

    subsets = {
        "test_inhouse_all": pred_df,
        "test_inhouse": pred_df.loc[pred_df["number_of_barcodes"] >= min_barcodes],
    }

    records: list[dict[str, Any]] = []
    aggregate: dict[str, float] = {}
    for split_name, subset_df in subsets.items():
        pearsons = []
        spearmans = []
        cod_r2_values = []
        losses = []
        for head in heads:
            predictor = f"pred_activity_{head}"
            metrics = regression_metrics(subset_df[target_column], subset_df[predictor])
            pearsons.append(metrics["pearson"])
            spearmans.append(metrics["spearman"])
            cod_r2_values.append(metrics["cod_r2"])
            losses.append(metrics["mse"])
            record = {
                "split": split_name,
                "scope": "per_head_proxy",
                "head": head,
                "cell_type_name": CELL_TYPE_NAMES.get(head, ""),
                "target": target_column,
                "predictor": predictor,
                "metric_target": "inhouse_fiveprime_proxy",
                "n": metrics["n"],
                "pearson": metrics["pearson"],
                "spearman": metrics["spearman"],
                "pearson_r2": metrics["pearson_r2"],
                "cod_r2": metrics["cod_r2"],
                "loss": metrics["mse"],
                "loss_kind": f"proxy_mse_vs_{target_column}",
                "mse_raw": metrics["mse"],
                "mae_raw": metrics["mae"],
                "rmse_raw": metrics["rmse"],
            }
            records.append(record)

            prefix = f"{split_name}_{head}"
            aggregate[f"{prefix}_pearson"] = record["pearson"]
            aggregate[f"{prefix}_spearman"] = record["spearman"]
            aggregate[f"{prefix}_pearson_r2"] = record["pearson_r2"]
            aggregate[f"{prefix}_cod_r2"] = record["cod_r2"]
            aggregate[f"{prefix}_loss"] = record["loss"]
            aggregate[f"{prefix}_loss_kind"] = record["loss_kind"]

        aggregate[f"{split_name}_mean_per_head_pearson"] = float(np.nanmean(pearsons))
        aggregate[f"{split_name}_mean_per_head_spearman"] = float(np.nanmean(spearmans))
        aggregate[f"{split_name}_mean_per_head_cod_r2"] = float(np.nanmean(cod_r2_values))
        aggregate[f"{split_name}_mean_per_head_loss"] = float(np.nanmean(losses))

    return records, aggregate


def evaluate_epoch_diagnostics(
    model: torch.nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    inhouse_df: pd.DataFrame,
    scaler: TargetScaler,
    heads: list[str],
    batch_size: int,
    device: str,
    inhouse_min_barcodes: int,
    hani_eval_sets: list[tuple[str, pd.DataFrame]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    all_records: list[dict[str, Any]] = []
    aggregate: dict[str, float] = {}
    if hani_eval_sets is None:
        hani_eval_sets = [("train", train_df)]
        if val_df is not None and not val_df.empty:
            hani_eval_sets.append(("val", val_df))
        if test_df is not None and not test_df.empty:
            hani_eval_sets.append(("test", test_df))

    for split_name, split_df in hani_eval_sets:
        if split_df is None or split_df.empty:
            continue
        split_records, split_aggregate = evaluate_hani_epoch_split(
            model=model,
            split_name=split_name,
            df=split_df,
            scaler=scaler,
            heads=heads,
            batch_size=batch_size,
            device=device,
        )
        all_records.extend(split_records)
        aggregate.update(split_aggregate)

    inhouse_records, inhouse_aggregate = evaluate_inhouse_epoch_split(
        model=model,
        inhouse_df=inhouse_df,
        scaler=scaler,
        heads=heads,
        batch_size=batch_size,
        device=device,
        min_barcodes=inhouse_min_barcodes,
    )
    all_records.extend(inhouse_records)
    aggregate.update(inhouse_aggregate)
    return all_records, aggregate


def annotate_epoch_records(
    records: list[dict[str, Any]],
    epoch_record: dict[str, Any],
) -> list[dict[str, Any]]:
    context_keys = [
        "epoch",
        "run_seed",
        "training_seed",
        "stage",
        "outer_split_seed",
        "inner_split_seed",
        "split_id",
        "unfreeze_scope",
        "head_lr",
        "backbone_lr",
        "target_scaler_source",
        "head_only_warmup",
        "freeze_backbone_epochs",
        "weight_decay",
    ]
    context = {key: epoch_record[key] for key in context_keys}
    return [{**context, **record} for record in records]


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
    checkpoint: dict[str, Any],
    spec: ExperimentSpec,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    test_df: pd.DataFrame | None,
    inhouse_df: pd.DataFrame,
    heads: list[str],
    scaler: TargetScaler,
    device: str,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    freeze_backbone_epochs: int | None,
    train_batch_size: int,
    pred_batch_size: int,
    weight_decay: float | None,
    monitor_metric: str,
    inhouse_min_barcodes: int,
    use_early_stopping: bool = True,
    hani_eval_sets: list[tuple[str, pd.DataFrame]] | None = None,
) -> tuple[torch.nn.Module, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    set_global_seed(spec.seed)
    freeze_epochs = spec.freeze_backbone_epochs if freeze_backbone_epochs is None else int(freeze_backbone_epochs)
    wd = spec.weight_decay if weight_decay is None else float(weight_decay)
    model = build_model_from_checkpoint(checkpoint, device=device)
    model_hparams = namespace_to_dict(checkpoint["model_hparams"])
    optimizer = build_optimizer(
        model,
        head_lr=spec.head_lr,
        backbone_lr=spec.backbone_lr,
        weight_decay=wd,
    )
    criterion = torch.nn.MSELoss()
    train_loader = make_loader(
        train_df,
        heads=heads,
        scaler=scaler,
        batch_size=train_batch_size,
        shuffle=True,
        seed=spec.seed,
    )

    monitor_mode = "min" if monitor_metric == "val_loss_standardized" else "max"
    best_state = copy.deepcopy(model.state_dict())
    best_value = math.inf if monitor_mode == "min" else -math.inf
    best_epoch = -1
    patience_counter = 0
    history = []
    diagnostics = []

    desc = f"{spec.tag()}"
    for epoch in tqdm(range(max_epochs), desc=desc):
        head_only_warmup = epoch < freeze_epochs
        set_trainable_scope(
            model,
            scope=spec.unfreeze_scope,
            model_hparams=model_hparams,
            head_only_warmup=head_only_warmup,
        )
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred_batch = model(x_batch)
            loss = criterion(pred_batch, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(x_batch)
            train_items += len(x_batch)

        diagnostic_records, diagnostic_summary = evaluate_epoch_diagnostics(
            model=model,
            train_df=train_df,
            scaler=scaler,
            test_df=test_df,
            inhouse_df=inhouse_df,
            val_df=val_df,
            heads=heads,
            batch_size=pred_batch_size,
            device=device,
            inhouse_min_barcodes=inhouse_min_barcodes,
            hani_eval_sets=hani_eval_sets,
        )
        epoch_record = {
            "epoch": int(epoch),
            "train_loss_batch_standardized": float(train_loss_sum / max(train_items, 1)),
            "run_seed": int(spec.seed),
            "training_seed": int(spec.seed),
            "unfreeze_scope": spec.unfreeze_scope,
            "head_lr": float(spec.head_lr),
            "backbone_lr": float(spec.backbone_lr),
            "target_scaler_source": spec.target_scaler_source,
            "head_only_warmup": bool(head_only_warmup),
            **spec.context(),
            **diagnostic_summary,
        }
        history.append(epoch_record)
        diagnostics.extend(annotate_epoch_records(diagnostic_records, epoch_record))

        if use_early_stopping:
            if monitor_metric not in epoch_record:
                raise ValueError(
                    f"Monitor metric {monitor_metric!r} was not produced. "
                    "Use a validation split or disable early stopping for this stage."
                )
            monitor_value = float(epoch_record[monitor_metric])
            improved = monitor_improved(monitor_value, best_value, monitor_mode)
        else:
            monitor_value = float("nan")
            improved = True

        if improved or best_epoch < 0:
            best_value = monitor_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if use_early_stopping and epoch + 1 >= min_epochs and patience_counter >= patience:
            break

    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    diagnostics_df = pd.DataFrame(diagnostics)
    fit_info = {
        "run_seed": int(spec.seed),
        "unfreeze_scope": spec.unfreeze_scope,
        "head_lr": float(spec.head_lr),
        "backbone_lr": float(spec.backbone_lr),
        "target_scaler_source": spec.target_scaler_source,
        **spec.context(),
        "target_scaler": scaler.to_dict(),
        "best_epoch": int(best_epoch),
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "best_monitor_value": float(best_value),
        "max_epochs": int(max_epochs),
        "min_epochs": int(min_epochs),
        "patience": int(patience),
        "freeze_backbone_epochs": int(freeze_epochs),
        "train_batch_size": int(train_batch_size),
        "pred_batch_size": int(pred_batch_size),
        "weight_decay": float(wd),
        "use_early_stopping": bool(use_early_stopping),
        "epoch_diagnostics": {
            "hani_splits": [
                split_name
                for split_name, split_df in (
                    hani_eval_sets
                    if hani_eval_sets is not None
                    else [("train", train_df), ("val", val_df), ("test", test_df)]
                )
                if split_df is not None and not split_df.empty
            ],
            "inhouse_splits": ["test_inhouse_all", "test_inhouse"],
            "inhouse_target": "log2_RNA_DNA",
            "heldout_and_inhouse_epoch_metrics_are_diagnostic_only": True,
        },
    }
    return model, history_df, diagnostics_df, fit_info


def select_scaler(
    source: str,
    pretrained_scaler: TargetScaler,
    lib2_train_scaler: TargetScaler,
) -> TargetScaler:
    if source == "pretrained_lib1_train":
        return pretrained_scaler
    if source == "lib2_train":
        return lib2_train_scaler
    raise ValueError(f"Unknown target scaler source: {source}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_if_not_empty(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frame.empty:
        frame.to_csv(path, index=False)


def combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True, sort=False)


def current_split_id(args: argparse.Namespace) -> str:
    if args.split_id:
        return args.split_id
    if args.stage in {"screening", "confirmation"}:
        try:
            index = list(args.inner_split_seeds).index(args.inner_split_seed)
        except ValueError:
            index = None
        return make_split_id(args.inner_split_seed, index)
    if args.stage == "final_eval":
        return "hpo_pool_to_final_test"
    return f"legacy_split_seed_{args.split_seed}"


def default_split_manifest_dir(args: argparse.Namespace) -> Path:
    return args.split_manifest_dir or (args.outdir / "split_manifests")


def load_manifest(path: Path, heads: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for column in [
        "sequence_len",
        "gc_fraction",
        "average_activity",
        "activity_quantile",
        "gc_quantile",
        "outer_split_seed",
        "inner_split_seed",
        "outer_split_hash",
        "inner_split_hash",
        "n_raw_rows",
        *[f"n_obs_{head}" for head in heads],
        *heads,
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def phase2_v2_split_frames(
    args: argparse.Namespace,
    lib2_wide: pd.DataFrame,
    heads: list[str],
) -> dict[str, Any]:
    split_manifest_dir = default_split_manifest_dir(args)
    build_phase2_v2_split_manifests(
        lib2_wide,
        heads=heads,
        split_manifest_dir=split_manifest_dir,
        outer_split_seed=args.outer_split_seed,
        final_test_frac=args.final_test_frac,
        inner_split_seeds=args.inner_split_seeds,
        inner_val_frac=args.inner_val_frac,
        activity_quantile_bins=args.activity_quantile_bins,
        gc_quantile_bins=args.gc_quantile_bins,
        force=args.force,
    )

    outer_path = args.outer_final_test_manifest or (split_manifest_dir / "outer_final_test_manifest.csv")
    outer_df = load_manifest(outer_path, heads)
    if "outer_split" not in outer_df.columns:
        raise ValueError(f"{outer_path} must contain an outer_split column.")

    split_id = current_split_id(args)
    inner_df = None
    train_df: pd.DataFrame
    val_df: pd.DataFrame | None
    final_test_df = outer_df.loc[outer_df["outer_split"] == "final_test"].copy()
    hpo_pool_df = outer_df.loc[outer_df["outer_split"] == "hpo_pool"].copy()
    if hpo_pool_df.empty or final_test_df.empty:
        raise ValueError("Phase 2 v2 outer split requires non-empty hpo_pool and final_test partitions.")

    if args.stage in {"screening", "confirmation"}:
        inner_path = args.inner_split_manifest or (split_manifest_dir / f"inner_split_manifest_{split_id}.csv")
        inner_df = load_manifest(inner_path, heads)
        if "split" not in inner_df.columns:
            raise ValueError(f"{inner_path} must contain a split column.")
        if "final_test" in set(inner_df["split"].astype(str)):
            raise ValueError(f"{inner_path} unexpectedly contains final_test rows.")
        train_df = inner_df.loc[inner_df["split"] == "train"].copy()
        val_df = inner_df.loc[inner_df["split"] == "val"].copy()
        if train_df.empty or val_df.empty:
            raise ValueError(f"{inner_path} must contain non-empty train and val splits.")
        lib2_eval_sets = [("lib2_inner_val", val_df)]
        epoch_eval_sets = [("train", train_df), ("val", val_df)]
        use_early_stopping = True
    elif args.stage == "final_eval":
        train_df = hpo_pool_df.copy()
        train_df["split"] = "train"
        val_df = None
        lib2_eval_sets = [("lib2_final_test", final_test_df)]
        epoch_eval_sets = [("train", train_df)]
        use_early_stopping = False
    else:
        raise ValueError(f"Unsupported v2 stage: {args.stage}")

    return {
        "split_manifest_dir": split_manifest_dir,
        "split_id": split_id,
        "outer_df": outer_df,
        "inner_df": inner_df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": final_test_df if args.stage == "final_eval" else None,
        "hpo_pool_df": hpo_pool_df,
        "final_test_df": final_test_df,
        "lib2_eval_sets": lib2_eval_sets,
        "epoch_eval_sets": epoch_eval_sets,
        "use_early_stopping": use_early_stopping,
        "outer_path": outer_path,
        "inner_path": args.inner_split_manifest or (split_manifest_dir / f"inner_split_manifest_{split_id}.csv"),
    }


def build_specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    specs = []
    target_scaler_sources = args.target_scaler_sources
    if not target_scaler_sources:
        target_scaler_sources = [args.target_scaler_source]
    freeze_epochs_values = args.freeze_backbone_epochs_list
    if not freeze_epochs_values:
        freeze_epochs_values = [args.freeze_backbone_epochs]
    weight_decay_values = args.weight_decays
    if not weight_decay_values:
        weight_decay_values = [args.weight_decay]
    split_id = current_split_id(args)

    for seed in args.seeds:
        for scope in args.unfreeze_scopes:
            scope_backbone_lrs = args.backbone_lrs
            if scope == "head_only" and args.stage != "legacy_v1":
                scope_backbone_lrs = [args.backbone_lrs[0]]
            for head_lr in args.head_lrs:
                for backbone_lr in scope_backbone_lrs:
                    for scaler_source in target_scaler_sources:
                        for freeze_epochs in freeze_epochs_values:
                            for weight_decay in weight_decay_values:
                                specs.append(
                                    ExperimentSpec(
                                        seed=int(seed),
                                        unfreeze_scope=scope,
                                        head_lr=float(head_lr),
                                        backbone_lr=float(backbone_lr),
                                        target_scaler_source=scaler_source,
                                        stage=args.stage,
                                        outer_split_seed=(
                                            int(args.outer_split_seed)
                                            if args.stage != "legacy_v1"
                                            else None
                                        ),
                                        inner_split_seed=(
                                            int(args.inner_split_seed)
                                            if args.stage in {"screening", "confirmation"} and args.inner_split_seed is not None
                                            else None
                                        ),
                                        split_id=split_id,
                                        freeze_backbone_epochs=int(freeze_epochs),
                                        weight_decay=float(weight_decay),
                                    )
                                )
    return specs


def spec_record(spec: ExperimentSpec) -> dict[str, Any]:
    record = asdict(spec)
    record["training_seed"] = spec.training_seed
    record["tag"] = spec.tag()
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BODA 5'UTR ResNet1D run 1mmy39ku on Hani/Goodarzi 5'UTR Lib2."
    )
    parser.add_argument("--artifact_path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--lib1_path", type=Path, default=DEFAULT_LIB1_PATH)
    parser.add_argument("--lib2_path", type=Path, default=DEFAULT_LIB2_PATH)
    parser.add_argument("--inhouse_path", type=Path, default=DEFAULT_INHOUSE_PATH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument(
        "--stage",
        choices=["legacy_v1", "screening", "confirmation", "final_eval"],
        default="screening",
        help="Phase 2 v2 stage. Use legacy_v1 to reproduce the old 80/10/10 hash split behavior.",
    )
    parser.add_argument("--split_manifest_dir", type=Path, default=None)
    parser.add_argument("--outer_final_test_manifest", type=Path, default=None)
    parser.add_argument("--inner_split_manifest", type=Path, default=None)
    parser.add_argument("--outer_split_seed", "--final_test_seed", dest="outer_split_seed", type=int, default=20260526)
    parser.add_argument("--final_test_frac", type=float, default=0.10)
    parser.add_argument("--inner_split_seed", type=int, default=101)
    parser.add_argument("--inner_split_seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--inner_val_frac", type=float, default=0.10)
    parser.add_argument("--split_id", type=str, default=None)
    parser.add_argument("--activity_quantile_bins", type=int, default=5)
    parser.add_argument("--gc_quantile_bins", type=int, default=5)

    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.10)
    parser.add_argument("--test_frac", type=float, default=0.10)
    parser.add_argument("--inhouse_min_barcodes", type=int, default=8)

    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument(
        "--unfreeze_scopes",
        nargs="+",
        choices=["head_only", "last_stage_plus_head", "full"],
        default=["head_only", "last_stage_plus_head"],
    )
    parser.add_argument("--head_lrs", nargs="+", type=float, default=[1e-4, 3e-4])
    parser.add_argument("--backbone_lrs", nargs="+", type=float, default=[1e-5])
    parser.add_argument(
        "--target_scaler_source",
        choices=["pretrained_lib1_train", "lib2_train"],
        default="pretrained_lib1_train",
        help="Use the pretrained Lib1 train scaler to preserve the original output coordinate, or a Lib2 train scaler.",
    )
    parser.add_argument(
        "--target_scaler_sources",
        nargs="+",
        choices=["pretrained_lib1_train", "lib2_train"],
        default=None,
        help="Grid over target scaler sources. Defaults to --target_scaler_source.",
    )
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--min_epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=3)
    parser.add_argument("--freeze_backbone_epochs_list", nargs="+", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--pred_batch_size", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--weight_decays", nargs="+", type=float, default=None)
    parser.add_argument(
        "--monitor_metric",
        choices=[
            "val_average_activity_pearson",
            "val_mean_per_head_pearson",
            "val_flattened_activity_pearson",
            "val_loss_standardized",
        ],
        default="val_average_activity_pearson",
    )
    parser.add_argument("--preview_only", action="store_true", help="Write splits/manifest and planned specs, then exit.")
    parser.add_argument("--prepare_splits_only", action="store_true", help="Alias for --preview_only with emphasis on v2 split materialization.")
    parser.add_argument("--force", action="store_true", help="Recompute run directories even if outputs already exist.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def validate_paths(args: argparse.Namespace) -> None:
    for path in [args.artifact_path, args.lib1_path, args.lib2_path, args.inhouse_path]:
        if not path.exists():
            raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    args.preview_only = bool(args.preview_only or args.prepare_splits_only)
    validate_paths(args)
    device = resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "predictions").mkdir(parents=True, exist_ok=True)
    (args.outdir / "runs").mkdir(parents=True, exist_ok=True)

    specs = build_specs(args)
    write_json(
        args.outdir / "run_manifest.json",
        {
            "repo_root": str(REPO_ROOT),
            "work_root": str(WORK_ROOT),
            "script": str(Path(__file__).resolve()),
            "created_unix": time.time(),
            "device": device,
            "heads": HEADS,
            "cell_type_names": CELL_TYPE_NAMES,
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "planned_specs": [spec_record(spec) for spec in specs],
        },
    )

    print(f"Repo root: {REPO_ROOT}")
    print(f"Using device: {device}")
    print(f"Output dir: {args.outdir}")
    print(f"Planned fine-tune runs: {len(specs)}")

    checkpoint = load_checkpoint_from_tar(args.artifact_path, map_location="cpu")
    data_hparams = namespace_to_dict(checkpoint["data_hparams"])
    artifact_heads = [str(head) for head in data_hparams.get("activity_columns", [])]
    if artifact_heads != HEADS:
        raise ValueError(f"Artifact head order {artifact_heads} does not match expected {HEADS}")

    lib1_df = load_lib1_wide(args.lib1_path, HEADS)
    lib1_train_df = lib1_df.loc[lib1_df["fold"] == "train"].copy()
    lib1_test_df = lib1_df.loc[lib1_df["fold"] == "test"].copy()
    if lib1_train_df.empty or lib1_test_df.empty:
        raise ValueError("Lib1 train/test splits are required for scaler and retention evaluation.")
    pretrained_scaler = TargetScaler.from_frame(lib1_train_df, HEADS, source="pretrained_lib1_train")

    lib2_wide, lib2_audit = load_lib2_wide(
        args.lib2_path,
        HEADS,
        require_all_heads=True,
    )
    split_id = current_split_id(args)
    if args.stage == "legacy_v1":
        lib2_split_df = split_by_sequence_hash(
            lib2_wide,
            split_seed=args.split_seed,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
        )
        lib2_train_df = lib2_split_df.loc[lib2_split_df["split"] == "train"].copy()
        lib2_val_df = lib2_split_df.loc[lib2_split_df["split"] == "val"].copy()
        lib2_test_df = lib2_split_df.loc[lib2_split_df["split"] == "test"].copy()
        if lib2_train_df.empty or lib2_val_df.empty or lib2_test_df.empty:
            raise ValueError("Lib2 train/val/test split is empty.")

        split_manifest_cols = [
            "seq_upper",
            "seq_original_example",
            "split",
            "sequence_len",
            "gc_fraction",
            "split_hash",
            "n_raw_rows",
            *[f"n_obs_{head}" for head in HEADS],
            *HEADS,
        ]
        lib2_split_df[split_manifest_cols].to_csv(args.outdir / "lib2_sequence_split_manifest.csv", index=False)
        split_audit_df = split_audit(lib2_split_df, HEADS)
        split_audit_df.to_csv(args.outdir / "lib2_sequence_split_audit.csv", index=False)
        lib2_eval_sets = [
            ("lib2_val", lib2_val_df),
            ("lib2_test", lib2_test_df),
        ]
        epoch_eval_sets = [
            ("train", lib2_train_df),
            ("val", lib2_val_df),
            ("test", lib2_test_df),
        ]
        use_early_stopping = True
        split_artifacts = {
            "legacy_manifest": str(args.outdir / "lib2_sequence_split_manifest.csv"),
            "legacy_split_audit": str(args.outdir / "lib2_sequence_split_audit.csv"),
            "split_seed": int(args.split_seed),
            "val_frac": float(args.val_frac),
            "test_frac": float(args.test_frac),
        }
    else:
        split_frames = phase2_v2_split_frames(args, lib2_wide, HEADS)
        lib2_split_df = None
        lib2_train_df = split_frames["train_df"]
        lib2_val_df = split_frames["val_df"]
        lib2_test_df = split_frames["test_df"]
        lib2_eval_sets = split_frames["lib2_eval_sets"]
        epoch_eval_sets = split_frames["epoch_eval_sets"]
        use_early_stopping = split_frames["use_early_stopping"]
        split_artifacts = {
            "split_manifest_dir": str(split_frames["split_manifest_dir"]),
            "outer_final_test_manifest": str(split_frames["outer_path"]),
            "inner_split_manifest": (
                str(split_frames["inner_path"])
                if args.stage in {"screening", "confirmation"}
                else None
            ),
            "split_policy": str(split_frames["split_manifest_dir"] / "split_policy.json"),
            "split_audit": str(split_frames["split_manifest_dir"] / "split_audit.csv"),
            "split_id": split_frames["split_id"],
            "outer_split_seed": int(args.outer_split_seed),
            "inner_split_seed": (
                int(args.inner_split_seed)
                if args.stage in {"screening", "confirmation"}
                else None
            ),
            "final_test_frac": float(args.final_test_frac),
            "inner_val_frac": float(args.inner_val_frac),
        }

    lib2_train_scaler = TargetScaler.from_frame(lib2_train_df, HEADS, source="lib2_train")
    write_json(
        args.outdir / "data_audit.json",
        {
            "lib2": lib2_audit,
            "stage": args.stage,
            "split_id": split_id,
            "split_artifacts": split_artifacts,
            "selected_lib2_counts": {
                "train_rows": int(len(lib2_train_df)),
                "val_rows": int(len(lib2_val_df)) if lib2_val_df is not None else 0,
                "final_test_rows": int(len(lib2_test_df)) if lib2_test_df is not None else 0,
            },
            "lib1": {
                "usable_rows": int(len(lib1_df)),
                "train_rows": int(len(lib1_train_df)),
                "test_rows": int(len(lib1_test_df)),
            },
            "pretrained_scaler": pretrained_scaler.to_dict(),
            "lib2_train_scaler": lib2_train_scaler.to_dict(),
        },
    )

    inhouse_df, inhouse_audit = load_inhouse_fiveprime(args.inhouse_path)
    write_json(args.outdir / "inhouse_fiveprime_audit.json", inhouse_audit)

    pd.DataFrame([spec_record(spec) for spec in specs]).to_csv(
        args.outdir / "planned_finetune_specs.csv",
        index=False,
    )

    if args.preview_only:
        print("Preview only: wrote manifests and planned specs without training.")
        return

    all_summary_frames = []
    all_per_head_frames = []
    all_inhouse_frames = []
    all_epoch_diagnostics_frames = []
    baseline_context = {
        "stage": args.stage,
        "outer_split_seed": int(args.outer_split_seed) if args.stage != "legacy_v1" else None,
        "inner_split_seed": (
            int(args.inner_split_seed)
            if args.stage in {"screening", "confirmation"}
            else None
        ),
        "training_seed": None,
        "split_id": split_id,
        "freeze_backbone_epochs": None,
        "weight_decay": None,
    }

    print("Evaluating pretrained 1mmy39ku baseline...")
    baseline_model = build_model_from_checkpoint(checkpoint, device=device)
    baseline_metrics = evaluate_model_everywhere(
        model=baseline_model,
        scaler=pretrained_scaler,
        heads=HEADS,
        model_label=f"pretrained_{DEFAULT_RUN_ID}",
        lib2_split_df=lib2_split_df,
        lib1_test_df=lib1_test_df,
        inhouse_df=inhouse_df,
        lib2_eval_sets=lib2_eval_sets,
        pred_batch_size=args.pred_batch_size,
        device=device,
        prediction_dir=args.outdir / "predictions",
        run_seed=None,
        unfreeze_scope=None,
        head_lr=None,
        backbone_lr=None,
        inhouse_min_barcodes=args.inhouse_min_barcodes,
        run_context=baseline_context,
    )
    all_summary_frames.append(baseline_metrics["summary"])
    all_per_head_frames.append(baseline_metrics["per_head"])
    all_inhouse_frames.append(baseline_metrics["inhouse"])
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for spec in specs:
        run_dir = args.outdir / "runs" / spec.tag()
        run_dir.mkdir(parents=True, exist_ok=True)
        existing_summary = run_dir / "model_comparison_summary.csv"
        existing_per_head = run_dir / "per_head_metrics.csv"
        existing_inhouse = run_dir / "inhouse_fiveprime_metrics.csv"
        existing_epoch_diagnostics = run_dir / "per_epoch_diagnostics.csv"
        if (
            existing_summary.exists()
            and existing_per_head.exists()
            and existing_inhouse.exists()
            and existing_epoch_diagnostics.exists()
            and not args.force
        ):
            print(f"Skipping existing run {spec.tag()}")
            all_summary_frames.append(pd.read_csv(existing_summary))
            all_per_head_frames.append(pd.read_csv(existing_per_head))
            all_inhouse_frames.append(pd.read_csv(existing_inhouse))
            all_epoch_diagnostics_frames.append(pd.read_csv(existing_epoch_diagnostics))
            continue

        print(f"Training {spec.tag()}")
        scaler = select_scaler(
            spec.target_scaler_source,
            pretrained_scaler=pretrained_scaler,
            lib2_train_scaler=lib2_train_scaler,
        )
        model, history_df, diagnostics_df, fit_info = train_one_spec(
            checkpoint=checkpoint,
            spec=spec,
            train_df=lib2_train_df,
            val_df=lib2_val_df,
            test_df=lib2_test_df,
            inhouse_df=inhouse_df,
            heads=HEADS,
            scaler=scaler,
            device=device,
            max_epochs=args.max_epochs,
            min_epochs=args.min_epochs,
            patience=args.patience,
            freeze_backbone_epochs=None,
            train_batch_size=args.train_batch_size,
            pred_batch_size=args.pred_batch_size,
            weight_decay=None,
            monitor_metric=args.monitor_metric,
            inhouse_min_barcodes=args.inhouse_min_barcodes,
            use_early_stopping=use_early_stopping,
            hani_eval_sets=epoch_eval_sets,
        )
        history_df.to_csv(run_dir / "history.csv", index=False)
        diagnostics_df.to_csv(existing_epoch_diagnostics, index=False)
        write_json(run_dir / "fit_info.json", fit_info)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_hparams": namespace_to_dict(checkpoint["model_hparams"]),
                "source_artifact_path": str(args.artifact_path),
                "source_run_id": DEFAULT_RUN_ID,
                "heads": HEADS,
                "fit_info": fit_info,
                "target_scaler": scaler.to_dict(),
                "split_context": spec.context(),
            },
            run_dir / "finetuned_model.pt",
        )

        model_label = f"finetuned_{DEFAULT_RUN_ID}__{spec.tag()}"
        run_metrics = evaluate_model_everywhere(
            model=model,
            scaler=scaler,
            heads=HEADS,
            model_label=model_label,
            lib2_split_df=lib2_split_df,
            lib1_test_df=lib1_test_df,
            inhouse_df=inhouse_df,
            lib2_eval_sets=lib2_eval_sets,
            pred_batch_size=args.pred_batch_size,
            device=device,
            prediction_dir=run_dir / "predictions",
            run_seed=spec.seed,
            unfreeze_scope=spec.unfreeze_scope,
            head_lr=spec.head_lr,
            backbone_lr=spec.backbone_lr,
            inhouse_min_barcodes=args.inhouse_min_barcodes,
            run_context=spec.context(),
        )
        write_if_not_empty(existing_summary, run_metrics["summary"])
        write_if_not_empty(existing_per_head, run_metrics["per_head"])
        write_if_not_empty(existing_inhouse, run_metrics["inhouse"])
        all_summary_frames.append(run_metrics["summary"])
        all_per_head_frames.append(run_metrics["per_head"])
        all_inhouse_frames.append(run_metrics["inhouse"])
        all_epoch_diagnostics_frames.append(diagnostics_df)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_df = combine_frames(all_summary_frames)
    per_head_df = combine_frames(all_per_head_frames)
    inhouse_metrics_df = combine_frames(all_inhouse_frames)
    epoch_diagnostics_df = combine_frames(all_epoch_diagnostics_frames)
    write_if_not_empty(args.outdir / "model_comparison_summary.csv", summary_df)
    write_if_not_empty(args.outdir / "per_head_metrics.csv", per_head_df)
    write_if_not_empty(args.outdir / "inhouse_fiveprime_metrics.csv", inhouse_metrics_df)
    write_if_not_empty(args.outdir / "per_epoch_diagnostics.csv", epoch_diagnostics_df)

    if not summary_df.empty:
        if args.stage == "legacy_v1":
            ranking_split = "lib2_test"
            ranking_name = "lib2_test_model_ranking.csv"
        elif args.stage in {"screening", "confirmation"}:
            ranking_split = "lib2_inner_val"
            ranking_name = "lib2_validation_model_ranking.csv"
        else:
            ranking_split = "lib2_final_test"
            ranking_name = "lib2_final_test_model_ranking.csv"
        ranking = summary_df.loc[summary_df["split"] == ranking_split].copy()
        ranking = ranking.sort_values(
            ["average_activity_pearson", "mean_per_head_pearson"],
            ascending=[False, False],
        )
        ranking.to_csv(args.outdir / ranking_name, index=False)

    print("Wrote:")
    for label, artifact in split_artifacts.items():
        if isinstance(artifact, str):
            print(f"  {label}: {artifact}")
    print(f"  {args.outdir / 'model_comparison_summary.csv'}")
    print(f"  {args.outdir / 'per_head_metrics.csv'}")
    print(f"  {args.outdir / 'inhouse_fiveprime_metrics.csv'}")
    print(f"  {args.outdir / 'per_epoch_diagnostics.csv'}")


if __name__ == "__main__":
    main()
