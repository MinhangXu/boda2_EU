#!/usr/bin/env python3
"""Mixed-training fine-tuning with disjoint low/medium/high test-quality sets.

This runner keeps the training set broadly mixed by barcode count, then reports
performance separately on held-out barcode-quality bins. It is meant to answer:
does the apparent model performance change when the test set itself is low,
medium, or high barcode quality?
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
FILTERED_SCRIPT_PATH = THIS_DIR / "lib1_enhancer_learning_curve_filtered_raw_ratio_split_options.py"
DEFAULT_DATA_PATH = Path(
    "/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/enhancers/"
    "L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.csv"
)
DEFAULT_MODEL_PATH = (
    Path("/home/minhang/synBio_AL/boda2_EU")
    / "tutorials"
    / "malinois_artifacts__20211113_021200__287348.tar.gz"
)
DEFAULT_OUTDIR = (
    Path("/home/minhang/synBio_AL/boda2_EU")
    / "src"
    / "finetune"
    / "learning_curve"
    / "lib1_enhancer_random_train_test_quality_cap1000_b1_b2_may2026"
)

DEFAULT_SEEDS = [23, 19, 31]
DEFAULT_SPLIT_SEED = 7
DEFAULT_TEST_QUALITY_BINS = ["test_bc_1_3", "test_bc_4_6", "test_bc_ge7"]
DEFAULT_TRAIN_SIZE_FRACS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
DEFAULT_MIN_TRAIN_SIZE = 10
DEFAULT_MAX_EPOCHS = 70
DEFAULT_PATIENCE = 10
DEFAULT_FROZEN_EPOCHS = 2
DEFAULT_TRAIN_BATCH_SIZE = 256
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_HEAD_LRS = [5e-4]
DEFAULT_BACKBONE_LRS = [1e-4]
CACHE_LAYOUT_VERSION = "filtered_raw_ratio_random_train_test_quality_v1"

filtered_runner = None


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_filtered_runner():
    global filtered_runner
    if filtered_runner is None:
        filtered_runner = _load_module("lib1_filtered_raw_ratio_runner_for_test_quality", FILTERED_SCRIPT_PATH)
    return filtered_runner


@dataclass(frozen=True)
class TestQualityBinSpec:
    name: str
    label: str
    query: str
    sort_order: int
    min_barcodes: int
    max_barcodes: int | None = None

    def mask(self, values: pd.Series) -> pd.Series:
        bc = pd.to_numeric(values, errors="coerce")
        mask = bc >= self.min_barcodes
        if self.max_barcodes is not None:
            mask &= bc <= self.max_barcodes
        return mask


TEST_QUALITY_BIN_SPECS = {
    "test_bc_eq1": TestQualityBinSpec(
        "test_bc_eq1",
        "test: 1 barcode",
        "number_of_barcodes == 1",
        5,
        min_barcodes=1,
        max_barcodes=1,
    ),
    "test_bc_1_2": TestQualityBinSpec(
        "test_bc_1_2",
        "test: 1-2 barcodes",
        "1 <= number_of_barcodes <= 2",
        8,
        min_barcodes=1,
        max_barcodes=2,
    ),
    "test_bc_1_3": TestQualityBinSpec(
        "test_bc_1_3",
        "test: 1-3 barcodes",
        "1 <= number_of_barcodes <= 3",
        10,
        min_barcodes=1,
        max_barcodes=3,
    ),
    "test_bc_3_4": TestQualityBinSpec(
        "test_bc_3_4",
        "test: 3-4 barcodes",
        "3 <= number_of_barcodes <= 4",
        15,
        min_barcodes=3,
        max_barcodes=4,
    ),
    "test_bc_4_6": TestQualityBinSpec(
        "test_bc_4_6",
        "test: 4-6 barcodes",
        "4 <= number_of_barcodes <= 6",
        20,
        min_barcodes=4,
        max_barcodes=6,
    ),
    "test_bc_5_6": TestQualityBinSpec(
        "test_bc_5_6",
        "test: 5-6 barcodes",
        "5 <= number_of_barcodes <= 6",
        25,
        min_barcodes=5,
        max_barcodes=6,
    ),
    "test_bc_7_8": TestQualityBinSpec(
        "test_bc_7_8",
        "test: 7-8 barcodes",
        "7 <= number_of_barcodes <= 8",
        28,
        min_barcodes=7,
        max_barcodes=8,
    ),
    "test_bc_ge7": TestQualityBinSpec(
        "test_bc_ge7",
        "test: >=7 barcodes",
        "number_of_barcodes >= 7",
        30,
        min_barcodes=7,
        max_barcodes=None,
    ),
    "test_bc_ge9": TestQualityBinSpec(
        "test_bc_ge9",
        "test: >=9 barcodes",
        "number_of_barcodes >= 9",
        35,
        min_barcodes=9,
        max_barcodes=None,
    ),
}

TEST_QUALITY_BIN_CHOICES = sorted(TEST_QUALITY_BIN_SPECS)


@dataclass(frozen=True)
class MixedTrainExperimentSpec:
    seed: int
    head_idx: int
    init_head: str
    setting_name: str
    train_size: int
    train_fraction: float
    unfreeze_scope: str
    train_sampling_mode: str
    head_lr: float
    backbone_lr: float

    def tag(self) -> str:
        frac = f"{self.train_fraction:.4f}".replace(".", "p")
        head_lr_tag = sanitize_tag(f"{self.head_lr:.2e}".replace(".", "p"))
        backbone_lr_tag = sanitize_tag(f"{self.backbone_lr:.2e}".replace(".", "p"))
        return (
            f"seed{self.seed}__head{self.init_head}__{self.setting_name}"
            f"__mixed_n{self.train_size}__frac{frac}"
            f"__{self.unfreeze_scope}__{self.train_sampling_mode}"
            f"__hlr{head_lr_tag}__blr{backbone_lr_tag}"
        )


def sanitize_tag(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.=-]+", "_", str(value)).strip("_")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_row_ids(df: pd.DataFrame) -> str:
    if "row_id" not in df.columns:
        return ""
    joined = "\n".join(map(str, sorted(df["row_id"].tolist())))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def make_seeded_split_seed(base_split_seed: int, run_seed: int, offset: int = 0) -> int:
    return int(base_split_seed) * 100_003 + int(run_seed) * 1_009 + int(offset)


def parse_quality_bins(values: Iterable[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        if value == "all":
            parsed.extend(DEFAULT_TEST_QUALITY_BINS)
            continue
        if value not in TEST_QUALITY_BIN_SPECS:
            valid = ", ".join([*TEST_QUALITY_BIN_CHOICES, "all"])
            raise argparse.ArgumentTypeError(f"Unknown test quality bin {value!r}. Valid choices: {valid}")
        parsed.append(value)
    out: list[str] = []
    seen: set[str] = set()
    for value in parsed:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def selected_head_indices(base_module, requested_heads: list[str] | None) -> list[tuple[int, str]]:
    all_heads = list(base_module.PRETRAINED_HEADS)
    if not requested_heads:
        return list(enumerate(all_heads))
    bad = sorted(set(requested_heads) - set(all_heads))
    if bad:
        raise ValueError(f"Unknown pretrained heads {bad}. Valid heads: {all_heads}")
    return [(idx, head) for idx, head in enumerate(all_heads) if head in requested_heads]


def barcode_count_summary(df: pd.DataFrame, barcode_column: str) -> dict[str, Any]:
    bc = pd.to_numeric(df[barcode_column], errors="coerce")
    return {
        "n_bc_eq1": int((bc == 1).sum()),
        "n_bc_1_3": int(((bc >= 1) & (bc <= 3)).sum()),
        "n_bc_4_6": int(((bc >= 4) & (bc <= 6)).sum()),
        "n_bc_ge7": int((bc >= 7).sum()),
        "barcode_min": float(bc.min()) if len(bc) else np.nan,
        "barcode_max": float(bc.max()) if len(bc) else np.nan,
        "barcode_mean": float(bc.mean()) if len(bc) else np.nan,
        "barcode_median": float(bc.median()) if len(bc) else np.nan,
    }


def select_quality_split_payload(
    clean_df: pd.DataFrame,
    quality_bins: list[str],
    barcode_column: str,
    split_seed: int,
    test_n_per_quality: int | None,
    test_frac_per_quality: float | None,
    val_min_barcodes: int,
    val_n: int | None,
    val_frac_within_pool: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(split_seed)
    used_idx: set[int] = set()
    test_parts: list[pd.DataFrame] = []
    test_counts: dict[str, Any] = {}

    for bin_name in quality_bins:
        spec = TEST_QUALITY_BIN_SPECS[bin_name]
        candidate = clean_df.loc[spec.mask(clean_df[barcode_column]) & ~clean_df.index.isin(used_idx)].copy()
        if len(candidate) < 3:
            raise ValueError(f"Not enough candidate rows for {bin_name}: {len(candidate)}")
        if test_n_per_quality is not None:
            n_test = min(int(test_n_per_quality), len(candidate) - 1)
        else:
            frac = float(test_frac_per_quality if test_frac_per_quality is not None else 0.10)
            n_test = max(1, int(round(len(candidate) * frac)))
            n_test = min(n_test, len(candidate) - 1)
        sampled_idx = rng.choice(candidate.index.to_numpy(), size=n_test, replace=False)
        used_idx.update(int(idx) for idx in sampled_idx)
        part = candidate.loc[sampled_idx].copy()
        part["test_quality_bin"] = spec.name
        part["test_quality_bin_label"] = spec.label
        part["test_quality_bin_query"] = spec.query
        part["test_quality_bin_sort_order"] = spec.sort_order
        test_parts.append(part)
        test_counts[f"{bin_name}_candidate_size"] = int(len(candidate))
        test_counts[f"{bin_name}_test_size"] = int(len(part))

    val_candidate = clean_df.loc[
        (pd.to_numeric(clean_df[barcode_column], errors="coerce") >= int(val_min_barcodes))
        & ~clean_df.index.isin(used_idx)
    ].copy()
    if len(val_candidate) < 3:
        raise ValueError(f"Not enough validation candidates after test selection: {len(val_candidate)}")
    if val_n is not None:
        n_val = min(int(val_n), len(val_candidate) - 1)
    else:
        n_val = max(1, int(round(len(val_candidate) * float(val_frac_within_pool))))
        n_val = min(n_val, len(val_candidate) - 1)
    val_idx = rng.choice(val_candidate.index.to_numpy(), size=n_val, replace=False)
    used_idx.update(int(idx) for idx in val_idx)

    val_df = val_candidate.loc[val_idx].copy()
    test_df = pd.concat(test_parts, ignore_index=True)
    train_rest_df = clean_df.loc[~clean_df.index.isin(used_idx)].copy()

    return {
        "split_seed": int(split_seed),
        "split_strategy": "random_mixed_train_disjoint_test_quality_per_seed",
        "split_pool": "test_quality_bins_then_hq_validation",
        "split_seed_effective": int(split_seed),
        "val_seed_effective": int(split_seed),
        "test_seed_effective": int(split_seed),
        "test_is_fixed_across_seeds": False,
        "val_is_fixed_across_seeds": False,
        "train_rest_df": train_rest_df.reset_index(drop=True),
        "val_df": val_df.reset_index(drop=True),
        "test_df": test_df.reset_index(drop=True),
        "test_quality_bins": list(quality_bins),
        "test_counts": test_counts,
    }


def build_velocity_segments(runs_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "val_pearson",
        "val_r2",
        "val_pearson_sq",
        "test_pearson",
        "test_r2",
        "test_pearson_sq",
    ]
    available_metrics = [metric for metric in metrics if metric in runs_df.columns]
    if not available_metrics:
        return pd.DataFrame()

    curve_group_cols = [
        "split_strategy",
        "split_seed_base",
        "split_seed_effective",
        "seed",
        "test_quality_bin",
        "test_quality_bin_label",
        "test_quality_bin_query",
        "train_sampling_mode",
        "setting",
        "b_cap",
        "head_lr",
        "backbone_lr",
        "unfreeze_scope",
        "init_head",
    ]
    curve_group_cols = [col for col in curve_group_cols if col in runs_df.columns]
    records: list[dict[str, Any]] = []

    for group_key, group in runs_df.groupby(curve_group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        key_payload = dict(zip(curve_group_cols, group_key))
        ordered = group.sort_values(["train_size", "train_fraction"]).drop_duplicates(
            subset=["train_size"],
            keep="last",
        )
        if len(ordered) < 2:
            continue
        rows = ordered.to_dict("records")
        for segment_idx, (left, right) in enumerate(zip(rows[:-1], rows[1:]), start=1):
            delta_n = int(right["train_size"]) - int(left["train_size"])
            if delta_n <= 0:
                continue
            for metric in available_metrics:
                y0 = float(left[metric]) if pd.notna(left[metric]) else np.nan
                y1 = float(right[metric]) if pd.notna(right[metric]) else np.nan
                delta_metric = y1 - y0 if np.isfinite(y0) and np.isfinite(y1) else np.nan
                slope = delta_metric / delta_n if np.isfinite(delta_metric) else np.nan
                records.append(
                    {
                        **key_payload,
                        "segment_index": segment_idx,
                        "metric": metric,
                        "train_size_start": int(left["train_size"]),
                        "train_size_end": int(right["train_size"]),
                        "train_fraction_start": float(left["train_fraction"]),
                        "train_fraction_end": float(right["train_fraction"]),
                        "delta_train_size": delta_n,
                        "metric_start": y0,
                        "metric_end": y1,
                        "delta_metric": delta_metric,
                        "slope_per_construct": slope,
                        "slope_per_100_constructs": slope * 100.0 if np.isfinite(slope) else np.nan,
                    }
                )
    return pd.DataFrame(records)


def aggregate_if_not_empty(base_module, frame: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    group_cols = [col for col in group_cols if col in frame.columns]
    metrics = [metric for metric in metrics if metric in frame.columns]
    if not group_cols or not metrics:
        return pd.DataFrame()
    return base_module.aggregate_metric_summary(frame, group_cols=group_cols, metrics=metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--test_quality_bins", nargs="+", default=DEFAULT_TEST_QUALITY_BINS)
    parser.add_argument("--test_n_per_quality", type=int, default=250)
    parser.add_argument("--test_frac_per_quality", type=float, default=None)
    parser.add_argument("--val_min_barcodes", type=int, default=8)
    parser.add_argument("--val_n", type=int, default=250)
    parser.add_argument("--val_frac_within_pool", type=float, default=0.10)
    parser.add_argument("--train_pool_cap", type=int, default=1000)
    parser.add_argument("--train_pool_cap_seed", type=int, default=104729)
    parser.add_argument("--train_size_fracs", nargs="*", type=float, default=DEFAULT_TRAIN_SIZE_FRACS)
    parser.add_argument("--train_sampling_mode", choices=["random", "hq_first"], default="random")
    parser.add_argument("--min_train_size", type=int, default=DEFAULT_MIN_TRAIN_SIZE)
    parser.add_argument(
        "--unfreeze_scopes",
        nargs="+",
        type=str,
        default=["branched_only"],
        choices=["head_only", "branched_only", "linear_all_head", "conv3_plus", "full"],
    )
    parser.add_argument("--pretrained_heads", nargs="+", default=["K562"], help="Optional subset of K562 HepG2 SKNSH.")
    parser.add_argument("--include_b1", action="store_true")
    parser.add_argument("--include_b2", action="store_true")
    parser.add_argument("--include_b3", action="store_true")
    parser.add_argument("--b3_bcaps", nargs="+", type=float, default=[10.0])
    parser.add_argument("--min_weight", type=float, default=0.1)
    parser.add_argument("--head_lrs", nargs="+", type=float, default=DEFAULT_HEAD_LRS)
    parser.add_argument("--backbone_lrs", nargs="+", type=float, default=DEFAULT_BACKBONE_LRS)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--frozen_epochs", type=int, default=DEFAULT_FROZEN_EPOCHS)
    parser.add_argument("--train_batch_size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--preview_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    overall_start_time = time.time()
    args = parse_args()

    filtered = get_filtered_runner()
    base = filtered.get_base()
    filtered.configure_base_for_filtered_ratio(base)

    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)
    if not args.model_path.exists() and not args.preview_only:
        raise FileNotFoundError(args.model_path)

    args.test_quality_bins = parse_quality_bins(args.test_quality_bins)
    if not any([args.include_b1, args.include_b2, args.include_b3]):
        args.include_b1 = True
        args.include_b2 = True

    settings = base.build_settings(
        args.include_b1,
        args.include_b2,
        args.include_b3,
        b3_bcaps=list(args.b3_bcaps),
        min_weight=args.min_weight,
    )
    selected_heads = selected_head_indices(base, args.pretrained_heads)
    head_lr_grid = [float(value) for value in args.head_lrs]
    backbone_lr_grid = [float(value) for value in args.backbone_lrs]

    device = base.resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_cache_dir = cache_dir / "splits"
    split_cache_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_summary = filtered.collect_preprocessing_summary(args.data_path)
    clean_df = base.load_clean_df(args.data_path)
    dataset_barcode_counts = barcode_count_summary(clean_df, base.BARCODE_COLUMN)

    run_started_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"

    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest["repo_root"] = str(base.REPO_ROOT)
    manifest["device_resolved"] = device
    manifest["run_id"] = run_id
    manifest["run_started_at"] = run_started_at
    manifest["settings"] = [asdict(x) for x in settings]
    manifest["selected_pretrained_heads"] = [head for _, head in selected_heads]
    manifest["cache_layout_version"] = CACHE_LAYOUT_VERSION
    manifest["data_sha256"] = sha256_file(args.data_path)
    manifest["dataset_sequence_column"] = filtered.DATASET_SEQUENCE_COLUMN
    manifest["dataset_barcode_column"] = filtered.DATASET_BARCODE_COLUMN
    manifest["dataset_raw_ratio_column"] = filtered.DATASET_RAW_RATIO_COLUMN
    manifest["modeled_target_column"] = filtered.MODELING_TARGET_COLUMN
    manifest["modeled_target_transform"] = "log10(RNA_bc_counts_sum / DNA_bc_counts_sum)"
    manifest["target_standardization"] = "train_only_zscore_after_split"
    manifest["test_quality_bin_specs"] = {name: asdict(spec) for name, spec in TEST_QUALITY_BIN_SPECS.items()}
    manifest["dataset_barcode_counts"] = dataset_barcode_counts
    manifest["preprocessing_summary"] = preprocessing_summary
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Repo root: {base.REPO_ROOT}")
    print(f"Using device: {device}")
    print(f"Output dir: {args.outdir}")
    print(f"Test quality bins: {args.test_quality_bins}")
    print(f"Validation min barcodes: {args.val_min_barcodes}")
    print(f"Train pool cap: {args.train_pool_cap}")
    print(f"Settings: {[setting.name for setting in settings]}")
    print(f"Pretrained heads: {[head for _, head in selected_heads]}")
    print(f"Unfreeze scopes: {args.unfreeze_scopes}")
    print(f"Train size fractions: {args.train_size_fracs}")

    checkpoint = None if args.preview_only else base.load_checkpoint_from_tar(args.model_path, map_location="cpu")

    run_records: list[dict[str, Any]] = []
    history_records: list[pd.DataFrame] = []
    zero_shot_records: list[pd.DataFrame] = []
    preview_records: list[dict[str, Any]] = []

    for seed in args.seeds:
        split_seed_effective = make_seeded_split_seed(args.split_seed, seed, offset=83)
        bins_tag = "-".join(args.test_quality_bins)
        test_tag = f"testn{args.test_n_per_quality}" if args.test_n_per_quality is not None else f"testfrac{args.test_frac_per_quality}"
        val_tag = f"valmin{args.val_min_barcodes}__valn{args.val_n}" if args.val_n is not None else f"valfrac{args.val_frac_within_pool}"
        split_cache_path = split_cache_dir / (
            f"random_mixed_test_quality__{sanitize_tag(bins_tag)}__{test_tag}__{val_tag}"
            f"__seed{seed}__splitseed{args.split_seed}.pkl"
        )
        split_payload = base.maybe_load_or_build(
            split_cache_path,
            force=args.force,
            builder=lambda split_seed_effective=split_seed_effective: select_quality_split_payload(
                clean_df=clean_df,
                quality_bins=args.test_quality_bins,
                barcode_column=base.BARCODE_COLUMN,
                split_seed=split_seed_effective,
                test_n_per_quality=args.test_n_per_quality,
                test_frac_per_quality=args.test_frac_per_quality,
                val_min_barcodes=args.val_min_barcodes,
                val_n=args.val_n,
                val_frac_within_pool=args.val_frac_within_pool,
            ),
        )

        train_rest_df = split_payload["train_rest_df"]
        val_df_raw = split_payload["val_df"]
        test_df_raw = split_payload["test_df"]
        split_counts = {
            "train_rest_size": int(len(train_rest_df)),
            "val_size_raw": int(len(val_df_raw)),
            "test_size_raw": int(len(test_df_raw)),
            **{f"train_rest_{k}": v for k, v in barcode_count_summary(train_rest_df, base.BARCODE_COLUMN).items()},
            **{f"val_{k}": v for k, v in barcode_count_summary(val_df_raw, base.BARCODE_COLUMN).items()},
            **{f"test_{k}": v for k, v in barcode_count_summary(test_df_raw, base.BARCODE_COLUMN).items()},
            **split_payload["test_counts"],
        }

        print(
            f"Seed {seed}: train_rest={len(train_rest_df)}, val={len(val_df_raw)}, "
            f"test={len(test_df_raw)}, split_seed_effective={split_payload['split_seed_effective']}"
        )

        if not args.preview_only:
            for bin_name in args.test_quality_bins:
                test_bin_raw = test_df_raw.loc[test_df_raw["test_quality_bin"] == bin_name].copy()
                zdf = base.run_zero_shot_eval_on_fixed_test(
                    checkpoint,
                    base.add_padded_sequences(test_bin_raw),
                    device=device,
                )
                spec = TEST_QUALITY_BIN_SPECS[bin_name]
                zdf["run_id"] = run_id
                zdf["seed"] = seed
                zdf["split_strategy"] = split_payload["split_strategy"]
                zdf["split_seed_effective"] = split_payload["split_seed_effective"]
                zdf["val_seed_effective"] = split_payload["val_seed_effective"]
                zdf["test_seed_effective"] = split_payload["test_seed_effective"]
                zdf["test_quality_bin"] = spec.name
                zdf["test_quality_bin_label"] = spec.label
                zdf["test_quality_bin_query"] = spec.query
                zdf["test_quality_bin_sort_order"] = spec.sort_order
                zero_shot_records.append(zdf)

        eligible_uncapped = train_rest_df.copy().reset_index(drop=True)
        if args.train_pool_cap > 0 and len(eligible_uncapped) > args.train_pool_cap:
            cap_seed = int(args.train_pool_cap_seed) + int(seed) * 1_009
            eligible = (
                eligible_uncapped.sample(n=args.train_pool_cap, random_state=cap_seed, replace=False)
                .sort_values("row_id")
                .reset_index(drop=True)
            )
        else:
            eligible = eligible_uncapped

        pool_components = {
            "eligible": eligible,
            "leftover_hq": eligible.loc[eligible[base.BARCODE_COLUMN] >= args.val_min_barcodes].copy().reset_index(drop=True),
            "lower_quality": eligible.loc[eligible[base.BARCODE_COLUMN] < args.val_min_barcodes].copy().reset_index(drop=True),
            "test_min_barcodes": int(args.val_min_barcodes),
        }
        size_grid = base.make_train_size_grid(
            n_available=len(eligible),
            min_train_size=args.min_train_size,
            train_size_fracs=list(args.train_size_fracs) if args.train_size_fracs else None,
        )
        print(f"Seed {seed}: mixed eligible={len(eligible)} from uncapped={len(eligible_uncapped)}, sizes={size_grid}")

        preview_records.append(
            {
                "seed": seed,
                "split_strategy": split_payload["split_strategy"],
                "split_seed_effective": split_payload["split_seed_effective"],
                "train_pool_eligible_size": len(eligible),
                "train_pool_uncapped_size": len(eligible_uncapped),
                "train_pool_cap": int(args.train_pool_cap) if args.train_pool_cap > 0 else np.nan,
                "train_size_grid": " ".join(map(str, size_grid)),
                "test_quality_bins": " ".join(args.test_quality_bins),
                "val_min_barcodes": int(args.val_min_barcodes),
                "test_n_per_quality": int(args.test_n_per_quality) if args.test_n_per_quality is not None else np.nan,
                **split_counts,
            }
        )

        if args.preview_only:
            continue

        for train_size in size_grid:
            train_raw = base.build_train_pool(
                train_pool_components=pool_components,
                train_size=train_size,
                subsample_seed=seed,
                sampling_mode=args.train_sampling_mode,
            )
            train_df, val_df, test_df, scaler = base.prepare_train_val_test_for_run(
                train_raw,
                val_df_raw,
                test_df_raw,
            )
            train_counts = barcode_count_summary(train_raw, base.BARCODE_COLUMN)
            split_hashes = {
                "train_row_id_hash": hash_row_ids(train_df),
                "val_row_id_hash": hash_row_ids(val_df),
                "test_row_id_hash": hash_row_ids(test_df),
            }
            scaler_mean = float(scaler.mean)
            scaler_std = float(scaler.std)

            for setting in settings:
                for head_idx, head_name in selected_heads:
                    for unfreeze_scope in args.unfreeze_scopes:
                        for head_lr in head_lr_grid:
                            for backbone_lr in backbone_lr_grid:
                                spec = MixedTrainExperimentSpec(
                                    seed=seed,
                                    head_idx=head_idx,
                                    init_head=head_name,
                                    setting_name=setting.name,
                                    train_size=len(train_df),
                                    train_fraction=len(train_df) / len(eligible),
                                    unfreeze_scope=unfreeze_scope,
                                    train_sampling_mode=args.train_sampling_mode,
                                    head_lr=float(head_lr),
                                    backbone_lr=float(backbone_lr),
                                )
                                cache_path = (
                                    cache_dir
                                    / "runs"
                                    / CACHE_LAYOUT_VERSION
                                    / sanitize_tag(bins_tag)
                                    / f"{test_tag}__{sanitize_tag(val_tag)}"
                                    / f"split{split_payload['split_seed_effective']}"
                                    / f"{spec.tag()}.pkl"
                                )

                                def _builder(
                                    spec=spec,
                                    setting=setting,
                                    train_df=train_df,
                                    val_df=val_df,
                                    test_df=test_df,
                                    scaler=scaler,
                                ):
                                    training_seed = spec.seed * 1_000_003 + spec.head_idx * 101
                                    model, history_df, fit_info = base.train_single_head_model(
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
                                        test_df=test_df,
                                        log_test_metrics_per_epoch=True,
                                        log_train_metrics_per_epoch=True,
                                    )
                                    val_metrics, _ = base.evaluate_single_head_model(model, val_df, scaler, device=device)
                                    train_metrics, _ = base.evaluate_single_head_model(model, train_df, scaler, device=device)
                                    combined_test_metrics, _ = base.evaluate_single_head_model(
                                        model,
                                        test_df,
                                        scaler,
                                        device=device,
                                    )
                                    quality_metrics = {}
                                    for q_name, q_df in test_df.groupby("test_quality_bin", dropna=False):
                                        metrics, pred_df = base.evaluate_single_head_model(
                                            model,
                                            q_df.copy(),
                                            scaler,
                                            device=device,
                                        )
                                        quality_metrics[str(q_name)] = {
                                            "metrics": metrics,
                                            "pred_df": pred_df,
                                        }
                                    return {
                                        "fit_info": fit_info,
                                        "history_df": history_df,
                                        "train_metrics": train_metrics,
                                        "val_metrics": val_metrics,
                                        "combined_test_metrics": combined_test_metrics,
                                        "quality_metrics": quality_metrics,
                                    }

                                payload = base.maybe_load_or_build(cache_path, force=args.force, builder=_builder)
                                fit_info = payload["fit_info"]
                                train_m = payload.get("train_metrics")

                                common_row = {
                                    "run_id": run_id,
                                    "seed": spec.seed,
                                    "split_strategy": split_payload["split_strategy"],
                                    "split_pool": split_payload["split_pool"],
                                    "split_seed_base": args.split_seed,
                                    "split_seed_effective": split_payload["split_seed_effective"],
                                    "val_seed_effective": split_payload["val_seed_effective"],
                                    "test_seed_effective": split_payload["test_seed_effective"],
                                    "val_is_fixed_across_seeds": split_payload["val_is_fixed_across_seeds"],
                                    "test_is_fixed_across_seeds": split_payload["test_is_fixed_across_seeds"],
                                    "split_val_fraction": np.nan,
                                    "split_test_fraction": np.nan,
                                    "val_min_barcodes": int(args.val_min_barcodes),
                                    "test_n_per_quality": (
                                        int(args.test_n_per_quality)
                                        if args.test_n_per_quality is not None
                                        else np.nan
                                    ),
                                    "target_column": filtered.MODELING_TARGET_COLUMN,
                                    "target_transform": "log10_recomputed_rna_over_dna",
                                    "target_scaler_mean": scaler_mean,
                                    "target_scaler_std": scaler_std,
                                    **split_hashes,
                                    "init_head": spec.init_head,
                                    "head_idx": spec.head_idx,
                                    "setting": spec.setting_name,
                                    "use_rc_augmentation": setting.use_rc_augmentation,
                                    "use_barcode_weighting": setting.use_barcode_weighting,
                                    "b_cap": (
                                        float(setting.b_cap)
                                        if setting.use_barcode_weighting and setting.b_cap is not None
                                        else np.nan
                                    ),
                                    "min_weight": float(setting.min_weight) if setting.use_barcode_weighting else np.nan,
                                    "train_barcode_bin": "mixed_all",
                                    "train_barcode_bin_label": "mixed barcode counts",
                                    "train_barcode_bin_query": "all remaining training rows after quality heldouts",
                                    "train_threshold": np.nan,
                                    "train_size": spec.train_size,
                                    "train_fraction": spec.train_fraction,
                                    "unfreeze_scope": spec.unfreeze_scope,
                                    "train_sampling_mode": spec.train_sampling_mode,
                                    "train_pool_eligible_size": len(eligible),
                                    "train_pool_uncapped_size": len(eligible_uncapped),
                                    "train_pool_cap": int(args.train_pool_cap) if args.train_pool_cap > 0 else np.nan,
                                    "val_size": len(val_df),
                                    "test_size": len(test_df),
                                    "head_lr": spec.head_lr,
                                    "backbone_lr": spec.backbone_lr,
                                    "weight_decay": args.weight_decay,
                                    **{f"train_{k}": v for k, v in train_counts.items()},
                                    **split_counts,
                                    **(
                                        {f"train_{k}": v for k, v in train_m.items()}
                                        if train_m is not None
                                        else {}
                                    ),
                                    **{f"val_{k}": v for k, v in payload["val_metrics"].items()},
                                    **{f"combined_test_{k}": v for k, v in payload["combined_test_metrics"].items()},
                                    **fit_info,
                                }

                                for bin_name in args.test_quality_bins:
                                    q_payload = payload["quality_metrics"][bin_name]
                                    q_metrics = q_payload["metrics"]
                                    q_spec = TEST_QUALITY_BIN_SPECS[bin_name]
                                    run_records.append(
                                        {
                                            **common_row,
                                            "test_quality_bin": q_spec.name,
                                            "test_quality_bin_label": q_spec.label,
                                            "test_quality_bin_query": q_spec.query,
                                            "test_quality_bin_sort_order": q_spec.sort_order,
                                            **{f"test_{k}": v for k, v in q_metrics.items()},
                                        }
                                    )

                                hist = payload["history_df"].copy()
                                hist["run_id"] = run_id
                                hist["seed"] = spec.seed
                                hist["split_strategy"] = split_payload["split_strategy"]
                                hist["split_pool"] = split_payload["split_pool"]
                                hist["split_seed_effective"] = split_payload["split_seed_effective"]
                                hist["val_seed_effective"] = split_payload["val_seed_effective"]
                                hist["test_seed_effective"] = split_payload["test_seed_effective"]
                                hist["val_is_fixed_across_seeds"] = split_payload["val_is_fixed_across_seeds"]
                                hist["test_is_fixed_across_seeds"] = split_payload["test_is_fixed_across_seeds"]
                                hist["val_min_barcodes"] = int(args.val_min_barcodes)
                                hist["test_n_per_quality"] = (
                                    int(args.test_n_per_quality) if args.test_n_per_quality is not None else np.nan
                                )
                                hist["train_row_id_hash"] = split_hashes["train_row_id_hash"]
                                hist["val_row_id_hash"] = split_hashes["val_row_id_hash"]
                                hist["test_row_id_hash"] = split_hashes["test_row_id_hash"]
                                hist["target_column"] = filtered.MODELING_TARGET_COLUMN
                                hist["target_transform"] = "log10_recomputed_rna_over_dna"
                                hist["target_scaler_mean"] = scaler_mean
                                hist["target_scaler_std"] = scaler_std
                                hist["init_head"] = spec.init_head
                                hist["head_idx"] = spec.head_idx
                                hist["setting"] = spec.setting_name
                                hist["use_rc_augmentation"] = setting.use_rc_augmentation
                                hist["use_barcode_weighting"] = setting.use_barcode_weighting
                                hist["b_cap"] = (
                                    float(setting.b_cap)
                                    if setting.use_barcode_weighting and setting.b_cap is not None
                                    else np.nan
                                )
                                hist["min_weight"] = float(setting.min_weight) if setting.use_barcode_weighting else np.nan
                                hist["train_barcode_bin"] = "mixed_all"
                                hist["train_barcode_bin_label"] = "mixed barcode counts"
                                hist["train_size"] = spec.train_size
                                hist["train_fraction"] = spec.train_fraction
                                hist["unfreeze_scope"] = spec.unfreeze_scope
                                hist["train_sampling_mode"] = spec.train_sampling_mode
                                hist["head_lr"] = spec.head_lr
                                hist["backbone_lr"] = spec.backbone_lr
                                history_records.append(hist)

    preview_df = pd.DataFrame(preview_records)
    if not preview_df.empty:
        preview_df.to_csv(args.outdir / "quality_split_planned_grid.csv", index=False)

    if args.preview_only:
        print("\nPreview only. Wrote:")
        print(f"  {args.outdir / 'quality_split_planned_grid.csv'}")
        print(f"  {args.outdir / 'run_manifest.json'}")
        elapsed_seconds = time.time() - overall_start_time
        print(f"\nTotal runtime: {elapsed_seconds:.1f} seconds ({elapsed_seconds / 60.0:.2f} minutes)")
        return

    if len(run_records) == 0:
        raise RuntimeError("No runs were executed. Check train sizes/settings.")

    runs_df = pd.DataFrame(run_records).sort_values(
        [
            "test_quality_bin_sort_order",
            "split_strategy",
            "setting",
            "b_cap",
            "head_lr",
            "backbone_lr",
            "train_sampling_mode",
            "unfreeze_scope",
            "train_size",
            "init_head",
            "seed",
        ]
    ).reset_index(drop=True)
    history_df = pd.concat(history_records, ignore_index=True)

    metric_cols = [
        "train_mae",
        "train_rmse",
        "train_pearson",
        "train_spearman",
        "train_r2",
        "train_r2_cod",
        "train_pearson_sq",
        "train_loss_standardized",
        "val_mae",
        "val_rmse",
        "val_pearson",
        "val_spearman",
        "val_r2",
        "val_r2_cod",
        "val_pearson_sq",
        "val_loss_standardized",
        "test_mae",
        "test_rmse",
        "test_pearson",
        "test_spearman",
        "test_r2",
        "test_r2_cod",
        "test_pearson_sq",
        "test_loss_standardized",
        "best_epoch",
        "best_val_loss_standardized",
        "initial_trainable_params",
        "final_trainable_params",
    ]

    group_cols = [
        "test_quality_bin",
        "test_quality_bin_label",
        "test_quality_bin_query",
        "test_quality_bin_sort_order",
        "split_strategy",
        "split_pool",
        "val_min_barcodes",
        "test_n_per_quality",
        "train_barcode_bin",
        "train_barcode_bin_label",
        "train_barcode_bin_query",
        "setting",
        "b_cap",
        "head_lr",
        "backbone_lr",
        "train_sampling_mode",
        "unfreeze_scope",
        "train_size",
        "init_head",
    ]
    aggregate_df = aggregate_if_not_empty(base, runs_df, group_cols=group_cols, metrics=metric_cols)

    full_fraction_df = runs_df.loc[runs_df["train_size"] == runs_df["train_pool_eligible_size"]].copy()
    full_fraction_summary_df = aggregate_if_not_empty(
        base,
        full_fraction_df,
        group_cols=[col for col in group_cols if col != "train_size"],
        metrics=[
            "val_pearson",
            "val_r2",
            "test_pearson",
            "test_r2",
            "test_pearson_sq",
            "best_epoch",
            "train_pool_eligible_size",
        ],
    )

    velocity_df = build_velocity_segments(runs_df)
    velocity_summary_df = aggregate_if_not_empty(
        base,
        velocity_df,
        group_cols=[
            "test_quality_bin",
            "test_quality_bin_label",
            "test_quality_bin_query",
            "train_sampling_mode",
            "setting",
            "b_cap",
            "head_lr",
            "backbone_lr",
            "unfreeze_scope",
            "metric",
        ],
        metrics=["delta_metric", "slope_per_construct", "slope_per_100_constructs"],
    )

    if zero_shot_records:
        pd.concat(zero_shot_records, ignore_index=True).to_csv(args.outdir / "zero_shot_evaluations.csv", index=False)
    runs_df.to_csv(args.outdir / "learning_curve_runs.csv", index=False)
    history_df.to_csv(args.outdir / "learning_curve_histories.csv", index=False)
    aggregate_df.to_csv(args.outdir / "learning_curve_summary_mean_std.csv", index=False)
    full_fraction_summary_df.to_csv(args.outdir / "full_fraction_summary_mean_std.csv", index=False)
    velocity_df.to_csv(args.outdir / "learning_curve_velocity_segments.csv", index=False)
    velocity_summary_df.to_csv(args.outdir / "learning_curve_velocity_summary_mean_std.csv", index=False)

    print("\nWrote outputs:")
    for path in [
        args.outdir / "quality_split_planned_grid.csv",
        args.outdir / "zero_shot_evaluations.csv",
        args.outdir / "learning_curve_runs.csv",
        args.outdir / "learning_curve_histories.csv",
        args.outdir / "learning_curve_summary_mean_std.csv",
        args.outdir / "full_fraction_summary_mean_std.csv",
        args.outdir / "learning_curve_velocity_segments.csv",
        args.outdir / "learning_curve_velocity_summary_mean_std.csv",
        args.outdir / "run_manifest.json",
    ]:
        if path.exists():
            print(f"  {path}")

    elapsed_seconds = time.time() - overall_start_time
    print(f"\nTotal runtime: {elapsed_seconds:.1f} seconds ({elapsed_seconds / 60.0:.2f} minutes)")


if __name__ == "__main__":
    main()
