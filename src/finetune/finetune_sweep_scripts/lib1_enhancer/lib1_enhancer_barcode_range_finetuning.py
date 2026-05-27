#!/usr/bin/env python3
"""Barcode-range learning-curve fine-tuning for filtered lib1 enhancer data.

This runner is intentionally narrow: it reuses the filtered raw-ratio
fine-tuning stack, but swaps cumulative training thresholds for named barcode
training bins and repeats high-barcode validation/test holdouts across seeds.

Default run shape matches the Stage 1 barcode-range diagnostic:
  heldout_min_barcodes: 4 and 10
  seeds: 3
  train_barcode_bins: bc_eq1, bc_2_3, bc_4_10, bc_gt10, bc_ge4
  train_size_fracs: 0.25, 0.50, 0.75, 1.0
  settings: B2_with_RC
  unfreeze_scopes: branched_only and full
  LR: head_lr=5e-4, backbone_lr=1e-4

It writes the normal run/history/summary CSVs plus finite-difference
learning-velocity summaries by barcode bin.
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
    / "lib1_enhancer_barcode_range_stage1_apr2026"
)

DEFAULT_SEEDS = [23, 19, 31]
DEFAULT_SPLIT_SEED = 7
DEFAULT_HELDOUT_MIN_BARCODES = [4, 10]
DEFAULT_TRAIN_BARCODE_BINS = ["bc_eq1", "bc_2_3", "bc_4_10", "bc_gt10", "bc_ge4"]
DEFAULT_TRAIN_SIZE_FRACS = [0.25, 0.50, 0.75, 1.0]
DEFAULT_MIN_TRAIN_SIZE = 32
DEFAULT_MAX_EPOCHS = 70
DEFAULT_PATIENCE = 10
DEFAULT_FROZEN_EPOCHS = 2
DEFAULT_TRAIN_BATCH_SIZE = 256
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_HEAD_LRS = [5e-4]
DEFAULT_BACKBONE_LRS = [1e-4]

CACHE_LAYOUT_VERSION = "filtered_raw_ratio_exact_barcode_bins_per_epoch_metrics_v1"

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
        filtered_runner = _load_module("lib1_filtered_raw_ratio_runner_for_barcode_bins", FILTERED_SCRIPT_PATH)
    return filtered_runner


@dataclass(frozen=True)
class BarcodeBinSpec:
    name: str
    label: str
    query: str
    sort_order: int

    def mask(self, values: pd.Series) -> pd.Series:
        bc = pd.to_numeric(values, errors="coerce")
        if self.name == "bc_eq1":
            return bc == 1
        if self.name == "bc_2_3":
            return (bc >= 2) & (bc <= 3)
        if self.name == "bc_4_10":
            return (bc >= 4) & (bc <= 10)
        if self.name == "bc_gt10":
            return bc > 10
        if self.name == "bc_ge4":
            return bc >= 4
        raise ValueError(f"Unknown barcode bin: {self.name}")


BARCODE_BIN_SPECS: dict[str, BarcodeBinSpec] = {
    "bc_eq1": BarcodeBinSpec("bc_eq1", "barcode == 1", "number_of_barcodes == 1", 10),
    "bc_2_3": BarcodeBinSpec("bc_2_3", "2 <= barcode <= 3", "2 <= number_of_barcodes <= 3", 20),
    "bc_4_10": BarcodeBinSpec("bc_4_10", "4 <= barcode <= 10", "4 <= number_of_barcodes <= 10", 30),
    "bc_gt10": BarcodeBinSpec("bc_gt10", "barcode > 10", "number_of_barcodes > 10", 40),
    "bc_ge4": BarcodeBinSpec("bc_ge4", "barcode >= 4", "number_of_barcodes >= 4", 50),
}

BARCODE_BIN_CHOICES = sorted(BARCODE_BIN_SPECS)


@dataclass(frozen=True)
class BarcodeRangeExperimentSpec:
    seed: int
    heldout_min_barcodes: int
    head_idx: int
    init_head: str
    setting_name: str
    train_barcode_bin: str
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
            f"seed{self.seed}__heldoutbc{self.heldout_min_barcodes}"
            f"__head{self.init_head}__{self.setting_name}"
            f"__bin{self.train_barcode_bin}__n{self.train_size}__frac{frac}"
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


def parse_train_bins(values: Iterable[str]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        if value == "all":
            parsed.extend(DEFAULT_TRAIN_BARCODE_BINS)
            continue
        if value not in BARCODE_BIN_SPECS:
            valid = ", ".join([*BARCODE_BIN_CHOICES, "all"])
            raise argparse.ArgumentTypeError(f"Unknown train barcode bin {value!r}. Valid choices: {valid}")
        parsed.append(value)
    out: list[str] = []
    seen: set[str] = set()
    for value in parsed:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def build_barcode_bin_train_pool_components(
    train_rest_df: pd.DataFrame,
    train_bin: BarcodeBinSpec,
    barcode_column: str,
    heldout_min_barcodes: int,
) -> dict[str, Any]:
    eligible = train_rest_df.loc[train_bin.mask(train_rest_df[barcode_column])].copy().reset_index(drop=True)
    if len(eligible) == 0:
        raise ValueError(f"No train rows available for barcode bin {train_bin.name}.")

    leftover_hq = eligible.loc[eligible[barcode_column] >= heldout_min_barcodes].copy().reset_index(drop=True)
    lower_quality = eligible.loc[eligible[barcode_column] < heldout_min_barcodes].copy().reset_index(drop=True)
    return {
        "eligible": eligible,
        "leftover_hq": leftover_hq,
        "lower_quality": lower_quality,
        "test_min_barcodes": int(heldout_min_barcodes),
        "train_barcode_bin": train_bin.name,
        "train_barcode_bin_label": train_bin.label,
        "train_barcode_bin_query": train_bin.query,
    }


def barcode_count_summary(df: pd.DataFrame, barcode_column: str) -> dict[str, int]:
    bc = pd.to_numeric(df[barcode_column], errors="coerce")
    return {
        "n_bc_eq1": int((bc == 1).sum()),
        "n_bc_2_3": int(((bc >= 2) & (bc <= 3)).sum()),
        "n_bc_4_10": int(((bc >= 4) & (bc <= 10)).sum()),
        "n_bc_gt10": int((bc > 10).sum()),
        "n_bc_ge4": int((bc >= 4).sum()),
        "n_bc_ge10": int((bc >= 10).sum()),
    }


def compute_train_bin_counts(
    train_raw: pd.DataFrame,
    barcode_column: str,
    heldout_min_barcodes: int,
) -> dict[str, Any]:
    bc = pd.to_numeric(train_raw[barcode_column], errors="coerce")
    counts = barcode_count_summary(train_raw, barcode_column)
    hq_count = int((bc >= heldout_min_barcodes).sum())
    counts.update(
        {
            "train_hq_count": hq_count,
            "train_lower_quality_count": int(len(train_raw) - hq_count),
            "train_hq_fraction": hq_count / max(len(train_raw), 1),
            "train_barcode_min": float(bc.min()) if len(bc) else np.nan,
            "train_barcode_max": float(bc.max()) if len(bc) else np.nan,
            "train_barcode_mean": float(bc.mean()) if len(bc) else np.nan,
            "train_barcode_median": float(bc.median()) if len(bc) else np.nan,
        }
    )
    return counts


def format_frac_tag(value: float, split_runner_module) -> str:
    if hasattr(split_runner_module, "format_frac_tag"):
        return split_runner_module.format_frac_tag(value)
    return f"{float(value):.4f}".replace(".", "p")


def selected_head_indices(base_module, requested_heads: list[str] | None) -> list[tuple[int, str]]:
    all_heads = list(base_module.PRETRAINED_HEADS)
    if not requested_heads:
        return list(enumerate(all_heads))
    bad = sorted(set(requested_heads) - set(all_heads))
    if bad:
        raise ValueError(f"Unknown pretrained heads {bad}. Valid heads: {all_heads}")
    return [(idx, head) for idx, head in enumerate(all_heads) if head in requested_heads]


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
        "heldout_min_barcodes",
        "split_strategy",
        "split_seed_base",
        "split_seed_effective",
        "val_seed_effective",
        "test_seed_effective",
        "seed",
        "train_barcode_bin",
        "train_barcode_bin_label",
        "train_barcode_bin_query",
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


def aggregate_if_not_empty(
    base_module,
    frame: pd.DataFrame,
    group_cols: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group_cols, *metrics])
    group_cols = [col for col in group_cols if col in frame.columns]
    metrics = [metric for metric in metrics if metric in frame.columns]
    if not group_cols or not metrics:
        return pd.DataFrame()
    return base_module.aggregate_metric_summary(frame, group_cols=group_cols, metrics=metrics)


def parse_args() -> argparse.Namespace:
    filtered = get_filtered_runner()
    split_runner = filtered.get_split_runner()

    parser = argparse.ArgumentParser(
        description=(
            "Run exact barcode-bin learning curves against repeated high-barcode "
            "validation/test holdouts for the filtered lib1 raw-ratio enhancer dataset."
        )
    )
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="random_hq_val_test_per_seed",
        choices=split_runner.SPLIT_STRATEGY_CHOICES,
        help=(
            "Validation/test split policy. The default redraws both validation and test "
            "from the high-barcode pool for every seed."
        ),
    )
    parser.add_argument(
        "--heldout_min_barcodes",
        nargs="+",
        type=int,
        default=DEFAULT_HELDOUT_MIN_BARCODES,
        help="High-barcode pool thresholds to use for validation/test holdouts.",
    )
    parser.add_argument(
        "--train_barcode_bins",
        nargs="+",
        default=DEFAULT_TRAIN_BARCODE_BINS,
        help=(
            "Named training barcode bins. Valid values: "
            + ", ".join([*BARCODE_BIN_CHOICES, "all"])
            + "."
        ),
    )
    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--test_frac", type=float, default=None)
    parser.add_argument("--val_frac_within_hq", type=float, default=0.10)
    parser.add_argument("--test_frac_within_hq", type=float, default=0.10)
    parser.add_argument("--train_size_fracs", nargs="*", type=float, default=DEFAULT_TRAIN_SIZE_FRACS)
    parser.add_argument(
        "--train_sampling_mode",
        type=str,
        default="random",
        choices=["random", "hq_first"],
        help="How to subsample within each eligible barcode-bin training pool.",
    )
    parser.add_argument("--min_train_size", type=int, default=DEFAULT_MIN_TRAIN_SIZE)
    parser.add_argument(
        "--unfreeze_scopes",
        nargs="+",
        type=str,
        default=["branched_only", "full"],
        choices=["head_only", "branched_only", "linear_all_head", "conv3_plus", "full"],
    )
    parser.add_argument("--pretrained_heads", nargs="+", default=None, help="Optional subset of K562 HepG2 SKNSH.")
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
    parser.add_argument(
        "--preview_only",
        action="store_true",
        help="Build split/bin count previews and exit before loading the model or training.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    overall_start_time = time.time()
    args = parse_args()

    filtered = get_filtered_runner()
    base = filtered.get_base()
    filtered.configure_base_for_filtered_ratio(base)

    split_runner = filtered.get_split_runner()
    split_runner.base = base

    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)
    if not args.model_path.exists() and not args.preview_only:
        raise FileNotFoundError(args.model_path)

    args.train_barcode_bins = parse_train_bins(args.train_barcode_bins)
    args.heldout_min_barcodes = sorted({int(value) for value in args.heldout_min_barcodes})
    if not any([args.include_b1, args.include_b2, args.include_b3]):
        args.include_b2 = True

    val_frac = float(args.val_frac if args.val_frac is not None else args.val_frac_within_hq)
    test_frac = float(args.test_frac if args.test_frac is not None else args.test_frac_within_hq)
    split_runner.validate_split_fracs(val_frac=val_frac, test_frac=test_frac)

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
    manifest["resolved_val_frac"] = val_frac
    manifest["resolved_test_frac"] = test_frac
    manifest["cache_layout_version"] = CACHE_LAYOUT_VERSION
    manifest["data_sha256"] = sha256_file(args.data_path)
    manifest["per_epoch_train_metrics_logged"] = True
    manifest["per_epoch_val_metrics_logged"] = True
    manifest["per_epoch_test_metrics_logged"] = True
    manifest["dataset_sequence_column"] = filtered.DATASET_SEQUENCE_COLUMN
    manifest["dataset_barcode_column"] = filtered.DATASET_BARCODE_COLUMN
    manifest["dataset_raw_ratio_column"] = filtered.DATASET_RAW_RATIO_COLUMN
    manifest["modeled_target_column"] = filtered.MODELING_TARGET_COLUMN
    manifest["modeled_target_transform"] = "log10(RNA_bc_counts_sum / DNA_bc_counts_sum)"
    manifest["target_standardization"] = "train_only_zscore_after_split"
    manifest["invalid_enhancer_filter"] = (
        f"{filtered.DATASET_SEQUENCE_COLUMN} matches {filtered.VALID_ENHANCER_PATTERN!r}"
    )
    manifest["barcode_bin_specs"] = {name: asdict(spec) for name, spec in BARCODE_BIN_SPECS.items()}
    manifest["dataset_barcode_counts"] = dataset_barcode_counts
    manifest["preprocessing_summary"] = preprocessing_summary
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Repo root: {base.REPO_ROOT}")
    print(f"Using device: {device}")
    print(f"Output dir: {args.outdir}")
    print(f"Split strategy: {args.split_strategy}")
    print(f"Heldout min barcodes: {args.heldout_min_barcodes}")
    print(f"Train barcode bins: {args.train_barcode_bins}")
    print(f"Validation fraction within heldout pool: {val_frac}")
    print(f"Test fraction within heldout pool: {test_frac}")
    print(f"Settings: {[setting.name for setting in settings]}")
    print(f"Pretrained heads: {[head for _, head in selected_heads]}")
    print(f"Unfreeze scopes: {args.unfreeze_scopes}")
    print(f"Head LR grid: {head_lr_grid}")
    print(f"Backbone LR grid: {backbone_lr_grid}")
    print(
        "Preprocessing summary: "
        f"raw_rows={preprocessing_summary['n_rows_raw']}, "
        f"clean_rows={preprocessing_summary['n_rows_clean']}, "
        f"dropped={preprocessing_summary['n_rows_dropped_total']}, "
        f"invalid_enhancers={preprocessing_summary['n_invalid_enhancer_rows']}, "
        f"ratio_mismatches={preprocessing_summary['n_ratio_mismatches_vs_reported_column']}"
    )

    checkpoint = None if args.preview_only else base.load_checkpoint_from_tar(args.model_path, map_location="cpu")

    run_records: list[dict[str, Any]] = []
    history_records: list[pd.DataFrame] = []
    zero_shot_records: list[pd.DataFrame] = []
    zero_shot_seen_keys: set[tuple[Any, ...]] = set()
    preview_records: list[dict[str, Any]] = []

    for heldout_min in args.heldout_min_barcodes:
        for seed in args.seeds:
            if args.split_strategy == "fixed_hq_val_test":
                seed_key = f"globalseed{args.split_seed}"
                run_seed_for_split = None
            else:
                seed_key = f"seed{seed}__splitseed{args.split_seed}"
                run_seed_for_split = seed

            split_cache_path = split_cache_dir / split_runner.split_cache_name(
                args.split_strategy,
                val_frac=val_frac,
                test_frac=test_frac,
                test_min_barcodes=heldout_min,
                seed_key=seed_key,
            )
            split_payload = base.maybe_load_or_build(
                split_cache_path,
                force=args.force,
                builder=lambda heldout_min=heldout_min, run_seed_for_split=run_seed_for_split: (
                    split_runner.build_split_payload(
                        clean_df,
                        strategy=args.split_strategy,
                        val_frac=val_frac,
                        test_frac=test_frac,
                        base_split_seed=args.split_seed,
                        run_seed=run_seed_for_split,
                        test_min_barcodes=heldout_min,
                    )
                ),
            )

            train_rest_df = split_payload["train_rest_df"]
            val_df_raw = split_payload["val_df"]
            test_df_raw = split_payload["test_df"]
            split_counts = {
                "train_rest_size": len(train_rest_df),
                "val_size_raw": len(val_df_raw),
                "test_size_raw": len(test_df_raw),
                **{f"train_rest_{k}": v for k, v in barcode_count_summary(train_rest_df, base.BARCODE_COLUMN).items()},
                **{f"val_{k}": v for k, v in barcode_count_summary(val_df_raw, base.BARCODE_COLUMN).items()},
                **{f"test_{k}": v for k, v in barcode_count_summary(test_df_raw, base.BARCODE_COLUMN).items()},
            }

            print(
                f"Seed {seed}, heldout >= {heldout_min}: "
                f"train_rest={len(train_rest_df)}, val={len(val_df_raw)}, test={len(test_df_raw)}, "
                f"split_seed_effective={split_payload['split_seed_effective']}"
            )

            zero_shot_split_key = (
                split_payload["test_seed_effective"]
                if split_payload["test_is_fixed_across_seeds"]
                else split_payload["split_seed_effective"]
            )
            zero_shot_key = (
                heldout_min,
                args.split_strategy,
                zero_shot_split_key,
                val_frac,
                test_frac,
            )
            if (not args.preview_only) and zero_shot_key not in zero_shot_seen_keys:
                zero_shot_seen_keys.add(zero_shot_key)
                zero_shot_df = base.run_zero_shot_eval_on_fixed_test(
                    checkpoint,
                    split_payload["test_padded"],
                    device=device,
                )
                zero_shot_df["run_id"] = run_id
                zero_shot_df["seed"] = seed
                zero_shot_df["heldout_min_barcodes"] = heldout_min
                zero_shot_df["split_strategy"] = args.split_strategy
                zero_shot_df["split_seed_effective"] = split_payload["split_seed_effective"]
                zero_shot_df["val_seed_effective"] = split_payload["val_seed_effective"]
                zero_shot_df["test_seed_effective"] = split_payload["test_seed_effective"]
                zero_shot_df["test_is_fixed_across_seeds"] = split_payload["test_is_fixed_across_seeds"]
                zero_shot_records.append(zero_shot_df)

            for train_bin_name in args.train_barcode_bins:
                train_bin = BARCODE_BIN_SPECS[train_bin_name]
                pool_components = build_barcode_bin_train_pool_components(
                    train_rest_df=train_rest_df,
                    train_bin=train_bin,
                    barcode_column=base.BARCODE_COLUMN,
                    heldout_min_barcodes=heldout_min,
                )
                pool = pool_components["eligible"]
                leftover_hq_pool = pool_components["leftover_hq"]
                lower_quality_pool = pool_components["lower_quality"]
                if len(pool) < args.min_train_size:
                    print(
                        f"Skipping seed {seed}, heldout >= {heldout_min}, bin {train_bin_name}: "
                        f"only {len(pool)} rows available."
                    )
                    continue

                size_grid = base.make_train_size_grid(
                    n_available=len(pool),
                    min_train_size=args.min_train_size,
                    train_size_fracs=list(args.train_size_fracs) if args.train_size_fracs else None,
                )
                print(
                    f"Seed {seed}, heldout >= {heldout_min}, bin {train_bin_name}: "
                    f"eligible={len(pool)} (leftover_HQ={len(leftover_hq_pool)}, "
                    f"lower_quality={len(lower_quality_pool)}), "
                    f"sampling={args.train_sampling_mode}, sizes={size_grid}"
                )

                preview_records.append(
                    {
                        "seed": seed,
                        "heldout_min_barcodes": heldout_min,
                        "train_barcode_bin": train_bin.name,
                        "train_barcode_bin_label": train_bin.label,
                        "train_barcode_bin_query": train_bin.query,
                        "train_pool_eligible_size": len(pool),
                        "train_pool_leftover_hq_size": len(leftover_hq_pool),
                        "train_pool_lower_quality_size": len(lower_quality_pool),
                        "train_size_grid": " ".join(map(str, size_grid)),
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
                    train_bin_counts = compute_train_bin_counts(
                        train_raw,
                        barcode_column=base.BARCODE_COLUMN,
                        heldout_min_barcodes=heldout_min,
                    )
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
                                        spec = BarcodeRangeExperimentSpec(
                                            seed=seed,
                                            heldout_min_barcodes=heldout_min,
                                            head_idx=head_idx,
                                            init_head=head_name,
                                            setting_name=setting.name,
                                            train_barcode_bin=train_bin.name,
                                            train_size=len(train_df),
                                            train_fraction=len(train_df) / len(pool),
                                            unfreeze_scope=unfreeze_scope,
                                            train_sampling_mode=args.train_sampling_mode,
                                            head_lr=float(head_lr),
                                            backbone_lr=float(backbone_lr),
                                        )
                                        split_tag = (
                                            f"split{split_payload['split_seed_effective']}"
                                            f"__val{split_payload['val_seed_effective']}"
                                            f"__test{split_payload['test_seed_effective']}"
                                        )
                                        cache_path = (
                                            cache_dir
                                            / "runs"
                                            / CACHE_LAYOUT_VERSION
                                            / args.split_strategy
                                            / f"heldout_minbc{heldout_min}"
                                            / f"val{format_frac_tag(val_frac, split_runner)}__test{format_frac_tag(test_frac, split_runner)}"
                                            / sanitize_tag(split_tag)
                                            / train_bin.name
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
                                            bin_offset = BARCODE_BIN_SPECS[spec.train_barcode_bin].sort_order
                                            training_seed = (
                                                spec.seed * 1_000_003
                                                + spec.heldout_min_barcodes * 10_007
                                                + bin_offset * 1_009
                                                + spec.head_idx * 101
                                            )
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
                                            val_metrics, _ = base.evaluate_single_head_model(
                                                model,
                                                val_df,
                                                scaler,
                                                device=device,
                                            )
                                            test_metrics, pred_df = base.evaluate_single_head_model(
                                                model,
                                                test_df,
                                                scaler,
                                                device=device,
                                            )
                                            train_metrics, _ = base.evaluate_single_head_model(
                                                model,
                                                train_df,
                                                scaler,
                                                device=device,
                                            )
                                            return {
                                                "fit_info": fit_info,
                                                "history_df": history_df,
                                                "train_metrics": train_metrics,
                                                "val_metrics": val_metrics,
                                                "test_metrics": test_metrics,
                                                "pred_df": pred_df,
                                            }

                                        payload = base.maybe_load_or_build(cache_path, force=args.force, builder=_builder)
                                        fit_info = payload["fit_info"]
                                        train_m = payload.get("train_metrics")
                                        row = {
                                            "run_id": run_id,
                                            "seed": spec.seed,
                                            "heldout_min_barcodes": heldout_min,
                                            "split_strategy": args.split_strategy,
                                            "split_pool": split_payload["split_pool"],
                                            "split_seed_base": args.split_seed,
                                            "split_seed_effective": split_payload["split_seed_effective"],
                                            "val_seed_effective": split_payload["val_seed_effective"],
                                            "test_seed_effective": split_payload["test_seed_effective"],
                                            "val_is_fixed_across_seeds": split_payload["val_is_fixed_across_seeds"],
                                            "test_is_fixed_across_seeds": split_payload["test_is_fixed_across_seeds"],
                                            "split_val_fraction": val_frac,
                                            "split_test_fraction": test_frac,
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
                                            "train_barcode_bin": train_bin.name,
                                            "train_barcode_bin_label": train_bin.label,
                                            "train_barcode_bin_query": train_bin.query,
                                            "train_barcode_bin_sort_order": train_bin.sort_order,
                                            "train_threshold": np.nan,
                                            "train_size": spec.train_size,
                                            "train_fraction": spec.train_fraction,
                                            "unfreeze_scope": spec.unfreeze_scope,
                                            "train_sampling_mode": spec.train_sampling_mode,
                                            "train_pool_eligible_size": len(pool),
                                            "train_pool_leftover_hq_size": len(leftover_hq_pool),
                                            "train_pool_lower_quality_size": len(lower_quality_pool),
                                            "val_size": len(val_df),
                                            "test_size": len(test_df),
                                            "head_lr": spec.head_lr,
                                            "backbone_lr": spec.backbone_lr,
                                            "weight_decay": args.weight_decay,
                                            **train_bin_counts,
                                            **split_counts,
                                            **(
                                                {f"train_{k}": v for k, v in train_m.items()}
                                                if train_m is not None
                                                else {
                                                    f"train_{k}": np.nan
                                                    for k in (
                                                        "n",
                                                        "mae",
                                                        "rmse",
                                                        "pearson",
                                                        "spearman",
                                                        "r2",
                                                        "r2_cod",
                                                        "pearson_sq",
                                                        "loss_standardized",
                                                    )
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
                                        hist["heldout_min_barcodes"] = heldout_min
                                        hist["split_strategy"] = args.split_strategy
                                        hist["split_pool"] = split_payload["split_pool"]
                                        hist["split_seed_effective"] = split_payload["split_seed_effective"]
                                        hist["val_seed_effective"] = split_payload["val_seed_effective"]
                                        hist["test_seed_effective"] = split_payload["test_seed_effective"]
                                        hist["val_is_fixed_across_seeds"] = split_payload["val_is_fixed_across_seeds"]
                                        hist["test_is_fixed_across_seeds"] = split_payload["test_is_fixed_across_seeds"]
                                        hist["split_val_fraction"] = val_frac
                                        hist["split_test_fraction"] = test_frac
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
                                        hist["min_weight"] = (
                                            float(setting.min_weight) if setting.use_barcode_weighting else np.nan
                                        )
                                        hist["train_barcode_bin"] = train_bin.name
                                        hist["train_barcode_bin_label"] = train_bin.label
                                        hist["train_barcode_bin_query"] = train_bin.query
                                        hist["train_barcode_bin_sort_order"] = train_bin.sort_order
                                        hist["train_size"] = spec.train_size
                                        hist["train_fraction"] = spec.train_fraction
                                        hist["unfreeze_scope"] = spec.unfreeze_scope
                                        hist["train_sampling_mode"] = spec.train_sampling_mode
                                        hist["head_lr"] = spec.head_lr
                                        hist["backbone_lr"] = spec.backbone_lr
                                        history_records.append(hist)

    preview_df = pd.DataFrame(preview_records)
    if not preview_df.empty:
        preview_df.to_csv(args.outdir / "barcode_range_planned_grid.csv", index=False)

    if args.preview_only:
        print("\nPreview only. Wrote:")
        print(f"  {args.outdir / 'barcode_range_planned_grid.csv'}")
        print(f"  {args.outdir / 'run_manifest.json'}")
        elapsed_seconds = time.time() - overall_start_time
        print(f"\nTotal runtime: {elapsed_seconds:.1f} seconds ({elapsed_seconds / 60.0:.2f} minutes)")
        return

    if len(run_records) == 0:
        raise RuntimeError("No runs were executed. Check bins/min_train_size/settings.")

    runs_df = pd.DataFrame(run_records).sort_values(
        [
            "heldout_min_barcodes",
            "train_barcode_bin_sort_order",
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

    aggregate_df = base.aggregate_metric_summary(
        runs_df,
        group_cols=[
            "heldout_min_barcodes",
            "split_strategy",
            "split_pool",
            "split_val_fraction",
            "split_test_fraction",
            "val_is_fixed_across_seeds",
            "test_is_fixed_across_seeds",
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
        ],
        metrics=metric_cols,
    )

    scope_summary_df = base.aggregate_metric_summary(
        runs_df,
        group_cols=[
            "heldout_min_barcodes",
            "split_strategy",
            "split_pool",
            "split_val_fraction",
            "split_test_fraction",
            "val_is_fixed_across_seeds",
            "test_is_fixed_across_seeds",
            "train_barcode_bin",
            "train_barcode_bin_label",
            "train_barcode_bin_query",
            "setting",
            "b_cap",
            "head_lr",
            "backbone_lr",
            "train_sampling_mode",
            "unfreeze_scope",
        ],
        metrics=[
            "test_mae",
            "test_rmse",
            "test_pearson",
            "test_spearman",
            "test_r2",
            "test_r2_cod",
            "test_pearson_sq",
            "test_loss_standardized",
            "best_epoch",
            "initial_trainable_params",
            "final_trainable_params",
        ],
    )

    full_fraction_df = runs_df.loc[runs_df["train_size"] == runs_df["train_pool_eligible_size"]].copy()
    full_fraction_summary_df = aggregate_if_not_empty(
        base,
        full_fraction_df,
        group_cols=[
            "heldout_min_barcodes",
            "train_barcode_bin",
            "train_barcode_bin_label",
            "train_barcode_bin_query",
            "setting",
            "b_cap",
            "head_lr",
            "backbone_lr",
            "train_sampling_mode",
            "unfreeze_scope",
        ],
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
            "heldout_min_barcodes",
            "train_barcode_bin",
            "train_barcode_bin_label",
            "train_barcode_bin_query",
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
    velocity_by_segment_df = aggregate_if_not_empty(
        base,
        velocity_df,
        group_cols=[
            "heldout_min_barcodes",
            "train_barcode_bin",
            "train_barcode_bin_label",
            "train_barcode_bin_query",
            "train_sampling_mode",
            "setting",
            "b_cap",
            "head_lr",
            "backbone_lr",
            "unfreeze_scope",
            "metric",
            "segment_index",
            "train_fraction_start",
            "train_fraction_end",
        ],
        metrics=["delta_train_size", "delta_metric", "slope_per_construct", "slope_per_100_constructs"],
    )

    zero_shot_path = args.outdir / "zero_shot_evaluations.csv"
    if zero_shot_records:
        pd.concat(zero_shot_records, ignore_index=True).to_csv(zero_shot_path, index=False)

    runs_df.to_csv(args.outdir / "learning_curve_runs.csv", index=False)
    history_df.to_csv(args.outdir / "learning_curve_histories.csv", index=False)
    aggregate_df.to_csv(args.outdir / "learning_curve_summary_mean_std.csv", index=False)
    scope_summary_df.to_csv(args.outdir / "unfreeze_scope_summary_mean_std.csv", index=False)
    full_fraction_summary_df.to_csv(args.outdir / "barcode_bin_full_fraction_summary_mean_std.csv", index=False)
    velocity_df.to_csv(args.outdir / "learning_curve_velocity_segments.csv", index=False)
    velocity_summary_df.to_csv(args.outdir / "learning_curve_velocity_summary_mean_std.csv", index=False)
    velocity_by_segment_df.to_csv(args.outdir / "learning_curve_velocity_by_segment_mean_std.csv", index=False)

    print("\nWrote outputs:")
    output_paths = [
        args.outdir / "barcode_range_planned_grid.csv",
        args.outdir / "learning_curve_runs.csv",
        args.outdir / "learning_curve_histories.csv",
        args.outdir / "learning_curve_summary_mean_std.csv",
        args.outdir / "unfreeze_scope_summary_mean_std.csv",
        args.outdir / "barcode_bin_full_fraction_summary_mean_std.csv",
        args.outdir / "learning_curve_velocity_segments.csv",
        args.outdir / "learning_curve_velocity_summary_mean_std.csv",
        args.outdir / "learning_curve_velocity_by_segment_mean_std.csv",
        args.outdir / "run_manifest.json",
    ]
    if zero_shot_records:
        output_paths.insert(1, zero_shot_path)
    for path in output_paths:
        print(f"  {path}")

    elapsed_seconds = time.time() - overall_start_time
    elapsed_minutes = elapsed_seconds / 60.0
    print(f"\nTotal runtime: {elapsed_seconds:.1f} seconds ({elapsed_minutes:.2f} minutes)")


if __name__ == "__main__":
    main()
