#!/usr/bin/env python3
"""Learning-curve fine-tuning for the filtered lib1 raw-ratio collaborator CSV.

This runner is based on `lib1_enhancer_learning_curve_finetune_split_options.py`,
but adapts the data-loading/preprocessing layer for the collaborator-provided CSV:

1. load `L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.csv`
2. drop invalid enhancer rows (non-ACGT enhancer sequence)
3. recompute the raw assay ratio from counts as `RNA_bc_counts_sum / DNA_bc_counts_sum`
4. use `log10(RNA/DNA)` as the modeling target
5. z-score that target with train-only mean/std after each split is created

Per-epoch train/validation/test metrics are logged through the shared base runner,
including standardized loss, Pearson, Spearman, and R2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT_PATH = THIS_DIR / "lib1_enhancer_learning_curve_finetune_updated.py"
SPLIT_SCRIPT_PATH = THIS_DIR / "lib1_enhancer_learning_curve_finetune_split_options.py"

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
    / "lib1_enhancer_filtered_raw_ratio_apr2026"
)

DEFAULT_SEEDS = [7, 8, 9]
DEFAULT_SPLIT_SEED = 7
DEFAULT_TRAIN_THRESHOLDS = [1, 2, 3]
DEFAULT_TRAIN_PRIORITY_MIN_BARCODES = 4
DEFAULT_TRAIN_SIZE_FRACS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
DEFAULT_MIN_TRAIN_SIZE = 32
DEFAULT_MAX_EPOCHS = 70
DEFAULT_PATIENCE = 10
DEFAULT_FROZEN_EPOCHS = 2
DEFAULT_TRAIN_BATCH_SIZE = 128
DEFAULT_WEIGHT_DECAY = 1e-4

DATASET_SEQUENCE_COLUMN = "Enhancer"
DATASET_BARCODE_COLUMN = "number_of_barcodes"
DATASET_RAW_RATIO_COLUMN = "RNA/DNA"
DATASET_RECOMPUTED_RATIO_COLUMN = "RNA_DNA_ratio_from_counts"
MODELING_TARGET_COLUMN = "RNA_DNA_log10"
VALID_ENHANCER_PATTERN = r"[ACGT]+"
CACHE_LAYOUT_VERSION = "filtered_raw_ratio_per_epoch_train_val_test_metrics_v1"

base = None
split_runner = None


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def get_base():
    global base
    if base is None:
        base = _load_module("lib1_learning_curve_base_filtered_ratio", BASE_SCRIPT_PATH)
    return base


def get_split_runner():
    global split_runner
    if split_runner is None:
        split_runner = _load_module("lib1_learning_curve_split_runner_filtered_ratio", SPLIT_SCRIPT_PATH)
    return split_runner


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


def collect_preprocessing_summary(data_path: Path) -> dict[str, Any]:
    raw_df = pd.read_csv(data_path).copy()
    required_columns = {
        DATASET_SEQUENCE_COLUMN,
        DATASET_BARCODE_COLUMN,
        "DNA_bc_counts_sum",
        "RNA_bc_counts_sum",
        DATASET_RAW_RATIO_COLUMN,
    }
    missing = sorted(required_columns - set(raw_df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    raw_df[DATASET_BARCODE_COLUMN] = pd.to_numeric(raw_df[DATASET_BARCODE_COLUMN], errors="coerce")
    raw_df["DNA_bc_counts_sum"] = pd.to_numeric(raw_df["DNA_bc_counts_sum"], errors="coerce")
    raw_df["RNA_bc_counts_sum"] = pd.to_numeric(raw_df["RNA_bc_counts_sum"], errors="coerce")
    raw_df[DATASET_RAW_RATIO_COLUMN] = pd.to_numeric(raw_df[DATASET_RAW_RATIO_COLUMN], errors="coerce")
    raw_df[DATASET_RECOMPUTED_RATIO_COLUMN] = raw_df["RNA_bc_counts_sum"] / raw_df["DNA_bc_counts_sum"]
    raw_df["ratio_matches_counts"] = np.isclose(
        raw_df[DATASET_RAW_RATIO_COLUMN],
        raw_df[DATASET_RECOMPUTED_RATIO_COLUMN],
        equal_nan=True,
    )
    raw_df["enhancer_is_acgt_only"] = raw_df[DATASET_SEQUENCE_COLUMN].astype(str).str.fullmatch(
        VALID_ENHANCER_PATTERN,
        na=False,
    )

    keep_mask = (
        raw_df[DATASET_SEQUENCE_COLUMN].notna()
        & raw_df["enhancer_is_acgt_only"]
        & raw_df[DATASET_BARCODE_COLUMN].notna()
        & np.isfinite(raw_df[DATASET_RECOMPUTED_RATIO_COLUMN])
        & (raw_df[DATASET_RECOMPUTED_RATIO_COLUMN] > 0)
    )

    return {
        "n_rows_raw": int(len(raw_df)),
        "n_rows_clean": int(keep_mask.sum()),
        "n_rows_dropped_total": int((~keep_mask).sum()),
        "n_invalid_enhancer_rows": int((~raw_df["enhancer_is_acgt_only"]).sum()),
        "n_missing_barcode_rows": int(raw_df[DATASET_BARCODE_COLUMN].isna().sum()),
        "n_nonpositive_or_nonfinite_ratio_rows": int(
            (~np.isfinite(raw_df[DATASET_RECOMPUTED_RATIO_COLUMN]) | (raw_df[DATASET_RECOMPUTED_RATIO_COLUMN] <= 0)).sum()
        ),
        "n_ratio_mismatches_vs_reported_column": int((~raw_df["ratio_matches_counts"]).sum()),
        "ratio_matches_counts_all": bool(raw_df["ratio_matches_counts"].all()),
    }


def configure_base_for_filtered_ratio(base_module) -> None:
    base_module.SEQUENCE_COLUMN = DATASET_SEQUENCE_COLUMN
    base_module.BARCODE_COLUMN = DATASET_BARCODE_COLUMN
    base_module.TARGET_COLUMN = MODELING_TARGET_COLUMN
    base_module.RAW_RATIO_COLUMN = DATASET_RAW_RATIO_COLUMN
    base_module.RECOMPUTED_RATIO_COLUMN = DATASET_RECOMPUTED_RATIO_COLUMN

    def add_padded_sequences(
        df: pd.DataFrame,
        sequence_column: str = DATASET_SEQUENCE_COLUMN,
        padded_seq_len: int | None = None,
    ) -> pd.DataFrame:
        out = df.copy()
        target_len = int(padded_seq_len if padded_seq_len is not None else base_module.INPUT_LEN)
        out["padded_seq"] = out.apply(
            lambda row: base_module.utils.row_pad_sequence(
                row,
                in_column_name=sequence_column,
                padded_seq_len=target_len,
            ),
            axis=1,
        )
        return out

    def load_clean_df(data_path: Path) -> pd.DataFrame:
        raw_df = pd.read_csv(data_path).copy()
        required_columns = {
            DATASET_SEQUENCE_COLUMN,
            DATASET_BARCODE_COLUMN,
            "DNA_bc_counts_sum",
            "RNA_bc_counts_sum",
            DATASET_RAW_RATIO_COLUMN,
        }
        missing = sorted(required_columns - set(raw_df.columns))
        if missing:
            raise ValueError(f"Missing required columns in {data_path}: {missing}")

        raw_df[DATASET_BARCODE_COLUMN] = pd.to_numeric(raw_df[DATASET_BARCODE_COLUMN], errors="coerce")
        raw_df["DNA_bc_counts_sum"] = pd.to_numeric(raw_df["DNA_bc_counts_sum"], errors="coerce")
        raw_df["RNA_bc_counts_sum"] = pd.to_numeric(raw_df["RNA_bc_counts_sum"], errors="coerce")
        raw_df[DATASET_RAW_RATIO_COLUMN] = pd.to_numeric(raw_df[DATASET_RAW_RATIO_COLUMN], errors="coerce")
        raw_df[DATASET_RECOMPUTED_RATIO_COLUMN] = raw_df["RNA_bc_counts_sum"] / raw_df["DNA_bc_counts_sum"]
        raw_df["ratio_matches_counts"] = np.isclose(
            raw_df[DATASET_RAW_RATIO_COLUMN],
            raw_df[DATASET_RECOMPUTED_RATIO_COLUMN],
            equal_nan=True,
        )
        raw_df["enhancer_is_acgt_only"] = raw_df[DATASET_SEQUENCE_COLUMN].astype(str).str.fullmatch(
            VALID_ENHANCER_PATTERN,
            na=False,
        )

        clean_df = raw_df.loc[
            raw_df[DATASET_SEQUENCE_COLUMN].notna()
            & raw_df["enhancer_is_acgt_only"]
            & raw_df[DATASET_BARCODE_COLUMN].notna()
            & np.isfinite(raw_df[DATASET_RECOMPUTED_RATIO_COLUMN])
            & (raw_df[DATASET_RECOMPUTED_RATIO_COLUMN] > 0)
        ].copy().reset_index(drop=True)

        clean_df[MODELING_TARGET_COLUMN] = np.log10(clean_df[DATASET_RECOMPUTED_RATIO_COLUMN].astype(float))
        clean_df["sequence_len"] = clean_df[DATASET_SEQUENCE_COLUMN].astype(str).str.len()
        clean_df["row_id"] = np.arange(len(clean_df))
        clean_df["target_transform"] = "log10_recomputed_rna_over_dna"
        return clean_df

    base_module.add_padded_sequences = add_padded_sequences
    base_module.load_clean_df = load_clean_df


def parse_args() -> argparse.Namespace:
    split_runner = get_split_runner()

    parser = argparse.ArgumentParser(
        description=(
            "Learning-curve fine-tuning with per-epoch train/val/test metrics for the "
            "filtered collaborator lib1 enhancer raw-ratio dataset."
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
        default="fixed_hq_test_random_hq_val_per_seed",
        choices=split_runner.SPLIT_STRATEGY_CHOICES,
        help=(
            "How validation/test splits are generated. "
            + " | ".join(f"{name}: {desc}" for name, desc in split_runner.SPLIT_STRATEGY_HELP.items())
        ),
    )
    parser.add_argument("--train_thresholds", nargs="+", type=int, default=DEFAULT_TRAIN_THRESHOLDS)
    parser.add_argument(
        "--train_priority_min_barcodes",
        type=int,
        default=DEFAULT_TRAIN_PRIORITY_MIN_BARCODES,
        help=(
            "Barcode threshold used to define the higher-priority subset for HQ-based split strategies "
            "and for `--train_sampling_mode hq_first`."
        ),
    )
    parser.add_argument("--val_frac", type=float, default=None)
    parser.add_argument("--test_frac", type=float, default=None)
    parser.add_argument("--val_frac_within_hq", type=float, default=0.20)
    parser.add_argument("--test_frac_within_hq", type=float, default=0.20)
    parser.add_argument("--train_size_fracs", nargs="*", type=float, default=DEFAULT_TRAIN_SIZE_FRACS)
    parser.add_argument(
        "--train_sampling_mode",
        type=str,
        default="hq_first",
        choices=["hq_first", "random"],
        help="How to grow the training set as train_size increases.",
    )
    parser.add_argument("--min_train_size", type=int, default=DEFAULT_MIN_TRAIN_SIZE)
    parser.add_argument(
        "--unfreeze_scopes",
        nargs="+",
        type=str,
        default=["head_only", "branched_only", "linear_all_head", "conv3_plus", "full"],
        choices=["head_only", "branched_only", "linear_all_head", "conv3_plus", "full"],
    )
    parser.add_argument("--include_b1", action="store_true")
    parser.add_argument("--include_b2", action="store_true")
    parser.add_argument("--include_b3", action="store_true")
    parser.add_argument("--b3_bcaps", nargs="+", type=float, default=None)
    parser.add_argument("--b3_bcap", type=float, default=8.0)
    parser.add_argument("--min_weight", type=float, default=0.1)
    parser.add_argument("--head_lrs", nargs="+", type=float, default=None)
    parser.add_argument("--backbone_lrs", nargs="+", type=float, default=None)
    parser.add_argument("--head_lr", type=float, default=2e-4)
    parser.add_argument("--backbone_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--frozen_epochs", type=int, default=DEFAULT_FROZEN_EPOCHS)
    parser.add_argument("--train_batch_size", type=int, default=DEFAULT_TRAIN_BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    overall_start_time = time.time()
    args = parse_args()
    base = get_base()
    configure_base_for_filtered_ratio(base)

    split_runner = get_split_runner()
    split_runner.base = base

    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)

    if not any([args.include_b1, args.include_b2, args.include_b3]):
        args.include_b2 = True

    val_frac = float(args.val_frac if args.val_frac is not None else args.val_frac_within_hq)
    test_frac = float(args.test_frac if args.test_frac is not None else args.test_frac_within_hq)
    split_runner.validate_split_fracs(val_frac=val_frac, test_frac=test_frac)

    b3_bcap_grid = list(args.b3_bcaps) if args.b3_bcaps else [args.b3_bcap]
    head_lr_grid = list(args.head_lrs) if args.head_lrs else [args.head_lr]
    backbone_lr_grid = list(args.backbone_lrs) if args.backbone_lrs else [args.backbone_lr]

    device = base.resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_cache_dir = cache_dir / "splits"
    split_cache_dir.mkdir(parents=True, exist_ok=True)

    preprocessing_summary = collect_preprocessing_summary(args.data_path)
    settings = base.build_settings(
        args.include_b1,
        args.include_b2,
        args.include_b3,
        b3_bcaps=b3_bcap_grid,
        min_weight=args.min_weight,
    )
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"

    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest["repo_root"] = str(base.REPO_ROOT)
    manifest["device_resolved"] = device
    manifest["run_id"] = run_id
    manifest["run_started_at"] = run_started_at
    manifest["settings"] = [asdict(x) for x in settings]
    manifest["resolved_val_frac"] = val_frac
    manifest["resolved_test_frac"] = test_frac
    manifest["cache_layout_version"] = CACHE_LAYOUT_VERSION
    manifest["data_sha256"] = sha256_file(args.data_path)
    manifest["per_epoch_train_metrics_logged"] = True
    manifest["per_epoch_val_metrics_logged"] = True
    manifest["per_epoch_test_metrics_logged"] = True
    manifest["dataset_sequence_column"] = DATASET_SEQUENCE_COLUMN
    manifest["dataset_barcode_column"] = DATASET_BARCODE_COLUMN
    manifest["dataset_raw_ratio_column"] = DATASET_RAW_RATIO_COLUMN
    manifest["modeled_target_column"] = MODELING_TARGET_COLUMN
    manifest["modeled_target_transform"] = "log10(RNA_bc_counts_sum / DNA_bc_counts_sum)"
    manifest["target_standardization"] = "train_only_zscore_after_split"
    manifest["invalid_enhancer_filter"] = f"{DATASET_SEQUENCE_COLUMN} matches {VALID_ENHANCER_PATTERN!r}"
    manifest["preprocessing_summary"] = preprocessing_summary
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Repo root: {base.REPO_ROOT}")
    print(f"Using device: {device}")
    print(f"Output dir: {args.outdir}")
    print(f"Split strategy: {args.split_strategy}")
    print(f"Validation fraction: {val_frac}")
    print(f"Test fraction: {test_frac}")
    print(f"Head LR grid: {head_lr_grid}")
    print(f"Backbone LR grid: {backbone_lr_grid}")
    print(f"B3 b_cap grid: {b3_bcap_grid}")
    print(
        "Preprocessing summary: "
        f"raw_rows={preprocessing_summary['n_rows_raw']}, "
        f"clean_rows={preprocessing_summary['n_rows_clean']}, "
        f"dropped={preprocessing_summary['n_rows_dropped_total']}, "
        f"invalid_enhancers={preprocessing_summary['n_invalid_enhancer_rows']}, "
        f"ratio_mismatches={preprocessing_summary['n_ratio_mismatches_vs_reported_column']}"
    )

    clean_df = base.load_clean_df(args.data_path)
    checkpoint = base.load_checkpoint_from_tar(args.model_path, map_location="cpu")

    fixed_split_payload = None
    if args.split_strategy == "fixed_hq_val_test":
        split_cache_path = split_cache_dir / split_runner.split_cache_name(
            args.split_strategy,
            val_frac=val_frac,
            test_frac=test_frac,
            test_min_barcodes=args.train_priority_min_barcodes,
            seed_key=f"globalseed{args.split_seed}",
        )
        fixed_split_payload = base.maybe_load_or_build(
            split_cache_path,
            force=args.force,
            builder=lambda: split_runner.build_split_payload(
                clean_df,
                strategy=args.split_strategy,
                val_frac=val_frac,
                test_frac=test_frac,
                base_split_seed=args.split_seed,
                run_seed=None,
                test_min_barcodes=args.train_priority_min_barcodes,
            ),
        )

    run_records: list[dict[str, Any]] = []
    history_records: list[pd.DataFrame] = []
    zero_shot_fixed_df: pd.DataFrame | None = None
    zero_shot_seed_records: list[pd.DataFrame] = []

    for seed in args.seeds:
        if fixed_split_payload is not None:
            split_payload = fixed_split_payload
        else:
            seed_key = f"seed{seed}__splitseed{args.split_seed}"
            split_cache_path = split_cache_dir / split_runner.split_cache_name(
                args.split_strategy,
                val_frac=val_frac,
                test_frac=test_frac,
                test_min_barcodes=args.train_priority_min_barcodes,
                seed_key=seed_key,
            )
            split_payload = base.maybe_load_or_build(
                split_cache_path,
                force=args.force,
                builder=lambda seed=seed: split_runner.build_split_payload(
                    clean_df,
                    strategy=args.split_strategy,
                    val_frac=val_frac,
                    test_frac=test_frac,
                    base_split_seed=args.split_seed,
                    run_seed=seed,
                    test_min_barcodes=args.train_priority_min_barcodes,
                ),
            )

        train_rest_df = split_payload["train_rest_df"]
        val_df_raw = split_payload["val_df"]
        test_df_raw = split_payload["test_df"]

        print(
            f"Seed {seed}: split_strategy={args.split_strategy}, "
            f"train_rest={len(train_rest_df)}, val={len(val_df_raw)}, test={len(test_df_raw)}, "
            f"split_seed_effective={split_payload['split_seed_effective']}"
        )

        if split_payload["test_is_fixed_across_seeds"]:
            if zero_shot_fixed_df is None:
                zero_shot_fixed_df = base.run_zero_shot_eval_on_fixed_test(
                    checkpoint,
                    split_payload["test_padded"],
                    device=device,
                )
                zero_shot_fixed_df["split_strategy"] = args.split_strategy
                zero_shot_fixed_df["split_seed_effective"] = split_payload["split_seed_effective"]
                zero_shot_fixed_df["test_seed_effective"] = split_payload["test_seed_effective"]
        else:
            zero_shot_df = base.run_zero_shot_eval_on_fixed_test(
                checkpoint,
                split_payload["test_padded"],
                device=device,
            )
            zero_shot_df["seed"] = seed
            zero_shot_df["split_strategy"] = args.split_strategy
            zero_shot_df["split_seed_effective"] = split_payload["split_seed_effective"]
            zero_shot_df["test_seed_effective"] = split_payload["test_seed_effective"]
            zero_shot_seed_records.append(zero_shot_df)

        for train_threshold in args.train_thresholds:
            pool_components = base.build_train_pool_components(
                train_rest_df=train_rest_df,
                train_threshold=train_threshold,
                test_min_barcodes=args.train_priority_min_barcodes,
            )
            pool = pool_components["eligible"]
            leftover_hq_pool = pool_components["leftover_hq"]
            lower_quality_pool = pool_components["lower_quality"]
            if len(pool) < args.min_train_size:
                print(f"Skipping seed {seed}, threshold {train_threshold}: only {len(pool)} rows available.")
                continue

            size_grid = base.make_train_size_grid(
                n_available=len(pool),
                min_train_size=args.min_train_size,
                train_size_fracs=list(args.train_size_fracs) if args.train_size_fracs else None,
            )
            print(
                f"Seed {seed}, threshold >= {train_threshold}: eligible={len(pool)} "
                f"(leftover_HQ={len(leftover_hq_pool)}, lower_quality={len(lower_quality_pool)}), "
                f"sampling={args.train_sampling_mode}, sizes={size_grid}"
            )

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
                train_hq_count = int((train_raw[base.BARCODE_COLUMN] >= args.train_priority_min_barcodes).sum())
                train_lq_count = int(len(train_raw) - train_hq_count)
                split_hashes = {
                    "train_row_id_hash": hash_row_ids(train_df),
                    "val_row_id_hash": hash_row_ids(val_df),
                    "test_row_id_hash": hash_row_ids(test_df),
                }
                scaler_mean = float(scaler.mean)
                scaler_std = float(scaler.std)

                for setting in settings:
                    for head_idx, head_name in enumerate(base.PRETRAINED_HEADS):
                        for unfreeze_scope in args.unfreeze_scopes:
                            for head_lr in head_lr_grid:
                                for backbone_lr in backbone_lr_grid:
                                    spec = base.ExperimentSpec(
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
                                    cache_path = (
                                        cache_dir
                                        / "runs"
                                        / CACHE_LAYOUT_VERSION
                                        / args.split_strategy
                                        / f"val{split_runner.format_frac_tag(val_frac)}__test{split_runner.format_frac_tag(test_frac)}"
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
                                        training_seed = spec.seed * 1000 + spec.head_idx * 100 + spec.train_threshold * 10
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
                                        test_metrics, pred_df = base.evaluate_single_head_model(model, test_df, scaler, device=device)
                                        train_metrics, _ = base.evaluate_single_head_model(model, train_df, scaler, device=device)
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
                                        "target_column": MODELING_TARGET_COLUMN,
                                        "target_transform": "log10_recomputed_rna_over_dna",
                                        "target_scaler_mean": scaler_mean,
                                        "target_scaler_std": scaler_std,
                                        **split_hashes,
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
                                                for k in (
                                                    "n",
                                                    "mae",
                                                    "rmse",
                                                    "pearson",
                                                    "spearman",
                                                    "r2",
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
                                    hist["target_column"] = MODELING_TARGET_COLUMN
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
            "split_strategy",
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

    aggregate_df = base.aggregate_metric_summary(
        runs_df,
        group_cols=[
            "split_strategy",
            "split_pool",
            "split_val_fraction",
            "split_test_fraction",
            "val_is_fixed_across_seeds",
            "test_is_fixed_across_seeds",
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
            "train_mae",
            "train_rmse",
            "train_pearson",
            "train_spearman",
            "train_r2",
            "train_pearson_sq",
            "train_loss_standardized",
            "val_mae",
            "val_rmse",
            "val_pearson",
            "val_spearman",
            "val_r2",
            "val_pearson_sq",
            "val_loss_standardized",
            "test_mae",
            "test_rmse",
            "test_pearson",
            "test_spearman",
            "test_r2",
            "test_pearson_sq",
            "test_loss_standardized",
            "best_epoch",
            "best_val_loss_standardized",
            "initial_trainable_params",
            "final_trainable_params",
        ],
    )

    scope_summary_df = base.aggregate_metric_summary(
        runs_df,
        group_cols=[
            "split_strategy",
            "split_pool",
            "split_val_fraction",
            "split_test_fraction",
            "val_is_fixed_across_seeds",
            "test_is_fixed_across_seeds",
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
            "test_pearson_sq",
            "test_loss_standardized",
            "initial_trainable_params",
            "final_trainable_params",
        ],
    )

    if zero_shot_fixed_df is not None:
        zero_shot_fixed_df.to_csv(args.outdir / "zero_shot_fixed_test.csv", index=False)
    elif zero_shot_seed_records:
        pd.concat(zero_shot_seed_records, ignore_index=True).to_csv(args.outdir / "zero_shot_by_seed.csv", index=False)

    runs_df.to_csv(args.outdir / "learning_curve_runs.csv", index=False)
    history_df.to_csv(args.outdir / "learning_curve_histories.csv", index=False)
    aggregate_df.to_csv(args.outdir / "learning_curve_summary_mean_std.csv", index=False)
    scope_summary_df.to_csv(args.outdir / "unfreeze_scope_summary_mean_std.csv", index=False)

    print("\nWrote outputs:")
    output_paths = []
    if zero_shot_fixed_df is not None:
        output_paths.append(args.outdir / "zero_shot_fixed_test.csv")
    elif zero_shot_seed_records:
        output_paths.append(args.outdir / "zero_shot_by_seed.csv")
    output_paths.extend(
        [
            args.outdir / "learning_curve_runs.csv",
            args.outdir / "learning_curve_histories.csv",
            args.outdir / "learning_curve_summary_mean_std.csv",
            args.outdir / "unfreeze_scope_summary_mean_std.csv",
            args.outdir / "run_manifest.json",
        ]
    )
    for path in output_paths:
        print(f"  {path}")

    elapsed_seconds = time.time() - overall_start_time
    elapsed_minutes = elapsed_seconds / 60.0
    print(f"\nTotal runtime: {elapsed_seconds:.1f} seconds ({elapsed_minutes:.2f} minutes)")


if __name__ == "__main__":
    main()
