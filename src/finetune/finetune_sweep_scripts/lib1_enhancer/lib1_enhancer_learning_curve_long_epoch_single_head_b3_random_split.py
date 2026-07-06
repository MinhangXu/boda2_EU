#!/usr/bin/env python3
"""Long-epoch single-head rerun for lib1 enhancer transfer learning.

This runner is a narrowed variant of
`lib1_enhancer_learning_curve_finetune_split_options.py` meant for a focused
follow-up experiment:

- one pretrained initialization head (`--init_head`)
- random train/val/test splits per seed (`random_all_per_seed`)
- B3 weighted setting only, with default `b_cap = 8`
- three unfreeze scopes: `branched_only`, `conv3_plus`, `full`
- one fixed LR pair instead of an LR sweep
- longer training window (`max_epochs = 800`, `patience = 300`)
- per-epoch test metrics are logged for visualization only; model selection still
  uses validation loss only

The default LR pair is intentionally not swept here:

- `head_lr = 1e-4`
- `backbone_lr = 2e-5`

Those defaults are taken as a pragmatic single-pair choice from the stronger
targeted April configurations so this rerun can focus on the long-epoch
question rather than repeat HPO.
"""

from __future__ import annotations

import argparse
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
    "/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/single_part_variant_level/enhancers/"
    "20251218_np_fastq1_500000NPreads_enh_variants_bc_sum_avg_expression.txt"
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
    / "lib1_enhancer_long_epoch_single_head_b3_random_split_apr2026_800epochs"
)

PRETRAINED_HEADS = ["K562", "HepG2", "SKNSH"]
DEFAULT_SEEDS = [17, 23, 19, 31]
DEFAULT_TRAIN_SIZE_FRACS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
DEFAULT_UNFREEZE_SCOPES = ["branched_only", "conv3_plus", "full"]

DEFAULT_B3_BCAP = 8.0
DEFAULT_HEAD_LR = 1e-4
DEFAULT_BACKBONE_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MIN_WEIGHT = 0.1
DEFAULT_SPLIT_SEED = 7
DEFAULT_VAL_FRAC = 0.15
DEFAULT_TEST_FRAC = 0.15
DEFAULT_TRAIN_THRESHOLDS = [1]
DEFAULT_TRAIN_PRIORITY_MIN_BARCODES = 4
DEFAULT_MIN_TRAIN_SIZE = 32
DEFAULT_FROZEN_EPOCHS = 2
DEFAULT_TRAIN_BATCH_SIZE = 128
DEFAULT_MAX_EPOCHS = 800
DEFAULT_PATIENCE = 300

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
        base = _load_module("lib1_learning_curve_base_long_epoch", BASE_SCRIPT_PATH)
    return base


def get_split_runner():
    global split_runner
    if split_runner is None:
        split_runner = _load_module("lib1_learning_curve_split_runner_long_epoch", SPLIT_SCRIPT_PATH)
    return split_runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Long-epoch single-head learning-curve rerun with random per-seed "
            "splits and the B3 weighted setting only."
        )
    )
    parser.add_argument("--data_path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--init_head",
        type=str,
        required=True,
        choices=PRETRAINED_HEADS,
        help="Single pretrained head to use for this focused rerun.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--val_frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--test_frac", type=float, default=DEFAULT_TEST_FRAC)
    parser.add_argument("--train_thresholds", nargs="+", type=int, default=DEFAULT_TRAIN_THRESHOLDS)
    parser.add_argument(
        "--train_priority_min_barcodes",
        type=int,
        default=DEFAULT_TRAIN_PRIORITY_MIN_BARCODES,
        help="Threshold used to define HQ rows for bookkeeping and `hq_first` sampling; retained for compatibility.",
    )
    parser.add_argument("--train_size_fracs", nargs="*", type=float, default=DEFAULT_TRAIN_SIZE_FRACS)
    parser.add_argument(
        "--train_sampling_mode",
        type=str,
        default="random",
        choices=["hq_first", "random"],
        help="How to grow the training set as train_size increases.",
    )
    parser.add_argument("--min_train_size", type=int, default=DEFAULT_MIN_TRAIN_SIZE)
    parser.add_argument(
        "--unfreeze_scopes",
        nargs="+",
        type=str,
        default=DEFAULT_UNFREEZE_SCOPES,
        choices=["head_only", "branched_only", "linear_all_head", "conv3_plus", "full"],
    )
    parser.add_argument("--b3_bcap", type=float, default=DEFAULT_B3_BCAP)
    parser.add_argument("--min_weight", type=float, default=DEFAULT_MIN_WEIGHT)
    parser.add_argument(
        "--head_lr",
        type=float,
        default=DEFAULT_HEAD_LR,
        help="Single head LR used for this rerun; default picked from prior targeted results.",
    )
    parser.add_argument(
        "--backbone_lr",
        type=float,
        default=DEFAULT_BACKBONE_LR,
        help="Single backbone LR used for this rerun; default picked from prior targeted results.",
    )
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
    split_runner = get_split_runner()

    if not args.data_path.exists():
        raise FileNotFoundError(args.data_path)
    if not args.model_path.exists():
        raise FileNotFoundError(args.model_path)

    split_runner.validate_split_fracs(val_frac=args.val_frac, test_frac=args.test_frac)

    device = base.resolve_device(args.device)
    args.outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.outdir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    split_cache_dir = cache_dir / "splits"
    split_cache_dir.mkdir(parents=True, exist_ok=True)

    settings = base.build_settings(
        include_b1=False,
        include_b2=False,
        include_b3=True,
        b3_bcaps=[args.b3_bcap],
        min_weight=args.min_weight,
    )
    if len(settings) != 1:
        raise RuntimeError(f"Expected exactly one B3 setting, got {len(settings)}")
    setting = settings[0]

    selected_head_idx = PRETRAINED_HEADS.index(args.init_head)
    split_strategy = "random_all_per_seed"
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"

    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest["repo_root"] = str(base.REPO_ROOT)
    manifest["device_resolved"] = device
    manifest["run_id"] = run_id
    manifest["run_started_at"] = run_started_at
    manifest["settings"] = [asdict(x) for x in settings]
    manifest["split_strategy"] = split_strategy
    manifest["split_strategy_description"] = split_runner.SPLIT_STRATEGY_HELP[split_strategy]
    manifest["selected_head_idx"] = selected_head_idx
    manifest["per_epoch_test_metrics_logged"] = True
    manifest["lr_selection_note"] = (
        "Single LR pair used to avoid another sweep. Defaults (`head_lr=1e-4`, "
        "`backbone_lr=2e-5`) were chosen as a more conservative long-epoch "
        "follow-up from the stronger targeted April results."
    )
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Repo root: {base.REPO_ROOT}")
    print(f"Using device: {device}")
    print(f"Output dir: {args.outdir}")
    print(f"Split strategy: {split_strategy}")
    print(f"Selected init head: {args.init_head} (head_idx={selected_head_idx})")
    print(f"Validation fraction: {args.val_frac}")
    print(f"Test fraction: {args.test_frac}")
    print(f"Head LR: {args.head_lr}")
    print(f"Backbone LR: {args.backbone_lr}")
    print(f"B3 b_cap: {args.b3_bcap}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Patience: {args.patience}")

    clean_df = base.load_clean_df(args.data_path)
    checkpoint = base.load_checkpoint_from_tar(args.model_path, map_location="cpu")

    run_records: list[dict[str, Any]] = []
    history_records: list[pd.DataFrame] = []
    zero_shot_seed_records: list[pd.DataFrame] = []

    for seed in args.seeds:
        seed_key = f"seed{seed}__splitseed{args.split_seed}"
        split_cache_path = split_cache_dir / split_runner.split_cache_name(
            split_strategy,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            test_min_barcodes=args.train_priority_min_barcodes,
            seed_key=seed_key,
        )
        split_payload = base.maybe_load_or_build(
            split_cache_path,
            force=args.force,
            builder=lambda seed=seed: split_runner.build_split_payload(
                clean_df,
                strategy=split_strategy,
                val_frac=args.val_frac,
                test_frac=args.test_frac,
                base_split_seed=args.split_seed,
                run_seed=seed,
                test_min_barcodes=args.train_priority_min_barcodes,
            ),
        )

        train_rest_df = split_payload["train_rest_df"]
        val_df_raw = split_payload["val_df"]
        test_df_raw = split_payload["test_df"]

        print(
            f"Seed {seed}: split_strategy={split_strategy}, "
            f"train_rest={len(train_rest_df)}, val={len(val_df_raw)}, test={len(test_df_raw)}, "
            f"split_seed_effective={split_payload['split_seed_effective']}"
        )

        zero_shot_df = base.run_zero_shot_eval_on_fixed_test(
            checkpoint,
            split_payload["test_padded"],
            device=device,
        )
        zero_shot_df = zero_shot_df.loc[zero_shot_df["init_head"] == args.init_head].copy()
        zero_shot_df["seed"] = seed
        zero_shot_df["split_strategy"] = split_strategy
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

                for unfreeze_scope in args.unfreeze_scopes:
                    spec = base.ExperimentSpec(
                        seed=seed,
                        head_idx=selected_head_idx,
                        init_head=args.init_head,
                        setting_name=setting.name,
                        train_threshold=train_threshold,
                        train_size=len(train_df),
                        train_fraction=len(train_df) / len(pool),
                        unfreeze_scope=unfreeze_scope,
                        train_sampling_mode=args.train_sampling_mode,
                        head_lr=float(args.head_lr),
                        backbone_lr=float(args.backbone_lr),
                    )
                    cache_path = (
                        cache_dir
                        / "runs"
                        / split_strategy
                        / f"val{split_runner.format_frac_tag(args.val_frac)}__test{split_runner.format_frac_tag(args.test_frac)}"
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
                        "split_strategy": split_strategy,
                        "split_pool": split_payload["split_pool"],
                        "split_seed_base": args.split_seed,
                        "split_seed_effective": split_payload["split_seed_effective"],
                        "val_seed_effective": split_payload["val_seed_effective"],
                        "test_seed_effective": split_payload["test_seed_effective"],
                        "val_is_fixed_across_seeds": split_payload["val_is_fixed_across_seeds"],
                        "test_is_fixed_across_seeds": split_payload["test_is_fixed_across_seeds"],
                        "split_val_fraction": args.val_frac,
                        "split_test_fraction": args.test_frac,
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
                    hist["split_strategy"] = split_strategy
                    hist["split_seed_effective"] = split_payload["split_seed_effective"]
                    hist["val_seed_effective"] = split_payload["val_seed_effective"]
                    hist["test_seed_effective"] = split_payload["test_seed_effective"]
                    hist["val_is_fixed_across_seeds"] = split_payload["val_is_fixed_across_seeds"]
                    hist["test_is_fixed_across_seeds"] = split_payload["test_is_fixed_across_seeds"]
                    hist["split_val_fraction"] = args.val_frac
                    hist["split_test_fraction"] = args.test_frac
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

    pd.concat(zero_shot_seed_records, ignore_index=True).to_csv(args.outdir / "zero_shot_by_seed.csv", index=False)
    runs_df.to_csv(args.outdir / "learning_curve_runs.csv", index=False)
    history_df.to_csv(args.outdir / "learning_curve_histories.csv", index=False)
    aggregate_df.to_csv(args.outdir / "learning_curve_summary_mean_std.csv", index=False)
    scope_summary_df.to_csv(args.outdir / "unfreeze_scope_summary_mean_std.csv", index=False)

    print("\nWrote outputs:")
    for path in [
        args.outdir / "zero_shot_by_seed.csv",
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
