#!/usr/bin/env python3
"""Curate decision tables from the frozen Lib1 dedup Stage 2 analyzer outputs.

The contract analyzer remains the source of truth for OOF, paired-RC, and
Intron sensitivity metrics.  This module adds presentation-layer fold
summaries and a reviewable (not launch-authorizing) Stage 3 shortlist.  It
never constructs a DataModule and never reads or scores the frozen audit set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LEARN_ROOT = REPO_ROOT / "src" / "learn"
DEFAULT_ANALYSIS_DIR = (
    LEARN_ROOT / "outputs/analysis/lib1_dedup_stage2_july2026"
)
DEFAULT_MANIFEST = (
    LEARN_ROOT
    / "outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ANALYSIS_DIR / "reporting"
DEFAULT_REGISTRY = LEARN_ROOT / "run_registry/runs.csv"
DEFAULT_WANDB_DIR = LEARN_ROOT / "wandb"

ARM_KEYS = ["analysis_lane", "part_slug", "base_config_id", "rc_mode"]
CONFIG_KEYS = ["analysis_lane", "part_slug", "base_config_id"]
PART_ORDER = ["enhancer", "promoter", "intron", "utr3", "utr5"]


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_analyzer_tables(root: str | Path = DEFAULT_ANALYSIS_DIR) -> dict[str, pd.DataFrame]:
    root = Path(root)
    tables = {
        "oof": pd.read_csv(root / "stage2_oof_metrics.csv"),
        "fold": pd.read_csv(root / "stage2_oof_fold_metrics.csv"),
        "rc": pd.read_csv(root / "stage2_rc_pair_metrics.csv"),
        "rc_fold": pd.read_csv(root / "stage2_rc_fold_pair_metrics.csv"),
        "intron": pd.read_csv(root / "stage2_intron_sensitivity_stratum_metrics.csv"),
        "intron_baseline": pd.read_csv(
            root / "stage2_intron_stratum_mean_baselines.csv"
        ),
    }
    if len(tables["oof"]) != 132 or len(tables["fold"]) != 660:
        raise ValueError("Stage 2 analyzer tables are not the complete 132-arm/660-fold product.")
    if len(tables["rc"]) != 66 or len(tables["rc_fold"]) != 330:
        raise ValueError("Stage 2 paired-RC tables are incomplete.")
    return tables


def arm_decision_table(oof: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Join pooled OOF metrics to the predeclared five-fold stability summaries."""
    grouped = folds.groupby(ARM_KEYS, sort=False, dropna=False)
    fold_summary = grouped["fold_pearson"].agg(
        fold_pearson_mean="mean",
        fold_pearson_sd="std",
        fold_pearson_min="min",
        fold_pearson_median="median",
    ).reset_index()
    fold_summary["fold_pearson_se"] = fold_summary["fold_pearson_sd"] / np.sqrt(5.0)
    fold_summary["fold_pearson_p20"] = grouped["fold_pearson"].quantile(0.20).to_numpy()
    fold_summary["positive_fold_count"] = grouped["fold_pearson"].apply(
        lambda values: int((values > 0).sum())
    ).to_numpy()
    result = oof.merge(fold_summary, on=ARM_KEYS, how="left", validate="one_to_one")
    if result["fold_pearson_mean"].isna().any():
        raise ValueError("A pooled OOF arm lacks its five-fold stability summary.")
    return result


def strict_rc_table(rc: pd.DataFrame, rc_fold: pd.DataFrame) -> pd.DataFrame:
    """Materialize the binding Pearson rule and the Intron robustness guard.

    ``no material RMSE/COD degradation`` does not yet have a numeric tolerance
    in the protocol.  Therefore the zero-tolerance guard is emitted separately
    and is never mislabeled as the only valid interpretation of materiality.
    """
    keys = CONFIG_KEYS
    grouped = rc_fold.groupby(keys, sort=False, dropna=False)
    evidence = grouped["delta_rc_on_minus_off_pooled_pearson"].agg(
        strict_mean_fold_delta_pearson="mean",
        strict_negative_fold_count=lambda values: int((values < 0).sum()),
        strict_positive_fold_count=lambda values: int((values > 0).sum()),
    ).reset_index()
    evidence["strict_pearson_fold_gate"] = (
        evidence["strict_mean_fold_delta_pearson"].ge(0.005)
        & evidence["strict_positive_fold_count"].ge(4)
    )
    result = rc.merge(evidence, on=keys, how="left", validate="one_to_one")
    result["zero_tolerance_rmse_cod_guard"] = (
        result["delta_rc_on_minus_off_pooled_oof_rmse"].le(0)
        & result["delta_rc_on_minus_off_pooled_oof_cod_r2"].ge(0)
    )
    result["strict_gate_with_zero_tolerance_error_guard"] = (
        result["strict_pearson_fold_gate"]
        & result["zero_tolerance_rmse_cod_guard"]
    )
    result["strict_intron_within_guard"] = True
    is_intron = result["part_slug"].eq("intron")
    result.loc[is_intron, "strict_intron_within_guard"] = (
        result.loc[
            is_intron,
            "mean_fold_delta_rc_on_minus_off_within_stratum_centered_pearson",
        ].ge(0)
        & result.loc[
            is_intron,
            "negative_fold_count_rc_on_minus_off_within_stratum_centered_pearson",
        ].le(2)
    )
    result["strict_part_specific_fold_gate"] = (
        result["strict_pearson_fold_gate"] & result["strict_intron_within_guard"]
    )
    return result


def lane_summary(arms: pd.DataFrame) -> pd.DataFrame:
    columns = ["pooled_oof_pearson", "pooled_oof_rmse", "pooled_oof_cod_r2"]
    rows = []
    for (part, lane, architecture), frame in arms.groupby(
        ["part_slug", "analysis_lane", "architecture"], sort=False, dropna=False
    ):
        row: dict[str, object] = {
            "part_slug": part,
            "analysis_lane": lane,
            "architecture": architecture,
            "n_arms": len(frame),
            "n_configs": frame["base_config_id"].nunique(),
        }
        for column in columns:
            stem = column.replace("pooled_oof_", "")
            row[f"{stem}_median"] = float(frame[column].median())
            row[f"{stem}_min"] = float(frame[column].min())
            row[f"{stem}_max"] = float(frame[column].max())
        rows.append(row)
    result = pd.DataFrame(rows)
    result["part_order"] = result["part_slug"].map(
        {part: index for index, part in enumerate(PART_ORDER)}
    )
    return result.sort_values(["part_order", "analysis_lane", "architecture"]).drop(
        columns="part_order"
    )


def manifest_hyperparameters(manifest_rows: Iterable[Mapping]) -> pd.DataFrame:
    """Flatten the trainer/optimizer fields needed for boundary diagnostics."""
    fields = [
        "lr",
        "backbone_lr",
        "head_lr",
        "weight_decay",
        "batch_size",
        "max_epochs",
        "min_epochs",
        "stopping_patience",
        "dropout_p",
        "linear_dropout_p",
    ]
    records = {}
    for row in manifest_rows:
        key = tuple(row[name] for name in CONFIG_KEYS)
        if key in records:
            continue
        identity = row["base_identity"]
        records[key] = {
            **{name: row.get(name, "") for name in CONFIG_KEYS},
            "architecture": row.get("architecture", ""),
            "policy_id": row.get("policy_id", ""),
            "source_head": row.get("source_head", ""),
            "unfreeze_scope": row.get("unfreeze_scope", ""),
            "input_policy": row.get("input_policy", ""),
            **{field: identity.get(field) for field in fields},
        }
    return pd.DataFrame(records.values())


def stage3_selection_review(arms: pd.DataFrame) -> pd.DataFrame:
    """Build a reviewable top-five table without authorizing a launch."""
    ordered = arms.sort_values(
        [*CONFIG_KEYS, "pooled_oof_pearson", "pooled_oof_rmse"],
        ascending=[True, True, True, False, True],
    )
    best = ordered.groupby(CONFIG_KEYS, sort=False, as_index=False).head(1).copy()
    best["pooled_rank_within_part"] = best.groupby("part_slug")[
        "pooled_oof_pearson"
    ].rank(method="first", ascending=False).astype(int)
    best["pure_pooled_top5"] = best["pooled_rank_within_part"].le(5)
    best["recommended_stage3_slot"] = best["pure_pooled_top5"]
    best["selection_reason"] = np.where(
        best["pure_pooled_top5"], "pure_pooled_top5", "not_selected"
    )

    # The 5'UTR protocol explicitly preserves an architecture-diverse option.
    # Replace numerical rank five with the best ResNet1D only in the recommended
    # review column; retain both rows and the pure ranking for transparency.
    utr5 = best.loc[best["part_slug"].eq("utr5")]
    resnet = utr5.loc[utr5["architecture"].eq("ResNet1DRegressor")]
    if not resnet.empty:
        resnet_index = resnet["pooled_oof_pearson"].idxmax()
        if not bool(best.loc[resnet_index, "pure_pooled_top5"]):
            fifth = utr5.loc[utr5["pure_pooled_top5"]].sort_values(
                "pooled_oof_pearson"
            ).index[0]
            best.loc[fifth, "recommended_stage3_slot"] = False
            best.loc[fifth, "selection_reason"] = "pure_rank5_diversity_alternate"
            best.loc[resnet_index, "recommended_stage3_slot"] = True
            best.loc[resnet_index, "selection_reason"] = (
                "architecture_diverse_resnet_recommended"
            )

    best["freeze_status"] = "ready_for_review_not_launch_authorized"
    best.loc[best["part_slug"].eq("utr3"), "freeze_status"] = (
        "hold_pending_approved_targeted_hpo_results"
    )
    best["base_config_prefix"] = best["base_config_id"].str.replace(
        "basecfg_", "", regex=False
    ).str[:8]
    best["recommended_rank"] = np.nan
    selected = best.loc[best["recommended_stage3_slot"]].copy()
    selected["recommended_rank"] = selected.groupby("part_slug")[
        "pooled_oof_pearson"
    ].rank(method="first", ascending=False)
    best.loc[selected.index, "recommended_rank"] = selected["recommended_rank"]
    return best.sort_values(
        ["part_slug", "recommended_stage3_slot", "pooled_rank_within_part"],
        ascending=[True, False, True],
    )


def _last_non_null(values: pd.Series) -> float:
    finite = values.dropna()
    return finite.iloc[-1] if len(finite) else np.nan


def export_learning_histories(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    registry_path: str | Path = DEFAULT_REGISTRY,
    wandb_dir: str | Path = DEFAULT_WANDB_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the 660 local W&B files into a compact epoch-level evidence table.

    This must run in ``boda_env`` because it uses W&B's protobuf reader.  It is
    deliberately local-only: no cloud query and no audit loader are involved.
    """
    from src.analysis.lib1_dedup_stage2_analysis import (
        resolve_analysis_cells,
        validate_analysis_manifest,
    )
    from src.learn.export_wandb_history import read_wandb_file

    manifest = _read_jsonl(manifest_path)
    validate_analysis_manifest(manifest)
    resolved = resolve_analysis_cells(manifest, registry_path)
    incomplete = [row for row in resolved if row["availability"] != "complete"]
    if incomplete:
        raise RuntimeError("Learning histories require all 660 resolved cells.")

    wanted = {str(row["resolved_run_id"]) for row in resolved}
    run_files = {}
    for path in Path(wandb_dir).glob("*run-*/run-*.wandb"):
        run_id = path.stem.replace("run-", "", 1)
        if run_id in wanted:
            run_files[run_id] = path
    missing = wanted - set(run_files)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} local W&B files; examples: {sorted(missing)[:5]}"
        )

    metric_columns = [
        "epoch",
        "trainer/global_step",
        "train_loss",
        "train_mse",
        "train_pearson",
        "train_spearman",
        "train_cod_r2",
        "val_loss",
        "val_mse",
        "val_pearson",
        "val_spearman",
        "val_cod_r2",
        "_runtime",
    ]
    histories = []
    summaries = []
    for index, row in enumerate(resolved):
        run_id = str(row["resolved_run_id"])
        try:
            local_metadata, _config, raw_rows, raw_columns = read_wandb_file(
                run_files[run_id], tolerate_truncated_tail=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse local W&B history for cell {row['cell_id']} "
                f"run {run_id}: {exc}"
            )
        raw = pd.DataFrame(raw_rows)
        lr_columns = sorted(column for column in raw_columns if column.startswith("lr-"))
        keep = [column for column in metric_columns + lr_columns if column in raw]
        history = raw[keep].copy()
        for column in keep:
            history[column] = pd.to_numeric(history[column], errors="coerce")
        history = history.loc[history.get("epoch", pd.Series(dtype=float)).notna()]
        if history.empty or "val_pearson" not in history:
            raise ValueError(f"Run {run_id} has no epoch-level validation history.")
        history["epoch"] = history["epoch"].astype(int)
        aggregations = {
            column: _last_non_null for column in history.columns if column != "epoch"
        }
        history = history.groupby("epoch", as_index=False).agg(aggregations)
        history = history.loc[
            history[[column for column in ["val_pearson", "train_mse"] if column in history]]
            .notna()
            .any(axis=1)
        ].copy()
        if history.empty:
            raise ValueError(f"Run {run_id} has no complete metric epochs.")

        metadata = {
            "analysis_cell": int(row["analysis_cell"]),
            "cell_id": row["cell_id"],
            "resolved_run_id": run_id,
            "analysis_lane": row["analysis_lane"],
            "part_slug": row["part_slug"],
            "base_config_id": row["base_config_id"],
            "architecture": row["architecture"],
            "policy_id": row["policy_id"],
            "source_head": row.get("source_head", ""),
            "unfreeze_scope": row.get("unfreeze_scope", ""),
            "rc_mode": row["rc_mode"],
            "development_fold": int(row["development_fold"]),
            "execution_disposition": row["execution_disposition"],
            "max_epochs_configured": int(row["base_identity"]["max_epochs"]),
        }
        for column, value in metadata.items():
            history[column] = value
        histories.append(history)

        valid_val = history.loc[history["val_pearson"].notna()]
        best_index = valid_val["val_pearson"].idxmax()
        best = history.loc[best_index]
        final = history.sort_values("epoch").iloc[-1]
        summaries.append(
            {
                **metadata,
                "history_epoch_rows": len(history),
                "history_best_epoch": int(best["epoch"]),
                "history_best_val_pearson": float(best["val_pearson"]),
                "train_mse_at_best_val_epoch": float(best.get("train_mse", np.nan)),
                "train_pearson_at_best_val_epoch": float(
                    best.get("train_pearson", np.nan)
                ),
                "stopping_epoch": int(final["epoch"]),
                "final_train_mse": float(final.get("train_mse", np.nan)),
                "minimum_logged_train_mse": float(history["train_mse"].min()),
                "runtime_seconds": float(history.get("_runtime", pd.Series([np.nan])).max()),
                "local_history_scan_warning": str(
                    local_metadata.get("history_scan_warning", "")
                ),
                "hit_configured_epoch_cap": bool(
                    int(final["epoch"]) >= int(row["base_identity"]["max_epochs"]) - 1
                ),
            }
        )

    history_frame = pd.concat(histories, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    if len(summary_frame) != 660 or summary_frame["cell_id"].nunique() != 660:
        raise ValueError("Learning-history export did not preserve all 660 cells.")
    return history_frame, summary_frame


def learning_diagnostic_tables(
    history_summary: pd.DataFrame, hyperparameters: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize convergence and the predeclared 3'UTR optimizer-boundary gate."""
    from scipy.stats import spearmanr

    route = (
        history_summary.groupby(
            ["part_slug", "analysis_lane", "architecture", "rc_mode"],
            dropna=False,
        )
        .agg(
            cells=("cell_id", "size"),
            median_best_epoch=("history_best_epoch", "median"),
            median_train_mse_at_best=("train_mse_at_best_val_epoch", "median"),
            train_mse_ge_0p90=(
                "train_mse_at_best_val_epoch",
                lambda values: int((values >= 0.90).sum()),
            ),
            median_stopping_epoch=("stopping_epoch", "median"),
            configured_epoch_cap_hits=("hit_configured_epoch_cap", "sum"),
            median_runtime_seconds=("runtime_seconds", "median"),
        )
        .reset_index()
    )

    joined = history_summary.merge(
        hyperparameters[CONFIG_KEYS + ["lr", "weight_decay"]],
        on=CONFIG_KEYS,
        how="left",
        validate="many_to_one",
    )
    utr3 = joined.loc[
        joined["analysis_lane"].eq("utr3_utrbasset_challenger")
    ].copy()
    utr3["log10_lr"] = np.log10(utr3["lr"].astype(float))
    rows = []
    for rc_mode, frame in utr3.groupby("rc_mode", sort=True):
        lr_val = spearmanr(frame["log10_lr"], frame["history_best_val_pearson"])
        lr_train = spearmanr(
            frame["log10_lr"], frame["train_mse_at_best_val_epoch"]
        )
        train_val = spearmanr(
            frame["train_mse_at_best_val_epoch"],
            frame["history_best_val_pearson"],
        )
        per_fold = []
        for _fold, fold_frame in frame.groupby("development_fold"):
            per_fold.append(
                float(
                    spearmanr(
                        fold_frame["log10_lr"],
                        fold_frame["history_best_val_pearson"],
                    ).correlation
                )
            )
        rows.append(
            {
                "part_slug": "utr3",
                "analysis_lane": "utr3_utrbasset_challenger",
                "rc_mode": rc_mode,
                "n_config_fold_cells": len(frame),
                "spearman_log10_lr_vs_best_val_pearson": float(lr_val.correlation),
                "unadjusted_p_log10_lr_vs_best_val_pearson": float(lr_val.pvalue),
                "spearman_log10_lr_vs_train_mse_at_best": float(
                    lr_train.correlation
                ),
                "spearman_train_mse_at_best_vs_best_val_pearson": float(
                    train_val.correlation
                ),
                "positive_per_fold_lr_val_association_count": int(
                    sum(value > 0 for value in per_fold)
                ),
                "per_fold_lr_val_spearman_min": float(min(per_fold)),
                "per_fold_lr_val_spearman_max": float(max(per_fold)),
                "tested_lr_min": float(frame["lr"].min()),
                "tested_lr_max": float(frame["lr"].max()),
            }
        )
    return route, pd.DataFrame(rows)


def write_reporting_tables(
    analysis_dir: str | Path = DEFAULT_ANALYSIS_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    export_histories: bool = False,
    registry_path: str | Path = DEFAULT_REGISTRY,
    wandb_dir: str | Path = DEFAULT_WANDB_DIR,
) -> dict[str, int | str | bool]:
    tables = load_analyzer_tables(analysis_dir)
    manifest = _read_jsonl(manifest_path)
    arms = arm_decision_table(tables["oof"], tables["fold"])
    rc = strict_rc_table(tables["rc"], tables["rc_fold"])
    lanes = lane_summary(arms)
    hyperparameters = manifest_hyperparameters(manifest)
    hyperparameter_fields = [
        column
        for column in hyperparameters.columns
        if column in CONFIG_KEYS
        or column
        not in {
            "architecture",
            "policy_id",
            "source_head",
            "unfreeze_scope",
            "input_policy",
        }
    ]
    selection = stage3_selection_review(arms).merge(
        hyperparameters[hyperparameter_fields],
        on=CONFIG_KEYS,
        how="left",
        validate="one_to_one",
    )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    products = {
        "stage2_arm_decision_metrics.csv": arms,
        "stage2_strict_rc_review.csv": rc,
        "stage2_lane_summary.csv": lanes,
        "stage2_manifest_hyperparameters.csv": hyperparameters,
        "stage3_selection_review.csv": selection,
    }
    for name, frame in products.items():
        frame.to_csv(root / name, index=False)

    history_summary_path = root / "stage2_learning_history_summary.csv"
    history_cells = 0
    if export_histories:
        histories, history_summary = export_learning_histories(
            manifest_path=manifest_path,
            registry_path=registry_path,
            wandb_dir=wandb_dir,
        )
        histories.to_csv(
            root / "stage2_learning_histories.tsv.gz",
            index=False,
            sep="\t",
            compression="gzip",
        )
        history_summary.to_csv(
            history_summary_path, index=False
        )
        route_diagnostics, boundary_diagnostics = learning_diagnostic_tables(
            history_summary, hyperparameters
        )
        route_diagnostics.to_csv(
            root / "stage2_learning_route_summary.csv", index=False
        )
        boundary_diagnostics.to_csv(
            root / "stage2_utr3_optimizer_boundary_diagnostics.csv", index=False
        )
        history_cells = len(history_summary)
    elif history_summary_path.is_file():
        history_summary = pd.read_csv(history_summary_path)
        route_diagnostics, boundary_diagnostics = learning_diagnostic_tables(
            history_summary, hyperparameters
        )
        route_diagnostics.to_csv(
            root / "stage2_learning_route_summary.csv", index=False
        )
        boundary_diagnostics.to_csv(
            root / "stage2_utr3_optimizer_boundary_diagnostics.csv", index=False
        )
        history_cells = len(history_summary)

    recommended = selection.loc[selection["recommended_stage3_slot"]]
    summary: dict[str, int | str | bool] = {
        "analysis_arms": len(arms),
        "paired_rc_configs": len(rc),
        "recommended_stage3_slots": len(recommended),
        "recommended_slots_per_part": int(
            recommended.groupby("part_slug").size().min()
        ),
        "utr3_selection_frozen": False,
        "utr3_targeted_hpo_decision": "approved_protocol_not_frozen",
        "learning_history_cells": history_cells,
        "audit_loader_instantiated": False,
        "output_dir": str(root.resolve()),
    }
    (root / "stage2_reporting_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--wandb-dir", default=str(DEFAULT_WANDB_DIR))
    parser.add_argument(
        "--export-histories",
        action="store_true",
        help="Parse all 660 local W&B files (run this in boda_env).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = write_reporting_tables(
        analysis_dir=args.analysis_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        export_histories=args.export_histories,
        registry_path=args.registry,
        wandb_dir=args.wandb_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
