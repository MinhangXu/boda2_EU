#!/usr/bin/env python3
"""Generate follow-up TAC figures requested after the first Lib1 deck review.

All final-test plotting is reporting-only and reads the already-frozen ensemble
predictions.  No model is selected, fit, recalibrated, or rescored here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.lib1_dedup_stage1_analysis import (
    assign_inferred_intron_subsets,
    load_predictions,
    load_stage1_results,
)
from src.analysis.lib1_dedup_tac_presentation_figures import (
    FINAL_DIR,
    FIGURE_DIR,
    OUTPUT_DIR,
    PART_COLOR,
    PART_LABEL,
    PART_ORDER,
    REPO_ROOT,
    STAGE2_DIR,
    STAGE3_DIR,
    TABLE_DIR,
    configure_style,
    deterministic_jitter,
    save_figure,
)


SPLIT_CATALOG = (
    REPO_ROOT / "src/learn/data_manifests/lib1_dedup_exact_v1_split_manifests.json"
)
SELECTED_EPOCH_HISTORY_PATH = TABLE_DIR / "selected_policy_epoch_histories.tsv"
FIT_COLOR = "#D55E00"
FOLD_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#6B6ECF"]
HIGH_SUPPORT_CONTROL_CUTOFF = 2_000
BARCODE_DISPLAY_MAX = 64


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def load_all_construct_support_table() -> pd.DataFrame:
    """Join frozen split assignments to all raw construct targets."""

    catalog = json.loads(SPLIT_CATALOG.read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    for part in PART_ORDER:
        info = catalog["parts"][part]
        split_path = Path(info["manifest_path"])
        dataset_path = Path(info["dataset_path"])
        assignments = pd.DataFrame(
            json.loads(split_path.read_text(encoding="utf-8"))["assignments"]
        )[
            [
                "construct_id",
                "partition",
                "development_fold",
                "n_barcodes",
            ]
        ]
        dataset = pd.read_csv(
            dataset_path,
            sep="\t",
            usecols=["construct_id", "log2_RNA_DNA", "n_barcodes"],
        )
        if len(assignments) != len(dataset):
            raise AssertionError(f"Split and dataset row counts differ for {part}")
        joined = assignments.merge(
            dataset,
            on="construct_id",
            how="left",
            validate="one_to_one",
            suffixes=("_manifest", "_dataset"),
        )
        if not joined["n_barcodes_manifest"].eq(joined["n_barcodes_dataset"]).all():
            raise AssertionError(f"Barcode support mismatch for {part}")
        if joined["log2_RNA_DNA"].isna().any():
            raise AssertionError(f"Missing construct expression after joining {part}")
        joined["n_barcodes"] = joined["n_barcodes_dataset"].astype(int)
        joined["part_slug"] = part
        joined["split_manifest_path"] = _relative(split_path)
        joined["dataset_path"] = _relative(dataset_path)
        frames.append(
            joined[
                [
                    "part_slug",
                    "construct_id",
                    "partition",
                    "development_fold",
                    "log2_RNA_DNA",
                    "n_barcodes",
                    "split_manifest_path",
                    "dataset_path",
                ]
            ]
        )
    result = pd.concat(frames, ignore_index=True)
    if result["n_barcodes"].min() < 1:
        raise AssertionError("All-construct table contains a row below one barcode")
    return result


def load_development_fold_table() -> pd.DataFrame:
    """Restrict the joined construct table to the five HQ8 development folds."""

    result = load_all_construct_support_table()
    result = result.loc[result["partition"].eq("development")].copy()
    if result["development_fold"].isna().any():
        raise AssertionError("Development-fold table contains a missing fold")
    result["development_fold"] = result["development_fold"].astype(int)
    result["display_fold"] = result["development_fold"] + 1
    if result["n_barcodes"].min() < 8:
        raise AssertionError("Development-fold table contains a construct below HQ8")
    return result


def exclude_high_support_control_rows(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Omit the two >2,000-barcode control-library entries from display."""

    high_support = data.loc[
        data["n_barcodes"].gt(HIGH_SUPPORT_CONTROL_CUTOFF)
    ].copy()
    if (
        len(high_support) != 2
        or high_support["construct_id"].nunique() != 1
        or set(high_support["part_slug"]) != {"promoter", "utr5"}
    ):
        raise AssertionError(
            "Expected one shared high-support control entry in each UTR5 and Promoter"
        )
    control_mask = data["construct_id"].eq(high_support["construct_id"].iloc[0])
    excluded = data.loc[control_mask].copy()
    if set(excluded["n_barcodes"]) != {2466}:
        raise AssertionError("The display-excluded control does not have 2,466 barcodes")
    if not excluded["n_barcodes"].gt(HIGH_SUPPORT_CONTROL_CUTOFF).all():
        raise AssertionError("The display-excluded control is not consistently high support")
    displayed = data.loc[~control_mask].copy()
    if displayed["n_barcodes"].max() > BARCODE_DISPLAY_MAX:
        raise AssertionError(
            "A non-control construct exceeds the fixed barcode display range"
        )
    return displayed, excluded


def figure_development_fold_balance() -> tuple[list[str], pd.DataFrame]:
    """Show expression and barcode-support coverage in all five OOF folds."""

    source_data = load_development_fold_table()
    data, excluded = exclude_high_support_control_rows(source_data)
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(18.5, 6.3),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    x_low = math.floor((float(data["log2_RNA_DNA"].min()) - 0.2) * 2) / 2
    x_high = math.ceil((float(data["log2_RNA_DNA"].max()) + 0.2) * 2) / 2

    summary_rows: list[dict[str, object]] = []
    for panel_index, (ax, part) in enumerate(zip(axes, PART_ORDER)):
        part_df = data.loc[data["part_slug"].eq(part)]
        for display_fold in range(1, 6):
            fold_df = part_df.loc[part_df["display_fold"].eq(display_fold)]
            ax.scatter(
                fold_df["log2_RNA_DNA"],
                fold_df["n_barcodes"],
                s=12,
                color=FOLD_COLORS[display_fold - 1],
                alpha=0.42,
                linewidths=0,
                zorder=2,
            )
            median_x = float(fold_df["log2_RNA_DNA"].median())
            median_y = float(fold_df["n_barcodes"].median())
            ax.scatter(
                [median_x],
                [median_y],
                marker="D",
                s=46,
                facecolor=FOLD_COLORS[display_fold - 1],
                edgecolor="white",
                linewidth=0.9,
                zorder=4,
            )
            summary_rows.append(
                {
                    "part_slug": part,
                    "display_fold": display_fold,
                    "n_constructs": len(fold_df),
                    "median_log2_RNA_DNA": median_x,
                    "median_n_barcodes": median_y,
                    "minimum_n_barcodes": int(fold_df["n_barcodes"].min()),
                    "maximum_n_barcodes": int(fold_df["n_barcodes"].max()),
                    "part_control_rows_excluded_from_display": int(
                        excluded["part_slug"].eq(part).sum()
                    ),
                }
            )

        ax.set_title(f"{PART_LABEL[part]}\nn={len(part_df):,}")
        ax.set_xlim(x_low, x_high)
        ax.set_yscale("log", base=2)
        ax.set_ylim(7.2, BARCODE_DISPLAY_MAX)
        ax.set_yticks([8, 16, 32, 64])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.grid(axis="x", alpha=0.45)
        if panel_index == 0:
            ax.set_ylabel("Distinct barcodes per construct (log₂ scale)")

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7,
            color=FOLD_COLORS[index],
            label=f"Fold {index + 1}",
        )
        for index in range(5)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="#52616B",
            markeredgecolor="white",
            label="Fold median",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=6,
        fontsize=10.5,
    )
    fig.suptitle(
        "Five development folds cover similar expression and barcode-support ranges",
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.91,
        "Each point is one ≥8-barcode construct, colored by the fold in which it was held out",
        ha="center",
        fontsize=12.5,
        color="#425466",
    )
    fig.supxlabel(
        "Construct expression, log₂(total RNA / total DNA)", fontsize=12.5, y=0.145
    )
    fig.text(
        0.5,
        0.018,
        "Balance diagnostic only: folds were hash-assigned, not stratified. Display only: the shared 2,466-barcode control is omitted from Promoter and 5′UTR; it remains in the modeled data.",
        ha="center",
        fontsize=9.8,
        color="#52616B",
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.80, bottom=0.24, wspace=0.08)
    return save_figure(fig, "main_development_fold_expression_barcode_balance"), pd.DataFrame(summary_rows)


def figure_all_construct_expression_barcode_support() -> tuple[list[str], pd.DataFrame]:
    """Show expression and barcode support for every modeled construct by part."""

    source_data = load_all_construct_support_table()
    data, excluded = exclude_high_support_control_rows(source_data)
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(18.5, 6.3),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    x_low = math.floor((float(data["log2_RNA_DNA"].min()) - 0.2) * 2) / 2
    x_high = math.ceil((float(data["log2_RNA_DNA"].max()) + 0.2) * 2) / 2

    summary_rows: list[dict[str, object]] = []
    for panel_index, (ax, part) in enumerate(zip(axes, PART_ORDER)):
        part_df = data.loc[data["part_slug"].eq(part)]
        ax.scatter(
            part_df["log2_RNA_DNA"],
            part_df["n_barcodes"],
            s=9,
            color=PART_COLOR[part],
            alpha=0.24,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        ax.axhline(
            8,
            color="#52616B",
            linewidth=1.15,
            linestyle=(0, (4, 3)),
            zorder=3,
        )
        ax.set_title(f"{PART_LABEL[part]}\nn={len(part_df):,}")
        ax.set_xlim(x_low, x_high)
        ax.set_yscale("log", base=2)
        ax.set_ylim(0.8, BARCODE_DISPLAY_MAX)
        ax.set_yticks([1, 2, 4, 8, 16, 32, 64])
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.grid(axis="x", alpha=0.45)
        if panel_index == 0:
            ax.set_ylabel("Distinct barcodes per construct (log₂ scale)")

        summary_rows.append(
            {
                "part_slug": part,
                "n_constructs_displayed": len(part_df),
                "n_control_rows_excluded_from_display": int(
                    excluded["part_slug"].eq(part).sum()
                ),
                "minimum_log2_RNA_DNA": float(part_df["log2_RNA_DNA"].min()),
                "median_log2_RNA_DNA": float(part_df["log2_RNA_DNA"].median()),
                "maximum_log2_RNA_DNA": float(part_df["log2_RNA_DNA"].max()),
                "minimum_n_barcodes": int(part_df["n_barcodes"].min()),
                "median_n_barcodes": float(part_df["n_barcodes"].median()),
                "maximum_n_barcodes": int(part_df["n_barcodes"].max()),
                "percent_hq8": 100.0 * float(part_df["n_barcodes"].ge(8).mean()),
            }
        )

    fig.suptitle(
        "Construct expression and barcode support in each Lib1 single-part library",
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.91,
        "Each point is one construct; the dashed line marks the 8-barcode evaluation threshold",
        ha="center",
        fontsize=12.5,
        color="#425466",
    )
    fig.supxlabel(
        "Construct expression, log₂(total RNA / total DNA)", fontsize=12.5, y=0.105
    )
    fig.text(
        0.5,
        0.018,
        "Display only: the shared 2,466-barcode control is omitted from Promoter and 5′UTR; it remains in the modeled data. Axes are shared across libraries.",
        ha="center",
        fontsize=9.8,
        color="#52616B",
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.80, bottom=0.17, wspace=0.08)
    return save_figure(fig, "main_all_construct_expression_barcode_support"), pd.DataFrame(summary_rows)


def _median_iqr(ax: plt.Axes, x: float, values: np.ndarray, color: str) -> None:
    q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
    ax.vlines(x, q1, q3, color=color, linewidth=5.0, zorder=5)
    ax.plot([x - 0.17, x + 0.17], [median, median], color="#172B4D", linewidth=2.2, zorder=6)


def figure_enhancer_route_comparison() -> tuple[list[str], pd.DataFrame]:
    """Compare scratch and transferred Enhancer routes on one OOF design."""

    source_path = STAGE2_DIR / "stage2_oof_metrics.csv"
    metrics = pd.read_csv(source_path)
    metrics = metrics.loc[metrics["part_slug"].eq("enhancer")].copy()
    scratch_off = metrics.loc[
        metrics["training_regime"].eq("scratch") & metrics["rc_mode"].eq("off")
    ].copy()
    transfer_off = metrics.loc[
        metrics["training_regime"].eq("transfer") & metrics["rc_mode"].eq("off")
    ].copy()
    transfer = metrics.loc[metrics["training_regime"].eq("transfer")].copy()

    selected_policy = pd.read_csv(STAGE3_DIR / "stage3_selected_part_policies.csv")
    selected_policy = selected_policy.loc[selected_policy["part_slug"].eq("enhancer")].iloc[0]
    selected_id = selected_policy["base_config_id"]

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.4), sharey=True)
    route_groups = [
        ("Scratch\nResNet1D", scratch_off, "#7B8794"),
        ("Pretrained\nBassetBranched", transfer_off, PART_COLOR["enhancer"]),
    ]
    for index, (label, group, color) in enumerate(route_groups):
        values = group["pooled_oof_pearson"].to_numpy(float)
        jitter = deterministic_jitter(len(values), 0.13, 2401 + index)
        axes[0].scatter(
            index + jitter,
            values,
            s=58,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        _median_iqr(axes[0], index, values, color)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels([group[0] for group in route_groups])
    axes[0].set_xlim(-0.55, 1.55)
    axes[0].set_ylabel("Pooled five-fold OOF Pearson r")
    axes[0].set_title("A. Route comparison with RC off")
    axes[0].text(
        0.5,
        0.04,
        f"medians: {scratch_off['pooled_oof_pearson'].median():.3f} vs {transfer_off['pooled_oof_pearson'].median():.3f}",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=10.5,
        color="#425466",
    )

    paired = transfer.pivot(
        index="base_config_id", columns="rc_mode", values="pooled_oof_pearson"
    ).dropna(subset=["off", "on"])
    for config_id, row in paired.iterrows():
        is_selected = config_id == selected_id
        axes[1].plot(
            [0, 1],
            [row["off"], row["on"]],
            color=PART_COLOR["enhancer"] if is_selected else "#9CB4C7",
            linewidth=2.6 if is_selected else 1.6,
            alpha=1.0 if is_selected else 0.78,
            marker="o",
            markersize=6.5,
            markeredgecolor="white",
            zorder=4 if is_selected else 2,
        )
    selected_y = float(paired.loc[selected_id, "on"])
    axes[1].scatter(
        [1],
        [selected_y],
        marker="*",
        s=210,
        facecolor="#E9B949",
        edgecolor="#172B4D",
        linewidth=1.0,
        zorder=6,
    )
    axes[1].annotate(
        "frozen selected policy",
        xy=(1, selected_y),
        xytext=(-8, 14),
        textcoords="offset points",
        ha="right",
        fontsize=9.8,
        color="#425466",
    )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["RC off", "RC on"])
    axes[1].set_xlim(-0.35, 1.35)
    axes[1].set_title("B. RC effect within the transfer route")
    axes[1].text(
        0.5,
        0.04,
        f"median paired Δr = {(paired['on'] - paired['off']).median():+.3f}",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=10.5,
        color="#425466",
    )

    for ax in axes:
        ax.axhline(0, color="#52616B", linewidth=0.9)
        ax.set_ylim(-0.08, 0.62)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        "A pretrained Enhancer route outperformed all tested scratch configurations",
        fontsize=20.5,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        "All points use the same five development folds and raw construct-expression target",
        ha="center",
        fontsize=12.2,
        color="#425466",
    )
    fig.text(
        0.5,
        0.018,
        "This is a route comparison—not a causal pretraining-only contrast—because architecture, input framing, and initialization differ. Each point is one configuration/policy.",
        ha="center",
        fontsize=9.8,
        color="#52616B",
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.15, wspace=0.12)

    table = metrics[
        [
            "training_regime",
            "architecture",
            "base_config_id",
            "source_head",
            "unfreeze_scope",
            "rc_mode",
            "pooled_oof_pearson",
            "pooled_oof_rmse",
            "pooled_oof_cod_r2",
        ]
    ].copy()
    table["frozen_selected_policy"] = (
        table["base_config_id"].eq(selected_id) & table["rc_mode"].eq("on")
    )
    return save_figure(fig, "main_enhancer_transfer_vs_scratch_oof"), table


def _plot_fold_trajectories_with_median(
    ax: plt.Axes,
    data: pd.DataFrame,
    train_column: str,
    validation_column: str,
    maximum_epoch: int,
) -> None:
    """Plot faint fold histories and a median only while at least 3 folds remain."""

    split_specs = [
        (train_column, "Train", "#0072B2"),
        (validation_column, "Development validation", "#E69F00"),
    ]
    for column, label, color in split_specs:
        for _, fold in data.groupby("development_fold"):
            fold = fold.loc[fold["epoch"].le(maximum_epoch)].sort_values("epoch")
            ax.plot(
                fold["epoch"],
                fold[column],
                color=color,
                alpha=0.17,
                linewidth=1.05,
                zorder=1,
            )
        aggregate = (
            data.loc[data["epoch"].le(maximum_epoch)]
            .groupby("epoch")[column]
            .agg(["median", "count"])
            .reset_index()
        )
        aggregate = aggregate.loc[aggregate["count"].ge(3)]
        ax.plot(
            aggregate["epoch"],
            aggregate["median"],
            color=color,
            linewidth=2.5,
            label=label,
            zorder=4,
        )


def figure_enhancer_unfreeze_training_dynamics() -> tuple[list[str], pd.DataFrame]:
    """Transpose the Enhancer unfreeze diagnostic into metric rows."""

    history_path = STAGE2_DIR / "reporting/stage2_learning_histories.tsv.gz"
    histories = pd.read_csv(history_path, sep="\t", low_memory=False)
    histories = histories.loc[
        histories["part_slug"].eq("enhancer")
        & histories["analysis_lane"].eq("enhancer_transfer_challenger")
        & histories["source_head"].eq("K562")
        & histories["rc_mode"].eq("on")
    ].copy()
    scope_specs = [
        ("branched_only", "Branch + output"),
        ("conv3_plus", "Top convolution block\n+ dense head"),
        ("full", "Full network"),
    ]
    metric_specs = [
        ("train_pearson", "val_pearson", "Pearson r"),
        ("train_mse", "val_mse", "MSE\n(standardized target)"),
        ("train_cod_r2", "val_cod_r2", "COD R²"),
    ]
    expected_scopes = {scope for scope, _ in scope_specs}
    if set(histories["unfreeze_scope"].dropna().unique()) != expected_scopes:
        raise AssertionError("Enhancer scope histories are incomplete")

    maximum_epoch = 80
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(15.7, 10.0),
        sharex=True,
        sharey="row",
        constrained_layout=False,
    )
    for column_index, (scope, scope_label) in enumerate(scope_specs):
        scope_df = histories.loc[histories["unfreeze_scope"].eq(scope)]
        for row_index, (train_column, val_column, metric_label) in enumerate(metric_specs):
            ax = axes[row_index, column_index]
            _plot_fold_trajectories_with_median(
                ax,
                scope_df,
                train_column,
                val_column,
                maximum_epoch=maximum_epoch,
            )
            if scope in {"conv3_plus", "full"}:
                ax.axvline(2, color="#52616B", linestyle="--", linewidth=1.0)
            ax.set_xlim(0, maximum_epoch)
            if column_index == 0:
                ax.set_ylabel(metric_label)
            if row_index == 0:
                ax.set_title(scope_label)
            if row_index == 2:
                ax.set_xlabel("Epoch")
    axes[0, 1].text(
        2.8,
        0.97,
        "backbone unfreezes",
        transform=axes[0, 1].get_xaxis_transform(),
        fontsize=8.8,
        color="#52616B",
        va="top",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=2,
        fontsize=10.8,
    )
    fig.suptitle(
        "Enhancer transfer training dynamics by fine-tuning scope",
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.942,
        "K562 initialization, RC on; rows share a y-axis so scope behavior can be compared directly",
        ha="center",
        fontsize=12.0,
        color="#425466",
    )
    fig.text(
        0.5,
        0.012,
        "Thin lines are held-out development folds; thick lines are the fold median while ≥3 folds remain. Dashed line marks backbone unfreezing after the two-epoch head warm-up where applicable. Train and development-validation only; display truncated at epoch 80 (one full-network fold continued to epoch 227).",
        ha="center",
        fontsize=9.4,
        color="#52616B",
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.12, hspace=0.15, wspace=0.08)

    table = histories.loc[histories["epoch"].le(maximum_epoch), [
        "epoch",
        "resolved_run_id",
        "base_config_id",
        "source_head",
        "unfreeze_scope",
        "rc_mode",
        "development_fold",
        "train_mse",
        "train_pearson",
        "train_cod_r2",
        "val_mse",
        "val_pearson",
        "val_cod_r2",
    ]].copy()
    return save_figure(fig, "supplement_enhancer_unfreeze_training_dynamics"), table


def figure_selected_policy_training_dynamics() -> tuple[list[str], pd.DataFrame]:
    """Show exact train/development histories for all five selected policies."""

    if not SELECTED_EPOCH_HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SELECTED_EPOCH_HISTORY_PATH}. Run "
            "`conda run -n boda_env python src/analysis/"
            "export_lib1_dedup_tac_epoch_histories.py` first."
        )
    histories = pd.read_csv(SELECTED_EPOCH_HISTORY_PATH, sep="\t", low_memory=False)
    if set(histories["part_slug"].unique()) != set(PART_ORDER):
        raise AssertionError("Selected-policy history export does not cover all five CRE parts")
    if not histories["history_status"].eq("exact_selected_policy_history").all():
        raise AssertionError("Selected-policy histories contain a fallback or unresolved source")
    if histories["fold"].nunique() != 5:
        raise AssertionError("Selected-policy histories do not cover five folds")
    plot_data = histories.rename(columns={"fold": "development_fold"}).copy()

    metric_specs = [
        ("train_pearson", "val_pearson", "Pearson r"),
        ("train_mse", "val_mse", "MSE\n(standardized target)"),
        ("train_cod_r2", "val_cod_r2", "COD R²"),
    ]
    architecture_label = {
        "enhancer": "Transferred Basset",
        "promoter": "PromoterBasset",
        "intron": "ResNet1D",
        "utr3": "UTRBasset",
        "utr5": "UTRBasset",
    }
    maximum_epoch = 140
    fig, axes = plt.subplots(
        3,
        5,
        figsize=(19.0, 10.0),
        sharex=True,
        sharey="row",
        constrained_layout=False,
    )
    for column_index, part in enumerate(PART_ORDER):
        part_df = plot_data.loc[plot_data["part_slug"].eq(part)]
        loss_mode = part_df["loss_mode"].dropna().unique()
        if len(loss_mode) != 1:
            raise AssertionError(f"Multiple selected loss modes for {part}")
        loss_label = "unweighted" if loss_mode[0] == "unweighted_mse" else "barcode-weighted"
        for row_index, (train_column, val_column, metric_label) in enumerate(metric_specs):
            ax = axes[row_index, column_index]
            _plot_fold_trajectories_with_median(
                ax,
                part_df,
                train_column,
                val_column,
                maximum_epoch=maximum_epoch,
            )
            ax.set_xlim(0, maximum_epoch)
            if column_index == 0:
                ax.set_ylabel(metric_label)
            if row_index == 0:
                ax.set_title(
                    f"{PART_LABEL[part]}\n{architecture_label[part]}\n{loss_label} loss",
                    fontsize=10.8,
                    linespacing=1.05,
                )
            if row_index == 2:
                ax.set_xlabel("Epoch")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.047),
        ncol=2,
        fontsize=10.8,
    )
    fig.suptitle(
        "Selected-policy training dynamics across all five CRE parts",
        fontsize=21,
        fontweight="bold",
        y=0.986,
    )
    fig.text(
        0.5,
        0.943,
        "Exact five-fold development histories; metric rows use common y-axes across model families",
        ha="center",
        fontsize=12.0,
        color="#425466",
    )
    fig.text(
        0.5,
        0.012,
        "Thin lines are held-out development folds; thick lines are the fold median while ≥3 folds remain. Train and development-validation only; no locked-final-test trajectory. For weighted policies, displayed MSE is the logged unweighted diagnostic—the optimized training objective used barcode weights.",
        ha="center",
        fontsize=9.4,
        color="#52616B",
    )
    fig.subplots_adjust(left=0.06, right=0.995, top=0.865, bottom=0.12, hspace=0.15, wspace=0.07)
    return save_figure(fig, "supplement_selected_policy_training_dynamics_all_parts"), histories


def load_intron_stage1_diagnostic() -> tuple[pd.DataFrame, dict[str, float]]:
    """Recreate the leakage-free fold-0 Intron composition diagnostic."""

    stratum_order = ["mask1_specific", "mask2_not_mask1", "mask3_residual"]
    stratum_labels = {
        "mask1_specific": "Sequence group 1",
        "mask2_not_mask1": "Sequence group 2",
        "mask3_residual": "Residual group",
    }
    results = load_stage1_results()
    intron_runs = (
        results.loc[
            results["run_kind"].eq("exact_replay")
            & results["part_slug"].eq("intron")
        ]
        .sort_values("val_pearson", ascending=False)
        .reset_index(drop=True)
    )
    leader = intron_runs.iloc[0]
    if leader["run_id"] != "zho9ew6n" or int(leader["development_fold"]) != 0:
        raise AssertionError("Unexpected Stage-1 Intron diagnostic leader")
    prediction = load_predictions(leader)
    dataset = assign_inferred_intron_subsets(pd.read_csv(leader["dataset_path"], sep="\t"))
    assignments = pd.DataFrame(
        json.loads(Path(leader["split_manifest_path"]).read_text(encoding="utf-8"))[
            "assignments"
        ]
    )[["construct_id", "partition", "development_fold"]]
    annotated = dataset.merge(assignments, on="construct_id", validate="one_to_one")
    validation = prediction.merge(
        annotated[
            [
                "construct_id",
                "inferred_intron_subset",
                "partition",
                "development_fold",
            ]
        ],
        on="construct_id",
        validate="one_to_one",
    )
    expected_ids = set(
        assignments.loc[
            assignments["partition"].eq("development")
            & assignments["development_fold"].eq(0),
            "construct_id",
        ]
    )
    if set(validation["construct_id"]) != expected_ids:
        raise AssertionError("Intron fold-0 predictions do not match the split manifest")
    fold_training = annotated.loc[
        annotated["partition"].eq("train_only")
        | (
            annotated["partition"].eq("development")
            & annotated["development_fold"].ne(0)
        )
    ].copy()
    training_means = fold_training.groupby("inferred_intron_subset")["log2_RNA_DNA"].mean()
    validation["stratum"] = validation["inferred_intron_subset"].map(stratum_labels)
    validation["training_fitted_group_mean"] = validation[
        "inferred_intron_subset"
    ].map(training_means)
    validation["observed_centered"] = validation["log2_RNA_DNA"] - validation.groupby(
        "inferred_intron_subset"
    )["log2_RNA_DNA"].transform("mean")
    validation["predicted_centered"] = validation["prediction_raw"] - validation.groupby(
        "inferred_intron_subset"
    )["prediction_raw"].transform("mean")
    validation["stratum_order"] = validation["inferred_intron_subset"].map(
        {key: index for index, key in enumerate(stratum_order)}
    )
    metrics = {
        "pooled_pearson": float(
            np.corrcoef(validation["log2_RNA_DNA"], validation["prediction_raw"])[0, 1]
        ),
        "training_mean_pearson": float(
            np.corrcoef(
                validation["log2_RNA_DNA"], validation["training_fitted_group_mean"]
            )[0, 1]
        ),
        "within_group_pearson": float(
            np.corrcoef(validation["observed_centered"], validation["predicted_centered"])[
                0, 1
            ]
        ),
    }
    return validation, metrics


def figure_intron_composition_triptych() -> tuple[list[str], pd.DataFrame]:
    """Show pooled, group-mean-only, and within-group Intron prediction."""

    data, metrics = load_intron_stage1_diagnostic()
    palette = {
        "Sequence group 1": "#4C78A8",
        "Sequence group 2": "#F58518",
        "Residual group": "#54A24B",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 6.2), constrained_layout=False)
    panels = [
        (
            "log2_RNA_DNA",
            "prediction_raw",
            f"A. CNN prediction\npooled r = {metrics['pooled_pearson']:.3f}",
            "Observed log₂(RNA/DNA)",
            "CNN-predicted log₂(RNA/DNA)",
        ),
        (
            "log2_RNA_DNA",
            "training_fitted_group_mean",
            f"B. Three training-fitted group means only\nr = {metrics['training_mean_pearson']:.3f}",
            "Observed log₂(RNA/DNA)",
            "Training-fitted group mean",
        ),
        (
            "observed_centered",
            "predicted_centered",
            f"C. Within-group ranking after centering\nr = {metrics['within_group_pearson']:.3f}",
            "Observed minus group mean",
            "Predicted minus group mean",
        ),
    ]
    for ax, (x_col, y_col, title, x_label, y_label) in zip(axes, panels):
        for label, group in data.groupby("stratum", sort=False):
            ax.scatter(
                group[x_col],
                group[y_col],
                s=31,
                alpha=0.68,
                color=palette[label],
                edgecolor="white",
                linewidth=0.25,
                label=f"{label} (n={len(group)})",
            )
        ax.set_title(title, fontsize=13.2)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect("equal", adjustable="box")

    raw_values = np.concatenate(
        [
            data["log2_RNA_DNA"].to_numpy(float),
            data["prediction_raw"].to_numpy(float),
            data["training_fitted_group_mean"].to_numpy(float),
        ]
    )
    raw_pad = max(0.1, 0.06 * float(np.ptp(raw_values)))
    raw_limits = (float(raw_values.min() - raw_pad), float(raw_values.max() + raw_pad))
    for ax in axes[:2]:
        ax.plot(raw_limits, raw_limits, "--", color="#52616B", linewidth=1.1)
        ax.set_xlim(raw_limits)
        ax.set_ylim(raw_limits)
    centered_limit = 1.08 * float(
        np.max(
            np.abs(
                np.concatenate(
                    [
                        data["observed_centered"].to_numpy(float),
                        data["predicted_centered"].to_numpy(float),
                    ]
                )
            )
        )
    )
    axes[2].axhline(0, color="#8795A1", linewidth=0.9)
    axes[2].axvline(0, color="#8795A1", linewidth=0.9)
    axes[2].set_xlim(-centered_limit, centered_limit)
    axes[2].set_ylim(-centered_limit, centered_limit)
    handles_by_label = {
        handle.get_label(): handle for handle in axes[0].get_legend_handles_labels()[0]
    }
    ordered_labels = ["Sequence group 1 (n=81)", "Sequence group 2 (n=60)", "Residual group (n=72)"]
    handles = [handles_by_label[label] for label in ordered_labels]
    fig.legend(
        handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=3,
        fontsize=10.5,
    )
    fig.suptitle(
        "Why the pooled Intron correlation is composition-assisted",
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        "Group separation explains much of the pooled score; the CNN still retains within-group ranking signal",
        ha="center",
        fontsize=12.2,
        color="#425466",
    )
    fig.text(
        0.5,
        0.018,
        "Illustrative Stage-1 held-out fold diagnostic (n=213), not final-policy performance. Groups are inferred from sequence masks, not verified synthesis sublibraries or measured splicing states.",
        ha="center",
        fontsize=9.6,
        color="#52616B",
    )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.82, bottom=0.22, wspace=0.22)

    summary = pd.DataFrame(
        [
            {"estimand": "pooled CNN prediction", "pearson": metrics["pooled_pearson"]},
            {
                "estimand": "three training-fitted group means",
                "pearson": metrics["training_mean_pearson"],
            },
            {
                "estimand": "within-group centered CNN prediction",
                "pearson": metrics["within_group_pearson"],
            },
        ]
    )
    return save_figure(fig, "main_intron_composition_triptych"), summary


def _nice_limits(values: np.ndarray) -> tuple[float, float]:
    data_low = float(np.nanmin(values))
    data_high = float(np.nanmax(values))
    pad = max(0.18, 0.07 * (data_high - data_low))
    low = math.floor((data_low - pad) * 2) / 2
    high = math.ceil((data_high + pad) * 2) / 2
    return low, high


def figure_final_test_scatter_hexbin() -> tuple[list[str], pd.DataFrame]:
    """Use the Notion-style scatter-over-hexbin locked-test layout."""

    predictions = pd.read_csv(FINAL_DIR / "audit_ensemble_predictions.tsv.gz", sep="\t")
    metrics = pd.read_csv(FINAL_DIR / "audit_metrics.csv")
    metrics = metrics.loc[metrics["primary_predictor"].eq(True)].set_index("part_slug")

    fig, axes = plt.subplots(2, 5, figsize=(19.2, 9.4), constrained_layout=False)
    hexbins = []
    for column, part in enumerate(PART_ORDER):
        part_df = predictions.loc[predictions["part_slug"].eq(part)]
        metric = metrics.loc[part]
        values = np.concatenate(
            [part_df["prediction_raw"].to_numpy(float), part_df["observed_raw"].to_numpy(float)]
        )
        low, high = _nice_limits(values)
        slope = float(metric["calibration_slope_observed_on_prediction"])
        intercept = float(metric["calibration_intercept_observed_on_prediction"])
        x_line = np.linspace(float(part_df["prediction_raw"].min()), float(part_df["prediction_raw"].max()), 100)

        top = axes[0, column]
        top.scatter(
            part_df["prediction_raw"],
            part_df["observed_raw"],
            s=24,
            color="#2C7FB8",
            alpha=0.35,
            edgecolor="none",
            zorder=2,
        )
        bottom = axes[1, column]
        hb = bottom.hexbin(
            part_df["prediction_raw"],
            part_df["observed_raw"],
            gridsize=28,
            extent=[low, high, low, high],
            mincnt=1,
            cmap="Blues",
            linewidths=0.22,
            edgecolors="white",
        )
        hexbins.append(hb)
        for ax in (top, bottom):
            ax.plot([low, high], [low, high], "--", color="#52616B", linewidth=1.1, zorder=3)
            ax.plot(x_line, intercept + slope * x_line, color=FIT_COLOR, linewidth=2.35, zorder=4)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_aspect("equal", adjustable="box")
            ax.locator_params(axis="both", nbins=5)
        top.tick_params(axis="x", labelbottom=False)
        top.set_title(
            f"{PART_LABEL[part]}\nn={int(metric['n'])}  •  r={metric['pearson']:.3f}  •  slope={slope:.2f}",
            fontsize=12.5,
        )

    max_count = max(float(np.max(hb.get_array())) for hb in hexbins)
    shared_norm = LogNorm(vmin=1, vmax=max(2, math.ceil(max_count)))
    for hb in hexbins:
        hb.set_norm(shared_norm)

    axes[0, 0].set_ylabel("Observed expression")
    axes[1, 0].set_ylabel("Observed expression")
    for ax in axes[1, :]:
        ax.set_xlabel("Ensemble prediction")

    fig.text(
        0.014,
        0.675,
        "Individual constructs",
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#172B4D",
    )
    fig.text(
        0.014,
        0.305,
        "Construct density",
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#172B4D",
    )

    legend_handles = [
        Line2D([0], [0], color="#52616B", linestyle="--", linewidth=1.4, label="Identity"),
        Line2D([0], [0], color=FIT_COLOR, linewidth=2.5, label="Observed-on-prediction fit"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.075, 0.012),
        ncol=2,
        fontsize=10.5,
    )

    density_ax = fig.add_axes([0.69, 0.035, 0.245, 0.026])
    density_edges = np.geomspace(1, max(2, math.ceil(max_count)), 25)
    density_ax.set_xscale("log")
    density_ax.set_xlim(density_edges[0], density_edges[-1])
    density_ax.set_ylim(0, 1)
    for left, right in zip(density_edges[:-1], density_edges[1:]):
        midpoint = math.sqrt(left * right)
        density_ax.add_patch(
            Rectangle(
                (left, 0),
                right - left,
                1,
                facecolor=mpl.colormaps["Blues"](shared_norm(midpoint)),
                edgecolor="none",
            )
        )
    density_ax.set_yticks([])
    density_ax.grid(False)
    density_ax.tick_params(axis="x", labelsize=8.5)
    density_ax.set_xlabel("Constructs per hexagon (log scale)", fontsize=9.3, labelpad=1)

    fig.suptitle(
        "One-time locked final test: raw-scale association and calibration",
        fontsize=21.5,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.946,
        "Top: individual constructs  •  Bottom: the same constructs summarized by hexagonal-bin density",
        ha="center",
        fontsize=12.2,
        color="#425466",
    )
    fig.text(
        0.5,
        0.915,
        "Frozen three-seed ensembles on the raw log₂(total RNA / total DNA) scale; no normalization or post-test recalibration",
        ha="center",
        fontsize=10.8,
        color="#52616B",
    )
    fig.subplots_adjust(left=0.065, right=0.992, top=0.855, bottom=0.12, hspace=0.19, wspace=0.22)

    summary = metrics.reset_index()[
        [
            "part_slug",
            "n",
            "pearson",
            "spearman",
            "rmse",
            "mae",
            "cod_r2",
            "bias_prediction_minus_observed",
            "calibration_slope_observed_on_prediction",
            "calibration_intercept_observed_on_prediction",
        ]
    ]
    return save_figure(fig, "main_locked_final_test_scatter_hexbin"), summary


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    plot_functions = [
        (
            "development_fold_balance",
            figure_development_fold_balance,
            "development_fold_expression_barcode_summary.csv",
        ),
        (
            "all_construct_expression_barcode_support",
            figure_all_construct_expression_barcode_support,
            "all_construct_expression_barcode_summary.csv",
        ),
        (
            "enhancer_route_comparison",
            figure_enhancer_route_comparison,
            "enhancer_transfer_scratch_oof_values.csv",
        ),
        (
            "enhancer_unfreeze_training_dynamics",
            figure_enhancer_unfreeze_training_dynamics,
            "enhancer_unfreeze_training_histories.tsv",
        ),
        (
            "selected_policy_training_dynamics",
            figure_selected_policy_training_dynamics,
            "selected_policy_training_histories_for_plot.tsv",
        ),
        (
            "intron_composition_triptych",
            figure_intron_composition_triptych,
            "intron_composition_triptych_summary.csv",
        ),
        (
            "final_test_scatter_hexbin",
            figure_final_test_scatter_hexbin,
            "locked_final_test_scatter_hexbin_metrics.csv",
        ),
    ]
    manifest: dict[str, object] = {
        "purpose": "Follow-up presentation figures for the July 2026 Lib1 deduplicated baseline TAC deck",
        "final_test_replot_is_reporting_only": True,
        "display_only_exclusions": {
            "selection_rule": (
                "the single construct shared by Promoter and UTR5 with "
                f"n_barcodes > {HIGH_SUPPORT_CONTROL_CUTOFF:,}"
            ),
            "barcode_count": 2466,
            "library_entries": ["promoter", "utr5"],
            "affected_figures": [
                "development_fold_balance",
                "all_construct_expression_barcode_support",
            ],
            "modeled_data_changed": False,
        },
        "figures": {},
        "source_files": {
            "split_catalog": _relative(SPLIT_CATALOG),
            "stage2_oof_metrics": _relative(STAGE2_DIR / "stage2_oof_metrics.csv"),
            "stage2_learning_histories": _relative(
                STAGE2_DIR / "reporting/stage2_learning_histories.tsv.gz"
            ),
            "selected_policy_epoch_histories": _relative(SELECTED_EPOCH_HISTORY_PATH),
            "epoch_history_exporter": _relative(
                REPO_ROOT / "src/analysis/export_lib1_dedup_tac_epoch_histories.py"
            ),
            "epoch_history_export_summary": _relative(
                TABLE_DIR / "epoch_history_export_summary.json"
            ),
            "enhancer_scope_config_audit": _relative(
                TABLE_DIR / "enhancer_k562_rc_on_unfreeze_scope_configs.tsv"
            ),
            "stage3_selected_policies": _relative(
                STAGE3_DIR / "stage3_selected_part_policies.csv"
            ),
            "final_predictions": _relative(FINAL_DIR / "audit_ensemble_predictions.tsv.gz"),
            "final_metrics": _relative(FINAL_DIR / "audit_metrics.csv"),
        },
    }
    for key, function, table_name in plot_functions:
        outputs, table = function()
        table_path = TABLE_DIR / table_name
        table.to_csv(table_path, index=False, sep="\t" if table_path.suffix == ".tsv" else ",")
        manifest["figures"][key] = {
            "outputs": outputs,
            "summary_table": _relative(table_path),
        }
    manifest_path = OUTPUT_DIR / "followup_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(plot_functions)} follow-up figure sets to {FIGURE_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
