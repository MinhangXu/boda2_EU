#!/usr/bin/env python3
"""Generate presentation-ready figures for the Lib1 deduplicated baseline TAC deck.

The script only reads frozen or development-stage reporting products.  In
particular, regenerating the final-test calibration figure is a reporting-only
operation: no models are fit and the locked-test predictions are not changed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

STAGE1_DIR = (
    REPO_ROOT
    / "tutorials/lib1_tasks/pretrain_CRE_inhouse_data/"
    "dedup_phase1_rerun_july2026/outputs"
)
STAGE2_DIR = (
    REPO_ROOT / "src/learn/outputs/analysis/lib1_dedup_stage2_july2026"
)
STAGE3_DIR = (
    REPO_ROOT
    / "src/learn/outputs/analysis/lib1_dedup_stage3_weighted_loss_july2026"
)
FINAL_DIR = (
    REPO_ROOT
    / "src/learn/outputs/audit/lib1_dedup_final_audit_july2026/frozen_products"
)
OUTPUT_DIR = (
    REPO_ROOT
    / "src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026"
)
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"


PART_ORDER = ["enhancer", "promoter", "intron", "utr3", "utr5"]
PART_LABEL = {
    "enhancer": "Enhancer",
    "promoter": "Promoter",
    "intron": "Intron",
    "utr3": "3′UTR",
    "utr5": "5′UTR",
}
PART_COLOR = {
    "enhancer": "#355F82",
    "promoter": "#2A9D8F",
    "intron": "#E9B949",
    "utr3": "#EF6C4D",
    "utr5": "#7A5195",
}
ARCH_LABEL = {
    "ResNet1DRegressor": "ResNet1D",
    "PromoterBassetVL": "PromoterBasset",
    "UTR_BassetVL": "UTRBasset",
    "BassetBranched": "Transferred Basset",
}
FROZEN_RC = {
    "enhancer": "on",
    "promoter": "off",
    "intron": "off",
    "utr3": "off",
    "utr5": "off",
}
FROZEN_ROUTE = {
    "enhancer": "Transferred\nEnhancer model",
    "promoter": "PromoterBasset",
    "intron": "ResNet1D",
    "utr3": "UTRBasset",
    "utr5": "UTRBasset",
}


def configure_style() -> None:
    """Use a projection-friendly, restrained Matplotlib style."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#DCE3EA",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    """Save a high-resolution PNG and a true vector SVG."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in (
        ("png", {"dpi": 240}),
        ("svg", {}),
    ):
        path = FIGURE_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path.relative_to(REPO_ROOT)))
    plt.close(fig)
    return outputs


def deterministic_jitter(n: int, width: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-width, width, n)


def figure_hpo_landscape() -> tuple[list[str], pd.DataFrame]:
    """Show all Stage-1 configurations and the portfolio advanced to Stage 2."""

    metrics = pd.read_csv(STAGE1_DIR / "stage1_exact_replay_metrics.csv")
    advanced = pd.read_csv(STAGE1_DIR / "stage2_candidate_selection_draft.csv")
    advanced_ids = set(advanced["base_config_id"])
    metrics["advanced"] = metrics["base_config_id"].isin(advanced_ids)

    fig, axes = plt.subplots(
        1, 5, figsize=(17.0, 6.4), sharey=True, constrained_layout=False
    )
    display_order = ["enhancer", "promoter", "intron", "utr3", "utr5"]

    for panel_index, (ax, part) in enumerate(zip(axes, display_order)):
        part_df = metrics.loc[metrics["part_slug"].eq(part)].copy()
        architectures = list(dict.fromkeys(part_df["architecture"].tolist()))
        if part == "utr5":
            architectures = ["ResNet1DRegressor", "UTR_BassetVL"]

        for arch_index, architecture in enumerate(architectures):
            arm = part_df.loc[part_df["architecture"].eq(architecture)].copy()
            values = arm["val_pearson"].dropna().to_numpy()
            if len(values) >= 2:
                violin = ax.violinplot(
                    values,
                    positions=[arch_index],
                    widths=0.74,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                )
                for body in violin["bodies"]:
                    body.set_facecolor(PART_COLOR[part])
                    body.set_edgecolor(PART_COLOR[part])
                    body.set_alpha(0.14)

            arm = arm.reset_index(drop=True)
            jitter = deterministic_jitter(
                len(arm), width=0.26, seed=1701 + panel_index * 31 + arch_index
            )
            not_advanced = ~arm["advanced"]
            ax.scatter(
                arch_index + jitter[not_advanced],
                arm.loc[not_advanced, "val_pearson"],
                s=16,
                color="#7B8794",
                alpha=0.36,
                edgecolor="none",
                zorder=2,
            )
            ax.scatter(
                arch_index + jitter[~not_advanced],
                arm.loc[~not_advanced, "val_pearson"],
                s=48,
                marker="D",
                facecolor=PART_COLOR[part],
                edgecolor="#172B4D",
                linewidth=0.7,
                alpha=0.95,
                zorder=4,
            )
            median = float(np.nanmedian(values))
            ax.plot(
                [arch_index - 0.22, arch_index + 0.22],
                [median, median],
                color="#172B4D",
                linewidth=2.1,
                zorder=3,
            )

        best = part_df["val_pearson"].max()
        suffix = "*" if part == "intron" else ""
        ax.set_title(
            f"{PART_LABEL[part]}{suffix}\n{len(part_df):,} settings; best r={best:.3f}"
        )
        ax.set_xticks(range(len(architectures)))
        ax.set_xticklabels([ARCH_LABEL[a] for a in architectures], fontsize=10)
        ax.set_xlim(-0.62, max(0.62, len(architectures) - 0.38))
        ax.axhline(0, color="#52616B", linewidth=0.8)
        ax.set_ylim(-0.10, 0.85)
        ax.grid(axis="x", visible=False)
        if panel_index == 0:
            ax.set_ylabel("Screening-fold Pearson r")

    fig.suptitle(
        "Broad hyperparameter screening across five Lib1 single-part models",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        "885 tested settings on one fixed screening fold per CRE part; RC off; unweighted MSE",
        ha="center",
        va="center",
        fontsize=13,
        color="#425466",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#7B8794",
            markeredgecolor="none",
            alpha=0.55,
            label="Tested configuration",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="#FFFFFF",
            markeredgecolor="#172B4D",
            label="Advanced to paired five-fold testing (10 per part)",
        ),
        Line2D(
            [0],
            [0],
            color="#172B4D",
            linewidth=2.2,
            label="Median",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=3,
        fontsize=11,
    )
    fig.text(
        0.5,
        0.025,
        "Screening selected a candidate portfolio; five-fold out-of-fold evaluation begins in Stage 2.  "
        "*Intron is pooled; see the sequence-group audit.",
        ha="center",
        fontsize=10.5,
        color="#52616B",
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.82, bottom=0.20, wspace=0.12)

    summary = (
        metrics.groupby(["part_slug", "architecture"], as_index=False)
        .agg(
            configurations=("base_config_id", "size"),
            median_screening_pearson=("val_pearson", "median"),
            best_screening_pearson=("val_pearson", "max"),
            advanced_to_stage2=("advanced", "sum"),
        )
        .sort_values(["part_slug", "architecture"])
    )
    return save_figure(fig, "supplement_hpo_configuration_landscape"), summary


def rc_route_label(row: pd.Series) -> str:
    part = row["part_slug"]
    if part == "enhancer" and row["training_regime"] == "transfer":
        return "Transferred\nEnhancer model"
    if part == "utr3" and row["challenger_family"] == "utr3_utrbasset":
        return "UTRBasset"
    return ARCH_LABEL.get(row["architecture"], row["architecture"])


def figure_rc_effect() -> tuple[list[str], pd.DataFrame]:
    """Audience-facing RC effect plot without the compound gate encoding."""

    pairs = pd.read_csv(STAGE2_DIR / "stage2_rc_pair_metrics.csv")
    pairs["route_label"] = pairs.apply(rc_route_label, axis=1)

    fig, axes = plt.subplots(
        1, 5, figsize=(17.0, 6.2), sharey=True, constrained_layout=False
    )

    for panel_index, (ax, part) in enumerate(zip(axes, PART_ORDER)):
        part_df = pairs.loc[pairs["part_slug"].eq(part)].copy()
        routes = list(dict.fromkeys(part_df["route_label"].tolist()))
        # Put the final-policy route last in multi-route panels so the eye ends there.
        selected_route = FROZEN_ROUTE[part]
        if selected_route in routes and len(routes) > 1:
            routes = [route for route in routes if route != selected_route] + [selected_route]

        for route_index, route in enumerate(routes):
            route_df = part_df.loc[part_df["route_label"].eq(route)].reset_index(drop=True)
            jitter = deterministic_jitter(
                len(route_df), width=0.19, seed=2701 + panel_index * 29 + route_index
            )
            is_selected_route = route == selected_route
            point_color = PART_COLOR[part] if is_selected_route else "#9AA5B1"
            ax.scatter(
                route_index + jitter,
                route_df["mean_fold_delta_rc_on_minus_off_pooled_pearson"],
                s=42 if is_selected_route else 30,
                facecolor=point_color,
                edgecolor="#FFFFFF",
                linewidth=0.5,
                alpha=0.82 if is_selected_route else 0.55,
                zorder=3,
            )
            median = route_df[
                "mean_fold_delta_rc_on_minus_off_pooled_pearson"
            ].median()
            ax.plot(
                [route_index - 0.20, route_index + 0.20],
                [median, median],
                color="#172B4D",
                linewidth=2.2,
                zorder=4,
            )

        xticklabels = [
            ("★ " if route == selected_route else "") + route for route in routes
        ]
        ax.set_xticks(range(len(routes)))
        ax.set_xticklabels(xticklabels, fontsize=9.4)
        ax.set_xlim(-0.55, max(0.55, len(routes) - 0.45))
        ax.axhline(0, color="#172B4D", linewidth=1.1)
        ax.set_ylim(-0.185, 0.055)
        ax.grid(axis="x", visible=False)
        ax.set_title(PART_LABEL[part])
        policy_color = "#1B7F3A" if FROZEN_RC[part] == "on" else "#52616B"
        ax.text(
            0.5,
            0.96,
            f"Frozen policy: RC {FROZEN_RC[part].upper()}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.2,
            color=policy_color,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": policy_color,
                "linewidth": 0.9,
                "alpha": 0.95,
            },
        )
        if panel_index == 0:
            ax.set_ylabel("Mean change in held-out-fold Pearson r\n(RC on − RC off)")

    fig.suptitle(
        "Reverse-complement augmentation was route-specific",
        fontsize=22,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.91,
        "Each point is an exact configuration pair with the same five folds, seed, and loss; bars show route medians",
        ha="center",
        fontsize=12.5,
        color="#425466",
    )
    fig.text(
        0.5,
        0.025,
        "★ Route used by the frozen policy. RC was retained only for the transferred Enhancer model. "
        "The transfer-versus-scratch contrast is diagnostic, not a pretraining-only causal comparison.",
        ha="center",
        fontsize=10.5,
        color="#52616B",
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.82, bottom=0.18, wspace=0.12)

    route_summary = (
        pairs.groupby(["part_slug", "route_label"], as_index=False)
        .agg(
            paired_configurations=("base_config_id", "size"),
            median_mean_fold_delta=(
                "mean_fold_delta_rc_on_minus_off_pooled_pearson",
                "median",
            ),
            minimum_mean_fold_delta=(
                "mean_fold_delta_rc_on_minus_off_pooled_pearson",
                "min",
            ),
            maximum_mean_fold_delta=(
                "mean_fold_delta_rc_on_minus_off_pooled_pearson",
                "max",
            ),
        )
        .sort_values(["part_slug", "route_label"])
    )
    return save_figure(fig, "main_rc_augmentation_effect"), route_summary


def selected_loss_effect_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(STAGE3_DIR / "stage3_selected_part_policies.csv")
    pairs = pd.read_csv(STAGE3_DIR / "stage3_loss_pair_metrics.csv")
    folds = pd.read_csv(STAGE3_DIR / "stage3_loss_fold_pair_metrics.csv")

    selected_pair_rows = []
    selected_fold_rows = []
    for _, policy in selected.iterrows():
        pair_match = pairs.loc[
            pairs["part_slug"].eq(policy["part_slug"])
            & pairs["base_config_id"].eq(policy["base_config_id"])
            & pairs["baseline_rc_mode"].eq(policy["rc_mode"])
            & pairs["intervention_rc_mode"].eq(policy["rc_mode"])
        ]
        if len(pair_match) != 1:
            raise RuntimeError(
                f"Expected one loss pair for {policy['part_slug']}; found {len(pair_match)}"
            )
        pair = pair_match.iloc[0].copy()
        pair["selected_loss_mode"] = policy["loss_mode"]
        pair["selected_rc_mode"] = policy["rc_mode"]
        selected_pair_rows.append(pair)

        fold_match = folds.loc[
            folds["part_slug"].eq(policy["part_slug"])
            & folds["base_config_id"].eq(policy["base_config_id"])
            & folds["baseline_rc_mode"].eq(policy["rc_mode"])
            & folds["intervention_rc_mode"].eq(policy["rc_mode"])
        ].copy()
        if len(fold_match) != 5:
            raise RuntimeError(
                f"Expected five fold pairs for {policy['part_slug']}; found {len(fold_match)}"
            )
        fold_match["selected_loss_mode"] = policy["loss_mode"]
        selected_fold_rows.append(fold_match)

    return pd.DataFrame(selected_pair_rows), pd.concat(selected_fold_rows, ignore_index=True)


def figure_weighted_loss_effect() -> tuple[list[str], pd.DataFrame]:
    """Show the exact weighted-vs-unweighted contrast for each selected model."""

    pairs, folds = selected_loss_effect_tables()
    order = PART_ORDER
    fig, ax = plt.subplots(figsize=(12.8, 6.9))
    fold_offsets = np.linspace(-0.16, 0.16, 5)

    for part_index, part in enumerate(order):
        part_folds = (
            folds.loc[folds["part_slug"].eq(part)]
            .sort_values("development_fold")
            .reset_index(drop=True)
        )
        pair = pairs.loc[pairs["part_slug"].eq(part)].iloc[0]
        values = part_folds["fold_pearson_delta"].to_numpy()
        ax.plot(
            [part_index, part_index],
            [values.min(), values.max()],
            color=PART_COLOR[part],
            linewidth=2,
            alpha=0.38,
            zorder=1,
        )
        ax.scatter(
            part_index + fold_offsets,
            values,
            s=52,
            facecolor="white",
            edgecolor=PART_COLOR[part],
            linewidth=1.6,
            zorder=3,
        )
        pooled = float(pair["pooled_oof_pearson_delta"])
        ax.scatter(
            [part_index],
            [pooled],
            s=120,
            marker="D",
            facecolor=PART_COLOR[part],
            edgecolor="#172B4D",
            linewidth=0.9,
            zorder=4,
        )
        label_offset = 0.010 if pooled >= 0 else -0.014
        ax.text(
            part_index,
            pooled + label_offset,
            f"{pooled:+.3f}",
            ha="center",
            va="bottom" if pooled >= 0 else "top",
            fontsize=10.5,
            fontweight="bold",
            color="#172B4D",
        )

    ax.axhline(0, color="#172B4D", linewidth=1.1)
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(-0.115, 0.105)
    selected_mode_by_part = {
        row["part_slug"]: (
            "weighted" if row["selected_loss_mode"] == "barcode_weighted_mse" else "unweighted"
        )
        for _, row in pairs.iterrows()
    }
    ax.set_xticks(range(5))
    ax.set_xticklabels(
        [
            f"{PART_LABEL[part]}\nretained: {selected_mode_by_part[part]}"
            for part in order
        ],
        fontsize=11,
    )
    ax.set_ylabel("Change in Pearson r\n(weighted loss − unweighted loss)")
    ax.grid(axis="x", visible=False)
    ax.set_title(
        "Barcode-weighted loss improved four selected model configurations",
        fontsize=21,
        pad=42,
    )
    ax.text(
        0.5,
        1.065,
        "Exact paired comparison: same model configuration, RC policy, seed, and five development folds",
        transform=ax.transAxes,
        ha="center",
        fontsize=12.5,
        color="#425466",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="#52616B",
            label="Held-out-fold change (five folds)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor="#7A5195",
            markeredgecolor="#172B4D",
            label="Pooled five-fold OOF change",
        ),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=10.5)
    fig.text(
        0.5,
        0.02,
        "The displayed effect is Pearson r; policy retention also required fold consistency and RMSE/COD R² guardrails. "
        "Folds are held-out partitions, not biological replicates.",
        ha="center",
        fontsize=10.2,
        color="#52616B",
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.80, bottom=0.20)

    summary_columns = [
        "part_slug",
        "base_config_id",
        "architecture",
        "baseline_rc_mode",
        "mean_fold_pearson_delta",
        "positive_fold_pearson_delta_count",
        "pooled_oof_pearson_delta",
        "selected_loss_mode",
        "gate_pass",
    ]
    summary = pairs[summary_columns].copy()
    return save_figure(fig, "main_barcode_weighted_loss_effect"), summary


def figure_final_test_calibration() -> tuple[list[str], pd.DataFrame]:
    """Replot locked-test calibration from frozen ensemble predictions."""

    predictions = pd.read_csv(
        FINAL_DIR / "audit_ensemble_predictions.tsv.gz", sep="\t"
    )
    metrics = pd.read_csv(FINAL_DIR / "audit_metrics.csv")
    metrics = metrics.loc[metrics["primary_predictor"].eq(True)].copy()
    metrics = metrics.set_index("part_slug")

    limit_low, limit_high = -5.25, 4.75
    fig = plt.figure(figsize=(15.7, 9.2), constrained_layout=False)
    grid = fig.add_gridspec(2, 3)
    first_ax = fig.add_subplot(grid[0, 0])
    plot_axes = [
        first_ax,
        fig.add_subplot(grid[0, 1], sharex=first_ax, sharey=first_ax),
        fig.add_subplot(grid[0, 2], sharex=first_ax, sharey=first_ax),
        fig.add_subplot(grid[1, 0], sharex=first_ax, sharey=first_ax),
        fig.add_subplot(grid[1, 1], sharex=first_ax, sharey=first_ax),
    ]
    # The interpretation key deliberately does not share axes with the data panels.
    info_ax = fig.add_subplot(grid[1, 2])
    hexbins = []

    for ax, part in zip(plot_axes, PART_ORDER):
        part_df = predictions.loc[predictions["part_slug"].eq(part)]
        metric = metrics.loc[part]
        hb = ax.hexbin(
            part_df["prediction_raw"],
            part_df["observed_raw"],
            gridsize=35,
            extent=[limit_low, limit_high, limit_low, limit_high],
            mincnt=1,
            cmap="Blues",
            linewidths=0.25,
            edgecolors="#FFFFFF",
        )
        hexbins.append(hb)
        ax.plot(
            [limit_low, limit_high],
            [limit_low, limit_high],
            linestyle="--",
            color="#52616B",
            linewidth=1.2,
            zorder=2,
        )
        x_min = float(part_df["prediction_raw"].min())
        x_max = float(part_df["prediction_raw"].max())
        x_line = np.linspace(x_min, x_max, 100)
        slope = float(metric["calibration_slope_observed_on_prediction"])
        intercept = float(metric["calibration_intercept_observed_on_prediction"])
        ax.plot(
            x_line,
            intercept + slope * x_line,
            color=PART_COLOR[part],
            linewidth=2.5,
            zorder=3,
        )
        ax.set_xlim(limit_low, limit_high)
        ax.set_ylim(limit_low, limit_high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{PART_LABEL[part]}\n"
            f"n={int(metric['n'])}  •  r={metric['pearson']:.3f}  •  slope={slope:.2f}",
            fontsize=13.2,
        )

    max_count = max(float(np.max(hb.get_array())) for hb in hexbins)
    shared_norm = LogNorm(vmin=1, vmax=max(2, math.ceil(max_count)))
    for hb in hexbins:
        hb.set_norm(shared_norm)

    info_ax.axis("off")
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)
    info_ax.text(
        0.04,
        0.95,
        "How to read this figure",
        fontsize=15,
        fontweight="bold",
        color="#172B4D",
        va="top",
    )
    info_ax.plot([0.06, 0.22], [0.80, 0.80], "--", color="#52616B", linewidth=1.5)
    info_ax.text(0.26, 0.80, "Identity: perfect calibration", va="center", fontsize=11)
    info_ax.plot([0.06, 0.22], [0.69, 0.69], color=PART_COLOR["utr3"], linewidth=2.6)
    info_ax.text(0.26, 0.69, "Observed-on-prediction fit", va="center", fontsize=11)
    info_ax.text(
        0.04,
        0.54,
        "Slope = 1 and intercept = 0 are ideal.\n"
        "A narrow prediction range appears as\n"
        "horizontal compression relative to identity.",
        fontsize=10.8,
        color="#425466",
        linespacing=1.45,
        va="top",
    )
    info_ax.text(
        0.04,
        0.33,
        "Frozen three-seed arithmetic ensemble\n"
        "Raw log₂(RNA/DNA) scale\n"
        "No post-test recalibration",
        fontsize=10.8,
        color="#172B4D",
        fontweight="bold",
        linespacing=1.30,
        va="top",
    )
    colorbar_ax = info_ax.inset_axes([0.06, 0.035, 0.72, 0.045])
    # Draw the density key as vector rectangles. Matplotlib's standard SVG
    # colorbar embeds a small raster gradient even when the data marks are vector.
    density_edges = np.geomspace(1, max(2, math.ceil(max_count)), 25)
    colorbar_ax.set_xscale("log")
    colorbar_ax.set_xlim(density_edges[0], density_edges[-1])
    colorbar_ax.set_ylim(0, 1)
    for left, right in zip(density_edges[:-1], density_edges[1:]):
        midpoint = math.sqrt(left * right)
        colorbar_ax.add_patch(
            Rectangle(
                (left, 0),
                right - left,
                1,
                facecolor=mpl.colormaps["Blues"](shared_norm(midpoint)),
                edgecolor="none",
            )
        )
    colorbar_ax.set_yticks([])
    colorbar_ax.grid(False)
    colorbar_ax.set_xlabel("Constructs per hexagonal bin (log scale)", fontsize=9.5)
    colorbar_ax.tick_params(axis="x", labelsize=8.5)

    fig.suptitle(
        "Locked final test: observed versus predicted expression",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.935,
        "Frozen selected policy for each CRE part; one-time evaluation; no normalization or recalibration",
        ha="center",
        fontsize=12.5,
        color="#425466",
    )
    fig.supxlabel(
        "Three-seed ensemble prediction, log₂(total RNA / total DNA)",
        fontsize=12.5,
        y=0.025,
    )
    fig.supylabel(
        "Observed expression target, log₂(total RNA / total DNA)",
        fontsize=12.5,
        x=0.02,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.875, bottom=0.095, hspace=0.26, wspace=0.16)

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
    return save_figure(fig, "main_locked_final_test_calibration_shared_axes"), summary


def figure_intron_group_audit() -> tuple[list[str], pd.DataFrame]:
    """Compare pooled and within-group Intron performance on common axes."""

    selected = pd.read_csv(STAGE3_DIR / "stage3_selected_part_policies.csv")
    intron_policy = selected.loc[selected["part_slug"].eq("intron")].iloc[0]
    dev_estimands = pd.read_csv(
        STAGE3_DIR / "stage3_intron_equal_stratum_estimands.csv"
    )
    dev_estimand = dev_estimands.loc[
        dev_estimands["base_config_id"].eq(intron_policy["base_config_id"])
        & dev_estimands["rc_mode"].eq(intron_policy["rc_mode"])
        & dev_estimands["loss_mode"].eq(intron_policy["loss_mode"])
    ].iloc[0]
    dev_strata = pd.read_csv(STAGE3_DIR / "stage3_intron_stratum_metrics.csv")
    dev_strata = dev_strata.loc[
        dev_strata["base_config_id"].eq(intron_policy["base_config_id"])
        & dev_strata["rc_mode"].eq(intron_policy["rc_mode"])
        & dev_strata["loss_mode"].eq(intron_policy["loss_mode"])
    ].copy()

    final_estimands = pd.read_csv(FINAL_DIR / "audit_intron_estimand_metrics.csv")
    final_estimand = final_estimands.loc[
        final_estimands["predictor"].eq("ensemble_mean")
    ].iloc[0]
    final_strata = pd.read_csv(FINAL_DIR / "audit_intron_stratum_metrics.csv")
    final_strata = final_strata.loc[
        final_strata["predictor"].eq("ensemble_mean")
    ].copy()

    estimand_labels = [
        "All constructs\n(pooled)",
        "Within groups\n(centered)",
        "Average of\ngroup-specific r",
        "Weakest\ngroup",
    ]
    dev_est_values = np.array(
        [
            dev_estimand["natural_pooled_pearson"],
            dev_estimand["within_stratum_centered_pearson"],
            dev_estimand["macro_stratum_pearson"],
            dev_estimand["minimum_stratum_pearson"],
        ],
        dtype=float,
    )
    final_est_values = np.array(
        [
            final_estimand["natural_pooled_pearson"],
            final_estimand["within_stratum_centered_pearson"],
            final_estimand["macro_stratum_pearson"],
            final_estimand["minimum_stratum_pearson"],
        ],
        dtype=float,
    )

    stratum_order = ["mask1_specific", "mask2_not_mask1", "mask3_residual"]
    stratum_label = {
        "mask1_specific": "Sequence group 1",
        "mask2_not_mask1": "Sequence group 2",
        "mask3_residual": "Residual group",
    }
    dev_strata = dev_strata.set_index("inferred_intron_sensitivity_stratum")
    final_strata = final_strata.set_index("inferred_stratum")
    dev_stratum_values = np.array(
        [dev_strata.loc[stratum, "pearson"] for stratum in stratum_order], dtype=float
    )
    final_stratum_values = np.array(
        [final_strata.loc[stratum, "pearson"] for stratum in stratum_order], dtype=float
    )
    final_stratum_n = [int(final_strata.loc[stratum, "n"]) for stratum in stratum_order]

    fig, axes = plt.subplots(1, 2, figsize=(14.7, 6.7), sharey=True)
    dev_color = "#355F82"
    final_color = "#EF6C4D"

    def paired_points(
        ax: plt.Axes,
        labels: list[str],
        development: np.ndarray,
        final: np.ndarray,
    ) -> None:
        x = np.arange(len(labels))
        for index in range(len(labels)):
            ax.plot(
                [x[index] - 0.09, x[index] + 0.09],
                [development[index], final[index]],
                color="#AAB7C4",
                linewidth=1.8,
                zorder=1,
            )
        ax.scatter(
            x - 0.09,
            development,
            s=78,
            marker="o",
            facecolor="white",
            edgecolor=dev_color,
            linewidth=2,
            zorder=3,
        )
        ax.scatter(
            x + 0.09,
            final,
            s=88,
            marker="D",
            facecolor=final_color,
            edgecolor="#7A2E1D",
            linewidth=0.8,
            zorder=4,
        )
        for index, value in enumerate(final):
            ax.text(
                x[index] + 0.09,
                value + 0.027,
                f"{value:.3f}",
                ha="center",
                fontsize=9.5,
                fontweight="bold",
                color="#7A2E1D",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10.2)
        ax.set_ylim(0, 0.76)
        ax.grid(axis="x", visible=False)

    paired_points(axes[0], estimand_labels, dev_est_values, final_est_values)
    axes[0].set_title("A. The pooled score is only one estimand")
    axes[0].set_ylabel("Pearson r")

    group_labels = [
        f"{stratum_label[stratum]}\n(final n={n})"
        for stratum, n in zip(stratum_order, final_stratum_n)
    ]
    paired_points(axes[1], group_labels, dev_stratum_values, final_stratum_values)
    axes[1].set_title("B. Predictive strength differs across groups")
    target_means = [float(dev_strata.loc[stratum, "target_mean"]) for stratum in stratum_order]
    axes[1].text(
        0.5,
        0.96,
        f"Development target means span {max(target_means) - min(target_means):.2f} log₂ units",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        color="#425466",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "#F3F6F9",
            "edgecolor": "#C6D0DA",
        },
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=dev_color,
            markeredgewidth=2,
            label="Development five-fold OOF",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=final_color,
            markeredgecolor="#7A2E1D",
            label="Locked final test",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.095),
        ncol=2,
        fontsize=11,
    )
    fig.suptitle(
        "Intron performance partly reflects three sequence-defined groups",
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.925,
        "Pooled correlation mixes separation of group means with prediction within each group",
        ha="center",
        fontsize=12.5,
        color="#425466",
    )
    fig.text(
        0.5,
        0.025,
        "Groups were inferred from sequence masks; they are not verified synthesis sublibraries or measured splicing states.",
        ha="center",
        fontsize=10.5,
        color="#52616B",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.22, wspace=0.10)

    rows = []
    for label, development, final in zip(
        estimand_labels, dev_est_values, final_est_values
    ):
        rows.append(
            {
                "panel": "estimand",
                "label": label.replace("\n", " "),
                "development_oof_pearson": development,
                "final_test_pearson": final,
            }
        )
    for label, development, final in zip(
        [stratum_label[s] for s in stratum_order],
        dev_stratum_values,
        final_stratum_values,
    ):
        rows.append(
            {
                "panel": "sequence_group",
                "label": label,
                "development_oof_pearson": development,
                "final_test_pearson": final,
            }
        )
    return save_figure(fig, "main_intron_sequence_group_audit"), pd.DataFrame(rows)


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "purpose": "Presentation-only replotting for the July 2026 Lib1 deduplicated baseline TAC deck",
        "final_test_replot_is_reporting_only": True,
        "figures": {},
        "source_files": {
            "stage1_metrics": str(
                (STAGE1_DIR / "stage1_exact_replay_metrics.csv").relative_to(REPO_ROOT)
            ),
            "stage1_candidates": str(
                (STAGE1_DIR / "stage2_candidate_selection_draft.csv").relative_to(
                    REPO_ROOT
                )
            ),
            "stage2_rc_pairs": str(
                (STAGE2_DIR / "stage2_rc_pair_metrics.csv").relative_to(REPO_ROOT)
            ),
            "stage3_loss_pairs": str(
                (STAGE3_DIR / "stage3_loss_pair_metrics.csv").relative_to(REPO_ROOT)
            ),
            "stage3_loss_fold_pairs": str(
                (STAGE3_DIR / "stage3_loss_fold_pair_metrics.csv").relative_to(
                    REPO_ROOT
                )
            ),
            "stage3_selected_policies": str(
                (STAGE3_DIR / "stage3_selected_part_policies.csv").relative_to(
                    REPO_ROOT
                )
            ),
            "final_predictions": str(
                (FINAL_DIR / "audit_ensemble_predictions.tsv.gz").relative_to(
                    REPO_ROOT
                )
            ),
            "final_metrics": str(
                (FINAL_DIR / "audit_metrics.csv").relative_to(REPO_ROOT)
            ),
        },
    }

    plot_functions = [
        ("hpo_landscape", figure_hpo_landscape, "hpo_configuration_summary.csv"),
        ("rc_effect", figure_rc_effect, "rc_route_summary.csv"),
        (
            "weighted_loss_effect",
            figure_weighted_loss_effect,
            "selected_weighted_loss_effects.csv",
        ),
        (
            "final_test_calibration",
            figure_final_test_calibration,
            "locked_final_test_metrics.csv",
        ),
        (
            "intron_group_audit",
            figure_intron_group_audit,
            "intron_group_audit_summary.csv",
        ),
    ]

    for key, function, table_name in plot_functions:
        figure_paths, table = function()
        table_path = TABLE_DIR / table_name
        table.to_csv(table_path, index=False)
        manifest["figures"][key] = {
            "outputs": figure_paths,
            "summary_table": str(table_path.relative_to(REPO_ROOT)),
        }

    manifest_path = OUTPUT_DIR / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(plot_functions)} figure sets to {FIGURE_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
