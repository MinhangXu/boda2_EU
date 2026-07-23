#!/usr/bin/env python3
"""Presentation reporting for the completed Lib1 dedup Stage 4 campaign.

This is deliberately a *display layer*.  It reads the compact products written
by ``lib1_dedup_stage4_downsampling_analysis.py`` and three explicitly
allow-listed, pre-dedup historical summaries.  For one compact full-N
calibration appendix, it follows only the analyzer-validated primary/full OOF
paths recorded in the completion table and verifies their SHA256 hashes.  It
does not read checkpoints, run registries, or any current final-test product,
and it does not rerun the paired bootstrap or fit a new learning curve.

The command fails closed unless the core analysis proves all 660 cells are
complete, the frozen 2,000-replicate bootstrap was used, and all final-test
isolation flags remain false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LEARN_ROOT = REPO_ROOT / "src" / "learn"
DEFAULT_CORE_DIR = (
    LEARN_ROOT / "outputs/analysis/lib1_dedup_stage4_downsampling_july2026"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CORE_DIR / "presentation"
DEFAULT_THRESHOLD_HISTORY = (
    REPO_ROOT
    / "tutorials/lib1_tasks/pretrain_CRE_inhouse_data/plots/"
    "lib1_barcode_threshold_downsample_june2026/tables/"
    "threshold_size_endpoint_summary.csv"
)
DEFAULT_EXACT_N1_HISTORY = (
    REPO_ROOT
    / "tutorials/lib1_tasks/pretrain_CRE_inhouse_data/plots/"
    "lib1_barcode_bin_matched_n1000_june2026/"
    "exact_n1_downsampling_learning_curve/exact_n1_downsampling_curve_summary.csv"
)
DEFAULT_ENHANCER_TRANSFER_HISTORY = (
    REPO_ROOT
    / "src/finetune/learning_curve/"
    "lib1_enhancer_threshold_hq8_random_mixed_b2_allheads_8seed_absgrid_may2026/"
    "combined/learning_curve_summary_mean_std.csv"
)

EXPECTED_CELLS = 660
EXPECTED_BOOTSTRAPS = 2_000
PARTS = ("enhancer", "promoter", "intron", "utr3", "utr5")
PART_LABELS = {
    "enhancer": "Enhancer",
    "promoter": "Promoter",
    "intron": "Intron",
    "utr3": "3′UTR",
    "utr5": "5′UTR",
}
HISTORICAL_PART_MAP = {
    "Enhancer": "enhancer",
    "Promoter": "promoter",
    "Intron": "intron",
    "3UTR": "utr3",
    "5UTR": "utr5",
}
STRATUM_LABELS = {
    "mask1_specific": "Mask 1 compatible",
    "mask2_not_mask1": "Mask 2, not 1",
    "mask3_residual": "Residual",
}
PART_COLORS = dict(zip(PARTS, ("#355C7D", "#2A9D8F", "#E9C46A", "#E76F51", "#7A5195")))
FAMILY_COLORS = {"power_law": "#4C78A8", "exponential": "#E45756"}
SIZE_ORDER = ("40", "250", "400", "2500", "4000", "full")

CORE_FILES = {
    "completion": "stage4_completion.csv",
    "points": "stage4_curve_points.csv",
    "pooled": "stage4_pooled_oof_metrics.csv",
    "contrasts": "stage4_observed_contrast_summary.csv",
    "intron": "stage4_intron_stratum_metrics.csv",
    "fits": "stage4_curve_fits.csv",
    "disagreement": "stage4_curve_family_disagreement.csv",
    "boot_metrics": "stage4_bootstrap_metric_intervals.csv",
    "boot_contrasts": "stage4_bootstrap_contrast_intervals.csv",
    "boot_curves": "stage4_bootstrap_curve_intervals.csv",
    "boot_failures": "stage4_bootstrap_fit_failures.csv",
    "boot_disagreement": "stage4_bootstrap_curve_family_disagreement_intervals.csv",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json(path: Path) -> dict:
    _require(path.is_file(), f"Required reporting input is absent: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    _require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_core_readiness(core_dir: str | Path) -> tuple[dict, dict]:
    """Prove completion and current-final-test isolation before any plotting."""
    root = Path(core_dir).expanduser().resolve()
    readiness = _read_json(root / "stage4_readiness.json")
    contract = _read_json(root / "stage4_analysis_contract.json")

    _require(readiness.get("analysis_mode") == "completed_oof_only", "Core analysis is not complete-OOF mode.")
    _require(int(readiness.get("manifest_rows", -1)) == EXPECTED_CELLS, "Unexpected Stage 4 manifest size.")
    _require(int(readiness.get("completed_cells", -1)) == EXPECTED_CELLS, "Stage 4 is not 660/660 complete.")
    _require(int(readiness.get("remaining_cells", -1)) == 0, "Stage 4 still has unfinished cells.")
    _require(int(readiness.get("complete_pooled_tracks", -1)) == 132, "Stage 4 pooled-track accounting changed.")
    _require(int(readiness.get("curve_point_rows", -1)) == 72, "Stage 4 curve-point accounting changed.")
    _require(int(readiness.get("bootstrap_resamples", -1)) == EXPECTED_BOOTSTRAPS, "Reporting requires the frozen 2,000-replicate bootstrap.")
    _require(readiness.get("manifest_validation_status") == "valid", "Frozen manifest did not validate.")

    for flag in (
        "global_registry_read",
        "final_test_loader_instantiated",
        "final_test_products_read",
        "final_test_metrics_computed",
    ):
        _require(readiness.get(flag) is False, f"Unsafe readiness flag {flag!r} is not false.")
    for flag in ("final_test_loader_instantiated", "final_test_products_read"):
        _require(contract.get(flag) is False, f"Unsafe analysis-contract flag {flag!r} is not false.")
    _require(contract.get("registry", {}).get("global_registry_read") is False, "Core contract does not prove dedicated-registry isolation.")
    _require(int(contract.get("bootstrap", {}).get("resamples", -1)) == EXPECTED_BOOTSTRAPS, "Core contract bootstrap count changed.")
    _require(contract.get("primary_estimand") == "pooled_five_fold_development_oof_pearson", "Primary estimand changed.")

    completion_path = root / CORE_FILES["completion"]
    _require(completion_path.is_file(), f"Missing completion table: {completion_path}")
    completion = pd.read_csv(completion_path)
    _require(len(completion) == EXPECTED_CELLS, "Completion table is not exactly 660 rows.")
    _require(completion["cell_id"].nunique() == EXPECTED_CELLS, "Completion table contains duplicate/missing cell IDs.")
    _require(completion["availability"].eq("complete").all(), "Completion table contains a non-complete cell.")
    return readiness, contract


def read_core_products(core_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(core_dir).expanduser().resolve()
    products: dict[str, pd.DataFrame] = {}
    for key, filename in CORE_FILES.items():
        path = root / filename
        _require(path.is_file(), f"Required core analysis table is absent: {path}")
        products[key] = pd.read_csv(path)
    for key in ("boot_metrics", "boot_contrasts", "boot_curves"):
        frame = products[key]
        _require(not frame.empty, f"Core bootstrap table {CORE_FILES[key]} is empty.")
        _require(frame["bootstrap_resamples"].eq(EXPECTED_BOOTSTRAPS).all(), f"Unexpected bootstrap count in {CORE_FILES[key]}.")
    return products


def _as_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    paths = [directory / f"{stem}.png", directory / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.20, linewidth=0.7)


def _interval_errors(
    point: Sequence[float], low: Sequence[float], high: Sequence[float]
) -> np.ndarray:
    values = np.asarray(point, dtype=float)
    return np.vstack(
        [
            np.maximum(0.0, values - np.asarray(low, dtype=float)),
            np.maximum(0.0, np.asarray(high, dtype=float) - values),
        ]
    )


def _primary_point_intervals(products: Mapping[str, pd.DataFrame], metric: str) -> pd.DataFrame:
    points = products["points"].copy()
    points["downsample_n_label"] = _as_label(points["downsample_n_label"])
    points = points.loc[points["stage4_lane"].eq("primary")].copy()
    intervals = products["boot_metrics"].copy()
    intervals["downsample_n_label"] = _as_label(intervals["downsample_n_label"])
    intervals = intervals.loc[
        intervals["stage4_lane"].eq("primary")
        & intervals["metric_scope"].eq("overall")
        & intervals["metric"].eq(metric)
    ].copy()
    keys = ["part_slug", "stage4_lane", "base_config_id", "downsample_n_label"]
    joined = points.merge(
        intervals[keys + ["ci_2_5", "ci_97_5", "bootstrap_mean"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    _require(joined["ci_2_5"].notna().all(), f"Missing primary {metric} bootstrap intervals.")
    return joined


def plot_primary_pearson(products: Mapping[str, pd.DataFrame], figure_dir: Path) -> list[Path]:
    frame = _primary_point_intervals(products, "pearson")
    low = float(frame["ci_2_5"].min())
    high = float(frame["ci_97_5"].max())
    margin = max((high - low) * 0.08, 0.025)
    limits = (max(-1.0, low - margin), min(1.0, high + margin))
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.1), sharey=True)
    for part, axis in zip(PARTS, axes):
        group = frame.loc[frame["part_slug"].eq(part)].sort_values("mean_actual_train_n")
        y = group["mean_pearson"].to_numpy(float)
        yerr = _interval_errors(y, group["ci_2_5"], group["ci_97_5"])
        axis.errorbar(
            group["mean_actual_train_n"], y, yerr=yerr,
            color=PART_COLORS[part], marker="o", linewidth=2.0, capsize=3,
            markersize=5, label="observed mean ± 95% CI",
        )
        axis.set_xscale("log")
        axis.set_ylim(*limits)
        axis.set_title(PART_LABELS[part], fontweight="semibold")
        axis.set_xlabel("Training constructs (log scale)")
        _style_axis(axis)
    axes[0].set_ylabel("Pooled five-fold development-OOF Pearson r")
    fig.suptitle("How predictive performance changes with training-set size", fontsize=15, fontweight="semibold")
    fig.text(0.5, 0.005, "Points average the three frozen nested subsets at finite N; bars are paired-bootstrap 95% intervals.", ha="center", fontsize=9)
    return _save_figure(fig, figure_dir, "01_primary_pearson_learning_curves")


def observed_10x_table(products: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    summary = products["contrasts"].copy()
    boot = products["boot_contrasts"].copy()
    summary = summary.loc[
        summary["stage4_lane"].eq("primary")
        & summary["multiplicative_contrast"].eq("10x")
    ].copy()
    boot = boot.loc[
        boot["stage4_lane"].eq("primary")
        & boot["multiplicative_contrast"].eq("10x")
        & boot["metric"].eq("pearson")
        & boot["metric_scope"].isin(["overall", "within_stratum_centered"])
    ].copy()
    keys = ["part_slug", "stage4_lane", "base_config_id", "low_n", "high_n", "multiplicative_contrast"]
    records = []
    for row in boot.itertuples(index=False):
        selected = summary
        for key in keys:
            selected = selected.loc[selected[key].eq(getattr(row, key))]
        _require(len(selected) == 1, "Could not match a direct 10× point estimate to its bootstrap interval.")
        point_field = "mean_delta_within_stratum_centered_pearson" if row.metric_scope == "within_stratum_centered" else "mean_delta_pearson"
        point = float(selected.iloc[0][point_field])
        records.append(
            {
                "part_slug": row.part_slug,
                "scope": row.metric_scope,
                "low_n": int(row.low_n),
                "high_n": int(row.high_n),
                "contrast": f"{int(row.low_n):,}→{int(row.high_n):,}",
                "delta_pearson": point,
                "ci_2_5": float(row.ci_2_5),
                "ci_97_5": float(row.ci_97_5),
                "evidence": (
                    "positive" if row.ci_2_5 > 0 else
                    "negative" if row.ci_97_5 < 0 else "uncertain"
                ),
            }
        )
    frame = pd.DataFrame(records)
    _require(len(frame) == 18, "Expected 15 overall plus three Intron-centered 10× contrasts.")
    return frame


def observed_100x_table(products: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the directly observed 40→4,000 contrast for each primary part."""
    summary = products["contrasts"].copy()
    boot = products["boot_contrasts"].copy()
    summary = summary.loc[
        summary["stage4_lane"].eq("primary")
        & summary["multiplicative_contrast"].eq("100x")
    ].copy()
    boot = boot.loc[
        boot["stage4_lane"].eq("primary")
        & boot["multiplicative_contrast"].eq("100x")
        & boot["metric"].eq("pearson")
        & boot["metric_scope"].eq("overall")
    ].copy()
    keys = [
        "part_slug", "stage4_lane", "base_config_id", "low_n", "high_n",
        "multiplicative_contrast",
    ]
    records = []
    for row in boot.itertuples(index=False):
        selected = summary
        for key in keys:
            selected = selected.loc[selected[key].eq(getattr(row, key))]
        _require(
            len(selected) == 1,
            "Could not match a direct 100× point estimate to its bootstrap interval.",
        )
        point = float(selected.iloc[0]["mean_delta_pearson"])
        records.append(
            {
                "part_slug": row.part_slug,
                "low_n": int(row.low_n),
                "high_n": int(row.high_n),
                "contrast": f"{int(row.low_n):,}→{int(row.high_n):,}",
                "delta_pearson": point,
                "ci_2_5": float(row.ci_2_5),
                "ci_97_5": float(row.ci_97_5),
                "evidence": (
                    "positive" if row.ci_2_5 > 0 else
                    "negative" if row.ci_97_5 < 0 else "uncertain"
                ),
            }
        )
    frame = pd.DataFrame(records)
    _require(len(frame) == 5, "Expected one directly observed 40→4,000 contrast per part.")
    return frame


def plot_observed_10x(frame: pd.DataFrame, figure_dir: Path) -> list[Path]:
    # The common 400→4,000 decade is the cleanest presentation comparison.
    # The complete three-decade table remains available as a CSV appendix.
    plot = frame.loc[frame["low_n"].eq(400) & frame["high_n"].eq(4000)].copy()
    plot["scope_order"] = plot["scope"].map({"overall": 0, "within_stratum_centered": 1})
    plot["part_order"] = plot["part_slug"].map({part: index for index, part in enumerate(PARTS)})
    plot = plot.sort_values(["part_order", "scope_order", "low_n"], ascending=[False, False, True]).reset_index(drop=True)
    labels = [
        f"{PART_LABELS[row.part_slug]}{' (centered)' if row.scope == 'within_stratum_centered' else ''}  {row.contrast}"
        for row in plot.itertuples(index=False)
    ]
    colors = pd.Series("#59A14F", index=plot.index)
    fig, axis = plt.subplots(figsize=(10.5, 7.2))
    y = np.arange(len(plot))
    for index, row in enumerate(plot.itertuples(index=False)):
        axis.plot([row.ci_2_5, row.ci_97_5], [index, index], color=colors.iloc[index], linewidth=2.2)
        axis.scatter(row.delta_pearson, index, color=colors.iloc[index], s=42, zorder=3, edgecolor="white", linewidth=0.6)
    axis.axvline(0, color="#333333", linestyle="--", linewidth=1)
    axis.set_yticks(y, labels)
    axis.set_xlabel("Observed Δ development-OOF Pearson r")
    axis.set_title("Observed gain over the common 400→4,000 decade", fontsize=15, fontweight="semibold")
    axis.text(0.01, -0.10, "Intervals pair the same OOF constructs and nested training-subset tracks. Centered Intron rows remove frozen stratum means.", transform=axis.transAxes, fontsize=9)
    _style_axis(axis)
    axis.grid(axis="x", alpha=0.22)
    axis.grid(axis="y", visible=False)
    return _save_figure(fig, figure_dir, "02_observed_10x_pearson_forest")


def intron_curve_table(products: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    pooled = products["pooled"].copy()
    pooled["downsample_n_label"] = _as_label(pooled["downsample_n_label"])
    pooled = pooled.loc[(pooled["part_slug"].eq("intron")) & pooled["stage4_lane"].eq("primary")]
    point_lookup = products["points"].copy()
    point_lookup["downsample_n_label"] = _as_label(point_lookup["downsample_n_label"])
    point_lookup = point_lookup.loc[(point_lookup["part_slug"].eq("intron")) & point_lookup["stage4_lane"].eq("primary")]
    n_map = point_lookup.set_index("downsample_n_label")["mean_actual_train_n"].to_dict()

    records = []
    for label, group in pooled.groupby("downsample_n_label", sort=False):
        records.extend(
            [
                {"downsample_n_label": label, "mean_actual_train_n": n_map[label], "scope": "overall", "stratum": "", "pearson": float(group["pearson"].mean())},
                {"downsample_n_label": label, "mean_actual_train_n": n_map[label], "scope": "within_stratum_centered", "stratum": "", "pearson": float(group["within_stratum_centered_pearson"].mean())},
            ]
        )
    strata = products["intron"].copy()
    strata["downsample_n_label"] = _as_label(strata["downsample_n_label"])
    strata = strata.loc[strata["stage4_lane"].eq("primary")]
    for (label, stratum), group in strata.groupby(["downsample_n_label", "inferred_intron_sensitivity_stratum"], sort=False):
        records.append(
            {"downsample_n_label": label, "mean_actual_train_n": n_map[label], "scope": "per_stratum", "stratum": stratum, "pearson": float(group["pearson"].mean())}
        )
    points = pd.DataFrame(records)

    boot = products["boot_metrics"].copy()
    boot["downsample_n_label"] = _as_label(boot["downsample_n_label"])
    boot = boot.loc[
        boot["part_slug"].eq("intron")
        & boot["stage4_lane"].eq("primary")
        & boot["metric"].eq("pearson")
        & boot["metric_scope"].isin(["overall", "within_stratum_centered", "per_stratum"])
    ].copy()
    boot["stratum"] = boot["inferred_intron_sensitivity_stratum"].fillna("")
    joined = points.merge(
        boot[["downsample_n_label", "metric_scope", "stratum", "ci_2_5", "ci_97_5"]],
        left_on=["downsample_n_label", "scope", "stratum"],
        right_on=["downsample_n_label", "metric_scope", "stratum"],
        how="left",
        validate="one_to_one",
    ).drop(columns="metric_scope")
    _require(joined["ci_2_5"].notna().all(), "Missing Intron scoped bootstrap intervals.")
    return joined


def plot_intron(frame: pd.DataFrame, figure_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharey=True)
    top_styles = {
        "overall": ("Pooled (between + within strata)", "#355C7D"),
        "within_stratum_centered": ("Within-stratum centered", "#D1495B"),
    }
    for scope, (label, color) in top_styles.items():
        group = frame.loc[frame["scope"].eq(scope)].sort_values("mean_actual_train_n")
        y = group["pearson"].to_numpy(float)
        axes[0].errorbar(group["mean_actual_train_n"], y, yerr=_interval_errors(y, group["ci_2_5"], group["ci_97_5"]), marker="o", capsize=3, linewidth=2, color=color, label=label)
    for stratum, group in frame.loc[frame["scope"].eq("per_stratum")].groupby("stratum"):
        group = group.sort_values("mean_actual_train_n")
        y = group["pearson"].to_numpy(float)
        axes[1].errorbar(group["mean_actual_train_n"], y, yerr=_interval_errors(y, group["ci_2_5"], group["ci_97_5"]), marker="o", capsize=2.5, linewidth=1.8, label=STRATUM_LABELS.get(stratum, stratum))
    for axis, title in zip(axes, ("What pooling contributes", "Performance inside each frozen stratum")):
        axis.set_xscale("log")
        axis.axhline(0, color="#555555", linewidth=0.8)
        axis.set_xlabel("Training constructs (log scale)")
        axis.set_title(title, fontweight="semibold")
        axis.legend(frameon=False, fontsize=8)
        _style_axis(axis)
    axes[0].set_ylabel("Development-OOF Pearson r")
    fig.suptitle("Intron learning curves: pooled signal versus within-stratum prediction", fontsize=15, fontweight="semibold")
    fig.text(
        0.5,
        0.005,
        "Inferred design/mask strata are not measured splicing classes and do not replace the proposed position-balanced 80-bp evaluation.",
        ha="center",
        fontsize=9,
    )
    return _save_figure(fig, figure_dir, "03_intron_scoped_learning_curves")


def enhancer_transfer_scratch_table(products: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    points = products["points"].copy()
    points["downsample_n_label"] = _as_label(points["downsample_n_label"])
    points = points.loc[
        points["part_slug"].eq("enhancer")
        & points["stage4_lane"].isin(["primary", "scratch_diagnostic"])
    ].copy()
    boot = products["boot_metrics"].copy()
    boot["downsample_n_label"] = _as_label(boot["downsample_n_label"])
    boot = boot.loc[
        boot["part_slug"].eq("enhancer")
        & boot["stage4_lane"].isin(["primary", "scratch_diagnostic"])
        & boot["metric_scope"].eq("overall")
        & boot["metric"].eq("pearson")
    ]
    keys = ["part_slug", "stage4_lane", "base_config_id", "downsample_n_label"]
    joined = points.merge(boot[keys + ["ci_2_5", "ci_97_5"]], on=keys, how="left", validate="one_to_one")
    _require(len(joined) == 12 and joined["ci_2_5"].notna().all(), "Enhancer transfer/scratch curve is incomplete.")
    return joined


def plot_enhancer_transfer_scratch(frame: pd.DataFrame, figure_dir: Path) -> list[Path]:
    fig, axis = plt.subplots(figsize=(7.8, 4.8))
    styles = {
        "primary": ("Selected K562/full transfer + RC", "#355C7D", "o"),
        "scratch_diagnostic": ("Scratch ResNet1D, RC off (diagnostic)", "#E76F51", "s"),
    }
    for lane, (label, color, marker) in styles.items():
        group = frame.loc[frame["stage4_lane"].eq(lane)].sort_values("mean_actual_train_n")
        y = group["mean_pearson"].to_numpy(float)
        axis.errorbar(group["mean_actual_train_n"], y, yerr=_interval_errors(y, group["ci_2_5"], group["ci_97_5"]), marker=marker, capsize=3, linewidth=2, color=color, label=label)
    axis.set_xscale("log")
    axis.set_xlabel("Training constructs (log scale)")
    axis.set_ylabel("Pooled five-fold development-OOF Pearson r")
    axis.set_title("Enhancer: transfer retains an advantage over the scratch diagnostic", fontsize=14, fontweight="semibold")
    axis.legend(frameon=False)
    _style_axis(axis)
    axis.text(0.01, -0.18, "Diagnostic—not a controlled pretraining-only contrast: architecture, input/RC policy, and initialization all differ.", transform=axis.transAxes, fontsize=9)
    return _save_figure(fig, figure_dir, "04_enhancer_transfer_vs_scratch")


def alternative_point_deltas(products: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    pooled = products["pooled"].copy()
    pooled["downsample_n_label"] = _as_label(pooled["downsample_n_label"])
    primary = pooled.loc[pooled["stage4_lane"].eq("primary")].copy()
    alternatives = pooled.loc[pooled["stage4_lane"].eq("alternative")].copy()
    records = []
    for row in alternatives.itertuples(index=False):
        comparison = primary.loc[
            primary["part_slug"].eq(row.part_slug)
            & primary["downsample_n_label"].eq(row.downsample_n_label)
        ]
        if str(row.downsample_n_label) != "full":
            comparison = comparison.loc[
                comparison["subset_replicate"].eq(int(row.subset_replicate))
                & comparison["train_subsample_seed"].eq(int(row.train_subsample_seed))
            ]
        _require(len(comparison) == 1, "Alternative sensitivity anchor lacks an exactly matched primary point.")
        baseline = comparison.iloc[0]
        records.append(
            {
                "part_slug": row.part_slug,
                "portfolio_rank": int(row.portfolio_rank),
                "alternative_base_config_id": row.base_config_id,
                "alternative_architecture": row.architecture,
                "downsample_n_label": str(row.downsample_n_label),
                "mean_actual_train_n": float(row.mean_actual_train_n),
                "alternative_pearson": float(row.pearson),
                "matched_primary_pearson": float(baseline.pearson),
                "alternative_minus_primary_pearson": float(row.pearson - baseline.pearson),
                "uncertainty_status": "point_delta_only_no_paired_bootstrap_interval",
            }
        )
    result = pd.DataFrame(records).sort_values(["part_slug", "portfolio_rank", "mean_actual_train_n"])
    _require(len(result) == 36, "Expected nine alternatives × four shared anchors.")
    return result


def plot_alternative_deltas(frame: pd.DataFrame, figure_dir: Path) -> list[Path]:
    rows = frame[["part_slug", "portfolio_rank", "alternative_base_config_id", "alternative_architecture"]].drop_duplicates().copy()
    rows["part_order"] = rows["part_slug"].map({part: i for i, part in enumerate(PARTS)})
    rows = rows.sort_values(["part_order", "portfolio_rank"])
    labels = [f"{PART_LABELS[row.part_slug]}  rank {row.portfolio_rank}\n{row.alternative_architecture} · {row.alternative_base_config_id[8:16]}" for row in rows.itertuples(index=False)]
    matrix = np.full((len(rows), 4), np.nan)
    for i, row in enumerate(rows.itertuples(index=False)):
        subset = frame.loc[frame["alternative_base_config_id"].eq(row.alternative_base_config_id)].set_index("downsample_n_label")
        matrix[i] = [float(subset.loc[label, "alternative_minus_primary_pearson"]) for label in ("40", "400", "4000", "full")]
    limit = max(float(np.nanmax(np.abs(matrix))), 0.01)
    fig, axis = plt.subplots(figsize=(9.2, 6.0))
    image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center", fontsize=8, color="white" if abs(matrix[i, j]) > 0.55 * limit else "#222222")
    axis.set_xticks(range(4), ("40", "400", "4,000", "full"))
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Shared training-size anchor")
    axis.set_title("Portfolio sensitivity: alternative − frozen primary Pearson r", fontsize=14, fontweight="semibold")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("Point ΔPearson")
    fig.text(0.5, 0.005, "Descriptive matched-track point deltas only—no paired-bootstrap CI and no post hoc reselection.", ha="center", fontsize=9)
    return _save_figure(fig, figure_dir, "05_portfolio_sensitivity_point_deltas")


def curve_scenario_table(products: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    fits = products["fits"].copy()
    fits = fits.loc[
        fits["stage4_lane"].eq("primary")
        & fits["metric"].eq("pearson")
        & fits["fit_status"].eq("success")
    ].copy()
    boot = products["boot_curves"].copy()
    boot = boot.loc[
        boot["stage4_lane"].eq("primary")
        & boot["metric"].eq("pearson")
        & boot["quantity"].eq("projected_gain_full_to_10x")
    ].copy()
    keys = ["part_slug", "stage4_lane", "base_config_id", "metric", "curve_family"]
    joined = fits.merge(boot[keys + ["ci_2_5", "ci_97_5", "successful_bootstrap_replicates"]], on=keys, how="left", validate="one_to_one")
    _require(len(joined) == 10 and joined["ci_2_5"].notna().all(), "Primary curve-scenario bootstrap table is incomplete.")
    _require(joined["successful_bootstrap_replicates"].gt(0).all(), "Every primary scenario fit failed in bootstrap.")
    return joined


def curve_disagreement_table(products: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    point = products["disagreement"].loc[
        products["disagreement"]["stage4_lane"].eq("primary")
        & products["disagreement"]["metric"].eq("pearson")
    ].copy()
    boot = products["boot_disagreement"].loc[
        products["boot_disagreement"]["stage4_lane"].eq("primary")
        & products["boot_disagreement"]["metric"].eq("pearson")
        & products["boot_disagreement"]["projection_horizon"].eq("10x")
    ].copy()
    keys = ["part_slug", "stage4_lane", "base_config_id", "metric"]
    joined = point.merge(
        boot[keys + ["ci_2_5", "ci_97_5", "successful_bootstrap_replicates"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    _require(len(joined) == 5 and joined["ci_2_5"].notna().all(), "Curve-family disagreement intervals are incomplete.")
    return joined


def plot_curve_scenarios(
    frame: pd.DataFrame, disagreement: pd.DataFrame, figure_dir: Path
) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.0))
    offsets = {"power_law": -0.13, "exponential": 0.13}
    for family in ("power_law", "exponential"):
        group = frame.loc[frame["curve_family"].eq(family)].set_index("part_slug").loc[list(PARTS)].reset_index()
        y = np.arange(len(PARTS)) + offsets[family]
        x = group["projected_gain_full_to_10x"].to_numpy(float)
        axes[0].errorbar(x, y, xerr=_interval_errors(x, group["ci_2_5"], group["ci_97_5"]), fmt="o", capsize=3, color=FAMILY_COLORS[family], label=family.replace("_", " "))
        axes[1].scatter(group["loo_rmse"], y, color=FAMILY_COLORS[family], label=family.replace("_", " "), s=45)
    for axis in axes[:2]:
        axis.set_yticks(np.arange(len(PARTS)), [PART_LABELS[p] for p in PARTS])
        axis.invert_yaxis()
        _style_axis(axis)
        axis.grid(axis="x", alpha=0.22)
        axis.grid(axis="y", visible=False)
        axis.legend(frameon=False, fontsize=8)
    axes[0].axvline(0, color="#333333", linestyle="--", linewidth=0.9)
    axes[0].set_xlabel("Scenario ΔPearson: fitted(full) → fitted(10× full)")
    axes[0].set_title("Tail scenario and paired-bootstrap 95% CI", fontweight="semibold")
    axes[1].set_xlabel("Leave-one-size-out RMSE (lower is better)")
    axes[1].set_title("How well each family predicts omitted sizes", fontweight="semibold")
    disagreement = disagreement.set_index("part_slug").loc[list(PARTS)].reset_index()
    y = np.arange(len(PARTS))
    x = disagreement["absolute_10x_gain_disagreement"].to_numpy(float)
    axes[2].errorbar(
        x, y,
        xerr=_interval_errors(x, disagreement["ci_2_5"], disagreement["ci_97_5"]),
        fmt="o", capsize=3, color="#7A5195",
    )
    axes[2].set_yticks(y, [PART_LABELS[p] for p in PARTS])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("|power-law gain − exponential gain|")
    axes[2].set_title("Tail-family disagreement and 95% CI", fontweight="semibold")
    axes[2].grid(axis="x", alpha=0.22)
    axes[2].grid(axis="y", visible=False)
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Curve families are sensitivity scenarios—not primary evidence", fontsize=15, fontweight="semibold")
    boundary_parts = frame.loc[
        frame["curve_family"].eq("power_law") & frame["asymptote"].ge(0.999),
        "part_slug",
    ].tolist()
    boundary_text = ", ".join(PART_LABELS[part] for part in boundary_parts)
    fig.text(0.5, 0.005, f"Family disagreement is tail-shape uncertainty. Power-law asymptote reached the allowed boundary for: {boundary_text}.", ha="center", fontsize=9)
    return _save_figure(fig, figure_dir, "06_curve_family_scenarios")


def load_historical_context(
    current_points: pd.DataFrame,
    threshold_path: str | Path,
    exact_n1_path: str | Path,
    enhancer_transfer_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load explicitly selected histories and normalize only within each series."""
    threshold_path = Path(threshold_path).expanduser().resolve()
    exact_n1_path = Path(exact_n1_path).expanduser().resolve()
    enhancer_transfer_path = Path(enhancer_transfer_path).expanduser().resolve()
    for path in (threshold_path, exact_n1_path, enhancer_transfer_path):
        _require(path.is_file(), f"Historical comparison source is absent: {path}")

    records: list[dict] = []
    current = current_points.loc[current_points["stage4_lane"].eq("primary")].copy()
    for row in current.itertuples(index=False):
        records.append({"study_id": "stage4_dedup_oof", "study_label": "Current dedup Stage 4\ndevelopment OOF", "part_slug": row.part_slug, "route": row.training_regime, "n": float(row.mean_actual_train_n), "pearson": float(row.mean_pearson), "evaluation": "pooled five-fold development OOF", "upstream_deduplicated": True})

    threshold = pd.read_csv(threshold_path)
    threshold = threshold.loc[threshold["threshold_display"].eq("1+")].copy()
    for row in threshold.itertuples(index=False):
        records.append({"study_id": "prededup_threshold_scratch", "study_label": "Pre-dedup threshold ≥1\nscratch (historical final test)", "part_slug": HISTORICAL_PART_MAP[row.part], "route": "scratch", "n": float(row.train_size_x), "pearson": float(row.test_pearson_mean), "evaluation": "historical final-test mean across configs/seeds", "upstream_deduplicated": False})

    exact = pd.read_csv(exact_n1_path)
    for row in exact.itertuples(index=False):
        records.append({"study_id": "prededup_exact_n1_scratch", "study_label": "Pre-dedup exact n=1\nscratch (historical final test)", "part_slug": HISTORICAL_PART_MAP[row.part], "route": "scratch", "n": float(row.train_rows_median), "pearson": float(row.test_pearson_mean), "evaluation": "historical final-test mean across configs/seeds", "upstream_deduplicated": False})

    enhancer = pd.read_csv(enhancer_transfer_path)
    enhancer = enhancer.loc[
        enhancer["setting"].eq("B2_with_RC")
        & enhancer["train_threshold"].eq(1)
        & enhancer["init_head"].eq("K562")
        & enhancer["unfreeze_scope"].eq("full")
    ].copy()
    _require(
        len(enhancer) == 5,
        "Frozen five-point historical K562/full Enhancer transfer slice is absent.",
    )
    for row in enhancer.itertuples(index=False):
        records.append({"study_id": "prededup_enhancer_transfer", "study_label": "Pre-dedup Enhancer K562\ntransfer (historical test)", "part_slug": "enhancer", "route": "transfer", "n": float(row.train_size), "pearson": float(row.test_pearson_mean), "evaluation": "historical per-seed test mean", "upstream_deduplicated": False})

    frame = pd.DataFrame(records).sort_values(["study_id", "part_slug", "n"]).reset_index(drop=True)
    normalized = []
    for _, group in frame.groupby(["study_id", "part_slug", "route"], sort=False):
        group = group.sort_values("n").copy()
        baseline = float(group.iloc[0]["pearson"])
        maximum_n = float(group["n"].max())
        group["fraction_of_series_max_n"] = group["n"] / maximum_n
        group["delta_pearson_from_smallest_n"] = group["pearson"] - baseline
        group["normalization"] = "x=N/max_N_within_series; y=r-r_at_smallest_N_within_series"
        normalized.append(group)
    frame = pd.concat(normalized, ignore_index=True)

    design = pd.DataFrame(
        [
            {
                "study_id": "stage4_dedup_oof", "upstream": "exact dedup", "parts": "Enhancer, Promoter, Intron, 3′UTR, 5′UTR", "route": "frozen Stage 3 part policy; Enhancer transfer", "sizes": "40, 250, 400, 2,500, 4,000, full", "replication": "3 nested subset tracks at finite N; 5 outer folds; model seed 1701", "evaluation": "pooled five-fold development OOF", "selection_or_context": "current primary inference", "absolute_cross_study_comparison": False,
            },
            {
                "study_id": "prededup_threshold_scratch", "upstream": "pre-dedup", "parts": "Promoter, Intron, 3′UTR, 5′UTR", "route": "scratch; displayed slice uses barcode threshold ≥1", "sizes": "100, 500, 1,500, 2,500, 3,500, full", "replication": "5 configs × 5 split seeds at each point", "evaluation": "historical final-test means", "selection_or_context": "shape context only", "absolute_cross_study_comparison": False,
            },
            {
                "study_id": "prededup_exact_n1_scratch", "upstream": "pre-dedup", "parts": "Promoter, 3′UTR, 5′UTR", "route": "scratch; exact one-barcode training rows", "sizes": "250, 500, 1,000, full", "replication": "3 configs × 5 split seeds", "evaluation": "historical final-test means", "selection_or_context": "data-quality context only", "absolute_cross_study_comparison": False,
            },
            {
                "study_id": "prededup_enhancer_transfer", "upstream": "pre-dedup", "parts": "Enhancer", "route": "K562 full transfer, B2 + RC, threshold 1", "sizes": "50, 400, 1,000, 2,000, full", "replication": "8 seeds; heldouts vary by seed", "evaluation": "historical per-seed test means", "selection_or_context": "transfer-learning context only", "absolute_cross_study_comparison": False,
            },
        ]
    )
    return frame, design


def plot_historical_context(frame: pd.DataFrame, figure_dir: Path) -> list[Path]:
    study_order = ("stage4_dedup_oof", "prededup_threshold_scratch", "prededup_exact_n1_scratch", "prededup_enhancer_transfer")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharey=True)
    for study, axis in zip(study_order, axes):
        subset = frame.loc[frame["study_id"].eq(study)]
        for part, group in subset.groupby("part_slug"):
            group = group.sort_values("fraction_of_series_max_n")
            axis.plot(group["fraction_of_series_max_n"], group["delta_pearson_from_smallest_n"], marker="o", linewidth=1.8, color=PART_COLORS[part], label=PART_LABELS[part])
        axis.axhline(0, color="#444444", linewidth=0.8)
        axis.set_xscale("log")
        minimum = float(subset["fraction_of_series_max_n"].min())
        lower = max(minimum * 0.8, 0.004)
        axis.set_xlim(lower, 1.08)
        if minimum >= 0.10:
            ticks, labels = [0.2, 0.5, 1.0], ["0.2", "0.5", "1"]
        elif minimum >= 0.008:
            ticks, labels = [0.02, 0.1, 0.5, 1.0], ["0.02", "0.1", "0.5", "1"]
        else:
            ticks, labels = [0.005, 0.02, 0.1, 0.5, 1.0], ["0.005", "0.02", "0.1", "0.5", "1"]
        visible = [(tick, label) for tick, label in zip(ticks, labels) if tick >= lower]
        axis.set_xticks([tick for tick, _ in visible])
        axis.set_xticklabels([label for _, label in visible])
        axis.tick_params(axis="x", which="minor", labelbottom=False)
        axis.set_title(str(subset.iloc[0]["study_label"]), fontsize=10.5, fontweight="semibold")
        axis.set_xlabel("N / study-specific max N")
        axis.legend(frameon=False, fontsize=7)
        _style_axis(axis)
    axes[0].set_ylabel("Pearson r − r at smallest N in that series")
    fig.suptitle("Historical learning-curve shape context (absolute levels are not comparable)", fontsize=15, fontweight="semibold")
    fig.text(0.5, 0.005, "Studies differ in upstream deduplication, model/config selection, targets/splits, replication, and evaluation sets.", ha="center", fontsize=9)
    return _save_figure(fig, figure_dir, "07_historical_shape_context_noncomparable")


def full_n_raw_calibration(products: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute raw slope/bias from analyzer-authorized primary/full OOF files.

    The regression is observed target on raw prediction, matching the prior
    calibration convention.  Bias is mean(prediction - target).  These are
    descriptive full-N OOF point estimates; the frozen paired bootstrap did
    not include slope or bias intervals.
    """
    completion = products["completion"].copy()
    completion["downsample_n_label"] = _as_label(completion["downsample_n_label"])
    selected = completion.loc[
        completion["stage4_lane"].eq("primary")
        & completion["downsample_n_label"].eq("full")
    ].copy()
    _require(len(selected) == 25, "Expected five primary/full OOF folds for each of five parts.")
    campaign_root = (
        LEARN_ROOT / "outputs/hpo_runs/lib1_dedup_stage4_downsampling_july2026"
    ).resolve()
    pooled_metrics = products["pooled"].copy()
    pooled_metrics["downsample_n_label"] = _as_label(pooled_metrics["downsample_n_label"])
    records = []
    for part in PARTS:
        rows = selected.loc[selected["part_slug"].eq(part)].sort_values("outer_oof_fold")
        _require(rows["outer_oof_fold"].astype(int).tolist() == [0, 1, 2, 3, 4], f"{part} full-N calibration folds are incomplete.")
        frames = []
        for row in rows.itertuples(index=False):
            path = Path(row.resolved_prediction_path).expanduser().resolve()
            try:
                path.relative_to(campaign_root)
            except ValueError as error:
                raise RuntimeError(f"OOF calibration path escaped the Stage 4 campaign root: {path}") from error
            _require(path.name.endswith("__oof_predictions.tsv") and path.parent.name == "predictions", f"Calibration input is not an OOF prediction export: {path}")
            _require("test" not in {piece.lower() for piece in path.parts}, f"Calibration path contains a forbidden test component: {path}")
            _require(path.is_file(), f"Validated OOF prediction export is absent: {path}")
            _require(_sha256_file(path) == str(row.resolved_prediction_sha256), f"OOF calibration SHA256 mismatch: {path}")
            frame = pd.read_csv(path, sep="\t")
            required = {"construct_id", "log2_RNA_DNA", "prediction_raw"}
            _require(required.issubset(frame.columns), f"OOF calibration schema changed: {path}")
            frame = frame[["construct_id", "log2_RNA_DNA", "prediction_raw"]].copy()
            frame["outer_oof_fold"] = int(row.outer_oof_fold)
            frames.append(frame)
        pooled = pd.concat(frames, ignore_index=True)
        _require(not pooled["construct_id"].duplicated().any(), f"{part} full-N OOF IDs are not unique.")
        target = pooled["log2_RNA_DNA"].to_numpy(float)
        prediction = pooled["prediction_raw"].to_numpy(float)
        _require(np.isfinite(target).all() and np.isfinite(prediction).all(), f"{part} full-N calibration contains non-finite values.")
        pearson = float(np.corrcoef(target, prediction)[0, 1])
        slope, intercept = np.polyfit(prediction, target, 1)
        reference = pooled_metrics.loc[
            pooled_metrics["part_slug"].eq(part)
            & pooled_metrics["stage4_lane"].eq("primary")
            & pooled_metrics["downsample_n_label"].eq("full")
        ]
        _require(len(reference) == 1, f"{part} lacks one primary/full pooled-metric row.")
        _require(len(pooled) == int(reference.iloc[0]["n_constructs"]), f"{part} calibration OOF row count changed.")
        _require(math.isclose(pearson, float(reference.iloc[0]["pearson"]), rel_tol=0, abs_tol=1e-10), f"{part} calibration Pearson does not reconcile with the core analyzer.")
        records.append(
            {
                "part_slug": part,
                "part": PART_LABELS[part],
                "n_oof_constructs": len(pooled),
                "pearson": pearson,
                "observed_on_prediction_slope": float(slope),
                "observed_on_prediction_intercept": float(intercept),
                "mean_prediction_minus_target_bias": float(np.mean(prediction - target)),
                "ideal_slope": 1.0,
                "ideal_bias": 0.0,
                "uncertainty_status": "descriptive_point_estimate_no_bootstrap_interval",
                "source": "analyzer_validated_primary_full_development_oof_exports",
                "current_final_test_products_read": False,
            }
        )
    return pd.DataFrame(records)


def plot_full_n_raw_calibration(frame: pd.DataFrame, figure_dir: Path) -> list[Path]:
    plot = frame.set_index("part_slug").loc[list(PARTS)].reset_index()
    y = np.arange(len(plot))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), sharey=True)
    axes[0].scatter(plot["observed_on_prediction_slope"], y, s=60, color=[PART_COLORS[p] for p in plot["part_slug"]], edgecolor="white", linewidth=0.7, zorder=3)
    axes[0].axvline(1, color="#333333", linestyle="--", linewidth=1, label="ideal slope = 1")
    axes[0].set_xlabel("Observed-on-prediction slope")
    axes[0].set_title("Scale calibration", fontweight="semibold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].scatter(plot["mean_prediction_minus_target_bias"], y, s=60, color=[PART_COLORS[p] for p in plot["part_slug"]], edgecolor="white", linewidth=0.7, zorder=3)
    axes[1].axvline(0, color="#333333", linestyle="--", linewidth=1, label="ideal bias = 0")
    axes[1].set_xlabel("Mean(prediction − target), raw log₂ RNA/DNA")
    axes[1].set_title("Mean calibration", fontweight="semibold")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_yticks(y, plot["part"])
        axis.grid(axis="x", alpha=0.22)
        axis.grid(axis="y", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
    # The axes share y, so invert once; inverting each shared axis would toggle
    # the common direction twice and put the canonical part order upside down.
    axes[0].invert_yaxis()
    fig.suptitle("Full-N raw-scale calibration on pooled development OOF", fontsize=15, fontweight="semibold")
    fig.text(0.5, 0.005, "Point estimates only; observed target is regressed on raw prediction. Current final test remains untouched.", ha="center", fontsize=9)
    return _save_figure(fig, figure_dir, "08_full_n_raw_calibration")


def plot_cod_appendix(products: Mapping[str, pd.DataFrame], figure_dir: Path) -> list[Path]:
    frame = _primary_point_intervals(products, "cod_r2")
    low = float(frame["ci_2_5"].min())
    high = float(frame["ci_97_5"].max())
    margin = max((high - low) * 0.05, 0.05)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.0), sharey=True)
    for part, axis in zip(PARTS, axes):
        group = frame.loc[frame["part_slug"].eq(part)].sort_values("mean_actual_train_n")
        y = group["mean_cod_r2"].to_numpy(float)
        axis.errorbar(group["mean_actual_train_n"], y, yerr=_interval_errors(y, group["ci_2_5"], group["ci_97_5"]), marker="o", capsize=3, linewidth=1.8, color=PART_COLORS[part])
        axis.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_ylim(low - margin, high + margin)
        axis.set_title(PART_LABELS[part], fontweight="semibold")
        axis.set_xlabel("Training constructs")
        _style_axis(axis)
    axes[0].set_ylabel("Raw-scale COD R² (common y-axis)")
    fig.suptitle("Calibration appendix: variance explained improves with sample size", fontsize=15, fontweight="semibold")
    fig.text(0.5, 0.005, "COD R² can be negative: zero is the train-independent target-mean reference; Pearson alone does not guarantee calibration.", ha="center", fontsize=9)
    return _save_figure(fig, figure_dir, "09_cod_calibration_appendix")


def build_decision_scorecard(
    products: Mapping[str, pd.DataFrame],
    primary_intervals: pd.DataFrame,
    contrast_table: pd.DataFrame,
    contrast_100x_table: pd.DataFrame,
    intron_table: pd.DataFrame,
    enhancer_table: pd.DataFrame,
    alternative_deltas: pd.DataFrame,
    scenarios: pd.DataFrame,
    disagreement_intervals: pd.DataFrame,
) -> pd.DataFrame:
    records = []
    disagreements = products["disagreement"].loc[
        products["disagreement"]["stage4_lane"].eq("primary")
        & products["disagreement"]["metric"].eq("pearson")
    ].set_index("part_slug")
    disagreement_intervals = disagreement_intervals.set_index("part_slug")
    for part in PARTS:
        point = primary_intervals.loc[primary_intervals["part_slug"].eq(part)].copy()
        by_label = point.set_index("downsample_n_label")
        contrasts = contrast_table.loc[
            contrast_table["part_slug"].eq(part) & contrast_table["scope"].eq("overall")
        ].sort_values("low_n")
        contrast_100x = contrast_100x_table.loc[
            contrast_100x_table["part_slug"].eq(part)
        ]
        _require(len(contrast_100x) == 1, f"{part} lacks its observed 40→4,000 contrast.")
        contrast_100x = contrast_100x.iloc[0]
        positive_count = int(contrasts["evidence"].eq("positive").sum())
        full = by_label.loc["full"]
        n4000 = by_label.loc["4000"]
        scenario = scenarios.loc[scenarios["part_slug"].eq(part)].set_index("curve_family")
        alt = alternative_deltas.loc[alternative_deltas["part_slug"].eq(part)]
        first = point.iloc[0]
        conclusion = (
            "Strong observed sample-size evidence; more unique constructs remain useful."
            if positive_count >= 2 else
            "Localized sample-size benefit; prioritize the ranges with positive paired intervals."
            if positive_count == 1 else
            "No tested 10× step has a clearly positive paired interval; treat extrapolated gains cautiously."
        )
        record = {
            "part_slug": part,
            "part": PART_LABELS[part],
            "primary_architecture": first["architecture"],
            "primary_training_regime": first["training_regime"],
            "primary_rc_mode": first["rc_mode"],
            "primary_loss_mode": first["loss_mode"],
            "pearson_at_40": float(by_label.loc["40", "mean_pearson"]),
            "pearson_at_40_ci_low": float(by_label.loc["40", "ci_2_5"]),
            "pearson_at_40_ci_high": float(by_label.loc["40", "ci_97_5"]),
            "pearson_at_full": float(full["mean_pearson"]),
            "pearson_at_full_ci_low": float(full["ci_2_5"]),
            "pearson_at_full_ci_high": float(full["ci_97_5"]),
            "delta_full_minus_4000": float(full["mean_pearson"] - n4000["mean_pearson"]),
            "clearly_positive_observed_10x_steps": positive_count,
            "observed_10x_steps_tested": 3,
            "observed_40_to_4000_delta_pearson": float(contrast_100x["delta_pearson"]),
            "observed_40_to_4000_ci_low": float(contrast_100x["ci_2_5"]),
            "observed_40_to_4000_ci_high": float(contrast_100x["ci_97_5"]),
            "observed_40_to_4000_evidence": str(contrast_100x["evidence"]),
            "power_law_gain_full_to_10x": float(scenario.loc["power_law", "projected_gain_full_to_10x"]),
            "power_law_gain_ci_low": float(scenario.loc["power_law", "ci_2_5"]),
            "power_law_gain_ci_high": float(scenario.loc["power_law", "ci_97_5"]),
            "exponential_gain_full_to_10x": float(scenario.loc["exponential", "projected_gain_full_to_10x"]),
            "exponential_gain_ci_low": float(scenario.loc["exponential", "ci_2_5"]),
            "exponential_gain_ci_high": float(scenario.loc["exponential", "ci_97_5"]),
            "curve_family_gain_disagreement": float(disagreements.loc[part, "absolute_10x_gain_disagreement"]),
            "curve_family_gain_disagreement_ci_low": float(disagreement_intervals.loc[part, "ci_2_5"]),
            "curve_family_gain_disagreement_ci_high": float(disagreement_intervals.loc[part, "ci_97_5"]),
            "power_law_asymptote": float(scenario.loc["power_law", "asymptote"]),
            "power_law_asymptote_at_allowed_boundary": bool(scenario.loc["power_law", "asymptote"] >= 0.999),
            "power_law_loo_rmse": float(scenario.loc["power_law", "loo_rmse"]),
            "exponential_loo_rmse": float(scenario.loc["exponential", "loo_rmse"]),
            "maximum_abs_alternative_point_delta": float(alt["alternative_minus_primary_pearson"].abs().max()) if len(alt) else math.nan,
            "full_rmse": float(full["mean_rmse"]),
            "full_cod_r2": float(full["mean_cod_r2"]),
            "current_evidence_summary": conclusion,
        }
        for row in contrasts.itertuples(index=False):
            prefix = f"observed_{row.low_n}_to_{row.high_n}"
            record[f"{prefix}_delta_pearson"] = row.delta_pearson
            record[f"{prefix}_ci_low"] = row.ci_2_5
            record[f"{prefix}_ci_high"] = row.ci_97_5
            record[f"{prefix}_evidence"] = row.evidence
        if part == "intron":
            centered = intron_table.loc[intron_table["scope"].eq("within_stratum_centered")].set_index("downsample_n_label")
            record["intron_centered_pearson_at_full"] = float(centered.loc["full", "pearson"])
            centered_contrasts = contrast_table.loc[(contrast_table["part_slug"].eq("intron")) & contrast_table["scope"].eq("within_stratum_centered")]
            record["intron_centered_clearly_positive_10x_steps"] = int(centered_contrasts["evidence"].eq("positive").sum())
        if part == "enhancer":
            route_full = enhancer_table.loc[enhancer_table["downsample_n_label"].eq("full")].set_index("stage4_lane")
            record["enhancer_transfer_minus_scratch_at_full"] = float(route_full.loc["primary", "mean_pearson"] - route_full.loc["scratch_diagnostic", "mean_pearson"])
        records.append(record)
    return pd.DataFrame(records)


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "—" if not math.isfinite(number) else f"{number:.{digits}f}"


def write_executive_summary(
    scorecard: pd.DataFrame,
    design: pd.DataFrame,
    readiness: Mapping,
    contract: Mapping,
    figures: Sequence[Path],
    output_dir: Path,
) -> None:
    scorecard.to_csv(output_dir / "stage4_decision_scorecard.csv", index=False)
    scorecard.to_csv(output_dir / "stage4_executive_summary.csv", index=False)
    design.to_csv(output_dir / "stage4_study_design_comparison.csv", index=False)
    scorecard_records = json.loads(scorecard.to_json(orient="records"))
    summary = {
        "schema_version": "lib1_dedup_stage4_presentation_report_v1",
        "readiness": {
            "completed_cells": int(readiness["completed_cells"]),
            "manifest_rows": int(readiness["manifest_rows"]),
            "bootstrap_resamples": int(readiness["bootstrap_resamples"]),
            "primary_estimand": contract["primary_estimand"],
            "current_final_test_products_read": False,
        },
        "interpretation_contract": {
            "direct_observed_paired_contrasts": "primary",
            "bounded_curve_projections": "secondary_sensitivity_scenarios",
            "historical_absolute_performance_comparable": False,
            "alternative_point_deltas_have_paired_bootstrap_ci": False,
        },
        "decision_scorecard": scorecard_records,
        "study_design_comparison": design.to_dict(orient="records"),
        "figures": [str(path.resolve()) for path in figures if path.suffix == ".png"],
    }
    with (output_dir / "stage4_executive_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    table_lines = [
        "| CRE part | r at 40 | r at full | observed 400→4,000 Δr (95% CI) | observed 40→4,000 Δr (95% CI) | full − 4,000 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in scorecard.itertuples(index=False):
        table_lines.append(
            f"| {row.part} | {_fmt(row.pearson_at_40)} | {_fmt(row.pearson_at_full)} | "
            f"{_fmt(row.observed_400_to_4000_delta_pearson)} "
            f"[{_fmt(row.observed_400_to_4000_ci_low)}, {_fmt(row.observed_400_to_4000_ci_high)}] | "
            f"{_fmt(row.observed_40_to_4000_delta_pearson)} "
            f"[{_fmt(row.observed_40_to_4000_ci_low)}, {_fmt(row.observed_40_to_4000_ci_high)}] | "
            f"{_fmt(row.delta_full_minus_4000)} |"
        )
    intron = scorecard.loc[scorecard["part_slug"].eq("intron")].iloc[0]
    enhancer = scorecard.loc[scorecard["part_slug"].eq("enhancer")].iloc[0]
    boundary_parts = scorecard.loc[
        scorecard["power_law_asymptote_at_allowed_boundary"], "part"
    ].tolist()
    markdown = f"""# Lib1 dedup Stage 4 downsampling — presentation summary

## Readiness and scope

- The campaign and analysis are complete: **{int(readiness['completed_cells'])}/{int(readiness['manifest_rows'])} cells**, 132 pooled OOF tracks, and the frozen {int(readiness['bootstrap_resamples']):,}-replicate paired bootstrap.
- Every current result is development-only OOF. **No current final-test product was read or computed** by either the core analysis or this report.
- The modeled target is construct-mean log₂ RNA/DNA. Training-subset mean/SD normalization is used for optimization, then predictions are inverse-transformed; every displayed metric here is on raw log₂ RNA/DNA predictions/targets (Pearson itself is scale-invariant).
- Direct observed 10× contrasts are the primary evidence. Power-law and exponential full→10× values are bounded sensitivity scenarios, not forecasts or selection criteria.
- The 40→4,000 result is a directly observed 100× training-size contrast. It is not a projection from full N to 100× full N.
- The power-law asymptote reached its allowed correlation boundary for **{', '.join(boundary_parts)}**. Those power-law extrapolations are boundary-sensitive appendix values—not estimated empirical ceilings.

## Decision scorecard

{chr(10).join(table_lines)}

## Two biologically important diagnostics

- **Intron:** pooled full-N r is {_fmt(intron.pearson_at_full)}, while frozen within-stratum-centered full-N r is {_fmt(intron.intron_centered_pearson_at_full)}. The centered analysis has {int(intron.intron_centered_clearly_positive_10x_steps)}/3 clearly positive observed 10× steps. This separates prediction within the inferred design/mask strata from the easier between-stratum expression separation. These are not measured splicing classes, and this sensitivity analysis does not replace the PI-requested position-balanced 80-bp evaluation set.
- **Enhancer:** the selected transfer route exceeds the scratch diagnostic by {_fmt(enhancer.enhancer_transfer_minus_scratch_at_full)} Pearson at full N. This is diagnostic rather than a clean pretraining causal contrast because architecture, RC/input policy, and initialization also differ.

## How to use the historical panels

Historical series are normalized to their own smallest-N baseline and maximum N. They establish whether earlier curves showed similar *shape*, but their absolute r values must not be compared with Stage 4: upstream deduplication, targets/splits, portfolios, seeds, and evaluation sets differ. The older panels include historical test summaries; the current final test remains untouched.

## Figure order for a presentation

1. `01_primary_pearson_learning_curves.png` — current headline curves and uncertainty.
2. `02_observed_10x_pearson_forest.png` — the common observed 400→4,000 decade; all three 10× contrasts remain in the companion CSV.
3. `03_intron_scoped_learning_curves.png` — pooled versus centered/per-stratum Intron evidence.
4. `04_enhancer_transfer_vs_scratch.png` — current route diagnostic.
5. `05_portfolio_sensitivity_point_deltas.png` — whether sparse alternatives change the story; point estimates only.
6. `06_curve_family_scenarios.png` — tail-shape sensitivity and leave-one-size-out fit error.
7. `07_historical_shape_context_noncomparable.png` — pre-dedup context with explicit non-comparability.
8. `08_full_n_raw_calibration.png` — full-N raw slope and prediction bias against slope=1/bias=0 ideals.
9. `09_cod_calibration_appendix.png` — raw-scale calibration guardrail on common axes.

## Statistical reading guide

- A 10× interval entirely above zero is “clearly positive” under the paired resampling design; an interval crossing zero is unresolved, not evidence of no effect.
- Bootstrap intervals quantify construct and nested-subset sampling uncertainty. They do not remove curve-family, initialization-seed, or distribution-shift uncertainty.
- Finite-N points average three nested subset tracks; full N has one training-subset realization per outer fold. Small 4,000-to-full reversals should not be interpreted as harmful-data effects.
- Alternative-minus-primary values have no paired-bootstrap interval in the frozen compact outputs, so they are labeled descriptive and cannot trigger post hoc reselection.
"""
    (output_dir / "stage4_executive_summary.md").write_text(markdown, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, default=DEFAULT_CORE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold-history", type=Path, default=DEFAULT_THRESHOLD_HISTORY)
    parser.add_argument("--exact-n1-history", type=Path, default=DEFAULT_EXACT_N1_HISTORY)
    parser.add_argument("--enhancer-transfer-history", type=Path, default=DEFAULT_ENHANCER_TRANSFER_HISTORY)
    return parser.parse_args(argv)


def run_report(args: argparse.Namespace) -> dict:
    readiness, contract = validate_core_readiness(args.core_dir)
    products = read_core_products(args.core_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"

    primary = _primary_point_intervals(products, "pearson")
    contrasts = observed_10x_table(products)
    contrasts_100x = observed_100x_table(products)
    intron = intron_curve_table(products)
    enhancer = enhancer_transfer_scratch_table(products)
    alternatives = alternative_point_deltas(products)
    scenarios = curve_scenario_table(products)
    disagreement_intervals = curve_disagreement_table(products)
    calibration = full_n_raw_calibration(products)
    history, design = load_historical_context(
        products["points"], args.threshold_history, args.exact_n1_history,
        args.enhancer_transfer_history,
    )

    figures: list[Path] = []
    figures += plot_primary_pearson(products, figure_dir)
    figures += plot_observed_10x(contrasts, figure_dir)
    figures += plot_intron(intron, figure_dir)
    figures += plot_enhancer_transfer_scratch(enhancer, figure_dir)
    figures += plot_alternative_deltas(alternatives, figure_dir)
    figures += plot_curve_scenarios(scenarios, disagreement_intervals, figure_dir)
    figures += plot_historical_context(history, figure_dir)
    figures += plot_full_n_raw_calibration(calibration, figure_dir)
    figures += plot_cod_appendix(products, figure_dir)

    contrasts.to_csv(output_dir / "stage4_observed_10x_presentation.csv", index=False)
    contrasts_100x.to_csv(
        output_dir / "stage4_observed_100x_presentation.csv", index=False
    )
    intron.to_csv(output_dir / "stage4_intron_scoped_curves.csv", index=False)
    enhancer.to_csv(output_dir / "stage4_enhancer_transfer_scratch_curves.csv", index=False)
    alternatives.to_csv(output_dir / "stage4_alternative_point_deltas.csv", index=False)
    scenarios.to_csv(output_dir / "stage4_curve_scenarios.csv", index=False)
    disagreement_intervals.to_csv(
        output_dir / "stage4_curve_family_disagreement_presentation.csv", index=False
    )
    history.to_csv(output_dir / "stage4_historical_context_normalized.csv", index=False)
    calibration.to_csv(output_dir / "stage4_full_n_raw_calibration.csv", index=False)

    scorecard = build_decision_scorecard(
        products, primary, contrasts, contrasts_100x, intron, enhancer,
        alternatives, scenarios, disagreement_intervals,
    )
    scorecard = scorecard.merge(
        calibration[[
            "part_slug", "observed_on_prediction_slope",
            "mean_prediction_minus_target_bias",
        ]],
        on="part_slug",
        how="left",
        validate="one_to_one",
    )
    write_executive_summary(scorecard, design, readiness, contract, figures, output_dir)
    return {
        "completed_cells": int(readiness["completed_cells"]),
        "bootstrap_resamples": int(readiness["bootstrap_resamples"]),
        "scorecard_rows": len(scorecard),
        "figures": len([path for path in figures if path.suffix == ".png"]),
        "output_dir": str(output_dir),
        "current_final_test_products_read": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    result = run_report(parse_args(argv))
    print("Lib1 dedup Stage 4 presentation report")
    print(f"  completed cells: {result['completed_cells']}/{EXPECTED_CELLS}")
    print(f"  paired bootstrap: {result['bootstrap_resamples']:,}")
    print(f"  figures: {result['figures']}")
    print(f"  output: {result['output_dir']}")
    print("  current final-test products read: false")


if __name__ == "__main__":
    main()
