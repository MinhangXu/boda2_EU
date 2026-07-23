#!/usr/bin/env python3
"""Report what pooled Intron OOF correlation does and does not establish.

This is a presentation and sensitivity layer over the completed Lib1 dedup
Stage 2 outputs.  It never constructs a DataModule, never instantiates the
frozen audit loader, and never changes model selection.  The same metric
functions can later be applied to locked audit predictions after the final
policy has been selected.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.lib1_reporting import comparison_subplots, save_figure


ANALYSIS_ROOT = (
    REPO_ROOT / "src/learn/outputs/analysis/lib1_dedup_stage2_july2026"
)
DEFAULT_OOF = ANALYSIS_ROOT / "stage2_oof_predictions.tsv.gz"
DEFAULT_METRICS = ANALYSIS_ROOT / "stage2_oof_metrics.csv"
DEFAULT_BASELINES = ANALYSIS_ROOT / "stage2_intron_stratum_mean_baselines.csv"
DEFAULT_BASELINE_PREDICTIONS = (
    ANALYSIS_ROOT / "stage2_intron_stratum_mean_baseline_predictions.tsv"
)
DEFAULT_SPLIT = (
    REPO_ROOT
    / "src/learn/data_manifests/splits/lib1_intron_dedup_exact_v1_split.json"
)
DEFAULT_OUTPUT_DIR = ANALYSIS_ROOT / "reporting"
DEFAULT_FIGURE_DIR = DEFAULT_OUTPUT_DIR / "figures"

TARGET = "log2_RNA_DNA"
PREDICTION = "prediction_raw"
STRATUM = "inferred_intron_sensitivity_stratum"
STRATUM_ORDER = ("mask1_specific", "mask2_not_mask1", "mask3_residual")
STRATUM_LABEL = {
    "mask1_specific": "Mask 1 compatible",
    "mask2_not_mask1": "Mask 2, not 1",
    "mask3_residual": "Residual exact-80",
}
STRATUM_COLOR = {
    "mask1_specific": "#4C78A8",
    "mask2_not_mask1": "#F58518",
    "mask3_residual": "#54A24B",
}


def _safe_pearson(target: Sequence[float], prediction: Sequence[float]) -> float:
    target_array = np.asarray(target, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    if (
        len(target_array) < 2
        or np.ptp(target_array) == 0
        or np.ptp(prediction_array) == 0
    ):
        return math.nan
    return float(np.corrcoef(target_array, prediction_array)[0, 1])


def weighted_pearson(
    target: Sequence[float], prediction: Sequence[float], weights: Sequence[float]
) -> float:
    """Population-weighted Pearson correlation with nonnegative weights."""
    target_array = np.asarray(target, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if not (
        len(target_array) == len(prediction_array) == len(weight_array)
        and len(target_array) >= 2
    ):
        raise ValueError("target, prediction, and weights must have equal length >= 2")
    if not np.isfinite(weight_array).all() or (weight_array < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    if weight_array.sum() <= 0:
        raise ValueError("weights must have positive total mass")
    normalized = weight_array / weight_array.sum()
    target_mean = float(np.sum(normalized * target_array))
    prediction_mean = float(np.sum(normalized * prediction_array))
    covariance = float(
        np.sum(
            normalized
            * (target_array - target_mean)
            * (prediction_array - prediction_mean)
        )
    )
    target_variance = float(
        np.sum(normalized * (target_array - target_mean) ** 2)
    )
    prediction_variance = float(
        np.sum(normalized * (prediction_array - prediction_mean) ** 2)
    )
    if target_variance == 0 or prediction_variance == 0:
        return math.nan
    return covariance / math.sqrt(target_variance * prediction_variance)


def _validate_prediction_frame(frame: pd.DataFrame) -> None:
    required = {
        "construct_id",
        "development_fold",
        TARGET,
        PREDICTION,
        STRATUM,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Intron prediction table is missing columns: {}".format(missing))
    if frame.empty:
        raise ValueError("Intron prediction table is empty")
    if frame["construct_id"].duplicated().any():
        raise ValueError("Expected exactly one held-out prediction per construct")
    observed = set(frame[STRATUM].dropna().unique())
    if observed != set(STRATUM_ORDER):
        raise ValueError(
            "Inferred sensitivity categories differ: observed={!r}".format(observed)
        )


def intron_estimands(frame: pd.DataFrame) -> Dict[str, float]:
    """Return natural-mixture, equal-mixture, and conditional correlations."""
    _validate_prediction_frame(frame)
    centered = frame.copy()
    centered["target_centered"] = centered[TARGET] - centered.groupby(STRATUM)[
        TARGET
    ].transform("mean")
    centered["prediction_centered"] = centered[PREDICTION] - centered.groupby(
        STRATUM
    )[PREDICTION].transform("mean")

    counts = frame[STRATUM].value_counts()
    group_weights = {
        group: (1.0 / len(STRATUM_ORDER)) / (count / float(len(frame)))
        for group, count in counts.items()
    }
    weights = frame[STRATUM].map(group_weights).to_numpy(float)
    per_stratum = {
        group: _safe_pearson(
            frame.loc[frame[STRATUM].eq(group), TARGET],
            frame.loc[frame[STRATUM].eq(group), PREDICTION],
        )
        for group in STRATUM_ORDER
    }
    return {
        "n_constructs": int(len(frame)),
        "natural_pooled_pearson": _safe_pearson(frame[TARGET], frame[PREDICTION]),
        "equal_stratum_pooled_pearson": weighted_pearson(
            frame[TARGET], frame[PREDICTION], weights
        ),
        "equal_stratum_weight_ess": float(weights.sum() ** 2 / np.sum(weights**2)),
        "equal_stratum_within_centered_pearson": weighted_pearson(
            centered["target_centered"], centered["prediction_centered"], weights
        ),
        "within_stratum_centered_pearson": _safe_pearson(
            centered["target_centered"], centered["prediction_centered"]
        ),
        "macro_stratum_pearson": float(np.mean(list(per_stratum.values()))),
        "minimum_stratum_pearson": float(np.min(list(per_stratum.values()))),
        **{
            "{}_pearson".format(group): value
            for group, value in per_stratum.items()
        },
    }


def covariance_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    """Decompose target variance, prediction variance, and covariance by stratum."""
    _validate_prediction_frame(frame)
    target_mean = float(frame[TARGET].mean())
    prediction_mean = float(frame[PREDICTION].mean())
    grouped = frame.groupby(STRATUM, sort=False)
    counts = grouped.size()
    proportions = counts / float(len(frame))
    target_group_mean = grouped[TARGET].mean()
    prediction_group_mean = grouped[PREDICTION].mean()

    target_between = float(
        (proportions * (target_group_mean - target_mean) ** 2).sum()
    )
    prediction_between = float(
        (proportions * (prediction_group_mean - prediction_mean) ** 2).sum()
    )
    covariance_between = float(
        (
            proportions
            * (target_group_mean - target_mean)
            * (prediction_group_mean - prediction_mean)
        ).sum()
    )
    target_within = 0.0
    prediction_within = 0.0
    covariance_within = 0.0
    for group, subset in grouped:
        probability = float(proportions.loc[group])
        target_delta = subset[TARGET].to_numpy(float) - target_group_mean.loc[group]
        prediction_delta = (
            subset[PREDICTION].to_numpy(float) - prediction_group_mean.loc[group]
        )
        target_within += probability * float(np.mean(target_delta**2))
        prediction_within += probability * float(np.mean(prediction_delta**2))
        covariance_within += probability * float(
            np.mean(target_delta * prediction_delta)
        )

    rows = []
    for component, between, within in (
        ("target_variance", target_between, target_within),
        ("prediction_variance", prediction_between, prediction_within),
        ("target_prediction_covariance", covariance_between, covariance_within),
    ):
        total = between + within
        rows.append(
            {
                "component": component,
                "between_stratum": between,
                "within_stratum": within,
                "total": total,
                "between_stratum_share": between / total if total else math.nan,
                "within_stratum_share": within / total if total else math.nan,
            }
        )
    return pd.DataFrame(rows)


def fold_estimands(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, subset in frame.groupby("development_fold", sort=True):
        values = intron_estimands(subset)
        rows.append({"development_fold": int(fold), **values})
    return pd.DataFrame(rows)


def barcode_threshold_sensitivity(
    frame: pd.DataFrame, thresholds: Sequence[int] = (8, 10, 12)
) -> pd.DataFrame:
    if "n_barcodes" not in frame:
        raise ValueError("Barcode sensitivity requires n_barcodes")
    rows = []
    for threshold in thresholds:
        subset = frame.loc[frame["n_barcodes"].ge(threshold)].copy()
        if set(subset[STRATUM].unique()) != set(STRATUM_ORDER):
            raise ValueError(
                "Barcode threshold {} removes an entire stratum".format(threshold)
            )
        values = intron_estimands(subset)
        rows.append(
            {
                "minimum_n_barcodes": int(threshold),
                "analysis_status": "development_only_post_stage2_descriptive",
                **values,
                **{
                    "{}_n".format(group): int(subset[STRATUM].eq(group).sum())
                    for group in STRATUM_ORDER
                },
            }
        )
    return pd.DataFrame(rows)


def literal_base_balance_constraints(assignments: pd.DataFrame) -> pd.DataFrame:
    """Document why literal 25% A/C/G/T conflicts with the two GT...AG masks."""
    required = {"sequence", STRATUM}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError("Assignment table is missing columns: {}".format(missing))
    structured = assignments[STRATUM].isin(STRATUM_ORDER[:2])
    expected = ((1, "G"), (2, "T"), (79, "A"), (80, "G"))
    rows = []
    for position, base in expected:
        if not assignments.loc[structured, "sequence"].str[position - 1].eq(base).all():
            raise ValueError(
                "Structured masks do not all have {} at position {}".format(
                    base, position
                )
            )
        rows.append(
            {
                "position_1_based": position,
                "fixed_base_in_mask1_and_mask2": base,
                "natural_structured_stratum_mass": float(structured.mean()),
                "equal_three_stratum_structured_mass": 2.0 / 3.0,
                "maximum_structured_mass_if_fixed_base_frequency_is_0p25": 0.25,
                "literal_equal_bases_compatible_with_equal_strata": False,
                "reason": (
                    "P(base at position) >= P(mask1 or mask2), so a 0.25 base "
                    "target forces combined mask1+mask2 mass <= 0.25"
                ),
            }
        )
    return pd.DataFrame(rows)


def literal_position_balance_linear_program(assignments: pd.DataFrame) -> pd.DataFrame:
    """Test 25% A/C/G/T positional balance on the observed sequence support.

    The first linear program asks whether arbitrary nonnegative construct
    weights can produce exact 0.25 base frequency at every position.  If not,
    the second minimizes the largest absolute marginal deviation.  The
    resulting weight diagnostics describe one HiGHS optimum; only the optimum
    deviation and the separately optimized residual-mass range are treated as
    invariant evidence.
    """
    from scipy.optimize import linprog

    if "sequence" not in assignments or STRATUM not in assignments:
        raise ValueError("Position-balance LP requires sequence and stratum columns")
    sequences = assignments["sequence"].astype(str)
    lengths = sequences.str.len().unique()
    if len(lengths) != 1 or int(lengths[0]) != 80:
        raise ValueError("Position-balance LP requires canonical 80-nt sequences")
    if not sequences.str.fullmatch("[ACGT]{80}").all():
        raise ValueError("Position-balance LP requires canonical A/C/G/T sequences")

    sequence_array = np.asarray([list(sequence) for sequence in sequences])
    bases = ("A", "C", "G", "T")
    feature_columns = [
        (sequence_array[:, position] == base).astype(float)
        for position in range(80)
        for base in bases
    ]
    features = np.column_stack(feature_columns)
    n_constructs = len(assignments)

    # T is implied by A/C/G plus sum(weights)=1, avoiding one redundant
    # equality per position in the exact-feasibility program.
    acg_indices = [
        position * len(bases) + base_index
        for position in range(80)
        for base_index in range(3)
    ]
    exact_equalities = np.vstack(
        [np.ones(n_constructs), features[:, acg_indices].T]
    )
    exact_targets = np.concatenate(
        [np.ones(1), np.full(len(acg_indices), 0.25)]
    )
    exact = linprog(
        np.zeros(n_constructs),
        A_eq=exact_equalities,
        b_eq=exact_targets,
        bounds=(0, None),
        method="highs",
    )

    # Variables are n construct weights followed by the maximum deviation d.
    upper = np.hstack([features.T, -np.ones((features.shape[1], 1))])
    lower = np.hstack([-features.T, -np.ones((features.shape[1], 1))])
    inequalities = np.vstack([upper, lower])
    inequality_targets = np.concatenate(
        [np.full(features.shape[1], 0.25), np.full(features.shape[1], -0.25)]
    )
    equality = np.zeros((1, n_constructs + 1))
    equality[0, :n_constructs] = 1.0
    objective = np.zeros(n_constructs + 1)
    objective[-1] = 1.0
    closest = linprog(
        objective,
        A_ub=inequalities,
        b_ub=inequality_targets,
        A_eq=equality,
        b_eq=np.ones(1),
        bounds=[(0, None)] * (n_constructs + 1),
        method="highs",
    )
    if not closest.success:
        raise RuntimeError(
            "Closest positional-balance linear program failed: {}".format(
                closest.message
            )
        )
    weights = closest.x[:n_constructs]
    optimum_deviation = float(closest.x[-1])
    achieved = features.T.dot(weights)
    residual_indicator = assignments[STRATUM].eq("mask3_residual").to_numpy(float)

    # Determine how strongly the optimum itself forces a residual-heavy target
    # distribution, without relying on the arbitrary basic solution returned
    # by HiGHS for a nonunique optimum.
    tolerance = optimum_deviation + 1e-8
    fixed_inequalities = np.vstack([features.T, -features.T])
    fixed_targets = np.concatenate(
        [
            np.full(features.shape[1], 0.25 + tolerance),
            np.full(features.shape[1], -0.25 + tolerance),
        ]
    )
    fixed_equality = np.ones((1, n_constructs))
    minimum_residual = linprog(
        residual_indicator,
        A_ub=fixed_inequalities,
        b_ub=fixed_targets,
        A_eq=fixed_equality,
        b_eq=np.ones(1),
        bounds=(0, None),
        method="highs",
    )
    maximum_residual = linprog(
        -residual_indicator,
        A_ub=fixed_inequalities,
        b_ub=fixed_targets,
        A_eq=fixed_equality,
        b_eq=np.ones(1),
        bounds=(0, None),
        method="highs",
    )
    if not minimum_residual.success or not maximum_residual.success:
        raise RuntimeError("Could not bound residual mass at the balance optimum")

    stratum_mass = {
        group: float(weights[assignments[STRATUM].eq(group).to_numpy()].sum())
        for group in STRATUM_ORDER
    }
    return pd.DataFrame(
        [
            {
                "n_development_sequences": n_constructs,
                "exact_25pct_each_base_each_position_feasible": bool(exact.success),
                "minimum_max_absolute_marginal_deviation": optimum_deviation,
                "minimum_max_absolute_marginal_deviation_percentage_points": 100
                * optimum_deviation,
                "achieved_max_absolute_marginal_deviation": float(
                    np.max(np.abs(achieved - 0.25))
                ),
                "one_optimum_mask1_mass": stratum_mass["mask1_specific"],
                "one_optimum_mask2_not_mask1_mass": stratum_mass[
                    "mask2_not_mask1"
                ],
                "one_optimum_residual_mass": stratum_mass["mask3_residual"],
                "minimum_residual_mass_at_optimum": float(minimum_residual.fun),
                "maximum_residual_mass_at_optimum": float(-maximum_residual.fun),
                "one_optimum_positive_weight_count": int((weights > 1e-10).sum()),
                "one_optimum_maximum_weight": float(weights.max()),
                "one_optimum_maximum_weight_multiple_of_uniform": float(
                    weights.max() * n_constructs
                ),
                "one_optimum_kish_effective_sample_size": float(
                    1.0 / np.sum(weights**2)
                ),
                "weight_diagnostic_status": (
                    "one_nonunique_minimax_solution; deviation and residual range are invariant"
                ),
            }
        ]
    )


def estimand_summary_table(
    frame: pd.DataFrame, baseline_pearson: float
) -> pd.DataFrame:
    values = intron_estimands(frame)
    rows = [
        (
            "natural_pooled",
            "natural mixed-library ranking",
            values["natural_pooled_pearson"],
        ),
        (
            "equal_stratum_pooled",
            "pooled ranking after assigning one-third mass to each inferred category",
            values["equal_stratum_pooled_pearson"],
        ),
        (
            "fold_trained_stratum_mean_baseline",
            "category recognition and category-mean calibration only",
            baseline_pearson,
        ),
        (
            "within_stratum_centered",
            "ranking after removing observed and predicted category means",
            values["within_stratum_centered_pearson"],
        ),
        (
            "equal_stratum_within_centered",
            "within-category ranking after also assigning one-third mass per category",
            values["equal_stratum_within_centered_pearson"],
        ),
        (
            "macro_stratum",
            "equal-weight mean of the three category-specific correlations",
            values["macro_stratum_pearson"],
        ),
        (
            "minimum_stratum",
            "worst category-specific correlation",
            values["minimum_stratum_pearson"],
        ),
    ]
    rows.extend(
        (
            group,
            "ranking within {}".format(STRATUM_LABEL[group]),
            values["{}_pearson".format(group)],
        )
        for group in STRATUM_ORDER
    )
    return pd.DataFrame(rows, columns=["estimand", "interpretation", "pearson"])


def _estimand_audit_figure(
    summary: pd.DataFrame,
    decomposition: pd.DataFrame,
    folds: pd.DataFrame,
) -> plt.Figure:
    fig, axes = comparison_subplots(
        1, 3, y_groups="independent", figsize=(16.5, 5.2)
    )

    ax = axes[0, 0]
    labels = ["Target\nvariance", "Prediction\nvariance", "Target-prediction\ncovariance"]
    between = decomposition["between_stratum_share"].to_numpy(float) * 100
    within = decomposition["within_stratum_share"].to_numpy(float) * 100
    positions = np.arange(len(labels))
    ax.bar(positions, between, color="#4C78A8", label="Between categories")
    ax.bar(positions, within, bottom=between, color="#D9E2F3", label="Within categories")
    for position, value in zip(positions, between):
        ax.text(position, value / 2, "{:.1f}%".format(value), ha="center", va="center")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of total (%)")
    ax.set_title("A. Where the pooled signal comes from")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[0, 1]
    order = [
        "natural_pooled",
        "equal_stratum_pooled",
        "fold_trained_stratum_mean_baseline",
        "within_stratum_centered",
        "macro_stratum",
        "mask1_specific",
        "mask2_not_mask1",
        "mask3_residual",
    ]
    label_map = {
        "natural_pooled": "Natural pooled",
        "equal_stratum_pooled": "Equal-stratum pooled",
        "fold_trained_stratum_mean_baseline": "Mask-mean baseline",
        "within_stratum_centered": "Within-centered",
        "macro_stratum": "Macro-stratum",
        "mask1_specific": "Mask 1 compatible",
        "mask2_not_mask1": "Mask 2, not 1",
        "mask3_residual": "Residual exact-80",
    }
    selected = summary.set_index("estimand").loc[order].reset_index()
    y_position = np.arange(len(selected))[::-1]
    colors = [
        "#1F1F1F",
        "#777777",
        "#B279A2",
        "#E45756",
        "#E45756",
        STRATUM_COLOR["mask1_specific"],
        STRATUM_COLOR["mask2_not_mask1"],
        STRATUM_COLOR["mask3_residual"],
    ]
    ax.axvline(0, color="#777777", lw=0.8)
    ax.scatter(selected["pearson"], y_position, c=colors, s=48, zorder=3)
    for x_value, y_value in zip(selected["pearson"], y_position):
        ax.text(x_value + 0.015, y_value, "{:.3f}".format(x_value), va="center", fontsize=8)
    ax.set_yticks(y_position)
    ax.set_yticklabels([label_map[value] for value in selected["estimand"]])
    ax.set_xlim(-0.05, 0.82)
    ax.set_xlabel("Pearson r")
    ax.set_title("B. Different claims, different scores")

    ax = axes[0, 2]
    for _, row in folds.iterrows():
        values = [
            row["natural_pooled_pearson"],
            row["within_stratum_centered_pearson"],
        ]
        ax.plot([0, 1], values, color="#A0A0A0", alpha=0.8, lw=1)
        ax.scatter([0, 1], values, color="#707070", s=24, zorder=3)
    full = summary.set_index("estimand")["pearson"]
    ax.plot(
        [0, 1],
        [full["natural_pooled"], full["within_stratum_centered"]],
        color="black",
        lw=2.2,
        marker="D",
        markersize=6,
        label="All OOF constructs",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pooled", "Within-centered"])
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0.25, 0.82)
    ax.set_ylabel("Pearson r")
    ax.set_title("C. The gap recurs in all five folds")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle(
        "Intron OOF estimand audit: pooled performance contains category-level signal",
        y=1.03,
    )
    return fig


def _pooled_baseline_centered_figure(
    frame: pd.DataFrame, baseline_predictions: pd.DataFrame
) -> plt.Figure:
    required = {
        "construct_id",
        TARGET,
        PREDICTION,
        STRATUM,
        "baseline_type",
    }
    missing = sorted(required - set(baseline_predictions.columns))
    if missing:
        raise ValueError(
            "Baseline prediction table is missing columns: {}".format(missing)
        )
    baseline = baseline_predictions.loc[
        baseline_predictions["baseline_type"].eq("fold_trained_stratum_mean")
    ].copy()
    if baseline["construct_id"].duplicated().any() or len(baseline) != len(frame):
        raise ValueError("Expected one fold-trained baseline prediction per construct")
    if set(baseline["construct_id"]) != set(frame["construct_id"]):
        raise ValueError("Model and baseline construct IDs differ")
    combined = frame.merge(
        baseline[["construct_id", PREDICTION]].rename(
            columns={PREDICTION: "baseline_prediction_raw"}
        ),
        on="construct_id",
        validate="one_to_one",
    )
    combined["target_centered"] = combined[TARGET] - combined.groupby(STRATUM)[
        TARGET
    ].transform("mean")
    combined["prediction_centered"] = combined[PREDICTION] - combined.groupby(
        STRATUM
    )[PREDICTION].transform("mean")

    fig, axes = comparison_subplots(
        1, 3, y_groups=((0, 1), (2,)), figsize=(15.8, 5.0)
    )
    raw_lower = float(
        min(
            combined[TARGET].min(),
            combined[PREDICTION].min(),
            combined["baseline_prediction_raw"].min(),
        )
    )
    raw_upper = float(
        max(
            combined[TARGET].max(),
            combined[PREDICTION].max(),
            combined["baseline_prediction_raw"].max(),
        )
    )
    raw_specs = (
        (
            PREDICTION,
            "A. Leader pooled OOF\nr = {:.3f}".format(
                _safe_pearson(combined[TARGET], combined[PREDICTION])
            ),
        ),
        (
            "baseline_prediction_raw",
            "B. Fold-trained category means only\nr = {:.3f}".format(
                _safe_pearson(combined[TARGET], combined["baseline_prediction_raw"])
            ),
        ),
    )
    for axis, (prediction_column, title) in zip(axes.ravel()[:2], raw_specs):
        for group in STRATUM_ORDER:
            subset = combined.loc[combined[STRATUM].eq(group)]
            axis.scatter(
                subset[TARGET],
                subset[prediction_column],
                s=19,
                alpha=0.55,
                color=STRATUM_COLOR[group],
                edgecolors="none",
                label=STRATUM_LABEL[group],
            )
        axis.plot(
            [raw_lower, raw_upper],
            [raw_lower, raw_upper],
            color="black",
            ls="--",
            lw=0.9,
        )
        axis.set_xlim(raw_lower, raw_upper)
        axis.set_ylim(raw_lower, raw_upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Observed log2 RNA/DNA")
        axis.set_title(title)
    axes[0, 0].set_ylabel("Predicted log2 RNA/DNA")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")

    axis = axes[0, 2]
    centered_limit = float(
        max(
            abs(combined["target_centered"]).max(),
            abs(combined["prediction_centered"]).max(),
        )
    )
    for group in STRATUM_ORDER:
        subset = combined.loc[combined[STRATUM].eq(group)]
        axis.scatter(
            subset["target_centered"],
            subset["prediction_centered"],
            s=19,
            alpha=0.55,
            color=STRATUM_COLOR[group],
            edgecolors="none",
        )
    axis.axhline(0, color="#777777", lw=0.8)
    axis.axvline(0, color="#777777", lw=0.8)
    axis.plot(
        [-centered_limit, centered_limit],
        [-centered_limit, centered_limit],
        color="black",
        ls="--",
        lw=0.9,
    )
    axis.set_xlim(-centered_limit, centered_limit)
    axis.set_ylim(-centered_limit, centered_limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Observed minus category mean")
    axis.set_ylabel("Predicted minus category mean")
    axis.set_title(
        "C. Within-category centered\nr = {:.3f}".format(
            _safe_pearson(
                combined["target_centered"], combined["prediction_centered"]
            )
        )
    )
    fig.suptitle(
        "Why pooled Intron Pearson is not the average of within-category Pearsons",
        y=1.03,
    )
    return fig


def _calibration_figure(frame: pd.DataFrame) -> plt.Figure:
    fig, axes = comparison_subplots(1, 3, y_groups="all", figsize=(14.5, 4.7))
    lower = float(min(frame[TARGET].min(), frame[PREDICTION].min()))
    upper = float(max(frame[TARGET].max(), frame[PREDICTION].max()))
    for axis, group in zip(axes.ravel(), STRATUM_ORDER):
        subset = frame.loc[frame[STRATUM].eq(group)]
        correlation = _safe_pearson(subset[TARGET], subset[PREDICTION])
        axis.scatter(
            subset[PREDICTION],
            subset[TARGET],
            s=20,
            alpha=0.58,
            color=STRATUM_COLOR[group],
            edgecolors="none",
        )
        axis.plot([lower, upper], [lower, upper], color="black", ls="--", lw=0.9)
        if np.ptp(subset[PREDICTION].to_numpy(float)) > 0:
            slope, intercept = np.polyfit(
                subset[PREDICTION].to_numpy(float), subset[TARGET].to_numpy(float), 1
            )
            grid = np.array([lower, upper])
            axis.plot(grid, intercept + slope * grid, color=STRATUM_COLOR[group], lw=1.5)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            "{}\nn={}, r={:.3f}".format(STRATUM_LABEL[group], len(subset), correlation)
        )
        axis.set_xlabel("Held-out prediction (raw log2)")
    axes[0, 0].set_ylabel("Observed log2 RNA/DNA")
    fig.suptitle("Intron leader: OOF calibration within each inferred category", y=1.03)
    return fig


def _barcode_sensitivity_figure(table: pd.DataFrame) -> plt.Figure:
    fig, axes = comparison_subplots(1, 2, y_groups="independent", figsize=(12.5, 4.6))
    x = np.arange(len(table))
    labels = [">={}".format(int(value)) for value in table["minimum_n_barcodes"]]

    ax = axes[0, 0]
    bottom = np.zeros(len(table))
    for group in STRATUM_ORDER:
        counts = table["{}_n".format(group)].to_numpy(float)
        ax.bar(x, counts, bottom=bottom, color=STRATUM_COLOR[group], label=STRATUM_LABEL[group])
        bottom += counts
    for position, total in zip(x, table["n_constructs"]):
        ax.text(position, total + 15, "n={}".format(int(total)), ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Minimum barcode count")
    ax.set_ylabel("Development OOF constructs")
    ax.set_title("A. Higher thresholds change the evaluated population")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    metrics = [
        ("natural_pooled_pearson", "Pooled", "#1F1F1F", "o"),
        ("within_stratum_centered_pearson", "Within-centered", "#E45756", "s"),
        ("macro_stratum_pearson", "Macro-stratum", "#72B7B2", "^"),
        ("minimum_stratum_pearson", "Minimum stratum", "#B279A2", "D"),
    ]
    for column, label, color, marker in metrics:
        ax.plot(x, table[column], marker=marker, color=color, label=label)
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Minimum barcode count")
    ax.set_ylabel("Pearson r")
    ax.set_title("B. High barcode support does not remove the estimand gap")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Post-Stage-2 barcode-threshold sensitivity (descriptive, not a new test set)",
        y=1.03,
    )
    return fig


def write_outputs(
    oof_path: Path = DEFAULT_OOF,
    metrics_path: Path = DEFAULT_METRICS,
    baseline_path: Path = DEFAULT_BASELINES,
    baseline_predictions_path: Path = DEFAULT_BASELINE_PREDICTIONS,
    split_path: Path = DEFAULT_SPLIT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    figure_dir: Path = DEFAULT_FIGURE_DIR,
    base_config_id: Optional[str] = None,
    rc_mode: Optional[str] = None,
    thresholds: Sequence[int] = (8, 10, 12),
) -> Mapping[str, object]:
    metrics = pd.read_csv(metrics_path)
    intron_metrics = metrics.loc[metrics["part_slug"].eq("intron")].copy()
    if base_config_id is None or rc_mode is None:
        leader = intron_metrics.sort_values(
            ["pooled_oof_pearson", "pooled_oof_rmse"], ascending=[False, True]
        ).iloc[0]
        base_config_id = str(leader["base_config_id"])
        rc_mode = str(leader["rc_mode"])
    else:
        match = intron_metrics.loc[
            intron_metrics["base_config_id"].eq(base_config_id)
            & intron_metrics["rc_mode"].eq(rc_mode)
        ]
        if len(match) != 1:
            raise ValueError("Requested Intron config/RC arm is not unique")

    all_predictions = pd.read_csv(oof_path, sep="\t", low_memory=False)
    frame = all_predictions.loc[
        all_predictions["part_slug"].eq("intron")
        & all_predictions["base_config_id"].eq(base_config_id)
        & all_predictions["rc_mode"].eq(rc_mode)
    ].copy()
    _validate_prediction_frame(frame)

    split_payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    assignments = pd.DataFrame(
        row for row in split_payload["assignments"] if row["partition"] == "development"
    )
    if len(assignments) != len(frame):
        raise ValueError("Development assignment and OOF row counts differ")
    if set(assignments["construct_id"]) != set(frame["construct_id"]):
        raise ValueError("Development assignment and OOF construct IDs differ")
    assignment_fields = assignments[["construct_id", "sequence", "n_barcodes"]]
    frame = frame.merge(assignment_fields, on="construct_id", validate="one_to_one")

    baselines = pd.read_csv(baseline_path)
    baseline_predictions = pd.read_csv(baseline_predictions_path, sep="\t")
    baseline = baselines.loc[
        baselines["baseline_type"].eq("fold_trained_stratum_mean")
    ]
    if len(baseline) != 1:
        raise ValueError("Expected one fold-trained stratum-mean baseline")
    baseline_pearson = float(baseline.iloc[0]["pooled_oof_pearson"])

    decomposition = covariance_decomposition(frame)
    folds = fold_estimands(frame)
    barcode = barcode_threshold_sensitivity(frame, thresholds=thresholds)
    estimands = estimand_summary_table(frame, baseline_pearson)
    constraints = literal_base_balance_constraints(frame)
    balance_linear_program = literal_position_balance_linear_program(frame)

    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    products = {
        "stage2_intron_leader_estimand_summary.csv": estimands,
        "stage2_intron_leader_signal_decomposition.csv": decomposition,
        "stage2_intron_leader_fold_estimands.csv": folds,
        "stage2_intron_leader_barcode_threshold_sensitivity.csv": barcode,
        "stage2_intron_literal_base_balance_constraints.csv": constraints,
        "stage2_intron_literal_position_balance_lp.csv": balance_linear_program,
    }
    for name, table in products.items():
        table.to_csv(output_dir / name, index=False)

    source_paths = {
        "oof_predictions": oof_path,
        "oof_metrics": metrics_path,
        "stratum_mean_baselines": baseline_path,
        "stratum_mean_baseline_predictions": baseline_predictions_path,
        "split_manifest": split_path,
    }
    figure_metadata = {
        "base_config_id": base_config_id,
        "rc_mode": rc_mode,
        "prediction_scope": "five-fold development OOF",
        "sensitivity_label_status": "inferred_sequence_mask_not_true_subset",
        "audit_loader_instantiated": False,
    }
    save_figure(
        _estimand_audit_figure(estimands, decomposition, folds),
        figure_dir / "stage2_intron_estimand_audit",
        source_paths=source_paths,
        metadata=figure_metadata,
        close=True,
    )
    save_figure(
        _pooled_baseline_centered_figure(frame, baseline_predictions),
        figure_dir / "stage2_intron_pooled_baseline_centered_triptych",
        source_paths=source_paths,
        metadata=figure_metadata,
        close=True,
    )
    save_figure(
        _calibration_figure(frame),
        figure_dir / "stage2_intron_per_stratum_oof_calibration",
        source_paths=source_paths,
        metadata=figure_metadata,
        close=True,
    )
    save_figure(
        _barcode_sensitivity_figure(barcode),
        figure_dir / "stage2_intron_barcode_threshold_sensitivity",
        source_paths=source_paths,
        metadata={
            **figure_metadata,
            "analysis_status": "development_only_post_stage2_descriptive",
            "locked_candidate_audit_thresholds": list(thresholds),
        },
        close=True,
    )

    values = intron_estimands(frame)
    summary = {
        "base_config_id": base_config_id,
        "rc_mode": rc_mode,
        "n_development_oof_constructs": len(frame),
        "natural_pooled_pearson": values["natural_pooled_pearson"],
        "equal_stratum_pooled_pearson": values["equal_stratum_pooled_pearson"],
        "equal_stratum_within_centered_pearson": values[
            "equal_stratum_within_centered_pearson"
        ],
        "within_stratum_centered_pearson": values[
            "within_stratum_centered_pearson"
        ],
        "macro_stratum_pearson": values["macro_stratum_pearson"],
        "minimum_stratum_pearson": values["minimum_stratum_pearson"],
        "fold_trained_stratum_mean_baseline_pearson": baseline_pearson,
        "target_variance_between_stratum_share": float(
            decomposition.loc[
                decomposition["component"].eq("target_variance"),
                "between_stratum_share",
            ].iloc[0]
        ),
        "prediction_variance_between_stratum_share": float(
            decomposition.loc[
                decomposition["component"].eq("prediction_variance"),
                "between_stratum_share",
            ].iloc[0]
        ),
        "covariance_between_stratum_share": float(
            decomposition.loc[
                decomposition["component"].eq("target_prediction_covariance"),
                "between_stratum_share",
            ].iloc[0]
        ),
        "literal_equal_position_bases_and_equal_strata_compatible": False,
        "literal_equal_position_bases_feasible_on_development_support": bool(
            balance_linear_program.iloc[0][
                "exact_25pct_each_base_each_position_feasible"
            ]
        ),
        "closest_position_balance_max_deviation_percentage_points": float(
            balance_linear_program.iloc[0][
                "minimum_max_absolute_marginal_deviation_percentage_points"
            ]
        ),
        "audit_loader_instantiated": False,
        "output_dir": str(output_dir.resolve()),
    }
    summary_path = output_dir / "stage2_intron_sensitivity_reporting_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-predictions", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--oof-metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument(
        "--baseline-predictions", type=Path, default=DEFAULT_BASELINE_PREDICTIONS
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--base-config-id")
    parser.add_argument("--rc-mode", choices=("off", "on"))
    parser.add_argument("--barcode-thresholds", type=int, nargs="+", default=(8, 10, 12))
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.base_config_id is None) != (args.rc_mode is None):
        raise SystemExit("--base-config-id and --rc-mode must be supplied together")
    summary = write_outputs(
        oof_path=args.oof_predictions,
        metrics_path=args.oof_metrics,
        baseline_path=args.baselines,
        baseline_predictions_path=args.baseline_predictions,
        split_path=args.split_manifest,
        output_dir=args.output_dir,
        figure_dir=args.figure_dir,
        base_config_id=args.base_config_id,
        rc_mode=args.rc_mode,
        thresholds=tuple(args.barcode_thresholds),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
