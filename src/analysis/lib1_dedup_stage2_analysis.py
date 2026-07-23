#!/usr/bin/env python3
"""Contract-first OOF analysis for the July 2026 Lib1 dedup Stage 2 runs.

This module reads prediction tables that were already exported by training.  It
never constructs a DataModule and therefore cannot instantiate the frozen audit
loader.  The only split information it reads is the immutable JSON assignment
manifest needed to prove held-out development-fold coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.lib1_dedup_stage1_analysis import assign_inferred_intron_subsets


LEARN_ROOT = REPO_ROOT / "src" / "learn"
CAMPAIGN_ID = "lib1_dedup_phase1_rerun_july2026"
CAMPAIGN_STAGE = "stage2_paired_rc"
DEFAULT_MANIFEST = (
    LEARN_ROOT
    / "outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl"
)
DEFAULT_REGISTRY = LEARN_ROOT / "run_registry/runs.csv"
DEFAULT_OUTPUT_DIR = LEARN_ROOT / "outputs/analysis/lib1_dedup_stage2_july2026"

EXPECTED_ANALYSIS_CELLS = 660
EXPECTED_FOLDS = frozenset(range(5))
EXPECTED_INTRON_DEVELOPMENT_CONSTRUCTS = 1061
ARM_KEYS = ("analysis_lane", "part_slug", "base_config_id", "rc_mode")
CONFIG_KEYS = ("analysis_lane", "part_slug", "base_config_id")
RAW_TARGET = "log2_RNA_DNA"
RAW_PREDICTION = "prediction_raw"
SENSITIVITY_STRATUM = "inferred_intron_sensitivity_stratum"
STRATUM_ORDER = ("mask1_specific", "mask2_not_mask1", "mask3_residual")
METADATA_FIELDS = (
    "analysis_lane",
    "challenger_family",
    "config_origin",
    "training_regime",
    "part_slug",
    "architecture",
    "base_config_id",
    "policy_id",
    "initialization",
    "source_head",
    "unfreeze_scope",
    "input_policy",
    "rc_mode",
)
METRIC_NAMES = ("n_constructs", "pearson", "spearman", "mse", "rmse", "mae", "cod_r2")


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _arm_key(row: Mapping) -> tuple[str, str, str, str]:
    return tuple(str(row[key]) for key in ARM_KEYS)


def _config_key(row: Mapping) -> tuple[str, str, str]:
    return tuple(str(row[key]) for key in CONFIG_KEYS)


def _metadata(row: Mapping) -> dict:
    return {field: row.get(field, "") for field in METADATA_FIELDS}


def validate_analysis_manifest(rows: Sequence[dict]) -> None:
    """Validate the 660-cell development-only analysis design."""
    if len(rows) != EXPECTED_ANALYSIS_CELLS:
        raise ValueError(
            f"Expected {EXPECTED_ANALYSIS_CELLS} Stage 2 analysis cells; found {len(rows)}"
        )
    if Counter(row.get("campaign_stage") for row in rows) != Counter(
        {CAMPAIGN_STAGE: EXPECTED_ANALYSIS_CELLS}
    ):
        raise ValueError("Analysis manifest contains a non-Stage-2 campaign stage.")
    if len({row.get("cell_id") for row in rows}) != len(rows):
        raise ValueError("Analysis manifest cell_id values are not unique.")

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("campaign_id") != CAMPAIGN_ID:
            raise ValueError("Analysis manifest contains an unexpected campaign_id.")
        if bool(row.get("evaluate_test_after_fit")):
            raise ValueError(f"Cell {row.get('cell_id')} enables audit/test evaluation.")
        if list(row.get("prediction_splits", [])) != ["val"]:
            raise ValueError(
                f"Cell {row.get('cell_id')} must export validation predictions only."
            )
        if "test" in {str(value).lower() for value in row.get("epoch_eval_splits", [])}:
            raise ValueError(f"Cell {row.get('cell_id')} includes test epoch evaluation.")
        grouped[_arm_key(row)].append(row)

    if len(grouped) != EXPECTED_ANALYSIS_CELLS // 5:
        raise ValueError(f"Expected 132 config/RC arms; found {len(grouped)}")
    for key, arm_rows in grouped.items():
        folds = {int(row["development_fold"]) for row in arm_rows}
        if len(arm_rows) != 5 or folds != EXPECTED_FOLDS:
            raise ValueError(f"Arm {key} does not contain exactly folds 0..4.")

    paired: dict[tuple, set[str]] = defaultdict(set)
    for row in rows:
        paired[_config_key(row)].add(str(row["rc_mode"]))
    for key, modes in paired.items():
        if modes != {"off", "on"}:
            raise ValueError(f"Config {key} does not contain paired RC off/on arms.")


def _read_registry(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pick_registry_record(candidates: Sequence[dict]) -> dict | None:
    """Prefer a completed record with an extant prediction, then the latest row."""
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda row: str(row.get("timestamp", "")))
    usable = [
        row
        for row in ordered
        if row.get("status") == "completed"
        and bool(row.get("prediction_path"))
        and Path(row["prediction_path"]).is_file()
    ]
    return (usable or ordered)[-1]


def _registry_identity_mismatches(record: Mapping, manifest: Mapping) -> dict[str, tuple[str, str]]:
    """Return registry/manifest identity differences for one launched cell."""
    comparisons = {
        "cell_id": (record.get("cell_id", ""), manifest.get("cell_id", "")),
        "rc_pair_id": (record.get("rc_pair_id", ""), manifest.get("rc_pair_id", "")),
        "analysis_lane": (
            record.get("analysis_lane", ""),
            manifest.get("analysis_lane", ""),
        ),
        "part_slug": (record.get("part_slug", ""), manifest.get("part_slug", "")),
        "base_config_id": (
            record.get("base_config_id", ""),
            manifest.get("base_config_id", ""),
        ),
        "development_fold": (
            record.get("development_fold", ""),
            manifest.get("development_fold", ""),
        ),
        "rc_mode": (record.get("rc_mode", ""), manifest.get("rc_mode", "")),
    }
    mismatches = {}
    for field, (actual, expected) in comparisons.items():
        if field == "development_fold":
            try:
                actual = str(int(actual))
                expected = str(int(expected))
            except (TypeError, ValueError):
                actual = str(actual)
                expected = str(expected)
        else:
            actual = str(actual)
            expected = str(expected)
        if actual != expected:
            mismatches[field] = (actual, expected)
    return mismatches


def resolve_analysis_cells(
    manifest_rows: Sequence[dict], registry_path: str | Path = DEFAULT_REGISTRY
) -> list[dict]:
    """Resolve each manifest cell to a prediction without treating absence as failure."""
    registry = _read_registry(registry_path)
    by_name: dict[str, list[dict]] = defaultdict(list)
    for row in registry:
        if row.get("run_name"):
            by_name[row["run_name"]].append(row)

    resolved = []
    for manifest in manifest_rows:
        disposition = manifest["execution_disposition"]
        record = None
        if disposition == "reuse_stage1":
            prediction_path = str(manifest.get("reuse_prediction_path", ""))
            run_id = str(manifest.get("reuse_source_run_id", ""))
            if prediction_path and Path(prediction_path).is_file():
                expected_sha = str(manifest.get("reuse_prediction_sha256", ""))
                if expected_sha and _sha256_file(prediction_path) != expected_sha:
                    raise ValueError(
                        f"Reused Stage 1 prediction hash changed for {manifest['cell_id']}"
                    )
                availability = "complete"
            else:
                availability = "missing_prediction_file"
        elif disposition == "launch":
            candidates = [
                row
                for row in by_name.get(manifest["planned_run_name"], [])
                if row.get("campaign_id") == CAMPAIGN_ID
                and row.get("campaign_stage") == CAMPAIGN_STAGE
            ]
            matching_candidates = []
            for candidate in candidates:
                mismatches = _registry_identity_mismatches(candidate, manifest)
                if mismatches:
                    if candidate.get("status") == "completed":
                        raise ValueError(
                            "Completed registry row has the right run_name but wrong cell "
                            f"provenance for {manifest['cell_id']}: {mismatches}"
                        )
                    continue
                matching_candidates.append(candidate)
            if candidates and not matching_candidates:
                raise ValueError(
                    f"Registry rows for {manifest['cell_id']} do not match its frozen identity."
                )
            record = _pick_registry_record(matching_candidates)
            prediction_path = str(record.get("prediction_path", "")) if record else ""
            run_id = str(record.get("run_id", "")) if record else ""
            if record is None:
                availability = "missing_registry_row"
            elif record.get("status") != "completed":
                availability = "registry_status_" + str(record.get("status") or "unknown")
            elif not prediction_path or not Path(prediction_path).is_file():
                availability = "missing_prediction_file"
            else:
                availability = "complete"
        else:
            raise ValueError(f"Unknown execution_disposition {disposition!r}")

        resolved.append(
            {
                **manifest,
                "resolved_run_id": run_id,
                "resolved_prediction_path": prediction_path,
                "resolved_registry_status": record.get("status", "") if record else "",
                "availability": availability,
            }
        )
    return resolved


def cell_completion_table(rows: Sequence[dict]) -> pd.DataFrame:
    columns = [
        "analysis_cell",
        "cell_id",
        *ARM_KEYS,
        "development_fold",
        "execution_disposition",
        "planned_run_name",
        "resolved_run_id",
        "resolved_prediction_path",
        "resolved_registry_status",
        "availability",
    ]
    return pd.DataFrame([{column: row.get(column, "") for column in columns} for row in rows])


def _load_split_manifest(path: str | Path, cache: dict[str, dict]) -> dict:
    key = str(Path(path).resolve())
    if key not in cache:
        cache[key] = json.loads(Path(path).read_text(encoding="utf-8"))
    return cache[key]


def _split_sets(split: Mapping, fold: int) -> tuple[set[str], set[str], dict[str, str]]:
    expected = {
        str(row["construct_id"])
        for row in split["assignments"]
        if row["partition"] == "development"
        and int(row["development_fold"]) == int(fold)
    }
    audit = {
        str(row["construct_id"])
        for row in split["assignments"]
        if row["partition"] == "audit_test"
    }
    sequences = {
        str(row["construct_id"]): str(row.get("sequence", ""))
        for row in split["assignments"]
        if row["partition"] == "development"
    }
    return expected, audit, sequences


def load_heldout_prediction(
    row: Mapping, split_cache: dict[str, dict] | None = None
) -> pd.DataFrame:
    """Load one cell and prove it contains that fold's held-out IDs only."""
    if row.get("availability") != "complete":
        raise ValueError(f"Cell {row.get('cell_id')} is not complete.")
    split_cache = split_cache if split_cache is not None else {}
    split = _load_split_manifest(row["split_manifest_path"], split_cache)
    fold = int(row["development_fold"])
    expected, audit, sequences = _split_sets(split, fold)

    frame = pd.read_csv(row["resolved_prediction_path"], sep="\t")
    required = {"construct_id", RAW_TARGET, RAW_PREDICTION}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Prediction {row['resolved_prediction_path']} lacks {sorted(missing)}"
        )
    frame["construct_id"] = frame["construct_id"].astype(str)
    if frame["construct_id"].duplicated().any():
        raise ValueError(f"Cell {row['cell_id']} contains duplicate construct IDs.")
    observed = set(frame["construct_id"])
    if observed & audit:
        raise ValueError(f"Cell {row['cell_id']} exported frozen audit IDs.")
    if observed != expected:
        raise ValueError(
            f"Cell {row['cell_id']} held-out IDs differ from fold {fold}: "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )
    for column in (RAW_TARGET, RAW_PREDICTION):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise ValueError(f"Cell {row['cell_id']} has non-finite {column} values.")
    frame["development_fold"] = fold
    frame["cell_id"] = row["cell_id"]
    frame["resolved_run_id"] = row.get("resolved_run_id", "")
    if row["part_slug"] == "intron":
        frame["intron_sequence"] = frame["construct_id"].map(sequences)
        if frame["intron_sequence"].eq("").any() or frame["intron_sequence"].isna().any():
            raise ValueError("Missing Intron sequence in frozen split assignments.")
    return frame


def assemble_complete_oof_arms(
    rows: Sequence[dict], require_complete: bool = False
) -> tuple[dict[tuple, pd.DataFrame], pd.DataFrame]:
    """Concatenate five disjoint held-out folds for every available config/RC arm."""
    incomplete = [row for row in rows if row["availability"] != "complete"]
    if require_complete and incomplete:
        counts = Counter(row["availability"] for row in incomplete)
        raise RuntimeError(f"Stage 2 is incomplete: {dict(counts)}")

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_arm_key(row)].append(row)

    split_cache: dict[str, dict] = {}
    arms: dict[tuple, pd.DataFrame] = {}
    availability_records = []
    for key, arm_rows in sorted(grouped.items()):
        complete = [row for row in arm_rows if row["availability"] == "complete"]
        record = {**_metadata(arm_rows[0]), "complete_folds": len(complete)}
        if len(complete) != 5:
            record.update(oof_available=False, oof_rows=0, reason="incomplete_folds")
            availability_records.append(record)
            continue
        pieces = [load_heldout_prediction(row, split_cache) for row in complete]
        oof = pd.concat(pieces, ignore_index=True)
        if oof["construct_id"].duplicated().any():
            raise ValueError(f"Arm {key} contains more than one held-out prediction per ID.")

        split = _load_split_manifest(complete[0]["split_manifest_path"], split_cache)
        expected_development = {
            str(item["construct_id"])
            for item in split["assignments"]
            if item["partition"] == "development"
        }
        if set(oof["construct_id"]) != expected_development:
            raise ValueError(f"Arm {key} does not cover its exact development set.")
        expected_count = int(split["expected"]["counts"]["development"])
        if len(oof) != expected_count:
            raise ValueError(f"Arm {key} has {len(oof)} rows; expected {expected_count}.")
        if key[1] == "intron" and len(oof) != EXPECTED_INTRON_DEVELOPMENT_CONSTRUCTS:
            raise ValueError(
                f"Intron arm {key} must have exactly "
                f"{EXPECTED_INTRON_DEVELOPMENT_CONSTRUCTS} OOF predictions."
            )
        if key[1] == "intron":
            oof = assign_inferred_intron_subsets(oof, "intron_sequence").rename(
                columns={"inferred_intron_subset": SENSITIVITY_STRATUM}
            )
        for field, value in _metadata(complete[0]).items():
            oof[field] = value
        arms[key] = oof.sort_values("construct_id").reset_index(drop=True)
        record.update(oof_available=True, oof_rows=len(oof), reason="")
        availability_records.append(record)
    return arms, pd.DataFrame(availability_records)


def _safe_pearson(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2 or np.ptp(target) == 0 or np.ptp(prediction) == 0:
        return math.nan
    return float(np.corrcoef(target, prediction)[0, 1])


def _safe_spearman(target: np.ndarray, prediction: np.ndarray) -> float:
    return _safe_pearson(rankdata(target), rankdata(prediction))


def raw_metrics(
    frame: pd.DataFrame,
    target_column: str = RAW_TARGET,
    prediction_column: str = RAW_PREDICTION,
) -> dict[str, float | int]:
    target = frame[target_column].to_numpy(float)
    prediction = frame[prediction_column].to_numpy(float)
    residual = prediction - target
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((target - target.mean()) ** 2))
    return {
        "n_constructs": int(len(frame)),
        "pearson": _safe_pearson(target, prediction),
        "spearman": _safe_spearman(target, prediction),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(residual))),
        "cod_r2": (
            float(1.0 - np.sum(residual**2) / denominator)
            if denominator > 0
            else math.nan
        ),
    }


def _calibration(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    denominator = float(np.sum((prediction - prediction.mean()) ** 2))
    if denominator == 0:
        slope = math.nan
        intercept = math.nan
    else:
        slope = float(
            np.sum((prediction - prediction.mean()) * (target - target.mean()))
            / denominator
        )
        intercept = float(target.mean() - slope * prediction.mean())
    return {
        # Definition: observed target = intercept + slope * prediction.
        "calibration_slope_observed_on_prediction": slope,
        "calibration_intercept_observed_on_prediction": intercept,
        "target_mean": float(target.mean()),
        "prediction_mean": float(prediction.mean()),
        "prediction_minus_target_bias": float(np.mean(prediction - target)),
    }


def intron_sensitivity_metrics(
    oof: pd.DataFrame, metadata: Mapping | None = None
) -> tuple[dict, list[dict]]:
    """Compute raw OOF metrics for inferred mask sensitivity categories."""
    if SENSITIVITY_STRATUM not in oof:
        raise ValueError(f"Intron OOF table lacks {SENSITIVITY_STRATUM}.")
    metadata = dict(metadata or {})
    per_stratum = []
    for stratum in STRATUM_ORDER:
        group = oof.loc[oof[SENSITIVITY_STRATUM].eq(stratum)].copy()
        if group.empty:
            raise ValueError(f"Intron sensitivity stratum {stratum} is empty.")
        target = group[RAW_TARGET].to_numpy(float)
        prediction = group[RAW_PREDICTION].to_numpy(float)
        per_stratum.append(
            {
                **metadata,
                "sensitivity_label_status": "inferred_sequence_mask_not_true_subset",
                SENSITIVITY_STRATUM: stratum,
                **raw_metrics(group),
                **_calibration(target, prediction),
            }
        )

    centered = oof.copy()
    centered["target_centered"] = centered[RAW_TARGET] - centered.groupby(
        SENSITIVITY_STRATUM
    )[RAW_TARGET].transform("mean")
    centered["prediction_centered"] = centered[RAW_PREDICTION] - centered.groupby(
        SENSITIVITY_STRATUM
    )[RAW_PREDICTION].transform("mean")
    centered_metrics = raw_metrics(centered, "target_centered", "prediction_centered")
    values = pd.DataFrame(per_stratum)
    summary = {
        **{
            f"within_stratum_centered_{key}": value
            for key, value in centered_metrics.items()
            if key != "n_constructs"
        },
        "macro_stratum_pearson": float(values["pearson"].mean()),
        "minimum_stratum_pearson": float(values["pearson"].min()),
        "macro_stratum_spearman": float(values["spearman"].mean()),
        "minimum_stratum_spearman": float(values["spearman"].min()),
        "macro_stratum_cod_r2": float(values["cod_r2"].mean()),
        "minimum_stratum_cod_r2": float(values["cod_r2"].min()),
        "macro_stratum_rmse": float(values["rmse"].mean()),
        "maximum_stratum_rmse": float(values["rmse"].max()),
        "macro_stratum_mae": float(values["mae"].mean()),
        "maximum_stratum_mae": float(values["mae"].max()),
    }
    return summary, per_stratum


def score_oof_arms(
    arms: Mapping[tuple, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    intron_rows = []
    for key, oof in sorted(arms.items()):
        metadata = _metadata(oof.iloc[0])
        pooled = raw_metrics(oof)
        row = {
            **metadata,
            "primary_metric_name": "pooled_five_fold_oof_pearson",
            **{f"pooled_oof_{name}": value for name, value in pooled.items()},
            **_calibration(
                oof[RAW_TARGET].to_numpy(float),
                oof[RAW_PREDICTION].to_numpy(float),
            ),
        }
        if key[1] == "intron":
            summary, strata = intron_sensitivity_metrics(oof, metadata)
            row.update(summary)
            intron_rows.extend(strata)
        metric_rows.append(row)
    metric_frame = pd.DataFrame(metric_rows)
    if metric_frame.empty:
        metric_frame = pd.DataFrame(
            columns=[
                *METADATA_FIELDS,
                "primary_metric_name",
                *(f"pooled_oof_{name}" for name in METRIC_NAMES),
            ]
        )
    intron_frame = pd.DataFrame(intron_rows)
    if intron_frame.empty:
        intron_frame = pd.DataFrame(
            columns=[
                *METADATA_FIELDS,
                "sensitivity_label_status",
                SENSITIVITY_STRATUM,
                *METRIC_NAMES,
                "calibration_slope_observed_on_prediction",
                "calibration_intercept_observed_on_prediction",
                "target_mean",
                "prediction_mean",
                "prediction_minus_target_bias",
            ]
        )
    return metric_frame, intron_frame


def score_oof_folds(arms: Mapping[tuple, pd.DataFrame]) -> pd.DataFrame:
    """Score each held-out fold within every config/RC arm.

    Pooled five-fold OOF Pearson remains the primary Stage 2 estimand. These
    rows are the predeclared fold-stability diagnostics for the Stage 3
    one-standard-error review; they must not replace the pooled result.
    """
    rows = []
    for _key, oof in sorted(arms.items()):
        metadata = _metadata(oof.iloc[0])
        for fold, frame in oof.groupby("development_fold", sort=True):
            row = {
                **metadata,
                "development_fold": int(fold),
                **{f"fold_{name}": value for name, value in raw_metrics(frame).items()},
            }
            if metadata["part_slug"] == "intron":
                summary, _ = intron_sensitivity_metrics(frame)
                row.update(
                    fold_within_stratum_centered_pearson=summary[
                        "within_stratum_centered_pearson"
                    ],
                    fold_macro_stratum_pearson=summary["macro_stratum_pearson"],
                    fold_minimum_stratum_pearson=summary[
                        "minimum_stratum_pearson"
                    ],
                )
            rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                *METADATA_FIELDS,
                "development_fold",
                *(f"fold_{name}" for name in METRIC_NAMES),
                "fold_within_stratum_centered_pearson",
                "fold_macro_stratum_pearson",
                "fold_minimum_stratum_pearson",
            ]
        )
    return frame


def score_intron_stratum_mean_baselines(
    arms: Mapping[tuple, pd.DataFrame],
    analysis_rows: Sequence[Mapping] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit leakage-safe and explanatory Intron stratum-mean baselines.

    For the leakage-safe baseline, each fold's stratum means are fit on its
    exact model-training pool: all ``train_only`` constructs plus the other
    four development folds. The oracle row uses all development targets and is
    labeled explanatory because it includes the held-out fold when estimating
    each mean. Neither path scores audit constructs or instantiates a loader.
    """
    intron_arms = [
        frame for key, frame in sorted(arms.items()) if key[1] == "intron"
    ]
    if not intron_arms:
        return pd.DataFrame(), pd.DataFrame()

    reference = intron_arms[0][
        [
            "construct_id",
            "development_fold",
            RAW_TARGET,
            SENSITIVITY_STRATUM,
        ]
    ].copy()
    if reference["construct_id"].duplicated().any():
        raise ValueError("Reference Intron OOF arm contains duplicate constructs.")
    if len(reference) != EXPECTED_INTRON_DEVELOPMENT_CONSTRUCTS:
        raise ValueError(
            "Reference Intron OOF arm does not contain all development constructs."
        )

    training_source = reference.copy()
    training_source["partition"] = "development"
    if analysis_rows is not None:
        intron_rows = [row for row in analysis_rows if row.get("part_slug") == "intron"]
        if not intron_rows:
            raise ValueError("No Intron analysis row is available for the baseline.")
        split_path = Path(intron_rows[0]["split_manifest_path"])
        dataset_path = Path(intron_rows[0]["dataset_path"])
        split = _load_split_manifest(split_path, {})
        assignments = pd.DataFrame(split["assignments"])
        allowed = assignments.loc[
            assignments["partition"].isin(["train_only", "development"]),
            ["construct_id", "partition", "development_fold", "sequence"],
        ].copy()
        allowed["construct_id"] = allowed["construct_id"].astype(str)
        dataset = pd.read_csv(
            dataset_path,
            sep="\t",
            usecols=["construct_id", RAW_TARGET],
        )
        dataset["construct_id"] = dataset["construct_id"].astype(str)
        training_source = allowed.merge(
            dataset,
            on="construct_id",
            how="left",
            validate="one_to_one",
        )
        if training_source[RAW_TARGET].isna().any():
            raise ValueError("A non-audit Intron training target is missing.")
        training_source = assign_inferred_intron_subsets(
            training_source, "sequence"
        ).rename(columns={"inferred_intron_subset": SENSITIVITY_STRATUM})

    fold_pieces = []
    for fold in sorted(reference["development_fold"].astype(int).unique()):
        is_train_only = training_source["partition"].eq("train_only")
        fold_values = pd.to_numeric(
            training_source["development_fold"], errors="coerce"
        )
        train = training_source.loc[is_train_only | ~fold_values.eq(fold)]
        heldout = reference.loc[
            reference["development_fold"].astype(int).eq(fold)
        ].copy()
        if analysis_rows is not None:
            expected_count = int(
                split["expected"]["per_fold"][str(fold)]["train_count"]
            )
            if len(train) != expected_count:
                raise ValueError(
                    f"Intron fold {fold} baseline has {len(train)} training rows; "
                    f"expected {expected_count}."
                )
        means = train.groupby(SENSITIVITY_STRATUM)[RAW_TARGET].mean()
        heldout[RAW_PREDICTION] = heldout[SENSITIVITY_STRATUM].map(means)
        if heldout[RAW_PREDICTION].isna().any():
            raise ValueError(f"Fold {fold} lacks a training stratum mean.")
        heldout["baseline_type"] = "fold_trained_stratum_mean"
        heldout["fit_scope"] = (
            "exact_non_audit_model_training_rows"
            if analysis_rows is not None
            else "other_four_development_folds_test_fixture"
        )
        fold_pieces.append(heldout)
    fold_trained = pd.concat(fold_pieces, ignore_index=True)

    oracle = reference.copy()
    oracle_means = oracle.groupby(SENSITIVITY_STRATUM)[RAW_TARGET].mean()
    oracle[RAW_PREDICTION] = oracle[SENSITIVITY_STRATUM].map(oracle_means)
    oracle["baseline_type"] = "development_oracle_stratum_mean"
    oracle["fit_scope"] = "all_development_targets_explanatory_only"

    summaries = []
    for frame in (fold_trained, oracle):
        intron_summary, _ = intron_sensitivity_metrics(frame)
        summaries.append(
            {
                "baseline_type": frame["baseline_type"].iloc[0],
                "fit_scope": frame["fit_scope"].iloc[0],
                **{f"pooled_oof_{key}": value for key, value in raw_metrics(frame).items()},
                **intron_summary,
            }
        )
    predictions = pd.concat([fold_trained, oracle], ignore_index=True)
    return pd.DataFrame(summaries), predictions


def compare_paired_rc(
    arms: Mapping[tuple, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare RC off/on as construct-paired raw predictions."""
    by_config: dict[tuple, dict[str, pd.DataFrame]] = defaultdict(dict)
    for key, frame in arms.items():
        by_config[key[:3]][key[3]] = frame

    summaries = []
    fold_summaries = []
    construct_rows = []
    for key, modes in sorted(by_config.items()):
        if set(modes) != {"off", "on"}:
            continue
        off = modes["off"].sort_values("construct_id").reset_index(drop=True)
        on = modes["on"].sort_values("construct_id").reset_index(drop=True)
        if not off["construct_id"].equals(on["construct_id"]):
            raise ValueError(f"RC pair {key} does not contain identical constructs.")
        if not off["development_fold"].equals(on["development_fold"]):
            raise ValueError(f"RC pair {key} has inconsistent development-fold labels.")
        if not np.allclose(
            off[RAW_TARGET].to_numpy(float),
            on[RAW_TARGET].to_numpy(float),
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError(f"RC pair {key} has inconsistent raw targets.")

        target = off[RAW_TARGET].to_numpy(float)
        prediction_off = off[RAW_PREDICTION].to_numpy(float)
        prediction_on = on[RAW_PREDICTION].to_numpy(float)
        abs_off = np.abs(prediction_off - target)
        abs_on = np.abs(prediction_on - target)
        sq_off = (prediction_off - target) ** 2
        sq_on = (prediction_on - target) ** 2
        delta_abs = abs_on - abs_off
        delta_sq = sq_on - sq_off
        off_metrics = raw_metrics(off)
        on_metrics = raw_metrics(on)
        metadata = _metadata(off.iloc[0])
        metadata.pop("rc_mode", None)
        summary = {**metadata, "n_paired_constructs": len(off)}
        for metric in ("pearson", "spearman", "rmse", "mae", "cod_r2"):
            summary[f"rc_off_pooled_oof_{metric}"] = off_metrics[metric]
            summary[f"rc_on_pooled_oof_{metric}"] = on_metrics[metric]
            summary[f"delta_rc_on_minus_off_pooled_oof_{metric}"] = (
                on_metrics[metric] - off_metrics[metric]
            )

        if key[1] == "intron":
            off_intron, _ = intron_sensitivity_metrics(off)
            on_intron, _ = intron_sensitivity_metrics(on)
            for metric in (
                "within_stratum_centered_pearson",
                "macro_stratum_pearson",
                "minimum_stratum_pearson",
            ):
                summary[f"rc_off_{metric}"] = off_intron[metric]
                summary[f"rc_on_{metric}"] = on_intron[metric]
                summary[f"delta_rc_on_minus_off_{metric}"] = (
                    on_intron[metric] - off_intron[metric]
                )

        config_fold_rows = []
        for fold in sorted(off["development_fold"].astype(int).unique()):
            fold_off = off.loc[off["development_fold"].astype(int).eq(fold)].copy()
            fold_on = on.loc[on["development_fold"].astype(int).eq(fold)].copy()
            fold_off = fold_off.sort_values("construct_id").reset_index(drop=True)
            fold_on = fold_on.sort_values("construct_id").reset_index(drop=True)
            if not fold_off["construct_id"].equals(fold_on["construct_id"]):
                raise ValueError(
                    f"RC pair {key} fold {fold} does not contain identical constructs."
                )
            if not np.allclose(
                fold_off[RAW_TARGET].to_numpy(float),
                fold_on[RAW_TARGET].to_numpy(float),
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError(f"RC pair {key} fold {fold} has inconsistent raw targets.")
            fold_off_metrics = raw_metrics(fold_off)
            fold_on_metrics = raw_metrics(fold_on)
            fold_row = {
                **metadata,
                "development_fold": int(fold),
                "n_paired_constructs": len(fold_off),
                "rc_off_pooled_pearson": fold_off_metrics["pearson"],
                "rc_on_pooled_pearson": fold_on_metrics["pearson"],
                "delta_rc_on_minus_off_pooled_pearson": (
                    fold_on_metrics["pearson"] - fold_off_metrics["pearson"]
                ),
            }
            if key[1] == "intron":
                fold_off_intron, _ = intron_sensitivity_metrics(fold_off)
                fold_on_intron, _ = intron_sensitivity_metrics(fold_on)
                fold_row.update(
                    rc_off_within_stratum_centered_pearson=fold_off_intron[
                        "within_stratum_centered_pearson"
                    ],
                    rc_on_within_stratum_centered_pearson=fold_on_intron[
                        "within_stratum_centered_pearson"
                    ],
                    delta_rc_on_minus_off_within_stratum_centered_pearson=(
                        fold_on_intron["within_stratum_centered_pearson"]
                        - fold_off_intron["within_stratum_centered_pearson"]
                    ),
                )
            fold_summaries.append(fold_row)
            config_fold_rows.append(fold_row)

        fold_frame = pd.DataFrame(config_fold_rows)
        pearson_fold_deltas = fold_frame[
            "delta_rc_on_minus_off_pooled_pearson"
        ].dropna()
        positive_pearson_folds = int((pearson_fold_deltas > 0).sum())
        formal_pearson_fold_gate = bool(
            pearson_fold_deltas.mean() >= 0.005
            and positive_pearson_folds >= 4
        )
        zero_tolerance_error_guard = bool(
            summary["delta_rc_on_minus_off_pooled_oof_rmse"] <= 0
            and summary["delta_rc_on_minus_off_pooled_oof_cod_r2"] >= 0
        )
        summary.update(
            mean_fold_delta_rc_on_minus_off_pooled_pearson=float(
                pearson_fold_deltas.mean()
            ),
            positive_fold_count_rc_on_minus_off_pooled_pearson=(
                positive_pearson_folds
            ),
            negative_fold_count_rc_on_minus_off_pooled_pearson=int(
                (pearson_fold_deltas < 0).sum()
            ),
            finite_fold_count_rc_on_minus_off_pooled_pearson=int(
                len(pearson_fold_deltas)
            ),
            positive_mean_and_no_more_than_two_negative_pearson_folds=bool(
                pearson_fold_deltas.mean() > 0
                and int((pearson_fold_deltas < 0).sum()) <= 2
            ),
            formal_pearson_fold_gate_mean_ge_0p005_and_positive_ge_4=(
                formal_pearson_fold_gate
            ),
            zero_tolerance_rmse_cod_guard=zero_tolerance_error_guard,
            formal_pearson_gate_and_zero_tolerance_error_guard=bool(
                formal_pearson_fold_gate and zero_tolerance_error_guard
            ),
        )
        if key[1] == "intron":
            within_fold_deltas = fold_frame[
                "delta_rc_on_minus_off_within_stratum_centered_pearson"
            ].dropna()
            summary.update(
                mean_fold_delta_rc_on_minus_off_within_stratum_centered_pearson=float(
                    within_fold_deltas.mean()
                ),
                negative_fold_count_rc_on_minus_off_within_stratum_centered_pearson=int(
                    (within_fold_deltas < 0).sum()
                ),
                finite_fold_count_rc_on_minus_off_within_stratum_centered_pearson=int(
                    len(within_fold_deltas)
                ),
                positive_mean_and_no_more_than_two_negative_within_stratum_folds=bool(
                    within_fold_deltas.mean() > 0
                    and int((within_fold_deltas < 0).sum()) <= 2
                ),
                formal_intron_pooled_and_within_fold_gate=bool(
                    formal_pearson_fold_gate
                    and within_fold_deltas.mean() >= 0
                    and int((within_fold_deltas < 0).sum()) <= 2
                ),
                formal_intron_gate_and_zero_tolerance_error_guard=bool(
                    formal_pearson_fold_gate
                    and within_fold_deltas.mean() >= 0
                    and int((within_fold_deltas < 0).sum()) <= 2
                    and zero_tolerance_error_guard
                ),
            )
        summary.update(
            mean_paired_abs_error_delta_on_minus_off=float(delta_abs.mean()),
            median_paired_abs_error_delta_on_minus_off=float(np.median(delta_abs)),
            paired_abs_error_delta_standard_error=(
                float(delta_abs.std(ddof=1) / math.sqrt(len(delta_abs)))
                if len(delta_abs) > 1
                else math.nan
            ),
            mean_paired_squared_error_delta_on_minus_off=float(delta_sq.mean()),
            paired_squared_error_delta_standard_error=(
                float(delta_sq.std(ddof=1) / math.sqrt(len(delta_sq)))
                if len(delta_sq) > 1
                else math.nan
            ),
            rc_on_lower_abs_error_fraction=float(np.mean(abs_on < abs_off)),
            rc_off_lower_abs_error_fraction=float(np.mean(abs_off < abs_on)),
            paired_abs_error_tie_fraction=float(np.mean(abs_off == abs_on)),
        )
        summaries.append(summary)

        paired = pd.DataFrame(
            {
                **{field: [value] * len(off) for field, value in metadata.items()},
                "construct_id": off["construct_id"],
                RAW_TARGET: target,
                "prediction_raw_rc_off": prediction_off,
                "prediction_raw_rc_on": prediction_on,
                "abs_error_rc_off": abs_off,
                "abs_error_rc_on": abs_on,
                "abs_error_delta_on_minus_off": delta_abs,
                "squared_error_rc_off": sq_off,
                "squared_error_rc_on": sq_on,
                "squared_error_delta_on_minus_off": delta_sq,
            }
        )
        if SENSITIVITY_STRATUM in off:
            paired[SENSITIVITY_STRATUM] = off[SENSITIVITY_STRATUM]
        construct_rows.append(paired)

    construct_frame = (
        pd.concat(construct_rows, ignore_index=True) if construct_rows else pd.DataFrame()
    )
    summary_frame = pd.DataFrame(summaries)
    fold_summary_frame = pd.DataFrame(fold_summaries)
    if summary_frame.empty:
        summary_frame = pd.DataFrame(
            columns=[
                *(field for field in METADATA_FIELDS if field != "rc_mode"),
                "n_paired_constructs",
                "mean_paired_abs_error_delta_on_minus_off",
                "mean_paired_squared_error_delta_on_minus_off",
            ]
        )
    if fold_summary_frame.empty:
        fold_summary_frame = pd.DataFrame(
            columns=[
                *(field for field in METADATA_FIELDS if field != "rc_mode"),
                "development_fold",
                "n_paired_constructs",
                "rc_off_pooled_pearson",
                "rc_on_pooled_pearson",
                "delta_rc_on_minus_off_pooled_pearson",
                "rc_off_within_stratum_centered_pearson",
                "rc_on_within_stratum_centered_pearson",
                "delta_rc_on_minus_off_within_stratum_centered_pearson",
            ]
        )
    return summary_frame, fold_summary_frame, construct_frame


def flatten_oof_arms(arms: Mapping[tuple, pd.DataFrame]) -> pd.DataFrame:
    if not arms:
        return pd.DataFrame()
    keep = [
        *ARM_KEYS,
        "architecture",
        "policy_id",
        "development_fold",
        "cell_id",
        "resolved_run_id",
        "construct_id",
        RAW_TARGET,
        RAW_PREDICTION,
    ]
    pieces = []
    for frame in arms.values():
        columns = keep + ([SENSITIVITY_STRATUM] if SENSITIVITY_STRATUM in frame else [])
        pieces.append(frame[columns])
    return pd.concat(pieces, ignore_index=True)


def _write_frame(frame: pd.DataFrame, path: Path, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, sep=sep)


def run_analysis(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    registry_path: str | Path = DEFAULT_REGISTRY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    require_complete: bool = False,
) -> dict:
    manifest_rows = _read_jsonl(manifest_path)
    validate_analysis_manifest(manifest_rows)
    resolved = resolve_analysis_cells(manifest_rows, registry_path)
    arms, arm_availability = assemble_complete_oof_arms(
        resolved, require_complete=require_complete
    )
    metrics, intron_metrics = score_oof_arms(arms)
    fold_metrics = score_oof_folds(arms)
    intron_baselines, intron_baseline_predictions = (
        score_intron_stratum_mean_baselines(arms, analysis_rows=resolved)
    )
    rc_metrics, rc_fold_metrics, rc_constructs = compare_paired_rc(arms)
    oof_predictions = flatten_oof_arms(arms)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _write_frame(cell_completion_table(resolved), root / "stage2_cell_completion.csv")
    _write_frame(arm_availability, root / "stage2_oof_arm_availability.csv")
    _write_frame(metrics, root / "stage2_oof_metrics.csv")
    _write_frame(fold_metrics, root / "stage2_oof_fold_metrics.csv")
    _write_frame(
        intron_metrics, root / "stage2_intron_sensitivity_stratum_metrics.csv"
    )
    _write_frame(
        intron_baselines, root / "stage2_intron_stratum_mean_baselines.csv"
    )
    _write_frame(
        intron_baseline_predictions,
        root / "stage2_intron_stratum_mean_baseline_predictions.tsv",
        sep="\t",
    )
    _write_frame(rc_metrics, root / "stage2_rc_pair_metrics.csv")
    _write_frame(rc_fold_metrics, root / "stage2_rc_fold_pair_metrics.csv")
    _write_frame(
        rc_constructs, root / "stage2_rc_paired_construct_errors.tsv", sep="\t"
    )
    if not oof_predictions.empty:
        oof_predictions.to_csv(
            root / "stage2_oof_predictions.tsv.gz",
            index=False,
            sep="\t",
            compression="gzip",
        )

    availability_counts = Counter(row["availability"] for row in resolved)
    summary = {
        "campaign_id": CAMPAIGN_ID,
        "campaign_stage": CAMPAIGN_STAGE,
        "analysis_manifest": str(Path(manifest_path).resolve()),
        "registry": str(Path(registry_path).resolve()),
        "analysis_cells": len(resolved),
        "cell_availability": dict(sorted(availability_counts.items())),
        "complete_oof_arms": len(arms),
        "expected_oof_arms": EXPECTED_ANALYSIS_CELLS // 5,
        "complete_oof_fold_rows": len(fold_metrics),
        "complete_paired_rc_configs": len(rc_metrics),
        "primary_metric": "pooled_five_fold_oof_pearson",
        "raw_target_column": RAW_TARGET,
        "raw_prediction_column": RAW_PREDICTION,
        "intron_sensitivity_label_status": "inferred_sequence_mask_not_true_subset",
        "audit_loader_instantiated": False,
        "require_complete": bool(require_complete),
    }
    (root / "stage2_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless all 660 cell predictions are complete before analysis.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(
        manifest_path=args.manifest,
        registry_path=args.registry,
        output_dir=args.output_dir,
        require_complete=args.require_complete,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
