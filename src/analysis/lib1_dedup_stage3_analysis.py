#!/usr/bin/env python3
"""Development-only analysis for the frozen Lib1 dedup Stage 3 campaign.

The default path is fail-closed: all 900 manifest cells must resolve before any
OOF gate or model selection is computed.  ``--readiness-only`` (also exposed
as ``--completion-only``) validates the frozen manifest and writes completion
evidence without requiring the 450 new weighted cells to exist.  This module
reads exported validation predictions and compact provenance only.  It never
imports a DataModule, constructs a loader, or evaluates the frozen audit set.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LEARN_ROOT = REPO_ROOT / "src" / "learn"
for import_root in (REPO_ROOT, LEARN_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.analysis.lib1_dedup_intron_sensitivity_reporting import intron_estimands
from src.analysis.lib1_dedup_stage2_analysis import (
    RAW_PREDICTION,
    RAW_TARGET,
    SENSITIVITY_STRATUM,
    STRATUM_ORDER,
    _calibration,
    intron_sensitivity_metrics,
    raw_metrics,
)
from src.analysis.lib1_dedup_utr3_targeted_hpo_analysis import model_parameter_count
from src.learn import verify_lib1_dedup_stage3_manifest as manifest_verifier
from src.learn.run_lib1_dedup_stage3_campaign import (
    expected_registry_fields,
    validate_completed_record,
)


MANIFEST_TAG = "lib1_dedup_stage3_weighted_loss_july2026"
PREFIX = LEARN_ROOT / "outputs" / "hpo_manifests" / MANIFEST_TAG
DEFAULT_ANALYSIS_MANIFEST = Path(str(PREFIX) + "__analysis_manifest.jsonl")
DEFAULT_DRY_RUN_MANIFEST = Path(str(PREFIX) + "__dry_run_manifest.jsonl")
DEFAULT_REUSE_MANIFEST = Path(str(PREFIX) + "__unweighted_reuse.jsonl")
DEFAULT_PORTFOLIO = Path(str(PREFIX) + "__portfolio.json")
DEFAULT_MANIFEST_SUMMARY = Path(str(PREFIX) + "__summary.json")
DEFAULT_STAGE2_MANIFEST = (
    LEARN_ROOT
    / "outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl"
)
DEFAULT_TARGETED_MANIFEST = (
    LEARN_ROOT
    / "outputs/hpo_manifests/"
    "lib1_dedup_utr3_targeted_hpo_july2026__dry_run_manifest.jsonl"
)
DEFAULT_STAGE2_METRICS = (
    LEARN_ROOT
    / "outputs/analysis/lib1_dedup_stage2_july2026/stage2_oof_metrics.csv"
)
DEFAULT_TARGETED_METRICS = (
    LEARN_ROOT
    / "outputs/analysis/lib1_dedup_utr3_targeted_hpo_july2026/"
    "utr3_targeted_hpo_combined_arm_metrics.csv"
)
DEFAULT_REGISTRY = LEARN_ROOT / "run_registry/runs.csv"
DEFAULT_INTRON_BASELINE_PREDICTIONS = (
    LEARN_ROOT
    / "outputs/analysis/lib1_dedup_stage2_july2026/"
    "stage2_intron_stratum_mean_baseline_predictions.tsv"
)
EXPECTED_INTRON_BASELINE_PREDICTIONS_SHA256 = (
    "82c228a3ba0cd0b0df403b52095f8efc1a9a3cdd20417a656b8cccb8f2d14e8c"
)
DEFAULT_OUTPUT_DIR = (
    LEARN_ROOT / "outputs/analysis/lib1_dedup_stage3_weighted_loss_july2026"
)

EXPECTED_CELLS = 900
EXPECTED_WEIGHTED_CELLS = 450
EXPECTED_ARMS = 180
EXPECTED_LOSS_ARM_PAIRS = 90
EXPECTED_RC_ARM_PAIRS = 80
EXPECTED_FACTORIALS = 40
EXPECTED_FOLDS = tuple(range(5))
PART_ORDER = {"enhancer": 0, "promoter": 1, "intron": 2, "utr3": 3, "utr5": 4}
LOSS_ORDER = {"unweighted_mse": 0, "barcode_weighted_mse": 1}
RC_ORDER = {"off": 0, "on": 1}
BOOTSTRAP_SEED = 20260714
BOOTSTRAP_RESAMPLES = 10_000
ARM_KEYS = ("part_slug", "base_config_id", "rc_mode", "loss_mode")
EXPECTED_INTRON_DEVELOPMENT_CONSTRUCTS = 1061

ARM_METADATA_FIELDS = (
    "part_slug",
    "portfolio_rank",
    "portfolio_role",
    "base_config_id",
    "architecture",
    "analysis_lane",
    "training_regime",
    "initialization",
    "source_head",
    "unfreeze_scope",
    "input_policy",
    "policy_id",
    "rc_mode",
    "loss_mode",
)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_id_hash(values: Sequence[object]) -> str:
    payload = json.dumps(
        sorted(str(value) for value in values),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def calibration_with_reason(target: np.ndarray, prediction: np.ndarray) -> dict:
    """Return calibration plus an explicit reason whenever it is undefined."""
    values = _calibration(target, prediction)
    if len(target) < 2:
        reason = "fewer_than_two_rows"
    elif not np.isfinite(target).all() or not np.isfinite(prediction).all():
        reason = "nonfinite_target_or_prediction"
    elif float(np.ptp(prediction)) == 0.0:
        reason = "constant_prediction"
    elif not np.isfinite(
        [
            values["calibration_slope_observed_on_prediction"],
            values["calibration_intercept_observed_on_prediction"],
        ]
    ).all():
        reason = "undefined_numeric_calibration"
    else:
        reason = ""
    return {
        **values,
        "calibration_defined": not bool(reason),
        "calibration_undefined_reason": reason,
    }


def strict_intron_sensitivity_metrics(
    frame: pd.DataFrame, metadata: Mapping | None = None
) -> tuple[dict, list[dict]]:
    """Make every pooled Intron aggregate propagate an undefined stratum."""
    summary, per_stratum = intron_sensitivity_metrics(frame, metadata)
    records = []
    for record in per_stratum:
        slope = record.get("calibration_slope_observed_on_prediction", math.nan)
        intercept = record.get(
            "calibration_intercept_observed_on_prediction", math.nan
        )
        defined = bool(np.isfinite([slope, intercept]).all())
        records.append(
            {
                **record,
                "sensitivity_label_status": (
                    "inferred_sequence_mask_not_verified_sublibrary"
                ),
                "calibration_defined": defined,
                "calibration_undefined_reason": (
                    "" if defined else "constant_or_nonfinite_stratum_prediction"
                ),
            }
        )
    values = pd.DataFrame(records)
    for metric, aggregate_names in {
        "pearson": ("macro_stratum_pearson", "minimum_stratum_pearson"),
        "spearman": ("macro_stratum_spearman", "minimum_stratum_spearman"),
        "cod_r2": ("macro_stratum_cod_r2", "minimum_stratum_cod_r2"),
        "rmse": ("macro_stratum_rmse", "maximum_stratum_rmse"),
        "mae": ("macro_stratum_mae", "maximum_stratum_mae"),
    }.items():
        if not np.isfinite(values[metric].to_numpy(float)).all():
            for name in aggregate_names:
                summary[name] = math.nan
    return summary, records


def arm_key(row: Mapping) -> tuple[str, str, str, str]:
    return tuple(str(row[field]) for field in ARM_KEYS)


def arm_metadata(row: Mapping) -> dict:
    return {field: row.get(field, "") for field in ARM_METADATA_FIELDS}


def write_json(path: Path, value: Mapping | Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_frame(frame: pd.DataFrame, path: Path, *, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, sep=sep)


def verifier_arguments(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=Path(args.dry_run_manifest),
        analysis_manifest=Path(args.analysis_manifest),
        reuse_manifest=Path(args.reuse_manifest),
        portfolio=Path(args.portfolio),
        summary=Path(args.manifest_summary),
        stage2_analysis_manifest=Path(args.stage2_analysis_manifest),
        targeted_utr3_manifest=Path(args.targeted_utr3_manifest),
        stage2_metrics=Path(args.stage2_metrics),
        targeted_metrics=Path(args.targeted_metrics),
    )


def validate_frozen_manifest(args: argparse.Namespace) -> dict:
    """Run the independent Stage 3 verifier before inspecting completion."""
    report = manifest_verifier.validate(verifier_arguments(args))
    if report.get("validation_status") != "passed":
        raise RuntimeError("The frozen Stage 3 manifest did not pass verification.")
    if report.get("analysis_cells") != EXPECTED_CELLS:
        raise RuntimeError("The verifier did not confirm the frozen 900-cell design.")
    if report.get("audit_loader_instantiated") is not False:
        raise RuntimeError("Manifest verification no longer proves audit isolation.")
    return report


def read_registry(path: str | Path) -> dict[str, list[dict]]:
    by_cell: dict[str, list[dict]] = defaultdict(list)
    registry_path = Path(path)
    if not registry_path.is_file():
        return by_cell
    with registry_path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record.get("cell_id"):
                by_cell[str(record["cell_id"])].append(record)
    return by_cell


def _weighted_completion(row: Mapping, candidates: Sequence[dict]) -> dict:
    if not candidates:
        return {
            "availability": "missing_registry_row",
            "resolved_run_id": "",
            "resolved_prediction_path": "",
            "resolved_prediction_sha256": "",
            "resolved_provenance_path": "",
            "resolved_provenance_sha256": "",
            "resolved_registry_status": "",
        }

    expected = expected_registry_fields(dict(row))
    matching = []
    for record in candidates:
        mismatches = {
            field: {"observed": record.get(field, ""), "expected": value}
            for field, value in expected.items()
            if record.get(field, "") != value
        }
        if mismatches:
            raise RuntimeError(
                "Stage 3 registry provenance collision for {}:\n{}".format(
                    row["cell_id"], json.dumps(mismatches, indent=2, sort_keys=True)
                )
            )
        matching.append(record)

    completed = [
        record
        for record in matching
        if str(record.get("status", "")).lower() == "completed"
    ]
    if len(completed) > 1:
        raise RuntimeError(
            f"Cell {row['cell_id']} resolves to multiple completed registry records."
        )
    if not completed:
        statuses = sorted(
            {str(record.get("status", "") or "unknown").lower() for record in matching}
        )
        return {
            "availability": "registry_status_" + "+".join(statuses),
            "resolved_run_id": "",
            "resolved_prediction_path": "",
            "resolved_prediction_sha256": "",
            "resolved_provenance_path": "",
            "resolved_provenance_sha256": "",
            "resolved_registry_status": "+".join(statuses),
        }

    record = completed[0]
    validate_completed_record(dict(row), record)
    prediction_path = Path(record["prediction_path"]).resolve()
    run_id = str(record["run_id"])
    provenance_path = (
        Path(row["default_root_dir"]) / "provenance" / f"{run_id}__run_provenance.json"
    ).resolve()
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if payload.get("data_split_summary", {}).get("n_test") != 0:
        raise RuntimeError(f"Cell {row['cell_id']} provenance exposes a test set.")
    return {
        "availability": "complete",
        "resolved_run_id": run_id,
        "resolved_prediction_path": str(prediction_path),
        "resolved_prediction_sha256": sha256_file(prediction_path),
        "resolved_provenance_path": str(provenance_path),
        "resolved_provenance_sha256": sha256_file(provenance_path),
        "resolved_registry_status": "completed",
    }


def resolve_cells(
    manifest_rows: Sequence[dict], registry_path: str | Path
) -> list[dict]:
    """Resolve immutable source cells and new weighted registry cells."""
    if len(manifest_rows) != EXPECTED_CELLS:
        raise ValueError(f"Expected {EXPECTED_CELLS} analysis cells; found {len(manifest_rows)}")
    registry = read_registry(registry_path)
    resolved = []
    for row in manifest_rows:
        if row["execution_disposition"] == "reuse_unweighted":
            prediction = Path(row["source_prediction_path"])
            provenance = Path(row["source_provenance_path"])
            if not prediction.is_file() or not provenance.is_file():
                raise FileNotFoundError(
                    f"Immutable source evidence is missing for {row['cell_id']}"
                )
            evidence = {
                "availability": "complete",
                "resolved_run_id": row["source_run_id"],
                "resolved_prediction_path": str(prediction.resolve()),
                "resolved_prediction_sha256": row["source_prediction_sha256"],
                "resolved_provenance_path": str(provenance.resolve()),
                "resolved_provenance_sha256": row["source_provenance_sha256"],
                "resolved_registry_status": "immutable_source_reuse",
            }
        elif row["execution_disposition"] == "launch":
            evidence = _weighted_completion(row, registry.get(row["cell_id"], []))
        else:
            raise ValueError(
                f"Unknown execution disposition {row['execution_disposition']!r}"
            )
        resolved.append({**row, **evidence})

    completed_launches = [
        row
        for row in resolved
        if row["execution_disposition"] == "launch"
        and row["availability"] == "complete"
    ]
    for field in (
        "resolved_run_id",
        "resolved_prediction_path",
        "resolved_provenance_path",
    ):
        duplicates = sorted(
            value
            for value, count in Counter(row[field] for row in completed_launches).items()
            if value and count > 1
        )
        if duplicates:
            raise RuntimeError(
                f"Completed weighted cells reuse {field}: {duplicates}"
            )
    return resolved


def completion_table(rows: Sequence[Mapping]) -> pd.DataFrame:
    fields = (
        "analysis_cell",
        "cell_id",
        "source_unweighted_cell_id",
        "loss_pair_id",
        "rc_pair_id",
        "part_slug",
        "portfolio_rank",
        "base_config_id",
        "development_fold",
        "rc_mode",
        "loss_mode",
        "execution_disposition",
        "planned_run_name",
        "availability",
        "resolved_registry_status",
        "resolved_run_id",
        "resolved_prediction_path",
        "resolved_prediction_sha256",
        "resolved_provenance_path",
        "resolved_provenance_sha256",
        "source_val_row_id_hash",
        "source_prediction_rows",
    )
    return pd.DataFrame(
        [{field: row.get(field, "") for field in fields} for row in rows]
    )


def arm_readiness(rows: Sequence[Mapping]) -> pd.DataFrame:
    records = []
    grouped: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[arm_key(row)].append(row)
    for key, pieces in sorted(
        grouped.items(),
        key=lambda item: (
            PART_ORDER[item[0][0]], item[0][1], RC_ORDER[item[0][2]], LOSS_ORDER[item[0][3]]
        ),
    ):
        complete_folds = sorted(
            int(row["development_fold"])
            for row in pieces
            if row["availability"] == "complete"
        )
        records.append(
            {
                **dict(zip(ARM_KEYS, key)),
                "manifest_cell_count": len(pieces),
                "complete_fold_count": len(complete_folds),
                "complete_folds_json": json.dumps(complete_folds),
                "complete_oof_arm": len(pieces) == 5 and complete_folds == list(EXPECTED_FOLDS),
                "availability_counts_json": json.dumps(
                    dict(sorted(Counter(row["availability"] for row in pieces).items())),
                    sort_keys=True,
                ),
            }
        )
    frame = pd.DataFrame(records)
    if len(frame) != EXPECTED_ARMS:
        raise ValueError(f"Expected {EXPECTED_ARMS} Stage 3 arms; found {len(frame)}")
    return frame


def readiness_summary(
    rows: Sequence[Mapping], readiness: pd.DataFrame, verifier_report: Mapping
) -> dict:
    complete_weighted = sum(
        row["availability"] == "complete"
        and row["loss_mode"] == "barcode_weighted_mse"
        for row in rows
    )
    return {
        "analysis_mode": "completion_only",
        "manifest_validation_status": verifier_report["validation_status"],
        "analysis_cells": len(rows),
        "cell_availability": dict(sorted(Counter(row["availability"] for row in rows).items())),
        "complete_immutable_unweighted_cells": sum(
            row["availability"] == "complete" and row["loss_mode"] == "unweighted_mse"
            for row in rows
        ),
        "complete_weighted_cells": int(complete_weighted),
        "remaining_weighted_cells": EXPECTED_WEIGHTED_CELLS - int(complete_weighted),
        "complete_oof_arms": int(readiness["complete_oof_arm"].sum()),
        "expected_oof_arms": EXPECTED_ARMS,
        "full_analysis_ready": bool(readiness["complete_oof_arm"].all()),
        "selection_performed": False,
        "audit_loader_instantiated": False,
        "audit_targets_loaded": False,
        "audit_predictions_scored": False,
        "audit_predictions_generated": False,
        "audit_metrics_computed": False,
        "audit_stratum_counts_inspected": False,
    }


def load_prediction_cell(row: Mapping) -> pd.DataFrame:
    """Load one validation export and prove its frozen fold identity."""
    if row.get("availability") != "complete":
        raise RuntimeError(f"Cell {row.get('cell_id')} is not complete.")
    path = Path(str(row["resolved_prediction_path"]))
    frame = pd.read_csv(path, sep="\t")
    required = {"construct_id", RAW_TARGET, RAW_PREDICTION}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction {path} lacks required columns {missing}")
    frame["construct_id"] = frame["construct_id"].astype(str)
    if frame["construct_id"].duplicated().any():
        raise ValueError(f"Cell {row['cell_id']} contains duplicate construct IDs.")
    if len(frame) != int(row["source_prediction_rows"]):
        raise ValueError(
            f"Cell {row['cell_id']} has {len(frame)} rows; "
            f"expected {row['source_prediction_rows']}."
        )
    observed_hash = canonical_id_hash(frame["construct_id"].tolist())
    if observed_hash != row["source_val_row_id_hash"]:
        raise ValueError(f"Cell {row['cell_id']} validation-ID hash changed.")
    frame[RAW_TARGET] = pd.to_numeric(frame[RAW_TARGET], errors="coerce")
    if not np.isfinite(frame[RAW_TARGET].to_numpy(float)).all():
        raise ValueError(f"Cell {row['cell_id']} has non-finite {RAW_TARGET} values.")
    # Non-finite predictions are a model outcome, not permission to drop rows.
    # They propagate to the frozen gates and make the arm inadmissible.
    frame[RAW_PREDICTION] = pd.to_numeric(frame[RAW_PREDICTION], errors="coerce")
    kept = ["construct_id", RAW_TARGET, RAW_PREDICTION]
    for optional in ("row_id", "n_barcodes"):
        if optional in frame:
            kept.append(optional)
    frame = frame[kept].copy()
    frame["development_fold"] = int(row["development_fold"])
    frame["cell_id"] = row["cell_id"]
    frame["resolved_run_id"] = row["resolved_run_id"]
    frame["loss_pair_id"] = row["loss_pair_id"]
    frame["rc_pair_id"] = row["rc_pair_id"]
    for field, value in arm_metadata(row).items():
        frame[field] = value
    return frame


def load_stage2_intron_baseline(
    path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Load the exact Stage 2 non-audit stratum map and OOF mean baseline."""
    baseline_path = Path(path)
    observed_sha256 = sha256_file(baseline_path)
    if observed_sha256 != EXPECTED_INTRON_BASELINE_PREDICTIONS_SHA256:
        raise ValueError(
            "Canonical Stage 2 Intron baseline/stratum-map SHA changed: "
            f"expected {EXPECTED_INTRON_BASELINE_PREDICTIONS_SHA256}, "
            f"observed {observed_sha256}."
        )
    frame = pd.read_csv(baseline_path, sep="\t")
    required = {
        "construct_id",
        "development_fold",
        RAW_TARGET,
        RAW_PREDICTION,
        SENSITIVITY_STRATUM,
        "baseline_type",
        "fit_scope",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Stage 2 Intron baseline lacks {missing}")
    frame["construct_id"] = frame["construct_id"].astype(str)
    fold_trained = frame.loc[
        frame["baseline_type"].eq("fold_trained_stratum_mean")
    ].copy()
    if len(fold_trained) != EXPECTED_INTRON_DEVELOPMENT_CONSTRUCTS:
        raise ValueError(
            "The Stage 2 fold-trained Intron baseline must contain exactly "
            f"{EXPECTED_INTRON_DEVELOPMENT_CONSTRUCTS} development rows."
        )
    if fold_trained["construct_id"].duplicated().any():
        raise ValueError("The Stage 2 fold-trained baseline duplicates a construct.")
    if set(fold_trained[SENSITIVITY_STRATUM]) != set(STRATUM_ORDER):
        raise ValueError("The Stage 2 Intron inferred-mask labels changed.")
    if set(fold_trained["fit_scope"]) != {"exact_non_audit_model_training_rows"}:
        raise ValueError("The Intron baseline is not the leakage-safe Stage 2 baseline.")
    if set(fold_trained["development_fold"].astype(int)) != set(EXPECTED_FOLDS):
        raise ValueError("The Stage 2 Intron baseline does not cover folds 0..4.")
    for column in (RAW_TARGET, RAW_PREDICTION):
        fold_trained[column] = pd.to_numeric(fold_trained[column], errors="coerce")
        if not np.isfinite(fold_trained[column].to_numpy(float)).all():
            raise ValueError(f"The Stage 2 Intron baseline has non-finite {column}.")

    mapping = fold_trained[
        ["construct_id", "development_fold", RAW_TARGET, SENSITIVITY_STRATUM]
    ].copy()
    pooled = raw_metrics(fold_trained)
    sensitivity, strata = strict_intron_sensitivity_metrics(fold_trained)
    summary = pd.DataFrame(
        [
            {
                "baseline_type": "fold_trained_stratum_mean",
                "fit_scope": "exact_non_audit_model_training_rows",
                **{f"pooled_oof_{name}": value for name, value in pooled.items()},
                **sensitivity,
            }
        ]
    )
    source = {
        "path": str(baseline_path.resolve()),
        "sha256": observed_sha256,
        "rows": str(len(fold_trained)),
        "label_status": "inferred_sequence_mask_not_verified_sublibrary",
        "audit_rows_loaded": "false",
    }
    return mapping, fold_trained, {**source, "per_stratum_rows": str(len(strata))}


def _attach_intron_strata(
    frame: pd.DataFrame, stratum_map: pd.DataFrame
) -> pd.DataFrame:
    map_columns = ["construct_id", "development_fold", RAW_TARGET, SENSITIVITY_STRATUM]
    merged = frame.merge(
        stratum_map[map_columns].rename(columns={RAW_TARGET: "stage2_baseline_target"}),
        on=["construct_id", "development_fold"],
        how="left",
        validate="one_to_one",
    )
    if merged[SENSITIVITY_STRATUM].isna().any():
        raise ValueError("An Intron OOF prediction is absent from the Stage 2 stratum map.")
    if not np.allclose(
        merged[RAW_TARGET].to_numpy(float),
        merged["stage2_baseline_target"].to_numpy(float),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Stage 3 Intron targets differ from the Stage 2 baseline map.")
    return merged.drop(columns=["stage2_baseline_target"])


def assemble_oof_arms(
    rows: Sequence[Mapping], stratum_map: pd.DataFrame
) -> dict[tuple[str, str, str, str], pd.DataFrame]:
    incomplete = [row for row in rows if row.get("availability") != "complete"]
    if incomplete:
        raise RuntimeError(
            "Stage 3 full analysis requires all 900 cells; incomplete counts: "
            + json.dumps(dict(sorted(Counter(row["availability"] for row in incomplete).items())))
        )
    grouped: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[arm_key(row)].append(row)
    if len(grouped) != EXPECTED_ARMS:
        raise ValueError(f"Expected {EXPECTED_ARMS} OOF arms; found {len(grouped)}")

    arms: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    part_reference: dict[str, pd.DataFrame] = {}
    for key, pieces in sorted(
        grouped.items(),
        key=lambda item: (
            PART_ORDER[item[0][0]], item[0][1], RC_ORDER[item[0][2]], LOSS_ORDER[item[0][3]]
        ),
    ):
        if len(pieces) != 5 or {int(row["development_fold"]) for row in pieces} != set(
            EXPECTED_FOLDS
        ):
            raise ValueError(f"Arm {key} does not contain exact folds 0..4.")
        frame = pd.concat(
            [load_prediction_cell(row) for row in sorted(pieces, key=lambda x: int(x["development_fold"]))],
            ignore_index=True,
        )
        if frame["construct_id"].duplicated().any():
            raise ValueError(f"Arm {key} predicts a development construct more than once.")
        if key[0] == "intron":
            frame = _attach_intron_strata(frame, stratum_map)
            if len(frame) != EXPECTED_INTRON_DEVELOPMENT_CONSTRUCTS:
                raise ValueError("An Intron Stage 3 arm does not contain exactly 1061 rows.")
        frame = frame.sort_values(["development_fold", "construct_id"]).reset_index(drop=True)
        reference = part_reference.get(key[0])
        if reference is None:
            part_reference[key[0]] = frame[
                ["construct_id", "development_fold", RAW_TARGET]
            ].copy()
        else:
            if not frame["construct_id"].equals(reference["construct_id"]):
                raise ValueError(f"Part {key[0]} arms do not share the exact development IDs.")
            if not frame["development_fold"].equals(reference["development_fold"]):
                raise ValueError(f"Part {key[0]} arms do not share exact fold assignments.")
            if not np.allclose(
                frame[RAW_TARGET].to_numpy(float),
                reference[RAW_TARGET].to_numpy(float),
                rtol=0,
                atol=1e-10,
            ):
                raise ValueError(f"Part {key[0]} arms do not share exact raw targets.")
        arms[key] = frame
    return arms


def _parameter_counts(rows: Sequence[Mapping]) -> dict[str, int]:
    by_config: dict[str, Mapping] = {}
    for row in rows:
        by_config.setdefault(str(row["base_config_id"]), row)
    identity_cache: dict[str, int] = {}
    counts = {}
    for config_id, row in by_config.items():
        identity_key = json.dumps(
            {"architecture": row["architecture"], "base_identity": row["base_identity"]},
            sort_keys=True,
        )
        if identity_key not in identity_cache:
            identity_cache[identity_key] = model_parameter_count(
                row["architecture"], row["base_identity"]
            )
        counts[config_id] = identity_cache[identity_key]
    return counts


def _equal_stratum_weight_diagnostics(frame: pd.DataFrame) -> dict[str, float]:
    counts = frame[SENSITIVITY_STRATUM].value_counts()
    if set(counts.index) != set(STRATUM_ORDER):
        raise ValueError("Intron equal-stratum weights require all three strata.")
    weights_by_stratum = {
        stratum: len(frame) / (len(STRATUM_ORDER) * int(count))
        for stratum, count in counts.items()
    }
    weights = frame[SENSITIVITY_STRATUM].map(weights_by_stratum).to_numpy(float)
    return {
        "equal_stratum_weight_min": float(weights.min()),
        "equal_stratum_weight_max": float(weights.max()),
        "equal_stratum_weight_ess": float(weights.sum() ** 2 / np.sum(weights**2)),
        **{f"{stratum}_n": int(counts[stratum]) for stratum in STRATUM_ORDER},
    }


def score_arms(
    arms: Mapping[tuple, pd.DataFrame], parameter_counts: Mapping[str, int]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    fold_rows = []
    intron_stratum_rows = []
    intron_equal_rows = []
    for key, frame in sorted(
        arms.items(),
        key=lambda item: (
            PART_ORDER[item[0][0]], item[0][1], RC_ORDER[item[0][2]], LOSS_ORDER[item[0][3]]
        ),
    ):
        metadata = arm_metadata(frame.iloc[0])
        pooled = raw_metrics(frame)
        row = {
            **metadata,
            "model_parameter_count": int(parameter_counts[key[1]]),
            "primary_metric_name": "pooled_five_fold_oof_pearson",
            **{f"pooled_oof_{name}": value for name, value in pooled.items()},
            **calibration_with_reason(
                frame[RAW_TARGET].to_numpy(float),
                frame[RAW_PREDICTION].to_numpy(float),
            ),
        }
        arm_fold_rows = []
        for fold in EXPECTED_FOLDS:
            subset = frame.loc[frame["development_fold"].astype(int).eq(fold)].copy()
            fold_metrics = raw_metrics(subset)
            fold_row = {
                **metadata,
                "development_fold": fold,
                **{f"fold_{name}": value for name, value in fold_metrics.items()},
            }
            if key[0] == "intron":
                sensitivity, _ = strict_intron_sensitivity_metrics(subset)
                fold_row.update(
                    fold_within_stratum_centered_pearson=sensitivity[
                        "within_stratum_centered_pearson"
                    ],
                    fold_macro_stratum_pearson=sensitivity["macro_stratum_pearson"],
                    fold_minimum_stratum_pearson=sensitivity[
                        "minimum_stratum_pearson"
                    ],
                )
            fold_rows.append(fold_row)
            arm_fold_rows.append(fold_row)
        fold_values = np.asarray([entry["fold_pearson"] for entry in arm_fold_rows], dtype=float)
        row.update(
            finite_fold_pearson_count=int(np.isfinite(fold_values).sum()),
            fold_pearson_mean=(float(np.mean(fold_values)) if np.isfinite(fold_values).all() else math.nan),
            fold_pearson_sd=(float(np.std(fold_values, ddof=1)) if np.isfinite(fold_values).all() else math.nan),
            minimum_fold_pearson=(float(np.min(fold_values)) if np.isfinite(fold_values).all() else math.nan),
        )
        if key[0] == "intron":
            sensitivity, strata = strict_intron_sensitivity_metrics(frame, metadata)
            estimands = intron_estimands(frame)
            weights = _equal_stratum_weight_diagnostics(frame)
            row.update(sensitivity)
            row.update(
                equal_stratum_pooled_pearson=estimands["equal_stratum_pooled_pearson"],
                equal_stratum_within_centered_pearson=estimands[
                    "equal_stratum_within_centered_pearson"
                ],
                **weights,
            )
            intron_stratum_rows.extend(strata)
            intron_equal_rows.append(
                {
                    **metadata,
                    "sensitivity_label_status": "inferred_sequence_mask_not_verified_sublibrary",
                    "natural_pooled_pearson": estimands["natural_pooled_pearson"],
                    "equal_stratum_pooled_pearson": estimands[
                        "equal_stratum_pooled_pearson"
                    ],
                    "within_stratum_centered_pearson": estimands[
                        "within_stratum_centered_pearson"
                    ],
                    "equal_stratum_within_centered_pearson": estimands[
                        "equal_stratum_within_centered_pearson"
                    ],
                    "macro_stratum_pearson": estimands["macro_stratum_pearson"],
                    "minimum_stratum_pearson": estimands["minimum_stratum_pearson"],
                    **weights,
                }
            )
        metric_rows.append(row)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(fold_rows),
        pd.DataFrame(intron_stratum_rows),
        pd.DataFrame(intron_equal_rows),
    )


def _aligned_pair(
    baseline: pd.DataFrame, intervention: pd.DataFrame, label: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = baseline.sort_values(["development_fold", "construct_id"]).reset_index(drop=True)
    right = intervention.sort_values(["development_fold", "construct_id"]).reset_index(drop=True)
    if not left["construct_id"].equals(right["construct_id"]):
        raise ValueError(f"{label} mates contain different construct IDs.")
    if not left["development_fold"].equals(right["development_fold"]):
        raise ValueError(f"{label} mates contain different fold assignments.")
    if not np.allclose(
        left[RAW_TARGET].to_numpy(float),
        right[RAW_TARGET].to_numpy(float),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(f"{label} mates contain different raw targets.")
    if SENSITIVITY_STRATUM in left or SENSITIVITY_STRATUM in right:
        if SENSITIVITY_STRATUM not in left or SENSITIVITY_STRATUM not in right:
            raise ValueError(f"{label} Intron mates have inconsistent stratum labels.")
        if not left[SENSITIVITY_STRATUM].equals(right[SENSITIVITY_STRATUM]):
            raise ValueError(f"{label} Intron mates have different stratum labels.")
    return left, right


def paired_gate(
    *,
    baseline: pd.DataFrame,
    intervention: pd.DataFrame,
    part: str,
    base_config_id: str,
    gate_kind: str,
    margin: Mapping,
    pair_id_column: str,
) -> tuple[dict, list[dict]]:
    """Apply one frozen intervention gate without dropping non-finite deltas."""
    baseline, intervention = _aligned_pair(baseline, intervention, gate_kind)
    baseline_pooled = raw_metrics(baseline)
    intervention_pooled = raw_metrics(intervention)
    fold_rows = []
    pair_ids = []
    for fold in EXPECTED_FOLDS:
        left = baseline.loc[baseline["development_fold"].astype(int).eq(fold)].copy()
        right = intervention.loc[intervention["development_fold"].astype(int).eq(fold)].copy()
        left, right = _aligned_pair(left, right, f"{gate_kind} fold {fold}")
        left_ids = set(left[pair_id_column].astype(str))
        right_ids = set(right[pair_id_column].astype(str))
        if len(left_ids) != 1 or left_ids != right_ids or "" in left_ids:
            raise ValueError(
                f"{gate_kind} fold {fold} does not preserve exact {pair_id_column}."
            )
        pair_id = next(iter(left_ids))
        pair_ids.append(pair_id)
        left_metrics = raw_metrics(left)
        right_metrics = raw_metrics(right)
        row = {
            "gate_kind": gate_kind,
            "part_slug": part,
            "base_config_id": base_config_id,
            "development_fold": fold,
            pair_id_column: pair_id,
            "baseline_rc_mode": str(left["rc_mode"].iloc[0]),
            "intervention_rc_mode": str(right["rc_mode"].iloc[0]),
            "baseline_loss_mode": str(left["loss_mode"].iloc[0]),
            "intervention_loss_mode": str(right["loss_mode"].iloc[0]),
            "baseline_fold_pearson": left_metrics["pearson"],
            "intervention_fold_pearson": right_metrics["pearson"],
            "fold_pearson_delta": right_metrics["pearson"] - left_metrics["pearson"],
        }
        if part == "intron":
            left_sensitivity, _ = strict_intron_sensitivity_metrics(left)
            right_sensitivity, _ = strict_intron_sensitivity_metrics(right)
            row.update(
                baseline_fold_within_stratum_centered_pearson=left_sensitivity[
                    "within_stratum_centered_pearson"
                ],
                intervention_fold_within_stratum_centered_pearson=right_sensitivity[
                    "within_stratum_centered_pearson"
                ],
                fold_within_stratum_centered_pearson_delta=(
                    right_sensitivity["within_stratum_centered_pearson"]
                    - left_sensitivity["within_stratum_centered_pearson"]
                ),
            )
        fold_rows.append(row)

    pearson_deltas = np.asarray(
        [row["fold_pearson_delta"] for row in fold_rows], dtype=float
    )
    all_five_fold_deltas_finite = bool(
        len(pearson_deltas) == 5 and np.isfinite(pearson_deltas).all()
    )
    mean_delta = (
        float(np.mean(pearson_deltas)) if all_five_fold_deltas_finite else math.nan
    )
    positive_count = int(np.sum(pearson_deltas > 0)) if all_five_fold_deltas_finite else 0
    pearson_mean_pass = bool(all_five_fold_deltas_finite and mean_delta >= 0.005)
    pearson_positive_pass = bool(all_five_fold_deltas_finite and positive_count >= 4)

    rmse_increase = intervention_pooled["rmse"] - baseline_pooled["rmse"]
    cod_decrease = baseline_pooled["cod_r2"] - intervention_pooled["cod_r2"]
    guardrails_finite = bool(np.isfinite([rmse_increase, cod_decrease]).all())
    rmse_pass = bool(
        guardrails_finite
        and rmse_increase
        <= float(margin["allowed_pooled_rmse_increase"])
        + float(margin["numeric_epsilon"])
    )
    cod_pass = bool(
        guardrails_finite
        and cod_decrease
        <= float(margin["allowed_pooled_cod_r2_decrease"])
        + float(margin["numeric_epsilon"])
    )

    intron_centered_finite = True
    intron_centered_mean = math.nan
    intron_centered_negative_count = 0
    intron_centered_pass = True
    if part == "intron":
        centered_deltas = np.asarray(
            [row["fold_within_stratum_centered_pearson_delta"] for row in fold_rows],
            dtype=float,
        )
        intron_centered_finite = bool(
            len(centered_deltas) == 5 and np.isfinite(centered_deltas).all()
        )
        intron_centered_mean = (
            float(np.mean(centered_deltas)) if intron_centered_finite else math.nan
        )
        intron_centered_negative_count = (
            int(np.sum(centered_deltas < 0)) if intron_centered_finite else 5
        )
        intron_centered_pass = bool(
            intron_centered_finite
            and intron_centered_mean >= 0.0
            and intron_centered_negative_count <= 2
        )

    first = baseline.iloc[0]
    summary = {
        "gate_kind": gate_kind,
        "part_slug": part,
        "portfolio_rank": int(first["portfolio_rank"]),
        "portfolio_role": first["portfolio_role"],
        "base_config_id": base_config_id,
        "architecture": first["architecture"],
        "training_regime": first["training_regime"],
        "baseline_rc_mode": first["rc_mode"],
        "intervention_rc_mode": intervention["rc_mode"].iloc[0],
        "baseline_loss_mode": first["loss_mode"],
        "intervention_loss_mode": intervention["loss_mode"].iloc[0],
        "fold_pair_ids_json": json.dumps(pair_ids),
        "finite_fold_delta_count": int(np.isfinite(pearson_deltas).sum()),
        "all_five_fold_pearson_deltas_finite": all_five_fold_deltas_finite,
        "mean_fold_pearson_delta": mean_delta,
        "positive_fold_pearson_delta_count": positive_count,
        "pearson_mean_delta_ge_0p005": pearson_mean_pass,
        "pearson_positive_fold_count_ge_4": pearson_positive_pass,
        "baseline_pooled_oof_pearson": baseline_pooled["pearson"],
        "intervention_pooled_oof_pearson": intervention_pooled["pearson"],
        "pooled_oof_pearson_delta": (
            intervention_pooled["pearson"] - baseline_pooled["pearson"]
        ),
        "baseline_pooled_oof_rmse": baseline_pooled["rmse"],
        "intervention_pooled_oof_rmse": intervention_pooled["rmse"],
        "pooled_oof_rmse_increase": rmse_increase,
        "allowed_pooled_rmse_increase": float(
            margin["allowed_pooled_rmse_increase"]
        ),
        "rmse_guardrail_pass": rmse_pass,
        "baseline_pooled_oof_cod_r2": baseline_pooled["cod_r2"],
        "intervention_pooled_oof_cod_r2": intervention_pooled["cod_r2"],
        "pooled_oof_cod_r2_decrease": cod_decrease,
        "allowed_pooled_cod_r2_decrease": float(
            margin["allowed_pooled_cod_r2_decrease"]
        ),
        "cod_r2_guardrail_pass": cod_pass,
        "guardrail_values_finite": guardrails_finite,
        "intron_all_five_centered_deltas_finite": intron_centered_finite,
        "intron_mean_fold_within_stratum_centered_pearson_delta": intron_centered_mean,
        "intron_negative_centered_delta_fold_count": intron_centered_negative_count,
        "intron_centered_gate_pass": intron_centered_pass,
    }
    summary["gate_pass"] = bool(
        pearson_mean_pass
        and pearson_positive_pass
        and rmse_pass
        and cod_pass
        and intron_centered_pass
    )
    return summary, fold_rows


def score_intervention_gates(
    arms: Mapping[tuple, pd.DataFrame], margins: Mapping[str, Mapping]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    loss_summaries = []
    loss_folds = []
    rc_summaries = []
    rc_folds = []
    configs = sorted(
        {(key[0], key[1]) for key in arms}, key=lambda value: (PART_ORDER[value[0]], value[1])
    )
    for part, config_id in configs:
        rc_modes = ("off",) if part == "utr3" else ("off", "on")
        for rc_mode in rc_modes:
            summary, folds = paired_gate(
                baseline=arms[(part, config_id, rc_mode, "unweighted_mse")],
                intervention=arms[(part, config_id, rc_mode, "barcode_weighted_mse")],
                part=part,
                base_config_id=config_id,
                gate_kind="weighted_minus_unweighted",
                margin=margins[part],
                pair_id_column="loss_pair_id",
            )
            loss_summaries.append(summary)
            loss_folds.extend(folds)
        if part == "utr3":
            continue
        for loss_mode in ("unweighted_mse", "barcode_weighted_mse"):
            summary, folds = paired_gate(
                baseline=arms[(part, config_id, "off", loss_mode)],
                intervention=arms[(part, config_id, "on", loss_mode)],
                part=part,
                base_config_id=config_id,
                gate_kind="rc_on_minus_off",
                margin=margins[part],
                pair_id_column="rc_pair_id",
            )
            rc_summaries.append(summary)
            rc_folds.extend(folds)
    loss_frame = pd.DataFrame(loss_summaries)
    loss_fold_frame = pd.DataFrame(loss_folds)
    rc_frame = pd.DataFrame(rc_summaries)
    rc_fold_frame = pd.DataFrame(rc_folds)
    expected_counts = (
        (len(loss_frame), EXPECTED_LOSS_ARM_PAIRS, "loss arm pairs"),
        (len(loss_fold_frame), EXPECTED_LOSS_ARM_PAIRS * 5, "loss fold pairs"),
        (len(rc_frame), EXPECTED_RC_ARM_PAIRS, "RC arm pairs"),
        (len(rc_fold_frame), EXPECTED_RC_ARM_PAIRS * 5, "RC fold pairs"),
    )
    for observed, expected, label in expected_counts:
        if observed != expected:
            raise ValueError(f"Expected {expected} {label}; found {observed}")
    return loss_frame, loss_fold_frame, rc_frame, rc_fold_frame


def score_factorial_differences(
    arms: Mapping[tuple, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe the 40 complete RC-by-loss difference-in-differences."""
    summaries = []
    fold_rows = []
    configs = sorted(
        {(key[0], key[1]) for key in arms if key[0] != "utr3"},
        key=lambda value: (PART_ORDER[value[0]], value[1]),
    )
    for part, config_id in configs:
        cells = {
            (rc_mode, loss_mode): arms[(part, config_id, rc_mode, loss_mode)]
            for rc_mode in ("off", "on")
            for loss_mode in ("unweighted_mse", "barcode_weighted_mse")
        }
        reference = cells[("off", "unweighted_mse")]
        for label, frame in cells.items():
            _aligned_pair(reference, frame, f"factorial {part}/{config_id}/{label}")
        pooled = {label: raw_metrics(frame) for label, frame in cells.items()}
        first = reference.iloc[0]
        row = {
            "factorial_id": "factorial_"
            + hashlib.sha256(f"{part}|{config_id}".encode("utf-8")).hexdigest()[:20],
            "analysis_status": "descriptive_not_a_gate",
            "part_slug": part,
            "portfolio_rank": int(first["portfolio_rank"]),
            "portfolio_role": first["portfolio_role"],
            "base_config_id": config_id,
            "architecture": first["architecture"],
            "training_regime": first["training_regime"],
        }
        for metric in ("pearson", "rmse", "cod_r2"):
            off_unweighted = pooled[("off", "unweighted_mse")][metric]
            off_weighted = pooled[("off", "barcode_weighted_mse")][metric]
            on_unweighted = pooled[("on", "unweighted_mse")][metric]
            on_weighted = pooled[("on", "barcode_weighted_mse")][metric]
            loss_effect_off = off_weighted - off_unweighted
            loss_effect_on = on_weighted - on_unweighted
            rc_effect_unweighted = on_unweighted - off_unweighted
            rc_effect_weighted = on_weighted - off_weighted
            row.update(
                {
                    f"rc_off_unweighted_pooled_oof_{metric}": off_unweighted,
                    f"rc_off_weighted_pooled_oof_{metric}": off_weighted,
                    f"rc_on_unweighted_pooled_oof_{metric}": on_unweighted,
                    f"rc_on_weighted_pooled_oof_{metric}": on_weighted,
                    f"loss_effect_at_rc_off_{metric}": loss_effect_off,
                    f"loss_effect_at_rc_on_{metric}": loss_effect_on,
                    f"rc_effect_unweighted_{metric}": rc_effect_unweighted,
                    f"rc_effect_weighted_{metric}": rc_effect_weighted,
                    f"difference_in_differences_{metric}": loss_effect_on
                    - loss_effect_off,
                }
            )
        if part == "intron":
            centered = {
                label: strict_intron_sensitivity_metrics(frame)[0][
                    "within_stratum_centered_pearson"
                ]
                for label, frame in cells.items()
            }
            off_unweighted = centered[("off", "unweighted_mse")]
            off_weighted = centered[("off", "barcode_weighted_mse")]
            on_unweighted = centered[("on", "unweighted_mse")]
            on_weighted = centered[("on", "barcode_weighted_mse")]
            row.update(
                rc_off_unweighted_within_stratum_centered_pearson=off_unweighted,
                rc_off_weighted_within_stratum_centered_pearson=off_weighted,
                rc_on_unweighted_within_stratum_centered_pearson=on_unweighted,
                rc_on_weighted_within_stratum_centered_pearson=on_weighted,
                loss_effect_at_rc_off_within_stratum_centered_pearson=(
                    off_weighted - off_unweighted
                ),
                loss_effect_at_rc_on_within_stratum_centered_pearson=(
                    on_weighted - on_unweighted
                ),
                difference_in_differences_within_stratum_centered_pearson=(
                    (on_weighted - on_unweighted)
                    - (off_weighted - off_unweighted)
                ),
            )

        config_fold_rows = []
        for fold in EXPECTED_FOLDS:
            fold_metrics = {}
            fold_centered = {}
            for label, frame in cells.items():
                subset = frame.loc[frame["development_fold"].astype(int).eq(fold)].copy()
                fold_metrics[label] = raw_metrics(subset)
                if part == "intron":
                    fold_centered[label] = strict_intron_sensitivity_metrics(subset)[0][
                        "within_stratum_centered_pearson"
                    ]
            loss_effect_off = (
                fold_metrics[("off", "barcode_weighted_mse")]["pearson"]
                - fold_metrics[("off", "unweighted_mse")]["pearson"]
            )
            loss_effect_on = (
                fold_metrics[("on", "barcode_weighted_mse")]["pearson"]
                - fold_metrics[("on", "unweighted_mse")]["pearson"]
            )
            fold_row = {
                "factorial_id": row["factorial_id"],
                "part_slug": part,
                "base_config_id": config_id,
                "development_fold": fold,
                "loss_effect_at_rc_off_fold_pearson": loss_effect_off,
                "loss_effect_at_rc_on_fold_pearson": loss_effect_on,
                "difference_in_differences_fold_pearson": loss_effect_on
                - loss_effect_off,
            }
            if part == "intron":
                centered_off = (
                    fold_centered[("off", "barcode_weighted_mse")]
                    - fold_centered[("off", "unweighted_mse")]
                )
                centered_on = (
                    fold_centered[("on", "barcode_weighted_mse")]
                    - fold_centered[("on", "unweighted_mse")]
                )
                fold_row.update(
                    loss_effect_at_rc_off_fold_within_stratum_centered_pearson=centered_off,
                    loss_effect_at_rc_on_fold_within_stratum_centered_pearson=centered_on,
                    difference_in_differences_fold_within_stratum_centered_pearson=(
                        centered_on - centered_off
                    ),
                )
            fold_rows.append(fold_row)
            config_fold_rows.append(fold_row)
        fold_did = np.asarray(
            [entry["difference_in_differences_fold_pearson"] for entry in config_fold_rows],
            dtype=float,
        )
        row.update(
            all_five_fold_pearson_interactions_finite=bool(np.isfinite(fold_did).all()),
            mean_fold_difference_in_differences_pearson=(
                float(np.mean(fold_did)) if np.isfinite(fold_did).all() else math.nan
            ),
        )
        if part == "intron":
            centered_did = np.asarray(
                [
                    entry[
                        "difference_in_differences_fold_within_stratum_centered_pearson"
                    ]
                    for entry in config_fold_rows
                ],
                dtype=float,
            )
            row.update(
                all_five_fold_centered_interactions_finite=bool(
                    np.isfinite(centered_did).all()
                ),
                mean_fold_difference_in_differences_within_stratum_centered_pearson=(
                    float(np.mean(centered_did))
                    if np.isfinite(centered_did).all()
                    else math.nan
                ),
            )
        summaries.append(row)
    summary_frame = pd.DataFrame(summaries)
    fold_frame = pd.DataFrame(fold_rows)
    if len(summary_frame) != EXPECTED_FACTORIALS or len(fold_frame) != 5 * EXPECTED_FACTORIALS:
        raise ValueError(
            "Expected 40 factorial summaries and 200 factorial fold rows; found "
            f"{len(summary_frame)} and {len(fold_frame)}."
        )
    return summary_frame, fold_frame


def apply_admissibility(
    metrics: pd.DataFrame,
    loss_gates: pd.DataFrame,
    rc_gates: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the frozen baseline/loss/RC admissibility graph to all 180 arms."""
    loss_lookup = {
        (row.part_slug, row.base_config_id, row.baseline_rc_mode): bool(row.gate_pass)
        for row in loss_gates.itertuples(index=False)
    }
    rc_lookup = {
        (row.part_slug, row.base_config_id, row.baseline_loss_mode): bool(row.gate_pass)
        for row in rc_gates.itertuples(index=False)
    }
    records = []
    for row in metrics.to_dict(orient="records"):
        part = row["part_slug"]
        config_id = row["base_config_id"]
        rc_mode = row["rc_mode"]
        loss_mode = row["loss_mode"]
        loss_required = loss_mode == "barcode_weighted_mse"
        rc_required = rc_mode == "on"
        loss_pass = (
            loss_lookup[(part, config_id, rc_mode)] if loss_required else True
        )
        rc_pass = rc_lookup[(part, config_id, loss_mode)] if rc_required else True
        complete_baseline = rc_mode == "off" and loss_mode == "unweighted_mse"
        intervention_graph_pass = bool(
            complete_baseline or (loss_pass and rc_pass)
        )
        required_selection_metrics = [
            row["pooled_oof_pearson"],
            row["minimum_fold_pearson"],
            row["pooled_oof_rmse"],
            row["pooled_oof_cod_r2"],
        ]
        if part == "intron":
            required_selection_metrics.extend(
                [
                    row["minimum_stratum_pearson"],
                    row["within_stratum_centered_pearson"],
                ]
            )
        selection_metrics_finite = bool(
            np.isfinite(np.asarray(required_selection_metrics, dtype=float)).all()
        )
        admissible = bool(intervention_graph_pass and selection_metrics_finite)
        selection_eligible = admissible
        if not selection_metrics_finite:
            reason = "nonfinite_required_selection_metric"
        elif complete_baseline:
            reason = "complete_rc_off_unweighted_baseline"
        elif loss_required and rc_required:
            reason = (
                "loss_and_rc_gates_pass"
                if intervention_graph_pass
                else "loss_or_rc_gate_failed"
            )
        elif loss_required:
            reason = (
                "loss_gate_pass" if intervention_graph_pass else "loss_gate_failed"
            )
        else:
            reason = "rc_gate_pass" if intervention_graph_pass else "rc_gate_failed"
        records.append(
            {
                **row,
                "complete_oof_arm": True,
                "loss_gate_required": loss_required,
                "loss_gate_pass": bool(loss_pass),
                "rc_gate_required": rc_required,
                "rc_gate_pass": bool(rc_pass),
                "admissible": admissible,
                "admissibility_reason": reason,
                "required_selection_metrics_finite": selection_metrics_finite,
                "selection_eligible": selection_eligible,
                "selection_ineligibility_reason": (
                    ""
                    if selection_eligible
                    else (
                        "intervention_gate_failed"
                        if not intervention_graph_pass
                        else "nonfinite_required_selection_metric"
                    )
                ),
            }
        )
    frame = pd.DataFrame(records)
    if len(frame) != EXPECTED_ARMS:
        raise ValueError("Admissibility did not return all 180 OOF arms.")
    baseline_row_count = int(
        (
            frame["rc_mode"].eq("off")
            & frame["loss_mode"].eq("unweighted_mse")
        ).sum()
    )
    if baseline_row_count != 50:
        raise ValueError(
            f"Expected 50 RC-off unweighted baseline arms; found {baseline_row_count}"
        )
    return frame


def pearson_array(target: np.ndarray, prediction: np.ndarray) -> float:
    if len(target) < 2 or np.ptp(target) == 0 or np.ptp(prediction) == 0:
        return math.nan
    return float(np.corrcoef(target, prediction)[0, 1])


def bootstrap_best_arm(
    frame: pd.DataFrame,
    *,
    part: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[np.ndarray, dict]:
    """Reinitialize PCG64 per part and resample rows within each fold."""
    ordered = frame.sort_values(["development_fold", "construct_id"]).reset_index(drop=True)
    if set(ordered["development_fold"].astype(int)) != set(EXPECTED_FOLDS):
        raise ValueError("Bootstrap arm does not cover folds 0..4.")
    rng = np.random.default_rng(seed)
    strata = [
        ordered.index[ordered["development_fold"].astype(int).eq(fold)].to_numpy()
        for fold in EXPECTED_FOLDS
    ]
    target = ordered[RAW_TARGET].to_numpy(float)
    prediction = ordered[RAW_PREDICTION].to_numpy(float)
    values = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = np.concatenate(
            [indices[rng.integers(0, len(indices), size=len(indices))] for indices in strata]
        )
        values[index] = pearson_array(target[sample], prediction[sample])
    if not np.isfinite(values).all():
        raise ValueError(f"{part} best-arm bootstrap produced a non-finite Pearson.")
    metadata = {
        "part_slug": part,
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "bootstrap_rng": "numpy.random.default_rng",
        "bootstrap_bit_generator": "PCG64",
        "bootstrap_within_fold_sort": "development_fold_then_construct_id_ascending",
        "bootstrap_standard_error_ddof": 1,
        "bootstrap_design": "row resampling with replacement separately within each development fold",
        "rng_reinitialized_for_part": True,
    }
    return values, metadata


def _performance_fields(part: str) -> list[str]:
    fields = ["minimum_fold_pearson"]
    if part == "intron":
        fields.extend(
            ["minimum_stratum_pearson", "within_stratum_centered_pearson"]
        )
    fields.extend(["pooled_oof_rmse", "pooled_oof_cod_r2"])
    return fields


def _performance_sort(part: str) -> tuple[list[str], list[bool]]:
    fields = _performance_fields(part)
    ascending = [False]
    if part == "intron":
        ascending.extend([False, False])
    ascending.extend([True, False])
    return fields, ascending


def _scope_rank(value: object) -> int:
    return {"branched_only": 0, "conv3_plus": 1, "full": 2}.get(str(value), 99)


def _order_exact_tie_block(block: pd.DataFrame, part: str) -> pd.DataFrame:
    records = block.to_dict(orient="records")
    all_enhancer_transfer = part == "enhancer" and all(
        row["training_regime"] == "transfer" for row in records
    )

    def compare(left: Mapping, right: Mapping) -> int:
        def numeric(value_left: float, value_right: float) -> int:
            return (value_left > value_right) - (value_left < value_right)

        if all_enhancer_transfer:
            difference = numeric(
                _scope_rank(left["unfreeze_scope"]), _scope_rank(right["unfreeze_scope"])
            )
            if difference:
                return difference
        else:
            difference = numeric(
                int(left["model_parameter_count"]), int(right["model_parameter_count"])
            )
            if difference:
                return difference
            if (
                part == "enhancer"
                and left["training_regime"] == "transfer"
                and right["training_regime"] == "transfer"
            ):
                difference = numeric(
                    _scope_rank(left["unfreeze_scope"]),
                    _scope_rank(right["unfreeze_scope"]),
                )
                if difference:
                    return difference
        for field, ordering in (("rc_mode", RC_ORDER), ("loss_mode", LOSS_ORDER)):
            difference = numeric(ordering[left[field]], ordering[right[field]])
            if difference:
                return difference
        return (left["base_config_id"] > right["base_config_id"]) - (
            left["base_config_id"] < right["base_config_id"]
        )

    ordered = sorted(records, key=functools.cmp_to_key(compare))
    return pd.DataFrame(ordered, columns=block.columns)


def order_one_se_band(frame: pd.DataFrame, part: str) -> pd.DataFrame:
    """Apply performance order, then exact-tie-only complexity preferences."""
    fields, ascending = _performance_sort(part)
    ordered = frame.sort_values(fields, ascending=ascending, kind="mergesort")
    pieces = []
    for _values, block in ordered.groupby(fields, sort=False, dropna=False):
        pieces.append(_order_exact_tie_block(block, part))
    result = pd.concat(pieces, ignore_index=True)
    result["one_se_tiebreak_rank"] = np.arange(1, len(result) + 1)
    return result


def select_parts(
    admissibility: pd.DataFrame,
    arms: Mapping[tuple, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Apply the exact part-specific one-SE rule to admissible arms only."""
    annotated = admissibility.copy()
    annotated["numerical_best"] = False
    annotated["one_se_threshold"] = math.nan
    annotated["within_one_se"] = False
    annotated["one_se_tiebreak_rank"] = pd.Series([pd.NA] * len(annotated), dtype="Int64")
    annotated["selected_part_arm"] = False
    bootstrap_rows = []
    bootstrap_summary = []
    selections = []
    for part in sorted(PART_ORDER, key=PART_ORDER.get):
        part_index = annotated.index[annotated["part_slug"].eq(part)]
        part_frame = annotated.loc[part_index].copy()
        candidates = part_frame.loc[part_frame["selection_eligible"]].copy()
        if candidates.empty:
            raise RuntimeError(
                f"No admissible {part} arm has all required finite selection metrics."
            )
        best_value = float(candidates["pooled_oof_pearson"].max())
        numerical_ties = candidates.loc[
            candidates["pooled_oof_pearson"].eq(best_value)
        ].copy()
        # Exact numerical-best ties use the same frozen part-specific ordering.
        best = order_one_se_band(numerical_ties, part).iloc[0]
        best_key = (
            part,
            best["base_config_id"],
            best["rc_mode"],
            best["loss_mode"],
        )
        samples, design = bootstrap_best_arm(arms[best_key], part=part)
        standard_error = float(np.std(samples, ddof=1))
        threshold = float(best["pooled_oof_pearson"] - standard_error)
        # Deliberately start from candidates: non-admissible arms never enter the band.
        band = candidates.loc[candidates["pooled_oof_pearson"].ge(threshold)].copy()
        ordered_band = order_one_se_band(band, part)
        selected = ordered_band.iloc[0]

        best_mask = (
            annotated["part_slug"].eq(part)
            & annotated["base_config_id"].eq(best["base_config_id"])
            & annotated["rc_mode"].eq(best["rc_mode"])
            & annotated["loss_mode"].eq(best["loss_mode"])
        )
        annotated.loc[best_mask, "numerical_best"] = True
        annotated.loc[part_index, "one_se_threshold"] = threshold
        for ordered_row in ordered_band.itertuples(index=False):
            mask = (
                annotated["part_slug"].eq(part)
                & annotated["base_config_id"].eq(ordered_row.base_config_id)
                & annotated["rc_mode"].eq(ordered_row.rc_mode)
                & annotated["loss_mode"].eq(ordered_row.loss_mode)
            )
            annotated.loc[mask, "within_one_se"] = True
            annotated.loc[mask, "one_se_tiebreak_rank"] = int(
                ordered_row.one_se_tiebreak_rank
            )
        selected_mask = (
            annotated["part_slug"].eq(part)
            & annotated["base_config_id"].eq(selected["base_config_id"])
            & annotated["rc_mode"].eq(selected["rc_mode"])
            & annotated["loss_mode"].eq(selected["loss_mode"])
        )
        annotated.loc[selected_mask, "selected_part_arm"] = True

        bootstrap_rows.extend(
            {
                "part_slug": part,
                "bootstrap_resample": index + 1,
                "best_arm_pearson": value,
            }
            for index, value in enumerate(samples)
        )
        bootstrap_summary.append(
            {
                **design,
                "numerical_best_base_config_id": best["base_config_id"],
                "numerical_best_rc_mode": best["rc_mode"],
                "numerical_best_loss_mode": best["loss_mode"],
                "numerical_best_pooled_oof_pearson": float(
                    best["pooled_oof_pearson"]
                ),
                "bootstrap_mean": float(np.mean(samples)),
                "bootstrap_standard_error": standard_error,
                "bootstrap_ci95_low": float(np.quantile(samples, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(samples, 0.975)),
                "one_se_threshold": threshold,
                "admissible_arm_count": int(part_frame["admissible"].sum()),
                "selection_eligible_arm_count": int(len(candidates)),
                "one_se_arm_count": int(len(ordered_band)),
                "selected_base_config_id": selected["base_config_id"],
                "selected_rc_mode": selected["rc_mode"],
                "selected_loss_mode": selected["loss_mode"],
            }
        )
        selections.append(
            {
                "part_slug": part,
                "base_config_id": selected["base_config_id"],
                "portfolio_rank": int(selected["portfolio_rank"]),
                "portfolio_role": selected["portfolio_role"],
                "architecture": selected["architecture"],
                "training_regime": selected["training_regime"],
                "unfreeze_scope": selected["unfreeze_scope"],
                "rc_mode": selected["rc_mode"],
                "loss_mode": selected["loss_mode"],
                "pooled_oof_pearson": float(selected["pooled_oof_pearson"]),
                "minimum_fold_pearson": float(selected["minimum_fold_pearson"]),
                "pooled_oof_rmse": float(selected["pooled_oof_rmse"]),
                "pooled_oof_cod_r2": float(selected["pooled_oof_cod_r2"]),
                "model_parameter_count": int(selected["model_parameter_count"]),
                "one_se_threshold": threshold,
                "bootstrap_standard_error": standard_error,
                "selection_status": "development_only_pre_audit",
            }
        )
    if int(annotated["selected_part_arm"].sum()) != 5:
        raise RuntimeError("Part-specific selection did not produce exactly five arms.")
    return (
        annotated,
        pd.DataFrame(bootstrap_summary),
        pd.DataFrame(bootstrap_rows),
        selections,
    )


def flatten_oof_arms(arms: Mapping[tuple, pd.DataFrame]) -> pd.DataFrame:
    pieces = []
    for key, frame in sorted(
        arms.items(),
        key=lambda item: (
            PART_ORDER[item[0][0]], item[0][1], RC_ORDER[item[0][2]], LOSS_ORDER[item[0][3]]
        ),
    ):
        keep = [
            *ARM_KEYS,
            "portfolio_rank",
            "portfolio_role",
            "architecture",
            "training_regime",
            "development_fold",
            "cell_id",
            "resolved_run_id",
            "construct_id",
            RAW_TARGET,
            RAW_PREDICTION,
        ]
        for optional in ("n_barcodes", SENSITIVITY_STRATUM):
            if optional in frame:
                keep.append(optional)
        pieces.append(frame[keep].copy())
    return pd.concat(pieces, ignore_index=True)


def write_artifact_index(
    output_dir: Path,
    paths: Sequence[Path],
    *,
    analysis_mode: str,
) -> Path:
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        unique.append(resolved)
    payload = {
        "schema_version": "lib1_dedup_stage3_analysis_artifacts_v1",
        "analysis_mode": analysis_mode,
        "artifacts": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(unique)
        ],
        "audit_loader_instantiated": False,
        "audit_targets_loaded": False,
        "audit_predictions_scored": False,
        "audit_predictions_generated": False,
        "audit_metrics_computed": False,
        "audit_stratum_counts_inspected": False,
    }
    index_path = output_dir / "stage3_output_artifact_index.json"
    write_json(index_path, payload)
    return index_path


def run_analysis(args: argparse.Namespace) -> dict:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    verifier_report = validate_frozen_manifest(args)
    validation_path = root / "stage3_manifest_validation.json"
    write_json(validation_path, verifier_report)
    written.append(validation_path)

    manifest_rows = read_jsonl(args.analysis_manifest)
    resolved = resolve_cells(manifest_rows, args.registry)
    cells = completion_table(resolved)
    readiness = arm_readiness(resolved)
    cells_path = root / "stage3_cell_completion.csv"
    readiness_path = root / "stage3_oof_arm_readiness.csv"
    write_frame(cells, cells_path)
    write_frame(readiness, readiness_path)
    written.extend([cells_path, readiness_path])

    ready_summary = readiness_summary(resolved, readiness, verifier_report)
    ready_summary.update(
        {
            "analysis_manifest": str(Path(args.analysis_manifest).resolve()),
            "analysis_manifest_sha256": sha256_file(args.analysis_manifest),
            "registry": str(Path(args.registry).resolve()),
            "completion_only_requested": bool(args.readiness_only),
        }
    )
    readiness_summary_path = root / "stage3_readiness_summary.json"
    write_json(readiness_summary_path, ready_summary)
    written.append(readiness_summary_path)
    if args.readiness_only:
        write_artifact_index(root, written, analysis_mode="completion_only")
        return ready_summary

    if not bool(readiness["complete_oof_arm"].all()):
        blocked = {
            **ready_summary,
            "analysis_mode": "full_analysis_blocked_incomplete",
            "selection_performed": False,
            "failure_reason": "all_900_cells_are_required_for_full_analysis",
        }
        blocked_path = root / "stage3_analysis_blocked_incomplete.json"
        write_json(blocked_path, blocked)
        written.append(blocked_path)
        write_artifact_index(root, written, analysis_mode="full_analysis_blocked_incomplete")
        raise RuntimeError(
            "Stage 3 full analysis is incomplete. Run with --readiness-only for status; "
            f"remaining weighted cells: {ready_summary['remaining_weighted_cells']}."
        )

    stratum_map, baseline_predictions, baseline_source = load_stage2_intron_baseline(
        args.stage2_intron_baseline_predictions
    )
    arms = assemble_oof_arms(resolved, stratum_map)
    parameter_counts = _parameter_counts(resolved)
    metrics, fold_metrics, intron_strata, intron_equal = score_arms(
        arms, parameter_counts
    )
    baseline_pooled = raw_metrics(baseline_predictions)
    baseline_sensitivity, baseline_strata = strict_intron_sensitivity_metrics(
        baseline_predictions
    )
    intron_mask = metrics["part_slug"].eq("intron")
    metrics.loc[
        intron_mask, "fold_trained_stratum_mean_baseline_pooled_oof_pearson"
    ] = baseline_pooled["pearson"]
    metrics.loc[
        intron_mask, "fold_trained_stratum_mean_baseline_pooled_oof_rmse"
    ] = baseline_pooled["rmse"]
    metrics.loc[
        intron_mask, "fold_trained_stratum_mean_baseline_pooled_oof_cod_r2"
    ] = baseline_pooled["cod_r2"]
    metrics.loc[
        intron_mask,
        "fold_trained_stratum_mean_baseline_within_stratum_centered_pearson",
    ] = baseline_sensitivity["within_stratum_centered_pearson"]
    metrics.loc[intron_mask, "delta_model_minus_baseline_pooled_oof_pearson"] = (
        metrics.loc[intron_mask, "pooled_oof_pearson"] - baseline_pooled["pearson"]
    )
    metrics.loc[
        intron_mask, "delta_model_minus_baseline_within_stratum_centered_pearson"
    ] = (
        metrics.loc[intron_mask, "within_stratum_centered_pearson"]
        - baseline_sensitivity["within_stratum_centered_pearson"]
    )
    margins = verifier_report["metric_margins"]
    loss_gates, loss_fold_gates, rc_gates, rc_fold_gates = score_intervention_gates(
        arms, margins
    )
    factorials, factorial_folds = score_factorial_differences(arms)
    admissibility = apply_admissibility(metrics, loss_gates, rc_gates)
    selection_table, bootstrap_summary, bootstrap_samples, selections = select_parts(
        admissibility, arms
    )

    baseline_summary = pd.DataFrame(
        [
            {
                "baseline_type": "fold_trained_stratum_mean",
                "fit_scope": "exact_non_audit_model_training_rows",
                **{f"pooled_oof_{name}": value for name, value in baseline_pooled.items()},
                **baseline_sensitivity,
                "source_path": baseline_source["path"],
                "source_sha256": baseline_source["sha256"],
            }
        ]
    )
    baseline_strata_frame = pd.DataFrame(baseline_strata)
    oof_predictions = flatten_oof_arms(arms)

    frame_outputs = (
        (metrics, "stage3_oof_metrics.csv", ","),
        (fold_metrics, "stage3_oof_fold_metrics.csv", ","),
        (intron_strata, "stage3_intron_stratum_metrics.csv", ","),
        (intron_equal, "stage3_intron_equal_stratum_estimands.csv", ","),
        (baseline_summary, "stage3_intron_stratum_mean_baseline.csv", ","),
        (baseline_strata_frame, "stage3_intron_baseline_per_stratum_metrics.csv", ","),
        (
            baseline_predictions,
            "stage3_intron_stratum_mean_baseline_predictions.tsv",
            "\t",
        ),
        (loss_gates, "stage3_loss_pair_metrics.csv", ","),
        (loss_fold_gates, "stage3_loss_fold_pair_metrics.csv", ","),
        (rc_gates, "stage3_rc_pair_metrics.csv", ","),
        (rc_fold_gates, "stage3_rc_fold_pair_metrics.csv", ","),
        (factorials, "stage3_rc_by_loss_factorial_metrics.csv", ","),
        (factorial_folds, "stage3_rc_by_loss_factorial_fold_differences.csv", ","),
        (selection_table, "stage3_arm_admissibility.csv", ","),
        (bootstrap_summary, "stage3_one_se_review.csv", ","),
        (bootstrap_samples, "stage3_one_se_bootstrap_samples.csv", ","),
        (pd.DataFrame(selections), "stage3_selected_part_policies.csv", ","),
    )
    for frame, name, separator in frame_outputs:
        path = root / name
        write_frame(frame, path, sep=separator)
        written.append(path)

    predictions_path = root / "stage3_oof_predictions.tsv.gz"
    oof_predictions.to_csv(
        predictions_path, index=False, sep="\t", compression="gzip"
    )
    written.append(predictions_path)

    selection_payload = {
        "schema_version": "lib1_dedup_stage3_part_selection_v1",
        "selection_status": "development_only_pre_audit",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples_per_part": BOOTSTRAP_RESAMPLES,
        "part_selections": selections,
        "audit_loader_instantiated": False,
        "audit_targets_loaded": False,
        "audit_predictions_scored": False,
        "audit_predictions_generated": False,
        "audit_metrics_computed": False,
        "audit_stratum_counts_inspected": False,
    }
    selection_path = root / "stage3_selected_part_policies.json"
    write_json(selection_path, selection_payload)
    written.append(selection_path)

    summary = {
        "schema_version": "lib1_dedup_stage3_analysis_v1",
        "analysis_status": "complete_development_only_pre_audit",
        "analysis_manifest": str(Path(args.analysis_manifest).resolve()),
        "analysis_manifest_sha256": sha256_file(args.analysis_manifest),
        "manifest_validation_status": verifier_report["validation_status"],
        "analysis_cells": len(resolved),
        "complete_oof_arms": len(arms),
        "raw_oof_metric_arms": len(metrics),
        "raw_oof_fold_metric_rows": len(fold_metrics),
        "loss_arm_pairs": len(loss_gates),
        "loss_fold_pairs": len(loss_fold_gates),
        "rc_arm_pairs": len(rc_gates),
        "rc_fold_pairs": len(rc_fold_gates),
        "rc_by_loss_factorials": len(factorials),
        "rc_by_loss_factorial_fold_rows": len(factorial_folds),
        "admissible_arm_count": int(selection_table["admissible"].sum()),
        "selection_eligible_arm_count": int(
            selection_table["selection_eligible"].sum()
        ),
        "selected_part_arm_count": int(selection_table["selected_part_arm"].sum()),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples_per_part": BOOTSTRAP_RESAMPLES,
        "bootstrap_rng_reinitialized_per_part": True,
        "bootstrap_sort": "development_fold_then_construct_id_ascending",
        "intron_development_constructs": EXPECTED_INTRON_DEVELOPMENT_CONSTRUCTS,
        "intron_stratum_map_source": baseline_source,
        "intron_equal_stratum_weight_min": float(
            intron_equal["equal_stratum_weight_min"].min()
        ),
        "intron_equal_stratum_weight_max": float(
            intron_equal["equal_stratum_weight_max"].max()
        ),
        "intron_equal_stratum_weight_ess": float(
            intron_equal["equal_stratum_weight_ess"].iloc[0]
        ),
        "primary_metric": "pooled_five_fold_raw_scale_oof_pearson",
        "full_analysis_required_all_cells": True,
        "nonfinite_fold_deltas_dropped": False,
        "only_admissible_arms_enter_one_se_band": True,
        "selection_status": "development_only_pre_audit",
        "audit_loader_instantiated": False,
        "audit_targets_loaded": False,
        "audit_predictions_scored": False,
        "audit_predictions_generated": False,
        "audit_metrics_computed": False,
        "audit_stratum_counts_inspected": False,
    }
    summary_path = root / "stage3_analysis_summary.json"
    write_json(summary_path, summary)
    written.append(summary_path)
    write_artifact_index(root, written, analysis_mode="full_analysis")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-manifest", type=Path, default=DEFAULT_ANALYSIS_MANIFEST)
    parser.add_argument("--dry-run-manifest", type=Path, default=DEFAULT_DRY_RUN_MANIFEST)
    parser.add_argument("--reuse-manifest", type=Path, default=DEFAULT_REUSE_MANIFEST)
    parser.add_argument("--portfolio", type=Path, default=DEFAULT_PORTFOLIO)
    parser.add_argument("--manifest-summary", type=Path, default=DEFAULT_MANIFEST_SUMMARY)
    parser.add_argument(
        "--stage2-analysis-manifest", type=Path, default=DEFAULT_STAGE2_MANIFEST
    )
    parser.add_argument(
        "--targeted-utr3-manifest", type=Path, default=DEFAULT_TARGETED_MANIFEST
    )
    parser.add_argument("--stage2-metrics", type=Path, default=DEFAULT_STAGE2_METRICS)
    parser.add_argument("--targeted-metrics", type=Path, default=DEFAULT_TARGETED_METRICS)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--stage2-intron-baseline-predictions",
        type=Path,
        default=DEFAULT_INTRON_BASELINE_PREDICTIONS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--readiness-only",
        "--completion-only",
        dest="readiness_only",
        action="store_true",
        help=(
            "Validate and resolve the 900-cell design, then write completion status "
            "without loading OOF predictions or performing selection."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_analysis(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
