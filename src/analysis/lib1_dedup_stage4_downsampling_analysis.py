#!/usr/bin/env python3
"""Development-only analysis for Lib1 dedup Stage 4 downsampling.

The module consumes only manifest-authorized outer-fold OOF prediction exports
and their compact provenance.  It never imports a DataModule and never reads a
final-test prediction, target, metric, checkpoint, or loader product.

The analysis is deliberately split into three layers:

* one-row (one outer fold) raw-scale metrics;
* five-fold pooled OOF metrics for a configuration/N/subset track;
* curve-level summaries across the three primary subset tracks.

Bounded curves and their projections are secondary descriptions.  Directly
observed 10x/100x contrasts remain the primary sample-efficiency evidence.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


REPO_ROOT = Path(__file__).resolve().parents[2]
LEARN_ROOT = REPO_ROOT / "src" / "learn"
for import_root in (REPO_ROOT, LEARN_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from src.analysis.lib1_dedup_stage2_analysis import (  # noqa: E402
    RAW_PREDICTION,
    RAW_TARGET,
    SENSITIVITY_STRATUM,
    STRATUM_ORDER,
    raw_metrics,
)
from src.learn import verify_lib1_dedup_stage4_downsampling_manifest as manifest_verifier  # noqa: E402
from src.learn.run_lib1_dedup_stage4_downsampling_campaign import (  # noqa: E402
    TEST_METRIC_FIELDS,
    expected_registry_fields,
    validate_completed_record,
)


MANIFEST_TAG = "lib1_dedup_stage4_downsampling_july2026"
PREFIX = LEARN_ROOT / "outputs" / "hpo_manifests" / MANIFEST_TAG
DEFAULT_MANIFEST = Path(str(PREFIX) + "__dry_run_manifest.jsonl")
DEFAULT_PORTFOLIO = Path(str(PREFIX) + "__portfolio.json")
DEFAULT_MANIFEST_SUMMARY = Path(str(PREFIX) + "__summary.json")
DEFAULT_REGISTRY = (
    LEARN_ROOT
    / "outputs/hpo_runs/status/lib1_dedup_stage4_downsampling_july2026/stage4_runs.csv"
)
DEFAULT_OUTPUT_DIR = LEARN_ROOT / "outputs" / "analysis" / MANIFEST_TAG
DEFAULT_INTRON_STRATA = (
    LEARN_ROOT
    / "outputs/analysis/lib1_dedup_stage2_july2026/"
    "stage2_intron_stratum_mean_baseline_predictions.tsv"
)
EXPECTED_INTRON_STRATA_SHA256 = (
    "82c228a3ba0cd0b0df403b52095f8efc1a9a3cdd20417a656b8cccb8f2d14e8c"
)
EXPECTED_ROWS = 660
EXPECTED_POOLED_TRACKS = 132
EXPECTED_CURVE_POINTS = 72
EXPECTED_CAMPAIGN_ID = "lib1_dedup_phase1_rerun_july2026"
EXPECTED_CAMPAIGN_STAGE = "stage4_downsampling"
EXPECTED_FOLDS = tuple(range(5))
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 20260717
PART_ORDER = {"enhancer": 0, "promoter": 1, "intron": 2, "utr3": 3, "utr5": 4}
PART_LABELS = {
    "enhancer": "Enhancer",
    "promoter": "Promoter",
    "intron": "Intron",
    "utr3": "3'UTR",
    "utr5": "5'UTR",
}
DIRECT_CONTRASTS = (
    (40, 400, "10x"),
    (250, 2500, "10x"),
    (400, 4000, "10x"),
    (40, 4000, "100x"),
)
CURVE_KEYS = ("part_slug", "stage4_lane", "base_config_id")
TRACK_KEYS = CURVE_KEYS + (
    "downsample_n_label",
    "subset_replicate",
    "train_subsample_seed",
)
METRICS = ("pearson", "rmse", "cod_r2")
REQUIRED_TRAINING_DIAGNOSTICS = (
    "best_epoch",
    "optimizer_steps",
    "train_pearson",
    "best_metric_value",
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


def write_json(path: Path, value: Mapping | Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def validate_frozen_manifest(args: argparse.Namespace) -> dict:
    report = manifest_verifier.validate(
        argparse.Namespace(
            manifest=Path(args.manifest),
            portfolio=Path(args.portfolio),
            summary=Path(args.manifest_summary),
        )
    )
    if report.get("status") != "valid" or int(report.get("rows", -1)) != EXPECTED_ROWS:
        raise RuntimeError("The frozen Stage 4 manifest did not validate.")
    if report.get("final_test_loader_instantiated") is not False:
        raise RuntimeError("The Stage 4 verifier no longer proves final-test isolation.")
    if report.get("final_test_metrics_read") is not False:
        raise RuntimeError("The Stage 4 verifier reports final-test metric access.")
    return report


def validate_registry_isolation(path: str | Path) -> Path:
    """Restrict reconciliation to the campaign's dedicated, ignored registry."""
    observed = Path(path).expanduser().resolve()
    expected = DEFAULT_REGISTRY.resolve()
    if observed != expected:
        raise ValueError(
            "Stage 4 analysis may read only its dedicated campaign registry: "
            f"{expected}; received {observed}. The global runs.csv is forbidden."
        )
    return observed


def read_registry(path: str | Path) -> dict[str, list[dict]]:
    by_cell: dict[str, list[dict]] = defaultdict(list)
    registry_path = Path(path)
    if not registry_path.is_file():
        return by_cell
    with registry_path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if record.get("campaign_id", "") != EXPECTED_CAMPAIGN_ID:
                raise RuntimeError("Dedicated Stage 4 registry contains another campaign.")
            if record.get("campaign_stage", "") != EXPECTED_CAMPAIGN_STAGE:
                raise RuntimeError("Dedicated Stage 4 registry contains another campaign stage.")
            populated_test = {
                field: record.get(field, "")
                for field in TEST_METRIC_FIELDS
                if str(record.get(field, "") or "").strip()
            }
            if populated_test:
                raise RuntimeError(
                    "Dedicated Stage 4 registry contains final-test metrics: "
                    + json.dumps(populated_test, sort_keys=True)
                )
            cell_id = str(record.get("cell_id", "") or "").strip()
            if not cell_id:
                raise RuntimeError("Dedicated Stage 4 registry row lacks cell_id.")
            by_cell[cell_id].append(record)
    return by_cell


def _finite_diagnostic(
    record: Mapping, payload: Mapping, field: str, *, integer: bool = False
) -> float | int:
    """Require one diagnostic in both registry and compact provenance."""
    values = []
    for source_name, source in (("registry", record), ("compact provenance", payload)):
        raw = source.get(field, "")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise RuntimeError(f"Completed cell lacks numeric {field} in {source_name}.")
        if not math.isfinite(value):
            raise RuntimeError(f"Completed cell has non-finite {field} in {source_name}.")
        values.append(value)
    if not math.isclose(values[0], values[1], rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"Registry/compact-provenance {field} mismatch: {values[0]} != {values[1]}."
        )
    value = values[0]
    if integer:
        if not value.is_integer():
            raise RuntimeError(f"Completed cell has non-integer {field}: {value}.")
        return int(value)
    return value


def _training_diagnostic_evidence(record: Mapping, payload: Mapping) -> dict:
    """Validate the frozen best-checkpoint training diagnostics.

    ``val_pearson`` in a run summary can refer to the last epoch.  The required
    inner-validation value is therefore the checkpoint's ``best_metric_value``
    after proving that both the configured and resolved monitor are Pearson.
    """
    for source_name, source in (("registry", record), ("compact provenance", payload)):
        if str(source.get("checkpoint_monitor", "")) != "val_pearson":
            raise RuntimeError(
                f"Completed cell checkpoint_monitor is not val_pearson in {source_name}."
            )
        if str(source.get("best_metric_name", "")) != "val_pearson":
            raise RuntimeError(
                f"Completed cell best_metric_name is not val_pearson in {source_name}."
            )
    best_epoch = _finite_diagnostic(record, payload, "best_epoch", integer=True)
    optimizer_steps = _finite_diagnostic(record, payload, "optimizer_steps", integer=True)
    train_pearson = _finite_diagnostic(record, payload, "train_pearson")
    best_inner_val_pearson = _finite_diagnostic(record, payload, "best_metric_value")
    if best_epoch < 0:
        raise RuntimeError(f"Completed cell has negative best_epoch: {best_epoch}.")
    if optimizer_steps <= 0:
        raise RuntimeError(f"Completed cell has nonpositive optimizer_steps: {optimizer_steps}.")
    for field, value in (
        ("train_pearson", train_pearson),
        ("best_metric_value", best_inner_val_pearson),
    ):
        if not -1.0 <= value <= 1.0:
            raise RuntimeError(f"Completed cell has out-of-range {field}: {value}.")
    return {
        "best_epoch": best_epoch,
        "optimizer_steps": optimizer_steps,
        "train_pearson": train_pearson,
        "best_inner_val_pearson": best_inner_val_pearson,
        "train_minus_best_inner_val_pearson_gap": train_pearson - best_inner_val_pearson,
    }


def _completed_evidence(row: Mapping, candidates: Sequence[dict]) -> dict:
    """Resolve one cell without opening any prediction until provenance passes."""
    if not candidates:
        return {
            "availability": "missing_registry_row",
            "resolved_run_id": "",
            "resolved_prediction_path": "",
            "resolved_prediction_sha256": "",
            "resolved_provenance_path": "",
            "resolved_provenance_sha256": "",
            "best_epoch": "",
            "optimizer_steps": "",
            "train_pearson": "",
            "best_inner_val_pearson": "",
            "train_minus_best_inner_val_pearson_gap": "",
        }

    expected = expected_registry_fields(dict(row))
    for record in candidates:
        mismatch = {
            field: {"observed": record.get(field, ""), "expected": value}
            for field, value in expected.items()
            if record.get(field, "") != value
        }
        if mismatch:
            raise RuntimeError(
                f"Registry provenance collision for {row['cell_id']}:\n"
                + json.dumps(mismatch, indent=2, sort_keys=True)
            )

    completed = [
        record for record in candidates if str(record.get("status", "")).lower() == "completed"
    ]
    if len(completed) > 1:
        raise RuntimeError(f"Cell {row['cell_id']} resolves to multiple completions.")
    if not completed:
        statuses = sorted(
            {str(record.get("status", "") or "unknown").lower() for record in candidates}
        )
        return {
            "availability": "registry_status_" + "+".join(statuses),
            "resolved_run_id": "",
            "resolved_prediction_path": "",
            "resolved_prediction_sha256": "",
            "resolved_provenance_path": "",
            "resolved_provenance_sha256": "",
            "best_epoch": "",
            "optimizer_steps": "",
            "train_pearson": "",
            "best_inner_val_pearson": "",
            "train_minus_best_inner_val_pearson_gap": "",
        }

    record = completed[0]
    # This validates empty final-test metric fields, exact OOF filename/location,
    # OOF IDs, physical final-test exclusion, n_test=0, and all split hashes.
    prediction, provenance = validate_completed_record(dict(row), record)
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    forbidden = {
        field: payload.get(field)
        for field in TEST_METRIC_FIELDS
        if str(payload.get(field, "") or "").strip()
    }
    if forbidden:
        raise RuntimeError(f"Compact provenance contains final-test metrics: {forbidden}")
    split = payload.get("data_split_summary", {})
    if split.get("n_test") != 0 or split.get("final_test_rows_physically_excluded") is not True:
        raise RuntimeError(f"Cell {row['cell_id']} does not prove physical final-test exclusion.")
    if split.get("audit_loader_authorized") is not False:
        raise RuntimeError(f"Cell {row['cell_id']} authorized a final-test loader.")
    diagnostics = _training_diagnostic_evidence(record, payload)
    return {
        "availability": "complete",
        "resolved_run_id": str(record["run_id"]),
        "resolved_prediction_path": str(prediction.resolve()),
        "resolved_prediction_sha256": sha256_file(prediction),
        "resolved_provenance_path": str(provenance.resolve()),
        "resolved_provenance_sha256": sha256_file(provenance),
        **diagnostics,
    }


def resolve_cells(manifest_rows: Sequence[dict], registry_path: str | Path) -> list[dict]:
    if len(manifest_rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} manifest rows; found {len(manifest_rows)}.")
    registry = read_registry(registry_path)
    resolved = [
        {**row, **_completed_evidence(row, registry.get(str(row["cell_id"]), []))}
        for row in manifest_rows
    ]
    completed = [row for row in resolved if row["availability"] == "complete"]
    for field in ("resolved_run_id", "resolved_prediction_path", "resolved_provenance_path"):
        duplicates = [
            value
            for value, count in Counter(str(row[field]) for row in completed).items()
            if value and count > 1
        ]
        if duplicates:
            raise RuntimeError(f"Completed cells reuse {field}: {duplicates}")
    return resolved


def completion_table(rows: Sequence[Mapping]) -> pd.DataFrame:
    fields = (
        "row",
        "cell_id",
        "part_slug",
        "stage4_lane",
        "base_config_id",
        "outer_oof_fold",
        "inner_validation_fold",
        "downsample_n_label",
        "expected_train_n",
        "subset_replicate",
        "train_subsample_seed",
        "availability",
        "resolved_run_id",
        "resolved_prediction_path",
        "resolved_prediction_sha256",
        "resolved_provenance_path",
        "resolved_provenance_sha256",
        "best_epoch",
        "optimizer_steps",
        "train_pearson",
        "best_inner_val_pearson",
        "train_minus_best_inner_val_pearson_gap",
    )
    return pd.DataFrame([{field: row.get(field, "") for field in fields} for row in rows])


def _assert_oof_path(path: Path) -> None:
    lower_parts = [part.lower() for part in path.parts]
    if "__oof_predictions.tsv" not in path.name:
        raise ValueError(f"Refusing non-OOF prediction product: {path}")
    if any("final_test" in part or "audit_test" in part for part in lower_parts):
        raise ValueError(f"Refusing path in a final-test product tree: {path}")


def load_prediction_cell(row: Mapping) -> pd.DataFrame:
    if row.get("availability") != "complete":
        raise RuntimeError(f"Cell {row.get('cell_id')} is not complete.")
    path = Path(str(row["resolved_prediction_path"]))
    _assert_oof_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != str(row["resolved_prediction_sha256"]):
        raise ValueError(f"OOF prediction SHA changed for {row['cell_id']}.")
    frame = pd.read_csv(path, sep="\t")
    required = {"construct_id", RAW_TARGET, RAW_PREDICTION}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"OOF prediction {path} lacks {missing}.")
    frame["construct_id"] = frame["construct_id"].astype(str)
    if frame["construct_id"].duplicated().any():
        raise ValueError(f"Cell {row['cell_id']} duplicates an OOF construct.")
    if len(frame) != int(row["expected_oof_n"]):
        raise ValueError(f"Cell {row['cell_id']} has the wrong OOF row count.")
    if canonical_id_hash(frame["construct_id"].tolist()) != row["expected_oof_id_hash"]:
        raise ValueError(f"Cell {row['cell_id']} has the wrong OOF construct IDs.")
    for column in (RAW_TARGET, RAW_PREDICTION):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(float)).all():
            raise ValueError(f"Cell {row['cell_id']} has non-finite {column}.")
    kept = ["construct_id", RAW_TARGET, RAW_PREDICTION]
    for optional in ("n_barcodes", "row_id"):
        if optional in frame:
            kept.append(optional)
    frame = frame[kept].copy()
    frame["outer_oof_fold"] = int(row["outer_oof_fold"])
    frame["cell_id"] = str(row["cell_id"])
    return frame


def load_intron_strata(path: str | Path) -> tuple[pd.DataFrame, dict]:
    """Load the frozen, development-only inferred-mask stratum map."""
    source = Path(path)
    observed_sha = sha256_file(source)
    if observed_sha != EXPECTED_INTRON_STRATA_SHA256:
        raise ValueError(
            "Frozen Intron development stratum source changed: "
            f"{observed_sha} != {EXPECTED_INTRON_STRATA_SHA256}"
        )
    frame = pd.read_csv(source, sep="\t")
    required = {
        "construct_id",
        "development_fold",
        RAW_TARGET,
        SENSITIVITY_STRATUM,
        "baseline_type",
        "fit_scope",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Intron stratum source lacks {missing}.")
    frame = frame.loc[frame["baseline_type"].eq("fold_trained_stratum_mean")].copy()
    if set(frame["fit_scope"]) != {"exact_non_audit_model_training_rows"}:
        raise ValueError("Intron stratum labels are not the frozen development-safe source.")
    if frame["construct_id"].duplicated().any():
        raise ValueError("Frozen Intron stratum map duplicates construct IDs.")
    if set(frame[SENSITIVITY_STRATUM]) != set(STRATUM_ORDER):
        raise ValueError("Frozen Intron strata changed.")
    frame["construct_id"] = frame["construct_id"].astype(str)
    frame["development_fold"] = frame["development_fold"].astype(int)
    return (
        frame[["construct_id", "development_fold", RAW_TARGET, SENSITIVITY_STRATUM]],
        {
            "path": str(source.resolve()),
            "sha256": observed_sha,
            "rows": int(len(frame)),
            "fit_scope": "exact_non_audit_model_training_rows",
            "final_test_rows_loaded": False,
        },
    )


def attach_intron_strata(frame: pd.DataFrame, stratum_map: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(
        stratum_map.rename(
            columns={"development_fold": "outer_oof_fold", RAW_TARGET: "frozen_stratum_target"}
        ),
        on=["construct_id", "outer_oof_fold"],
        how="left",
        validate="one_to_one",
    )
    if merged[SENSITIVITY_STRATUM].isna().any():
        raise ValueError("An Intron OOF row is absent from the frozen stratum map.")
    if not np.allclose(
        merged[RAW_TARGET].to_numpy(float),
        merged["frozen_stratum_target"].to_numpy(float),
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Intron Stage 4 targets differ from the frozen development source.")
    return merged.drop(columns="frozen_stratum_target")


def _metadata(row: Mapping) -> dict:
    fields = (
        "part_slug",
        "stage4_lane",
        "diagnostic_only",
        "portfolio_rank",
        "base_config_id",
        "architecture",
        "training_regime",
        "initialization",
        "source_head",
        "unfreeze_scope",
        "rc_mode",
        "loss_mode",
        "policy_id",
        "downsample_n_label",
        "subset_replicate",
        "train_subsample_seed",
        "model_seed",
    )
    return {field: row.get(field, "") for field in fields}


def intron_metrics(frame: pd.DataFrame) -> tuple[dict, list[dict]]:
    centered = frame.copy()
    centered["target_centered"] = centered[RAW_TARGET] - centered.groupby(
        SENSITIVITY_STRATUM
    )[RAW_TARGET].transform("mean")
    centered["prediction_centered"] = centered[RAW_PREDICTION] - centered.groupby(
        SENSITIVITY_STRATUM
    )[RAW_PREDICTION].transform("mean")
    centered_values = raw_metrics(centered, "target_centered", "prediction_centered")
    records = []
    for stratum in STRATUM_ORDER:
        subset = frame.loc[frame[SENSITIVITY_STRATUM].eq(stratum)]
        if subset.empty:
            raise ValueError(f"Intron stratum {stratum} is empty.")
        values = raw_metrics(subset)
        records.append(
            {
                SENSITIVITY_STRATUM: stratum,
                **values,
                "target_mean": float(subset[RAW_TARGET].mean()),
                "prediction_mean": float(subset[RAW_PREDICTION].mean()),
            }
        )
    per = pd.DataFrame(records)
    return (
        {
            **{
                f"within_stratum_centered_{key}": value
                for key, value in centered_values.items()
                if key != "n_constructs"
            },
            "macro_stratum_pearson": float(per["pearson"].mean()),
            "minimum_stratum_pearson": float(per["pearson"].min()),
        },
        records,
    )


def score_completed_cells(
    rows: Sequence[Mapping], stratum_map: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    metric_rows: list[dict] = []
    predictions: dict[str, pd.DataFrame] = {}
    for row in rows:
        if row.get("availability") != "complete":
            continue
        frame = load_prediction_cell(row)
        if row["part_slug"] == "intron":
            frame = attach_intron_strata(frame, stratum_map)
        predictions[str(row["cell_id"])] = frame
        values = raw_metrics(frame)
        record = {
            **_metadata(row),
            "cell_id": row["cell_id"],
            "outer_oof_fold": int(row["outer_oof_fold"]),
            "inner_validation_fold": int(row["inner_validation_fold"]),
            "expected_train_n": int(row["expected_train_n"]),
            "best_epoch": pd.to_numeric(row.get("best_epoch", math.nan), errors="coerce"),
            "optimizer_steps": pd.to_numeric(
                row.get("optimizer_steps", math.nan), errors="coerce"
            ),
            "train_pearson_diagnostic": pd.to_numeric(
                row.get("train_pearson", math.nan), errors="coerce"
            ),
            "best_inner_val_pearson_diagnostic": pd.to_numeric(
                row.get("best_inner_val_pearson", math.nan), errors="coerce"
            ),
            **values,
        }
        record["train_minus_best_inner_val_pearson_gap"] = (
            record["train_pearson_diagnostic"]
            - record["best_inner_val_pearson_diagnostic"]
        )
        if row["part_slug"] == "intron":
            summary, _ = intron_metrics(frame)
            record.update(summary)
        metric_rows.append(record)
    return pd.DataFrame(metric_rows), predictions


def summarize_training_diagnostics(per_run: pd.DataFrame) -> pd.DataFrame:
    """Report best-checkpoint exposure/overfit diagnostics per config and N."""
    if per_run.empty:
        return pd.DataFrame()
    group_fields = list(CURVE_KEYS) + [
        "downsample_n_label",
        "architecture",
        "training_regime",
        "rc_mode",
        "loss_mode",
        "policy_id",
    ]
    records = []
    for key, group in per_run.groupby(group_fields, sort=False, dropna=False):
        record = {
            **dict(zip(group_fields, key)),
            "completed_cell_count": int(len(group)),
            "completed_outer_fold_count": int(group["outer_oof_fold"].nunique()),
            "subset_track_count": int(group["subset_replicate"].nunique()),
            "mean_actual_train_n": float(group["expected_train_n"].mean()),
            "total_optimizer_steps": int(group["optimizer_steps"].sum()),
        }
        for field in (
            "best_epoch",
            "optimizer_steps",
            "train_pearson_diagnostic",
            "best_inner_val_pearson_diagnostic",
            "train_minus_best_inner_val_pearson_gap",
        ):
            values = group[field].to_numpy(float)
            if not np.isfinite(values).all():
                raise RuntimeError(f"Completed Stage 4 rows have missing {field}.")
            record[f"mean_{field}"] = float(np.mean(values))
            record[f"median_{field}"] = float(np.median(values))
            record[f"minimum_{field}"] = float(np.min(values))
            record[f"maximum_{field}"] = float(np.max(values))
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["part_slug", "stage4_lane", "base_config_id", "mean_actual_train_n"]
    ).reset_index(drop=True)


def _track_key(row: Mapping) -> tuple:
    return tuple(row[field] for field in TRACK_KEYS)


def assemble_complete_tracks(
    rows: Sequence[Mapping], predictions: Mapping[str, pd.DataFrame]
) -> tuple[dict[tuple, pd.DataFrame], pd.DataFrame]:
    grouped: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        if row.get("availability") == "complete":
            grouped[_track_key(row)].append(row)
    tracks: dict[tuple, pd.DataFrame] = {}
    readiness = []
    for key, pieces in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        folds = sorted(int(row["outer_oof_fold"]) for row in pieces)
        complete = len(pieces) == 5 and folds == list(EXPECTED_FOLDS)
        readiness.append(
            {
                **dict(zip(TRACK_KEYS, key)),
                "complete_fold_count": len(folds),
                "complete_folds_json": json.dumps(folds),
                "complete_pooled_track": complete,
            }
        )
        if not complete:
            continue
        frame = pd.concat(
            [
                predictions[str(row["cell_id"])]
                for row in sorted(pieces, key=lambda value: int(value["outer_oof_fold"]))
            ],
            ignore_index=True,
        )
        if frame["construct_id"].duplicated().any():
            raise ValueError(f"Track {key} predicts an OOF construct more than once.")
        frame = frame.sort_values(["outer_oof_fold", "construct_id"]).reset_index(drop=True)
        frame.attrs["rows"] = [dict(row) for row in pieces]
        tracks[key] = frame
    return tracks, pd.DataFrame(readiness)


def validate_shared_oof_targets(tracks: Mapping[tuple, pd.DataFrame]) -> None:
    references: dict[str, pd.DataFrame] = {}
    for key, frame in tracks.items():
        part = str(key[0])
        values = frame[["outer_oof_fold", "construct_id", RAW_TARGET]].copy()
        reference = references.get(part)
        if reference is None:
            references[part] = values
            continue
        if not values[["outer_oof_fold", "construct_id"]].equals(
            reference[["outer_oof_fold", "construct_id"]]
        ):
            raise ValueError(f"Complete {part} tracks do not share exact OOF IDs/folds.")
        if not np.allclose(
            values[RAW_TARGET].to_numpy(float), reference[RAW_TARGET].to_numpy(float),
            rtol=0, atol=1e-10,
        ):
            raise ValueError(f"Complete {part} tracks do not share exact raw targets.")


def score_pooled_tracks(
    tracks: Mapping[tuple, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict] = []
    stratum_rows: list[dict] = []
    for key, frame in tracks.items():
        first = frame.attrs["rows"][0]
        train_counts = [int(row["expected_train_n"]) for row in frame.attrs["rows"]]
        record = {
            **_metadata(first),
            "mean_actual_train_n": float(np.mean(train_counts)),
            "minimum_actual_train_n": int(min(train_counts)),
            "maximum_actual_train_n": int(max(train_counts)),
            **raw_metrics(frame),
        }
        if first["part_slug"] == "intron":
            summary, per_stratum = intron_metrics(frame)
            record.update(summary)
            stratum_rows.extend(
                [
                    {
                        **_metadata(first),
                        "mean_actual_train_n": float(np.mean(train_counts)),
                        "minimum_actual_train_n": int(min(train_counts)),
                        "maximum_actual_train_n": int(max(train_counts)),
                        **values,
                    }
                    for values in per_stratum
                ]
            )
        metric_rows.append(record)
    metrics = pd.DataFrame(metric_rows)
    if not metrics.empty:
        metrics = metrics.sort_values(
            ["part_slug", "stage4_lane", "base_config_id", "mean_actual_train_n", "subset_replicate"]
        ).reset_index(drop=True)
    return metrics, pd.DataFrame(stratum_rows)


def curve_points(pooled: pd.DataFrame) -> pd.DataFrame:
    if pooled.empty:
        return pd.DataFrame()
    records = []
    for key, group in pooled.groupby(list(CURVE_KEYS) + ["downsample_n_label"], sort=False):
        metadata = dict(zip(CURVE_KEYS + ("downsample_n_label",), key))
        first = group.iloc[0]
        record = {
            **metadata,
            **{
                field: first[field]
                for field in (
                    "diagnostic_only", "portfolio_rank", "architecture", "training_regime",
                    "rc_mode", "loss_mode", "policy_id",
                )
            },
            "mean_actual_train_n": float(group["mean_actual_train_n"].mean()),
            "minimum_actual_train_n": int(group["minimum_actual_train_n"].min()),
            "maximum_actual_train_n": int(group["maximum_actual_train_n"].max()),
            "subset_track_count": int(len(group)),
        }
        for metric in METRICS:
            values = group[metric].to_numpy(float)
            record[f"mean_{metric}"] = float(np.mean(values))
            record[f"sd_{metric}"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            record[f"minimum_{metric}"] = float(np.min(values))
            record[f"maximum_{metric}"] = float(np.max(values))
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["part_slug", "stage4_lane", "base_config_id", "mean_actual_train_n"]
    ).reset_index(drop=True)


def observed_paired_contrasts(
    tracks: Mapping[tuple, pd.DataFrame], pooled: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pooled.empty:
        return pd.DataFrame(), pd.DataFrame()
    by_identity = {
        (
            row.part_slug, row.stage4_lane, row.base_config_id,
            int(row.subset_replicate), int(row.train_subsample_seed),
            str(row.downsample_n_label),
        ): row
        for row in pooled.itertuples(index=False)
    }
    records = []
    for curve_key in sorted({key[:3] for key in tracks}):
        reps = sorted({(int(key[4]), int(key[5])) for key in tracks if key[:3] == curve_key})
        for replicate, seed in reps:
            for low, high, ratio in DIRECT_CONTRASTS:
                low_key = (*curve_key, replicate, seed, str(low))
                high_key = (*curve_key, replicate, seed, str(high))
                low_row = by_identity.get(low_key)
                high_row = by_identity.get(high_key)
                if low_row is None or high_row is None:
                    continue
                low_track_key = (*curve_key, str(low), replicate, seed)
                high_track_key = (*curve_key, str(high), replicate, seed)
                low_frame = tracks[low_track_key]
                high_frame = tracks[high_track_key]
                if not low_frame[["outer_oof_fold", "construct_id"]].equals(
                    high_frame[["outer_oof_fold", "construct_id"]]
                ):
                    raise ValueError(f"Contrast {low_key}->{high_key} is not OOF-paired.")
                if not np.allclose(
                    low_frame[RAW_TARGET], high_frame[RAW_TARGET], rtol=0, atol=1e-10
                ):
                    raise ValueError(f"Contrast {low_key}->{high_key} changed raw targets.")
                record = {
                    **dict(zip(CURVE_KEYS, curve_key)),
                    "subset_replicate": replicate,
                    "train_subsample_seed": seed,
                    "low_n": low,
                    "high_n": high,
                    "multiplicative_contrast": ratio,
                    "paired_oof_constructs": len(low_frame),
                }
                for metric in METRICS:
                    low_value = float(getattr(low_row, metric))
                    high_value = float(getattr(high_row, metric))
                    record[f"low_{metric}"] = low_value
                    record[f"high_{metric}"] = high_value
                    record[f"delta_{metric}"] = high_value - low_value
                if curve_key[0] == "intron":
                    metric = "within_stratum_centered_pearson"
                    low_value = float(getattr(low_row, metric))
                    high_value = float(getattr(high_row, metric))
                    record[f"low_{metric}"] = low_value
                    record[f"high_{metric}"] = high_value
                    record[f"delta_{metric}"] = high_value - low_value
                records.append(record)
    detail = pd.DataFrame(records)
    if detail.empty:
        return detail, pd.DataFrame()
    summaries = []
    group_fields = list(CURVE_KEYS) + ["low_n", "high_n", "multiplicative_contrast"]
    delta_fields = [column for column in detail if column.startswith("delta_")]
    for key, group in detail.groupby(group_fields, sort=False):
        record = {**dict(zip(group_fields, key)), "subset_track_count": int(len(group))}
        for field in delta_fields:
            values = group[field].to_numpy(float)
            record[f"mean_{field}"] = float(np.mean(values))
            record[f"sd_{field}"] = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        summaries.append(record)
    return detail, pd.DataFrame(summaries)


def _tail_linear_solution(
    x: np.ndarray, y: np.ndarray, family: str, log_shape: float, metric: str
) -> tuple[float, float, float]:
    if family == "power_law":
        shape = math.exp(log_shape)
        q = np.power(x, -shape)
    elif family == "exponential":
        shape = math.exp(log_shape)
        q = np.exp(-x / shape)
    else:
        raise ValueError(f"Unknown curve family {family!r}")
    sign = -1.0 if metric == "pearson" else 1.0
    design = np.column_stack([np.ones(len(x)), sign * q])
    asymptote, amplitude = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = asymptote + sign * amplitude * q
    if not np.isfinite([asymptote, amplitude, shape, *prediction]).all() or amplitude <= 0:
        return math.inf, math.nan, math.nan
    if metric == "pearson" and not (-5.0 <= asymptote <= 5.0):
        return math.inf, math.nan, math.nan
    if metric == "rmse" and asymptote < 0:
        return math.inf, math.nan, math.nan
    return float(np.sum((prediction - y) ** 2)), float(asymptote), float(amplitude)


def fit_saturating_curve(
    n: Sequence[float], values: Sequence[float], family: str, metric: str = "pearson"
) -> dict:
    """Fit a monotone saturating curve with a fast one-dimensional search.

    Pearson is fitted in Fisher-z space and inverse transformed, so every
    reported correlation prediction remains in [-1, 1].  RMSE is fitted in
    raw units with a nonnegative asymptote.
    """
    x = np.asarray(n, dtype=float)
    observed = np.asarray(values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(observed) & (x > 0)
    x, observed = x[valid], observed[valid]
    if len(x) < 3 or len(np.unique(x)) < 3:
        return {"fit_status": "insufficient_points", "curve_family": family, "metric": metric}
    order = np.argsort(x)
    x, observed = x[order], observed[order]
    y = np.arctanh(np.clip(observed, -0.999999, 0.999999)) if metric == "pearson" else observed
    if family == "power_law":
        bounds = (math.log(0.005), math.log(5.0))
    elif family == "exponential":
        bounds = (math.log(max(float(x.min()) / 100.0, 1e-3)), math.log(float(x.max()) * 1e4))
    else:
        raise ValueError(f"Unknown curve family {family!r}")

    def raw_objective(log_shape: float) -> float:
        return _tail_linear_solution(x, y, family, log_shape, metric)[0]

    def objective(log_shape: float) -> float:
        value = raw_objective(log_shape)
        return value if np.isfinite(value) else 1e100

    grid = np.linspace(bounds[0], bounds[1], 161)
    scores = np.asarray([raw_objective(point) for point in grid])
    finite = np.isfinite(scores)
    if not finite.any():
        return {"fit_status": "constraint_failure", "curve_family": family, "metric": metric}
    best_index = int(np.nanargmin(np.where(finite, scores, np.nan)))
    left = grid[max(best_index - 2, 0)]
    right = grid[min(best_index + 2, len(grid) - 1)]
    result = minimize_scalar(objective, bounds=(left, right), method="bounded")
    log_shape = float(result.x) if result.success and np.isfinite(result.fun) else float(grid[best_index])
    sse, asymptote, amplitude = _tail_linear_solution(x, y, family, log_shape, metric)
    if not np.isfinite(sse):
        return {"fit_status": "optimization_failure", "curve_family": family, "metric": metric}
    shape = math.exp(log_shape)

    def predict_raw(sample_n: float) -> float:
        q = sample_n ** (-shape) if family == "power_law" else math.exp(-sample_n / shape)
        fitted = asymptote + (-1.0 if metric == "pearson" else 1.0) * amplitude * q
        return math.tanh(fitted) if metric == "pearson" else fitted

    full_n = float(x.max())
    predicted_observed = np.asarray([predict_raw(value) for value in x])
    full_value = predict_raw(full_n)
    ten_value = predict_raw(10.0 * full_n)
    hundred_value = predict_raw(100.0 * full_n)
    record = {
        "fit_status": "success",
        "curve_family": family,
        "metric": metric,
        "n_points": int(len(x)),
        "minimum_n": float(x.min()),
        "observed_full_n": full_n,
        "asymptote_transformed": asymptote,
        "asymptote": math.tanh(asymptote) if metric == "pearson" else asymptote,
        "amplitude": amplitude,
        "shape_parameter": shape,
        "shape_parameter_name": "alpha" if family == "power_law" else "tau",
        "fit_sse_transformed": sse,
        "fit_rmse_observed_scale": float(np.sqrt(np.mean((predicted_observed - observed) ** 2))),
        "predicted_at_observed_full": full_value,
        "projected_at_10x_full": ten_value,
        "projected_at_100x_full": hundred_value,
        "projected_gain_full_to_10x": ten_value - full_value,
        "projected_gain_full_to_100x": hundred_value - full_value,
    }
    return record


def predict_curve(record: Mapping, n: Sequence[float]) -> np.ndarray:
    x = np.asarray(n, dtype=float)
    family = record["curve_family"]
    shape = float(record["shape_parameter"])
    q = np.power(x, -shape) if family == "power_law" else np.exp(-x / shape)
    sign = -1.0 if record["metric"] == "pearson" else 1.0
    y = float(record["asymptote_transformed"]) + sign * float(record["amplitude"]) * q
    return np.tanh(y) if record["metric"] == "pearson" else y


def fit_curves(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fits, loo_rows = [], []
    if points.empty:
        return pd.DataFrame(), pd.DataFrame()
    for key, group in points.groupby(list(CURVE_KEYS), sort=False):
        group = group.sort_values("mean_actual_train_n")
        x = group["mean_actual_train_n"].to_numpy(float)
        for metric in ("pearson", "rmse"):
            y = group[f"mean_{metric}"].to_numpy(float)
            for family in ("power_law", "exponential"):
                record = {**dict(zip(CURVE_KEYS, key)), **fit_saturating_curve(x, y, family, metric)}
                errors = []
                for index in range(len(x)):
                    keep = np.arange(len(x)) != index
                    fitted = fit_saturating_curve(x[keep], y[keep], family, metric)
                    if fitted.get("fit_status") == "success":
                        predicted = float(predict_curve(fitted, [x[index]])[0])
                        error = predicted - float(y[index])
                        errors.append(error)
                        status = "success"
                    else:
                        predicted, error, status = math.nan, math.nan, fitted.get("fit_status", "failed")
                    loo_rows.append(
                        {
                            **dict(zip(CURVE_KEYS, key)),
                            "metric": metric,
                            "curve_family": family,
                            "omitted_n": float(x[index]),
                            "observed_value": float(y[index]),
                            "predicted_value": predicted,
                            "prediction_error": error,
                            "fit_status": status,
                        }
                    )
                finite_errors = np.asarray([value for value in errors if np.isfinite(value)])
                record["loo_success_count"] = int(len(finite_errors))
                record["loo_failure_count"] = int(len(x) - len(finite_errors))
                record["loo_mae"] = float(np.mean(np.abs(finite_errors))) if len(finite_errors) else math.nan
                record["loo_rmse"] = float(np.sqrt(np.mean(finite_errors**2))) if len(finite_errors) else math.nan
                fits.append(record)
    return pd.DataFrame(fits), pd.DataFrame(loo_rows)


def curve_family_disagreement(fits: pd.DataFrame) -> pd.DataFrame:
    if fits.empty:
        return pd.DataFrame()
    success = fits.loc[fits["fit_status"].eq("success")].copy()
    records = []
    for key, group in success.groupby(list(CURVE_KEYS) + ["metric"], sort=False):
        by_family = {row.curve_family: row for row in group.itertuples(index=False)}
        if set(by_family) != {"power_law", "exponential"}:
            continue
        power = by_family["power_law"]
        exponential = by_family["exponential"]
        records.append(
            {
                **dict(zip(CURVE_KEYS + ("metric",), key)),
                "power_law_projected_gain_full_to_10x": power.projected_gain_full_to_10x,
                "exponential_projected_gain_full_to_10x": exponential.projected_gain_full_to_10x,
                "absolute_10x_gain_disagreement": abs(
                    power.projected_gain_full_to_10x - exponential.projected_gain_full_to_10x
                ),
                "power_law_projected_gain_full_to_100x": power.projected_gain_full_to_100x,
                "exponential_projected_gain_full_to_100x": exponential.projected_gain_full_to_100x,
                "absolute_100x_gain_disagreement": abs(
                    power.projected_gain_full_to_100x - exponential.projected_gain_full_to_100x
                ),
                "power_law_loo_rmse": power.loo_rmse,
                "exponential_loo_rmse": exponential.loo_rmse,
            }
        )
    return pd.DataFrame(records)


def _metrics_from_arrays(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if len(target) != len(prediction):
        raise ValueError("Target/prediction length mismatch.")
    if len(target) == 0:
        return {metric: math.nan for metric in METRICS}
    residual = prediction - target
    denominator = float(np.sum((target - target.mean()) ** 2))
    pearson = (
        float(np.corrcoef(target, prediction)[0, 1])
        if len(target) >= 2 and np.ptp(target) > 0 and np.ptp(prediction) > 0
        else math.nan
    )
    mse = float(np.mean(residual**2))
    return {
        "pearson": pearson,
        "rmse": math.sqrt(mse),
        "cod_r2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else math.nan,
    }


def _within_stratum_centered_pearson(
    target: np.ndarray, prediction: np.ndarray, strata: np.ndarray
) -> float:
    centered_target = target.astype(float, copy=True)
    centered_prediction = prediction.astype(float, copy=True)
    for stratum in np.unique(strata):
        selected = strata == stratum
        centered_target[selected] -= centered_target[selected].mean()
        centered_prediction[selected] -= centered_prediction[selected].mean()
    return _metrics_from_arrays(centered_target, centered_prediction)["pearson"]


def paired_bootstrap(
    tracks: Mapping[tuple, pd.DataFrame],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the deterministic paired construct/subset-track bootstrap.

    The same within-fold construct indices and subset-track draw are used by
    every N/configuration for a part in a replicate.  Returned tables are
    percentile summaries, not the potentially large raw replicate matrix.
    """
    if resamples < 0:
        raise ValueError("bootstrap resamples must be nonnegative")
    if resamples == 0 or not tracks:
        return (pd.DataFrame(),) * 5
    validate_shared_oof_targets(tracks)
    rng = np.random.default_rng(seed)
    curves: dict[tuple, dict[str, list[tuple]]] = defaultdict(lambda: defaultdict(list))
    for key in tracks:
        curves[key[:3]][str(key[3])].append(key)
    for sizes in curves.values():
        for keys in sizes.values():
            keys.sort(key=lambda value: (int(value[4]), int(value[5])))

    references = {}
    fold_positions = {}
    for part in sorted({key[0] for key in tracks}):
        reference = next(frame for key, frame in tracks.items() if key[0] == part)
        references[part] = reference
        fold_positions[part] = {
            fold: np.flatnonzero(reference["outer_oof_fold"].to_numpy(int) == fold)
            for fold in EXPECTED_FOLDS
        }

    point_values: dict[tuple, list[float]] = defaultdict(list)
    contrast_values: dict[tuple, list[float]] = defaultdict(list)
    curve_values: dict[tuple, list[float]] = defaultdict(list)
    disagreement_values: dict[tuple, list[float]] = defaultdict(list)
    fit_attempts: Counter = Counter()
    fit_successes: Counter = Counter()

    for _ in range(resamples):
        indices_by_part = {}
        track_draw_by_part = {}
        for part in references:
            sampled = [
                rng.choice(positions, size=len(positions), replace=True)
                for positions in fold_positions[part].values()
            ]
            indices_by_part[part] = np.concatenate(sampled)
            track_draw_by_part[part] = rng.integers(1, 4, size=3)

        # Keys are (scope, frozen Intron stratum or "", metric).  This keeps
        # overall, centered, and per-stratum uncertainty explicit rather than
        # encoding biological strata into opaque metric names.
        metric_cache: dict[tuple, dict[tuple[str, str, str], float]] = {}
        for key, frame in tracks.items():
            indices = indices_by_part[key[0]]
            target = frame[RAW_TARGET].to_numpy(float)[indices]
            prediction = frame[RAW_PREDICTION].to_numpy(float)[indices]
            overall = _metrics_from_arrays(target, prediction)
            metric_cache[key] = {
                ("overall", "", metric): value for metric, value in overall.items()
            }
            if key[0] == "intron":
                strata = frame[SENSITIVITY_STRATUM].to_numpy()[indices]
                metric_cache[key][
                    ("within_stratum_centered", "", "pearson")
                ] = _within_stratum_centered_pearson(target, prediction, strata)
                for stratum in STRATUM_ORDER:
                    selected = strata == stratum
                    stratum_metrics = _metrics_from_arrays(
                        target[selected], prediction[selected]
                    )
                    metric_cache[key].update(
                        {
                            ("per_stratum", str(stratum), metric): value
                            for metric, value in stratum_metrics.items()
                        }
                    )

        curve_points_rep: dict[tuple, dict[object, float]] = {}
        for curve_key, sizes in curves.items():
            part = curve_key[0]
            for label, keys in sizes.items():
                by_replicate = {int(key[4]): key for key in keys}
                if set(by_replicate) == {1, 2, 3}:
                    selected = [by_replicate[int(rep)] for rep in track_draw_by_part[part]]
                else:
                    selected = keys
                descriptors = tuple(metric_cache[selected[0]])
                if any(tuple(metric_cache[key]) != descriptors for key in selected):
                    raise RuntimeError("Bootstrap metric scopes changed across paired tracks.")
                values = {
                    descriptor: float(
                        np.mean([metric_cache[key][descriptor] for key in selected])
                    )
                    for descriptor in descriptors
                }
                values["n"] = float(np.mean([
                    np.mean([int(row["expected_train_n"]) for row in tracks[key].attrs["rows"]])
                    for key in selected
                ]))
                curve_points_rep[(*curve_key, label)] = values
                for descriptor in descriptors:
                    scope, stratum, metric = descriptor
                    point_values[
                        (*curve_key, label, scope, stratum, metric)
                    ].append(values[descriptor])

            for low, high, ratio in DIRECT_CONTRASTS:
                low_values = curve_points_rep.get((*curve_key, str(low)))
                high_values = curve_points_rep.get((*curve_key, str(high)))
                if low_values is None or high_values is None:
                    continue
                descriptors = [key for key in low_values if isinstance(key, tuple)]
                if set(descriptors) != {
                    key for key in high_values if isinstance(key, tuple)
                }:
                    raise RuntimeError("Bootstrap contrast metric scopes are not paired.")
                for descriptor in descriptors:
                    scope, stratum, metric = descriptor
                    contrast_values[
                        (*curve_key, low, high, ratio, scope, stratum, metric)
                    ].append(high_values[descriptor] - low_values[descriptor])

            labels = list(sizes)
            ordered = sorted(
                [curve_points_rep[(*curve_key, label)] for label in labels],
                key=lambda value: value["n"],
            )
            x = np.asarray([value["n"] for value in ordered])
            for metric in ("pearson", "rmse"):
                descriptor = ("overall", "", metric)
                y = np.asarray([value[descriptor] for value in ordered])
                family_fits = {}
                for family in ("power_law", "exponential"):
                    fit_key = (*curve_key, metric, family)
                    fit_attempts[fit_key] += 1
                    fitted = fit_saturating_curve(x, y, family, metric)
                    if fitted.get("fit_status") != "success":
                        continue
                    fit_successes[fit_key] += 1
                    family_fits[family] = fitted
                    for field in (
                        "asymptote", "projected_gain_full_to_10x", "projected_gain_full_to_100x"
                    ):
                        curve_values[(*fit_key, field)].append(float(fitted[field]))
                if set(family_fits) == {"power_law", "exponential"}:
                    for suffix, field in (
                        ("10x", "projected_gain_full_to_10x"),
                        ("100x", "projected_gain_full_to_100x"),
                    ):
                        disagreement_values[(*curve_key, metric, suffix)].append(
                            abs(
                                float(family_fits["power_law"][field])
                                - float(family_fits["exponential"][field])
                            )
                        )

    def interval(values: Sequence[float]) -> dict:
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        return {
            "successful_bootstrap_replicates": int(len(array)),
            "failed_or_degenerate_bootstrap_replicates": int(resamples - len(array)),
            "bootstrap_mean": float(np.mean(array)) if len(array) else math.nan,
            "bootstrap_median": float(np.median(array)) if len(array) else math.nan,
            "ci_2_5": float(np.percentile(array, 2.5)) if len(array) else math.nan,
            "ci_97_5": float(np.percentile(array, 97.5)) if len(array) else math.nan,
        }

    point_rows = [
        {
            **dict(zip(CURVE_KEYS, key[:3])),
            "downsample_n_label": key[3],
            "metric_scope": key[4],
            SENSITIVITY_STRATUM: key[5],
            "metric": key[6],
            "bootstrap_resamples": resamples,
            **interval(values),
        }
        for key, values in point_values.items()
    ]
    contrast_rows = [
        {
            **dict(zip(CURVE_KEYS, key[:3])),
            "low_n": key[3], "high_n": key[4],
            "multiplicative_contrast": key[5],
            "metric_scope": key[6],
            SENSITIVITY_STRATUM: key[7],
            "metric": key[8],
            "bootstrap_resamples": resamples,
            **interval(values),
        }
        for key, values in contrast_values.items()
    ]
    curve_rows = [
        {
            **dict(zip(CURVE_KEYS, key[:3])),
            "metric": key[3], "curve_family": key[4], "quantity": key[5],
            "bootstrap_resamples": resamples,
            **interval(values),
        }
        for key, values in curve_values.items()
    ]
    failure_rows = []
    for key, attempts in fit_attempts.items():
        success = fit_successes[key]
        failure_rows.append(
            {
                **dict(zip(CURVE_KEYS, key[:3])),
                "metric": key[3], "curve_family": key[4],
                "bootstrap_fit_attempts": attempts,
                "bootstrap_fit_successes": success,
                "bootstrap_fit_failures": attempts - success,
            }
        )
    disagreement_rows = [
        {
            **dict(zip(CURVE_KEYS, key[:3])),
            "metric": key[3],
            "projection_horizon": key[4],
            "quantity": "absolute_curve_family_gain_disagreement",
            "bootstrap_resamples": resamples,
            **interval(values),
        }
        for key, values in disagreement_values.items()
    ]
    return (
        pd.DataFrame(point_rows), pd.DataFrame(contrast_rows),
        pd.DataFrame(curve_rows), pd.DataFrame(failure_rows),
        pd.DataFrame(disagreement_rows),
    )


def _primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["stage4_lane"].eq("primary")].copy()


def make_plots(
    pooled: pd.DataFrame,
    points: pd.DataFrame,
    fits: pd.DataFrame,
    contrasts: pd.DataFrame,
    intron_strata: pd.DataFrame,
    disagreement: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    if points.empty:
        return []
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.7), sharey=True)
    for part, axis in zip(PART_ORDER, axes):
        subset = _primary_rows(points.loc[points["part_slug"].eq(part)])
        tracks = _primary_rows(pooled.loc[pooled["part_slug"].eq(part)])
        for _, track in tracks.groupby("subset_replicate"):
            axis.plot(track["mean_actual_train_n"], track["pearson"], color="#4C78A8", alpha=0.25)
        axis.plot(subset["mean_actual_train_n"], subset["mean_pearson"], "o-", color="#1F4E79", label="observed")
        fit_subset = fits.loc[
            fits["part_slug"].eq(part) & fits["stage4_lane"].eq("primary")
            & fits["metric"].eq("pearson") & fits["fit_status"].eq("success")
        ]
        if not subset.empty:
            x = np.geomspace(subset["mean_actual_train_n"].min(), subset["mean_actual_train_n"].max(), 150)
            for _, row in fit_subset.iterrows():
                axis.plot(x, predict_curve(row, x), linestyle="--", label=row["curve_family"])
        axis.set_xscale("log")
        axis.set_title(PART_LABELS[part])
        axis.set_xlabel("Unique training constructs (log scale)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Pooled five-fold OOF Pearson")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Stage 4 primary learning curves (development OOF only)")
    fig.tight_layout()
    path = figure_dir / "stage4_primary_learning_curves.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex="col")
    for column, part in enumerate(PART_ORDER):
        subset = _primary_rows(points.loc[points["part_slug"].eq(part)])
        for row, metric, label in ((0, "rmse", "Raw RMSE"), (1, "cod_r2", "Raw COD R²")):
            axes[row, column].plot(subset["mean_actual_train_n"], subset[f"mean_{metric}"], "o-", color="#4C78A8")
            axes[row, column].set_xscale("log")
            axes[row, column].grid(alpha=0.25)
            if column == 0:
                axes[row, column].set_ylabel(label)
        axes[0, column].set_title(PART_LABELS[part])
        axes[1, column].set_xlabel("Unique training constructs")
    fig.suptitle("Raw-scale calibration guardrails")
    fig.tight_layout()
    path = figure_dir / "stage4_primary_calibration_guardrails.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=True)
    for part, axis in zip(PART_ORDER, axes):
        subset = points.loc[points["part_slug"].eq(part)]
        for (lane, config), group in subset.groupby(["stage4_lane", "base_config_id"]):
            style = "-" if lane == "primary" else "--"
            axis.plot(group["mean_actual_train_n"], group["mean_pearson"], "o" + style, label=f"{lane}: {config[8:16]}")
        axis.set_xscale("log")
        axis.set_title(PART_LABELS[part])
        axis.set_xlabel("Unique training constructs")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6)
    axes[0].set_ylabel("Pooled OOF Pearson")
    fig.suptitle("Configuration-sensitivity anchors (not reselection)")
    fig.tight_layout()
    path = figure_dir / "stage4_configuration_sensitivity.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    written.append(path)

    if not contrasts.empty:
        primary = contrasts.loc[
            contrasts["stage4_lane"].eq("primary") & contrasts["multiplicative_contrast"].eq("10x")
        ].copy()
        if not primary.empty:
            fig, axis = plt.subplots(figsize=(10, 4.5))
            labels, positions = [], []
            for index, (key, group) in enumerate(primary.groupby(["part_slug", "low_n", "high_n"], sort=False)):
                x = np.full(len(group), index, dtype=float) + np.linspace(-0.12, 0.12, len(group))
                axis.scatter(x, group["delta_pearson"], alpha=0.8)
                axis.plot([index - 0.18, index + 0.18], [group["delta_pearson"].mean()] * 2, color="black")
                labels.append(f"{PART_LABELS[key[0]]}\n{key[1]}→{key[2]}")
                positions.append(index)
            axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
            axis.set_xticks(positions, labels, rotation=35, ha="right")
            axis.set_ylabel("Observed Δ pooled OOF Pearson")
            axis.set_title("Direct paired 10× sample-size contrasts")
            axis.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            path = figure_dir / "stage4_observed_10x_contrasts.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            written.append(path)

    if not intron_strata.empty:
        primary = intron_strata.loc[intron_strata["stage4_lane"].eq("primary")]
        fig, axis = plt.subplots(figsize=(7.5, 4.5))
        for stratum, group in primary.groupby(SENSITIVITY_STRATUM):
            summarized = group.groupby("mean_actual_train_n", as_index=False)["pearson"].mean()
            axis.plot(summarized["mean_actual_train_n"], summarized["pearson"], "o-", label=stratum)
        axis.set_xscale("log")
        axis.set_xlabel("Unique training constructs")
        axis.set_ylabel("Within-stratum OOF Pearson")
        axis.set_title("Intron inferred-mask stratum learning curves")
        axis.legend()
        axis.grid(alpha=0.25)
        fig.tight_layout()
        path = figure_dir / "stage4_intron_stratum_learning_curves.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    if not disagreement.empty:
        subset = disagreement.loc[disagreement["metric"].eq("pearson") & disagreement["stage4_lane"].eq("primary")]
        if not subset.empty:
            fig, axis = plt.subplots(figsize=(7, 4))
            axis.bar(
                [PART_LABELS[value] for value in subset["part_slug"]],
                subset["absolute_10x_gain_disagreement"], color="#E45756",
            )
            axis.set_ylabel("|power-law Δ10× − exponential Δ10×|")
            axis.set_title("Projection sensitivity to tail-shape assumption")
            axis.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            path = figure_dir / "stage4_curve_family_disagreement.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            written.append(path)
    return written


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--portfolio", type=Path, default=DEFAULT_PORTFOLIO)
    parser.add_argument("--manifest-summary", type=Path, default=DEFAULT_MANIFEST_SUMMARY)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--intron-strata", type=Path, default=DEFAULT_INTRON_STRATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--readiness-only", action="store_true")
    return parser.parse_args(argv)


def run_analysis(args: argparse.Namespace) -> dict:
    verifier_report = validate_frozen_manifest(args)
    registry_path = validate_registry_isolation(args.registry)
    manifest_rows = read_jsonl(args.manifest)
    rows = resolve_cells(manifest_rows, registry_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = completion_table(rows)
    write_frame(completion, output_dir / "stage4_completion.csv")
    complete_count = int((completion["availability"] == "complete").sum())
    readiness = {
        "analysis_mode": "readiness_only" if args.readiness_only else "completed_oof_only",
        "manifest_validation_status": verifier_report["status"],
        "manifest_sha256": verifier_report["manifest_sha256"],
        "manifest_rows": len(rows),
        "completed_cells": complete_count,
        "remaining_cells": len(rows) - complete_count,
        "availability_counts": dict(sorted(Counter(completion["availability"]).items())),
        "dedicated_registry_path": str(registry_path),
        "global_registry_read": False,
        "final_test_loader_instantiated": False,
        "final_test_products_read": False,
        "final_test_metrics_computed": False,
    }
    if args.require_complete and complete_count != len(rows):
        write_json(output_dir / "stage4_readiness.json", readiness)
        raise RuntimeError(f"Stage 4 is incomplete: {complete_count}/{len(rows)} cells.")
    if args.require_complete and (
        args.skip_bootstrap or int(args.bootstrap_resamples) != BOOTSTRAP_RESAMPLES
    ):
        raise RuntimeError(
            f"A complete contract analysis requires exactly {BOOTSTRAP_RESAMPLES} "
            "bootstrap replicates."
        )
    if args.readiness_only or complete_count == 0:
        write_json(output_dir / "stage4_readiness.json", readiness)
        return readiness

    stratum_map, stratum_source = load_intron_strata(args.intron_strata)
    per_run, predictions = score_completed_cells(rows, stratum_map)
    training_diagnostics = summarize_training_diagnostics(per_run)
    tracks, track_readiness = assemble_complete_tracks(rows, predictions)
    if tracks:
        validate_shared_oof_targets(tracks)
    pooled, intron_strata = score_pooled_tracks(tracks)
    points = curve_points(pooled)
    if complete_count == len(rows):
        if len(tracks) != EXPECTED_POOLED_TRACKS:
            raise RuntimeError(
                f"Complete Stage 4 has {len(tracks)} pooled tracks; "
                f"expected {EXPECTED_POOLED_TRACKS}."
            )
        if len(points) != EXPECTED_CURVE_POINTS or len(training_diagnostics) != EXPECTED_CURVE_POINTS:
            raise RuntimeError(
                "Complete Stage 4 does not have all required per-config/N curve points "
                "and training diagnostics."
            )
    contrast_detail, contrast_summary = observed_paired_contrasts(tracks, pooled)
    fits, loo = fit_curves(points)
    disagreement = curve_family_disagreement(fits)

    products = {
        "stage4_run_metrics.csv": per_run,
        "stage4_training_diagnostics.csv": training_diagnostics,
        "stage4_track_readiness.csv": track_readiness,
        "stage4_pooled_oof_metrics.csv": pooled,
        "stage4_curve_points.csv": points,
        "stage4_observed_contrasts.csv": contrast_detail,
        "stage4_observed_contrast_summary.csv": contrast_summary,
        "stage4_intron_stratum_metrics.csv": intron_strata,
        "stage4_curve_fits.csv": fits,
        "stage4_curve_leave_one_size_out.csv": loo,
        "stage4_curve_family_disagreement.csv": disagreement,
    }
    for name, frame in products.items():
        write_frame(frame, output_dir / name)

    bootstrap_resamples = 0 if args.skip_bootstrap else int(args.bootstrap_resamples)
    bootstrap_products = paired_bootstrap(
        tracks, resamples=bootstrap_resamples, seed=int(args.bootstrap_seed)
    )
    for name, frame in zip(
        (
            "stage4_bootstrap_metric_intervals.csv",
            "stage4_bootstrap_contrast_intervals.csv",
            "stage4_bootstrap_curve_intervals.csv",
            "stage4_bootstrap_fit_failures.csv",
            "stage4_bootstrap_curve_family_disagreement_intervals.csv",
        ),
        bootstrap_products,
    ):
        write_frame(frame, output_dir / name)

    figures = make_plots(
        pooled, points, fits, contrast_detail, intron_strata, disagreement, output_dir
    )
    complete_tracks = len(tracks)
    readiness.update(
        complete_pooled_tracks=complete_tracks,
        curve_point_rows=len(points),
        observed_contrast_rows=len(contrast_detail),
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=int(args.bootstrap_seed),
        intron_strata_source=stratum_source,
        figures=[str(path.resolve()) for path in figures],
    )
    write_json(output_dir / "stage4_readiness.json", readiness)
    write_json(
        output_dir / "stage4_analysis_contract.json",
        {
            "schema_version": "lib1_dedup_stage4_downsampling_analysis_v1",
            "primary_estimand": "pooled_five_fold_development_oof_pearson",
            "raw_guardrails": ["rmse", "cod_r2"],
            "direct_contrasts": [
                {"low_n": low, "high_n": high, "ratio": ratio}
                for low, high, ratio in DIRECT_CONTRASTS
            ],
            "curve_families": ["fisher_z_power_law", "fisher_z_exponential_saturation"],
            "curve_role": "secondary_projection_sensitivity_not_selection",
            "bootstrap": {
                "resamples": bootstrap_resamples,
                "seed": int(args.bootstrap_seed),
                "construct_resampling": "within_outer_fold_paired_across_all_N_tracks_configs",
                "subset_track_resampling": "paired_with_replacement_for_three_track_curves",
                "intron_metric_scopes": [
                    "overall",
                    "within_stratum_centered",
                    "per_frozen_inferred_mask_stratum",
                ],
                "curve_family_disagreement": (
                    "recomputed_within_each_replicate_then_percentile_summarized"
                ),
            },
            "training_diagnostics": {
                "best_validation_value": "best_metric_value_after_val_pearson_monitor_check",
                "gap": "best_checkpoint_train_pearson_minus_best_inner_validation_pearson",
                "optimizer_steps": "total_lightning_optimizer_updates_during_fit",
                "required_for_every_completed_cell": True,
            },
            "registry": {
                "path": str(registry_path),
                "scope": "dedicated_stage4_campaign_only",
                "global_registry_read": False,
            },
            "final_test_loader_instantiated": False,
            "final_test_products_read": False,
        },
    )
    return readiness


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_analysis(args)
    print("Lib1 dedup Stage 4 development-only analysis")
    print(f"  completed cells: {result['completed_cells']}/{result['manifest_rows']}")
    print(f"  pooled tracks: {result.get('complete_pooled_tracks', 0)}")
    print(f"  output: {Path(args.output_dir).resolve()}")
    print("  final-test products read: false")


if __name__ == "__main__":
    main()
