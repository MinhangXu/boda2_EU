#!/usr/bin/env python3
"""Contract-first analysis of the July 2026 targeted 3'UTR HPO.

This program reconciles the frozen 240-cell manifest against the run registry,
local compact provenance, local W&B summaries, and validation-only prediction
exports.  It constructs no DataModule and never loads or scores an audit
target.  The immutable split JSON is read only to verify the expected
development-fold IDs.

The primary estimand and the one-standard-error procedure are those frozen in
``lib1_dedup_targeted_utr3_hpo_protocol_amendment_july14_2026.md``.  The
program also recomputes the 40 historical Stage 2 3'UTR arms from the frozen
Stage 2 OOF prediction product, retaining their original provenance labels.
It produces a Stage 3 *review* table, but deliberately does not freeze a
selection or generate a Stage 3 launch manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.lib1_dedup_stage2_analysis import (  # noqa: E402
    _calibration,
    compare_paired_rc,
    raw_metrics,
)
from src.learn.run_lib1_dedup_utr3_targeted_hpo_campaign import (  # noqa: E402
    EXPECTED_ENTITY,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_PROJECT,
    EXPECTED_ROWS,
    EXPECTED_VAL_COUNT,
    EXPECTED_VAL_HASHES,
    TEST_METRIC_FIELDS,
    expected_registry_fields,
    validate_completed_record,
)


LEARN_ROOT = REPO_ROOT / "src" / "learn"
TARGETED_MANIFEST = (
    LEARN_ROOT
    / "outputs/hpo_manifests/lib1_dedup_utr3_targeted_hpo_july2026__dry_run_manifest.jsonl"
)
TARGETED_REGISTRY = LEARN_ROOT / "run_registry/runs.csv"
LOCAL_WANDB_ROOT = LEARN_ROOT / "wandb"
TARGETED_STATUS_ROOT = (
    LEARN_ROOT / "outputs/hpo_runs/status/lib1_dedup_utr3_targeted_hpo_july2026"
)
STAGE2_MANIFEST = (
    LEARN_ROOT
    / "outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl"
)
STAGE2_ANALYSIS_ROOT = LEARN_ROOT / "outputs/analysis/lib1_dedup_stage2_july2026"
STAGE2_OOF = STAGE2_ANALYSIS_ROOT / "stage2_oof_predictions.tsv.gz"
STAGE2_METRICS = STAGE2_ANALYSIS_ROOT / "stage2_oof_metrics.csv"
STAGE2_HISTORY = (
    STAGE2_ANALYSIS_ROOT / "reporting/stage2_learning_history_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    LEARN_ROOT / "outputs/analysis/lib1_dedup_utr3_targeted_hpo_july2026"
)

CAMPAIGN_STAGE = "targeted_utr3_hpo"
CAMPAIGN_ID = "lib1_dedup_phase1_rerun_july2026"
ANALYSIS_LANE = "utr3_utrbasset_targeted_hpo"
EXPECTED_CONFIGS = 24
EXPECTED_ARMS = 48
EXPECTED_RC_PAIRS = 24
EXPECTED_OOF_CONSTRUCTS = 525
EXPECTED_FOLDS = frozenset(range(5))
EXPECTED_LR = frozenset((0.001, 0.002, 0.004, 0.006))
EXPECTED_WEIGHT_DECAY = frozenset((0.0001, 0.0007, 0.003))
EXPECTED_DROPOUT = frozenset((0.35, 0.50))
EXPECTED_WANDB_GROUP = (
    "lib1_dedup_phase1_rerun_july2026__targeted_utr3_hpo__full_oof_rc"
)
EXPECTED_WANDB_JOB_TYPE = "targeted_utr3_hpo_cell"
BOOTSTRAP_SEED = 20260714
BOOTSTRAP_RESAMPLES = 10_000
TARGET_COLUMN = "log2_RNA_DNA"
PREDICTION_COLUMN = "prediction_raw"


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_registry(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_targeted_manifest(
    rows: Sequence[dict], manifest_path: str | Path = TARGETED_MANIFEST
) -> None:
    """Fail closed unless the manifest is the exact frozen 24x5x2 grid."""
    observed_sha = sha256_file(manifest_path)
    if observed_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "Targeted manifest SHA changed: {} != {}".format(
                observed_sha, EXPECTED_MANIFEST_SHA256
            )
        )
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("Expected {} rows; found {}".format(EXPECTED_ROWS, len(rows)))
    if len({row["cell_id"] for row in rows}) != EXPECTED_ROWS:
        raise ValueError("Targeted manifest cell IDs are not unique.")
    if len({row["planned_run_name"] for row in rows}) != EXPECTED_ROWS:
        raise ValueError("Targeted manifest run names are not unique.")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    configs: dict[str, dict] = {}
    observed_grid = set()
    for row in rows:
        if row.get("campaign_id") != CAMPAIGN_ID:
            raise ValueError("Unexpected campaign_id in targeted manifest.")
        if row.get("campaign_stage") != CAMPAIGN_STAGE:
            raise ValueError("Unexpected campaign_stage in targeted manifest.")
        if row.get("analysis_lane") != ANALYSIS_LANE:
            raise ValueError("Unexpected targeted analysis lane.")
        if row.get("part_slug") != "utr3" or row.get("architecture") != "UTR_BassetVL":
            raise ValueError("Targeted row is not a 3'UTR UTRBassetVL cell.")
        if row.get("wandb_entity") != EXPECTED_ENTITY:
            raise ValueError("Unexpected targeted W&B entity.")
        if row.get("logger_project") != EXPECTED_PROJECT:
            raise ValueError("Unexpected targeted W&B project.")
        if row.get("wandb_group") != EXPECTED_WANDB_GROUP:
            raise ValueError("Unexpected targeted W&B group.")
        if row.get("wandb_job_type") != EXPECTED_WANDB_JOB_TYPE:
            raise ValueError("Unexpected targeted W&B job type.")
        if bool(row.get("evaluate_test_after_fit")):
            raise ValueError("Targeted manifest enables test evaluation.")
        if list(row.get("epoch_eval_splits", [])) != ["train", "val"]:
            raise ValueError("Targeted epoch splits are not exactly train/val.")
        if list(row.get("prediction_splits", [])) != ["val"]:
            raise ValueError("Targeted prediction splits are not val-only.")
        if row.get("artifact_retention") != "none":
            raise ValueError("Targeted manifest retains an artifact.")
        if row.get("loss_mode") != "unweighted_mse":
            raise ValueError("Targeted search is not unweighted MSE.")
        if int(row["development_fold"]) not in EXPECTED_FOLDS:
            raise ValueError("Unexpected development fold.")
        if row.get("rc_mode") not in {"off", "on"}:
            raise ValueError("Unexpected RC mode.")
        grouped[(row["base_config_id"], row["rc_mode"])].append(row)

        search = row["search_values"]
        identity = row["base_identity"]
        lr = float(search["lr"])
        wd = float(search["weight_decay"])
        dropout = float(search["linear_dropout_p"])
        if not math.isclose(lr, float(identity["lr"]), rel_tol=0, abs_tol=0):
            raise ValueError("Search LR does not match base identity.")
        if not math.isclose(wd, float(identity["weight_decay"]), rel_tol=0, abs_tol=0):
            raise ValueError("Search weight decay does not match base identity.")
        if not math.isclose(
            dropout, float(identity["linear_dropout_p"]), rel_tol=0, abs_tol=0
        ):
            raise ValueError("Search dropout does not match base identity.")
        observed_grid.add((lr, wd, dropout))
        configs.setdefault(row["base_config_id"], search)

    expected_grid = {
        (lr, wd, dropout)
        for lr in EXPECTED_LR
        for wd in EXPECTED_WEIGHT_DECAY
        for dropout in EXPECTED_DROPOUT
    }
    if observed_grid != expected_grid or len(configs) != EXPECTED_CONFIGS:
        raise ValueError("Targeted hyperparameter grid differs from the frozen grid.")
    if len(grouped) != EXPECTED_ARMS:
        raise ValueError("Expected {} targeted arms; found {}".format(EXPECTED_ARMS, len(grouped)))
    for key, arm_rows in grouped.items():
        folds = {int(row["development_fold"]) for row in arm_rows}
        if len(arm_rows) != 5 or folds != EXPECTED_FOLDS:
            raise ValueError("Targeted arm {} lacks exact folds 0..4.".format(key))


def local_wandb_summaries(
    expected_cells: set[str], expected_run_ids: set[str]
) -> pd.DataFrame:
    """Read the compact local W&B summaries for the targeted campaign only."""
    records = []
    for path in LOCAL_WANDB_ROOT.glob("run-*/files/wandb-summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("campaign_stage") != CAMPAIGN_STAGE:
            continue
        run_id = str(payload.get("run_id_recorded", ""))
        cell_id = str(payload.get("cell_id", ""))
        records.append(
            {
                "cell_id": cell_id,
                "run_id": run_id,
                "local_wandb_summary_path": str(path.resolve()),
                "fit_wall_time_seconds": payload.get("fit_wall_time_seconds"),
                "wandb_runtime_seconds": payload.get("_runtime"),
                "stopping_epoch": payload.get("epoch"),
                "model_parameter_count_wandb": payload.get("model_parameter_count"),
                "model_trainable_parameter_count_wandb": payload.get(
                    "model_trainable_parameter_count"
                ),
                "val_predictions_path_wandb": payload.get("val_predictions_path", ""),
                "val_predictions_sha256_wandb": payload.get(
                    "val_predictions_sha256", ""
                ),
                "compact_provenance_path_wandb": payload.get(
                    "compact_provenance_path", ""
                ),
                "compact_provenance_sha256_wandb": payload.get(
                    "compact_provenance_sha256", ""
                ),
                "resolved_wandb_entity": payload.get("resolved_wandb_entity", ""),
                "resolved_wandb_project": payload.get("resolved_wandb_project", ""),
                "resolved_wandb_run_url": payload.get("resolved_wandb_run_url", ""),
                "evaluate_test_after_fit_wandb": payload.get(
                    "evaluate_test_after_fit"
                ),
                "model_artifact_retained": payload.get("model_artifact_retained"),
                "pruned_lightning_checkpoint_count": payload.get(
                    "pruned_lightning_checkpoint_count"
                ),
            }
        )
    frame = pd.DataFrame(records)
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            "Expected {} targeted local W&B summaries; found {}".format(
                EXPECTED_ROWS, len(frame)
            )
        )
    if frame["cell_id"].duplicated().any() or frame["run_id"].duplicated().any():
        raise ValueError("Targeted local W&B summaries are not one-to-one.")
    if set(frame["cell_id"]) != expected_cells:
        raise ValueError("Local W&B summary cells do not match the manifest.")
    if set(frame["run_id"]) != expected_run_ids:
        raise ValueError("Local W&B summary run IDs do not match the registry.")
    return frame


def model_parameter_count(architecture: str, identity: Mapping) -> int:
    """Instantiate a CPU model from its frozen identity and count parameters."""
    from boda import model as boda_model

    cls = getattr(boda_model, architecture)
    signature = inspect.signature(cls)
    kwargs = {}
    for name in signature.parameters:
        if name in {"loss_criterion", "loss_args"}:
            continue
        value = identity.get(name)
        if value is not None:
            kwargs[name] = value
    instance = cls(**kwargs)
    return int(sum(parameter.numel() for parameter in instance.parameters()))


def development_ids(split_payload: Mapping, fold: int | None = None) -> set[str]:
    ids = set()
    for assignment in split_payload["assignments"]:
        if assignment.get("partition") != "development":
            continue
        if fold is not None and int(assignment["development_fold"]) != int(fold):
            continue
        ids.add(str(assignment["construct_id"]))
    return ids


def reconcile_targeted_cells(
    manifest_rows: Sequence[dict], registry_path: str | Path
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    """Resolve all cells and return cell, arm-prediction, and W&B tables."""
    registry_rows = read_registry(registry_path)
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for record in registry_rows:
        if record.get("cell_id"):
            by_cell[record["cell_id"]].append(record)

    split_cache: dict[str, dict] = {}
    split_development_cache: dict[str, set[str]] = {}
    parameter_cache: dict[str, int] = {}
    cells = []
    arm_pieces: dict[tuple[str, str], list[pd.DataFrame]] = defaultdict(list)
    resolved_run_ids = set()

    for row in sorted(manifest_rows, key=lambda item: int(item["manifest_row"])):
        candidates = by_cell.get(row["cell_id"], [])
        if len(candidates) != 1:
            raise ValueError(
                "Cell {} has {} registry rows; expected exactly one.".format(
                    row["cell_id"], len(candidates)
                )
            )
        record = candidates[0]
        mismatches = {
            field: (record.get(field, ""), expected)
            for field, expected in expected_registry_fields(row).items()
            if record.get(field, "") != expected
        }
        if mismatches:
            raise ValueError(
                "Registry provenance mismatch for {}: {}".format(
                    row["cell_id"], mismatches
                )
            )
        if record.get("status", "").lower() != "completed":
            raise ValueError("Cell {} is not completed.".format(row["cell_id"]))
        validate_completed_record(row, record)
        run_id = record["run_id"]
        if run_id in resolved_run_ids:
            raise ValueError("Targeted registry run ID is reused: {}".format(run_id))
        resolved_run_ids.add(run_id)

        prediction_path = Path(record["prediction_path"])
        provenance_path = (
            Path(row["default_root_dir"])
            / "provenance"
            / "{}__run_provenance.json".format(run_id)
        )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        split_summary = provenance["data_split_summary"]
        if split_summary.get("n_test") != 0:
            raise ValueError("A targeted provenance record instantiated a test set.")

        split_path = str(Path(row["split_manifest_path"]).resolve())
        if split_path not in split_cache:
            if sha256_file(split_path) != row["split_manifest_sha256"]:
                raise ValueError("Frozen split manifest hash changed.")
            split_cache[split_path] = json.loads(
                Path(split_path).read_text(encoding="utf-8")
            )
            split_development_cache[split_path] = development_ids(split_cache[split_path])
        fold = int(row["development_fold"])
        expected_ids = development_ids(split_cache[split_path], fold)
        if len(expected_ids) != EXPECTED_VAL_COUNT:
            raise ValueError("Frozen fold does not contain 105 development IDs.")

        prediction = pd.read_csv(prediction_path, sep="\t")
        required = {"construct_id", TARGET_COLUMN, PREDICTION_COLUMN}
        if not required.issubset(prediction.columns):
            raise ValueError("Prediction export lacks required raw columns.")
        prediction["construct_id"] = prediction["construct_id"].astype(str)
        if prediction["construct_id"].duplicated().any():
            raise ValueError("Prediction export contains duplicate construct IDs.")
        if set(prediction["construct_id"]) != expected_ids:
            raise ValueError("Prediction IDs do not match the frozen development fold.")
        for column in (TARGET_COLUMN, PREDICTION_COLUMN):
            prediction[column] = pd.to_numeric(prediction[column], errors="coerce")
            if not np.isfinite(prediction[column].to_numpy(float)).all():
                raise ValueError("Prediction export contains non-finite {}.".format(column))

        identity_key = json.dumps(
            {"architecture": row["architecture"], "identity": row["base_identity"]},
            sort_keys=True,
        )
        if identity_key not in parameter_cache:
            parameter_cache[identity_key] = model_parameter_count(
                row["architecture"], row["base_identity"]
            )
        parameter_count = parameter_cache[identity_key]

        predictions = prediction[["construct_id", TARGET_COLUMN, PREDICTION_COLUMN]].copy()
        predictions["development_fold"] = fold
        predictions["cell_id"] = row["cell_id"]
        predictions["run_id"] = run_id
        predictions["base_config_id"] = row["base_config_id"]
        predictions["rc_mode"] = row["rc_mode"]
        arm_pieces[(row["base_config_id"], row["rc_mode"])].append(predictions)

        search = row["search_values"]
        prediction_values = prediction[PREDICTION_COLUMN].to_numpy(float)
        nonblank_test_metrics = sum(
            bool(str(record.get(field, "")).strip()) for field in TEST_METRIC_FIELDS
        )
        cells.append(
            {
                "manifest_row": int(row["manifest_row"]),
                "cell_id": row["cell_id"],
                "run_id": run_id,
                "base_config_id": row["base_config_id"],
                "base_config_prefix": row["base_config_id"].replace("basecfg_", "")[:8],
                "search_config_index": int(row["search_config_index"]),
                "search_config_label": row["search_config_label"],
                "development_fold": fold,
                "rc_mode": row["rc_mode"],
                "lr": float(search["lr"]),
                "weight_decay": float(search["weight_decay"]),
                "linear_dropout_p": float(search["linear_dropout_p"]),
                "status": record["status"],
                "val_prediction_rows": int(len(prediction)),
                "val_row_id_hash": record["val_row_id_hash"],
                "expected_val_row_id_hash": EXPECTED_VAL_HASHES[fold],
                "prediction_path": str(prediction_path.resolve()),
                "prediction_sha256": sha256_file(prediction_path),
                "provenance_path": str(provenance_path.resolve()),
                "provenance_sha256": sha256_file(provenance_path),
                "registry_run_url": record["run_url"],
                "local_checkpoint_file_count": sum(
                    1 for _ in Path(row["default_root_dir"]).rglob("*.ckpt")
                ),
                "local_test_prediction_file_count": sum(
                    1
                    for path in Path(row["default_root_dir"]).rglob("*prediction*")
                    if path.is_file() and "test" in path.name.lower()
                ),
                "provenance_n_test": int(split_summary["n_test"]),
                "provenance_n_val": int(split_summary["n_val"]),
                "nonblank_test_metric_count": nonblank_test_metrics,
                "best_epoch": int(record["best_epoch"]),
                "registry_train_mse": float(record["train_mse"]),
                "registry_val_pearson": float(record["val_pearson"]),
                "prediction_mean": float(np.mean(prediction_values)),
                "prediction_std": float(np.std(prediction_values, ddof=0)),
                "prediction_range": float(np.ptp(prediction_values)),
                "prediction_n_unique": int(pd.Series(prediction_values).nunique()),
                "constant_prediction": bool(pd.Series(prediction_values).nunique() == 1),
                "model_parameter_count": parameter_count,
                "audit_loader_instantiated": False,
                "audit_targets_loaded": False,
                "audit_predictions_scored": False,
            }
        )

    cell_frame = pd.DataFrame(cells)
    wandb = local_wandb_summaries(
        set(cell_frame["cell_id"]), set(cell_frame["run_id"])
    )
    cell_frame = cell_frame.merge(
        wandb, on=["cell_id", "run_id"], how="left", validate="one_to_one"
    )
    for row in cell_frame.itertuples(index=False):
        if Path(row.val_predictions_path_wandb).resolve() != Path(row.prediction_path):
            raise ValueError("Local W&B prediction path differs from registry provenance.")
        if row.val_predictions_sha256_wandb != row.prediction_sha256:
            raise ValueError("Local W&B prediction SHA differs from the local file.")
        if Path(row.compact_provenance_path_wandb).resolve() != Path(row.provenance_path):
            raise ValueError("Local W&B provenance path differs from the local file.")
        if row.compact_provenance_sha256_wandb != row.provenance_sha256:
            raise ValueError("Local W&B provenance SHA differs from the local file.")
        if row.resolved_wandb_entity != EXPECTED_ENTITY:
            raise ValueError("Local W&B summary has the wrong entity.")
        if row.resolved_wandb_project != EXPECTED_PROJECT:
            raise ValueError("Local W&B summary has the wrong project.")
        if row.resolved_wandb_run_url != row.registry_run_url:
            raise ValueError("Local W&B run URL differs from the registry.")
        if bool(row.evaluate_test_after_fit_wandb):
            raise ValueError("Local W&B summary says test evaluation was enabled.")
        if bool(row.model_artifact_retained):
            raise ValueError("Local W&B summary says a model artifact was retained.")
        if int(row.model_parameter_count_wandb) != int(row.model_parameter_count):
            raise ValueError("Independent model parameter count disagrees with W&B.")
        if int(row.local_checkpoint_file_count) != 0:
            raise ValueError("Targeted cell retained a local checkpoint.")
        if int(row.local_test_prediction_file_count) != 0:
            raise ValueError("Targeted cell retained a test prediction.")

    arms = {}
    for key, pieces in sorted(arm_pieces.items()):
        if len(pieces) != 5:
            raise ValueError("Targeted OOF arm does not have exactly five fold pieces.")
        frame = pd.concat(pieces, ignore_index=True)
        if frame["construct_id"].duplicated().any():
            raise ValueError("Targeted OOF arm predicts a construct more than once.")
        split_path = str(Path(manifest_rows[0]["split_manifest_path"]).resolve())
        if set(frame["construct_id"]) != split_development_cache[split_path]:
            raise ValueError("Targeted OOF arm does not cover the development set.")
        if len(frame) != EXPECTED_OOF_CONSTRUCTS:
            raise ValueError("Targeted OOF arm does not contain 525 predictions.")
        arms[key] = frame.sort_values("construct_id").reset_index(drop=True)

    # Prove RC mates use exactly the same IDs, fold labels, and raw targets.
    for base_config_id in sorted({key[0] for key in arms}):
        off = arms[(base_config_id, "off")]
        on = arms[(base_config_id, "on")]
        if not off["construct_id"].equals(on["construct_id"]):
            raise ValueError("Targeted RC mates contain different construct IDs.")
        if not off["development_fold"].equals(on["development_fold"]):
            raise ValueError("Targeted RC mates contain different fold labels.")
        if not np.array_equal(
            off[TARGET_COLUMN].to_numpy(float), on[TARGET_COLUMN].to_numpy(float)
        ):
            raise ValueError("Targeted RC mates contain different targets.")
    return cell_frame, arms, wandb


def base_metadata(manifest_rows: Sequence[dict]) -> dict[str, dict]:
    metadata = {}
    for row in manifest_rows:
        config_id = row["base_config_id"]
        if config_id in metadata:
            continue
        search = row.get("search_values", {})
        identity = row["base_identity"]
        metadata[config_id] = {
            "base_config_id": config_id,
            "base_config_prefix": config_id.replace("basecfg_", "")[:8],
            "architecture": row["architecture"],
            "analysis_lane": row.get("analysis_lane", ""),
            "policy_id": row.get("policy_id", config_id),
            "search_config_index": row.get("search_config_index"),
            "search_config_label": row.get("search_config_label", ""),
            "lr": identity.get("lr", search.get("lr")),
            "weight_decay": identity.get("weight_decay", search.get("weight_decay")),
            "linear_dropout_p": identity.get(
                "linear_dropout_p", search.get("linear_dropout_p")
            ),
            "dropout_p": identity.get("dropout_p"),
            "base_identity": identity,
            "model_parameter_count": model_parameter_count(row["architecture"], identity),
        }
    return metadata


def score_arms(
    arms: Mapping[tuple[str, str], pd.DataFrame],
    metadata: Mapping[str, Mapping],
    cell_evidence: pd.DataFrame,
    portfolio_source: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pooled, fold, calibration, convergence, and collapse diagnostics."""
    arm_rows = []
    fold_rows = []
    for (config_id, rc_mode), frame in sorted(arms.items()):
        info = dict(metadata[config_id])
        pooled = raw_metrics(frame)
        calibration = _calibration(
            frame[TARGET_COLUMN].to_numpy(float),
            frame[PREDICTION_COLUMN].to_numpy(float),
        )
        centered = frame.copy()
        centered["target_fold_centered"] = centered[TARGET_COLUMN] - centered.groupby(
            "development_fold"
        )[TARGET_COLUMN].transform("mean")
        centered["prediction_fold_centered"] = centered[PREDICTION_COLUMN] - centered.groupby(
            "development_fold"
        )[PREDICTION_COLUMN].transform("mean")
        within_fold = raw_metrics(
            centered, "target_fold_centered", "prediction_fold_centered"
        )

        for fold, fold_frame in frame.groupby("development_fold", sort=True):
            metrics = raw_metrics(fold_frame)
            values = fold_frame[PREDICTION_COLUMN].to_numpy(float)
            fold_rows.append(
                {
                    "portfolio_source": portfolio_source,
                    **{key: value for key, value in info.items() if key != "base_identity"},
                    "rc_mode": rc_mode,
                    "development_fold": int(fold),
                    **{"fold_{}".format(key): value for key, value in metrics.items()},
                    "prediction_std": float(np.std(values, ddof=0)),
                    "prediction_range": float(np.ptp(values)),
                    "constant_prediction": bool(pd.Series(values).nunique() == 1),
                }
            )

        fold_subset = pd.DataFrame(fold_rows)
        fold_subset = fold_subset.loc[
            fold_subset["portfolio_source"].eq(portfolio_source)
            & fold_subset["base_config_id"].eq(config_id)
            & fold_subset["rc_mode"].eq(rc_mode)
        ]
        fold_pearson = fold_subset["fold_pearson"].dropna()
        evidence = cell_evidence.loc[
            cell_evidence["base_config_id"].eq(config_id)
            & cell_evidence["rc_mode"].eq(rc_mode)
        ]
        arm_rows.append(
            {
                "portfolio_source": portfolio_source,
                **{key: value for key, value in info.items() if key != "base_identity"},
                "rc_mode": rc_mode,
                "primary_metric_name": "pooled_five_fold_oof_pearson",
                **{"pooled_oof_{}".format(key): value for key, value in pooled.items()},
                **calibration,
                "within_fold_centered_pearson": within_fold["pearson"],
                "within_fold_centered_spearman": within_fold["spearman"],
                "fold_pearson_mean": float(fold_pearson.mean()),
                "fold_pearson_sd": (
                    float(fold_pearson.std(ddof=1)) if len(fold_pearson) > 1 else math.nan
                ),
                "fold_pearson_min": (
                    float(fold_pearson.min()) if len(fold_pearson) == 5 else math.nan
                ),
                "fold_pearson_median": float(fold_pearson.median()),
                "fold_pearson_p20": float(fold_pearson.quantile(0.20)),
                "finite_fold_pearson_count": int(len(fold_pearson)),
                "constant_prediction_fold_count": int(
                    fold_subset["constant_prediction"].sum()
                ),
                "best_epoch_median": float(evidence["best_epoch"].median()),
                "best_epoch_min": int(evidence["best_epoch"].min()),
                "best_epoch_max": int(evidence["best_epoch"].max()),
                "stopping_epoch_median": float(evidence["stopping_epoch"].median()),
                "fit_wall_time_seconds_total": float(
                    evidence["fit_wall_time_seconds"].sum()
                ),
                "fit_wall_time_seconds_median": float(
                    evidence["fit_wall_time_seconds"].median()
                ),
            }
        )
    return pd.DataFrame(arm_rows), pd.DataFrame(fold_rows)


def stage2_comparators(
    targeted_reference: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str], pd.DataFrame],
    dict[str, dict],
]:
    """Recompute all 40 Stage 2 3'UTR arms from the frozen OOF product."""
    manifest = [row for row in read_jsonl(STAGE2_MANIFEST) if row["part_slug"] == "utr3"]
    metadata = base_metadata(manifest)
    prediction = pd.read_csv(STAGE2_OOF, sep="\t", low_memory=False)
    prediction = prediction.loc[prediction["part_slug"].eq("utr3")].copy()
    if len(prediction) != 40 * EXPECTED_OOF_CONSTRUCTS:
        raise ValueError("Stage 2 3'UTR OOF product has unexpected row count.")
    arms = {}
    for (config_id, rc_mode), frame in prediction.groupby(
        ["base_config_id", "rc_mode"], sort=True
    ):
        frame = frame[
            [
                "construct_id",
                TARGET_COLUMN,
                PREDICTION_COLUMN,
                "development_fold",
                "cell_id",
                "resolved_run_id",
            ]
        ].copy()
        frame = frame.rename(columns={"resolved_run_id": "run_id"})
        frame["construct_id"] = frame["construct_id"].astype(str)
        if len(frame) != EXPECTED_OOF_CONSTRUCTS or frame["construct_id"].duplicated().any():
            raise ValueError("Stage 2 comparator arm is not an exact 525-row OOF arm.")
        reference = targeted_reference.sort_values("construct_id").reset_index(drop=True)
        check = frame.sort_values("construct_id").reset_index(drop=True)
        if not check["construct_id"].equals(reference["construct_id"]):
            raise ValueError("Stage 2 and targeted OOF construct IDs differ.")
        if not np.array_equal(
            check[TARGET_COLUMN].to_numpy(float),
            reference[TARGET_COLUMN].to_numpy(float),
        ):
            raise ValueError("Stage 2 and targeted OOF targets differ.")
        arms[(config_id, rc_mode)] = check

    histories = pd.read_csv(STAGE2_HISTORY)
    histories = histories.loc[histories["part_slug"].eq("utr3")].copy()
    histories = histories.rename(
        columns={
            "resolved_run_id": "run_id",
            "history_best_epoch": "best_epoch",
            "runtime_seconds": "fit_wall_time_seconds",
        }
    )
    histories["stopping_epoch"] = histories["stopping_epoch"].astype(float)
    histories["model_parameter_count"] = histories["base_config_id"].map(
        {config: info["model_parameter_count"] for config, info in metadata.items()}
    )
    stage2_arms, stage2_folds = score_arms(
        arms, metadata, histories, portfolio_source="stage2_pending_label"
    )
    stage2_arms["portfolio_source"] = np.where(
        stage2_arms["architecture"].eq("UTR_BassetVL"),
        "stage2_utrbasset_challenger",
        "stage2_resnet_core",
    )
    stage2_folds["portfolio_source"] = np.where(
        stage2_folds["architecture"].eq("UTR_BassetVL"),
        "stage2_utrbasset_challenger",
        "stage2_resnet_core",
    )

    canonical = pd.read_csv(STAGE2_METRICS)
    canonical = canonical.loc[canonical["part_slug"].eq("utr3")]
    check = stage2_arms.merge(
        canonical[
            [
                "base_config_id",
                "rc_mode",
                "pooled_oof_pearson",
                "pooled_oof_spearman",
                "pooled_oof_rmse",
                "pooled_oof_cod_r2",
            ]
        ],
        on=["base_config_id", "rc_mode"],
        suffixes=("", "_canonical"),
        validate="one_to_one",
    )
    for metric in ("pearson", "spearman", "rmse", "cod_r2"):
        delta = (
            check["pooled_oof_{}".format(metric)]
            - check["pooled_oof_{}_canonical".format(metric)]
        ).abs()
        if float(delta.max()) > 1e-12:
            raise ValueError("Recomputed Stage 2 {} differs from canonical output.".format(metric))
    return stage2_arms, stage2_folds, arms, metadata


def stratified_bootstrap_indices(frame: pd.DataFrame) -> list[np.ndarray]:
    # ``frame`` is sorted by construct_id before this function is called.
    # Preserve that canonical within-fold order so the frozen seed is stable
    # even if prediction TSV or registry row order changes.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    strata = [
        frame.index[frame["development_fold"].astype(int).eq(fold)].to_numpy()
        for fold in sorted(EXPECTED_FOLDS)
    ]
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        samples.append(
            np.concatenate(
                [
                    indices[rng.integers(0, len(indices), size=len(indices))]
                    for indices in strata
                ]
            )
        )
    return samples


def pearson_array(target: np.ndarray, prediction: np.ndarray) -> float:
    if np.ptp(target) == 0 or np.ptp(prediction) == 0:
        return math.nan
    return float(np.corrcoef(target, prediction)[0, 1])


def bootstrap_selection(
    targeted_metrics: pd.DataFrame,
    targeted_arms: Mapping[tuple[str, str], pd.DataFrame],
    stage2_metrics: pd.DataFrame,
    stage2_arms: Mapping[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply the frozen one-SE rule and compare the winner with the incumbent."""
    best = targeted_metrics.sort_values(
        ["pooled_oof_pearson", "pooled_oof_rmse", "base_config_id"],
        ascending=[False, True, True],
    ).iloc[0]
    best_key = (best["base_config_id"], best["rc_mode"])
    best_frame = targeted_arms[best_key].sort_values("construct_id").reset_index(drop=True)
    indices = stratified_bootstrap_indices(best_frame)
    target = best_frame[TARGET_COLUMN].to_numpy(float)
    best_prediction = best_frame[PREDICTION_COLUMN].to_numpy(float)
    best_samples = np.asarray(
        [pearson_array(target[index], best_prediction[index]) for index in indices],
        dtype=float,
    )
    bootstrap_se = float(np.std(best_samples, ddof=1))
    threshold = float(best["pooled_oof_pearson"] - bootstrap_se)

    ranked = targeted_metrics.copy()
    ranked["numeric_targeted_rank"] = ranked["pooled_oof_pearson"].rank(
        ascending=False, method="first"
    ).astype(int)
    ranked["one_se_threshold"] = threshold
    ranked["within_one_se"] = ranked["pooled_oof_pearson"].ge(threshold)
    near = ranked.loc[ranked["within_one_se"]].copy()
    near["_fold_min_sort"] = near["fold_pearson_min"].fillna(-np.inf)
    near = near.sort_values(
        [
            "_fold_min_sort",
            "pooled_oof_rmse",
            "pooled_oof_cod_r2",
            "lr",
            "weight_decay",
            "linear_dropout_p",
            "base_config_id",
            "rc_mode",
        ],
        ascending=[False, True, False, True, True, False, True, True],
    )
    near["one_se_tiebreak_rank"] = np.arange(1, len(near) + 1)
    preferred = near.iloc[0]
    ranked = ranked.merge(
        near[["base_config_id", "rc_mode", "one_se_tiebreak_rank"]],
        on=["base_config_id", "rc_mode"],
        how="left",
        validate="one_to_one",
    )
    ranked["preferred_targeted_one_se_arm"] = (
        ranked["base_config_id"].eq(preferred["base_config_id"])
        & ranked["rc_mode"].eq(preferred["rc_mode"])
    )

    incumbent = stage2_metrics.loc[
        stage2_metrics["portfolio_source"].eq("stage2_utrbasset_challenger")
    ].sort_values("pooled_oof_pearson", ascending=False).iloc[0]
    incumbent_frame = stage2_arms[
        (incumbent["base_config_id"], incumbent["rc_mode"])
    ].sort_values("construct_id").reset_index(drop=True)
    if not incumbent_frame["construct_id"].equals(best_frame["construct_id"]):
        raise ValueError("Winner/incumbent bootstrap frames are not construct-paired.")
    incumbent_prediction = incumbent_frame[PREDICTION_COLUMN].to_numpy(float)
    incumbent_samples = np.asarray(
        [pearson_array(target[index], incumbent_prediction[index]) for index in indices],
        dtype=float,
    )
    delta_samples = best_samples - incumbent_samples
    samples = pd.DataFrame(
        {
            "bootstrap_resample": np.arange(1, BOOTSTRAP_RESAMPLES + 1),
            "targeted_best_pearson": best_samples,
            "stage2_incumbent_pearson": incumbent_samples,
            "delta_targeted_best_minus_incumbent": delta_samples,
        }
    )
    summary = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "bootstrap_rng": "numpy.random.default_rng",
        "bootstrap_bit_generator": "PCG64",
        "bootstrap_within_fold_sort": "construct_id_ascending",
        "bootstrap_standard_error_ddof": 1,
        "bootstrap_design": "construct resampling with replacement within each development fold",
        "targeted_numerical_best_base_config_id": best["base_config_id"],
        "targeted_numerical_best_rc_mode": best["rc_mode"],
        "targeted_numerical_best_pearson": float(best["pooled_oof_pearson"]),
        "targeted_numerical_best_bootstrap_se": bootstrap_se,
        "targeted_numerical_best_bootstrap_mean": float(np.mean(best_samples)),
        "targeted_numerical_best_bootstrap_ci95_low": float(
            np.quantile(best_samples, 0.025)
        ),
        "targeted_numerical_best_bootstrap_ci95_high": float(
            np.quantile(best_samples, 0.975)
        ),
        "targeted_one_se_threshold": threshold,
        "targeted_one_se_arm_count": int(ranked["within_one_se"].sum()),
        "targeted_one_se_config_count": int(
            ranked.loc[ranked["within_one_se"], "base_config_id"].nunique()
        ),
        "preferred_targeted_base_config_id": preferred["base_config_id"],
        "preferred_targeted_rc_mode": preferred["rc_mode"],
        "preferred_targeted_pearson": float(preferred["pooled_oof_pearson"]),
        "preferred_targeted_min_fold_pearson": float(preferred["fold_pearson_min"]),
        "stage2_incumbent_base_config_id": incumbent["base_config_id"],
        "stage2_incumbent_rc_mode": incumbent["rc_mode"],
        "stage2_incumbent_pearson": float(incumbent["pooled_oof_pearson"]),
        "observed_delta_targeted_best_minus_incumbent": float(
            best["pooled_oof_pearson"] - incumbent["pooled_oof_pearson"]
        ),
        "bootstrap_delta_ci95_low": float(np.quantile(delta_samples, 0.025)),
        "bootstrap_delta_ci95_high": float(np.quantile(delta_samples, 0.975)),
        "bootstrap_probability_delta_gt_zero": float(np.mean(delta_samples > 0)),
        "comparison_status": (
            "descriptive_selection_biased_same_oof_rows_not_confirmatory"
        ),
    }
    return ranked, samples, summary


def targeted_rc_tables(
    targeted_arms: Mapping[tuple[str, str], pd.DataFrame],
    metadata: Mapping[str, Mapping],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    adapter = {}
    for (config_id, rc_mode), frame in targeted_arms.items():
        copy = frame.copy()
        copy["analysis_lane"] = ANALYSIS_LANE
        copy["challenger_family"] = "utr3_utrbasset_optimizer_regularization"
        copy["config_origin"] = "targeted_grid_20260714"
        copy["training_regime"] = "scratch"
        copy["part_slug"] = "utr3"
        copy["architecture"] = "UTR_BassetVL"
        copy["base_config_id"] = config_id
        copy["policy_id"] = config_id
        copy["initialization"] = "scratch"
        copy["source_head"] = ""
        copy["unfreeze_scope"] = ""
        copy["input_policy"] = "exact100_v1"
        copy["rc_mode"] = rc_mode
        adapter[(ANALYSIS_LANE, "utr3", config_id, rc_mode)] = copy
    summaries, fold_summaries, _construct_errors = compare_paired_rc(adapter)
    hyper = pd.DataFrame(
        [
            {
                "base_config_id": config_id,
                "base_config_prefix": info["base_config_prefix"],
                "search_config_index": info["search_config_index"],
                "lr": info["lr"],
                "weight_decay": info["weight_decay"],
                "linear_dropout_p": info["linear_dropout_p"],
            }
            for config_id, info in metadata.items()
        ]
    )
    summaries = summaries.merge(hyper, on="base_config_id", validate="one_to_one")
    fold_summaries = fold_summaries.merge(
        hyper, on="base_config_id", validate="many_to_one"
    )
    return summaries, fold_summaries


def factor_summary(metrics: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in ("lr", "weight_decay", "linear_dropout_p"):
        for (rc_mode, level), frame in metrics.groupby(["rc_mode", factor], sort=True):
            relevant_cells = cells.loc[
                cells["rc_mode"].eq(rc_mode) & cells[factor].eq(level)
            ]
            rows.append(
                {
                    "factor": factor,
                    "level": float(level),
                    "rc_mode": rc_mode,
                    "n_arms": int(len(frame)),
                    "mean_pooled_oof_pearson": float(frame["pooled_oof_pearson"].mean()),
                    "median_pooled_oof_pearson": float(frame["pooled_oof_pearson"].median()),
                    "max_pooled_oof_pearson": float(frame["pooled_oof_pearson"].max()),
                    "mean_pooled_oof_rmse": float(frame["pooled_oof_rmse"].mean()),
                    "constant_prediction_cell_count": int(
                        relevant_cells["constant_prediction"].sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def combined_review(
    targeted_ranked: pd.DataFrame,
    stage2_metrics: pd.DataFrame,
    one_se_threshold: float,
    preferred_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([targeted_ranked, stage2_metrics], ignore_index=True, sort=False)
    combined["combined_arm_rank"] = combined["pooled_oof_pearson"].rank(
        ascending=False, method="first"
    ).astype(int)
    combined["within_combined_best_one_se_band"] = combined["pooled_oof_pearson"].ge(
        one_se_threshold
    )
    combined = combined.sort_values(
        ["pooled_oof_pearson", "pooled_oof_rmse", "base_config_id", "rc_mode"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)

    best_config = combined.groupby("base_config_id", sort=False, as_index=False).head(1).copy()
    best_config = best_config.sort_values(
        ["pooled_oof_pearson", "pooled_oof_rmse", "base_config_id"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    best_config["combined_base_config_rank"] = np.arange(1, len(best_config) + 1)
    best_config["pure_pooled_top5"] = best_config["combined_base_config_rank"].le(5)
    best_config["balanced_review_portfolio"] = False
    best_config["balanced_review_reason"] = "not_in_balanced_review_set"

    targeted = best_config.loc[best_config["portfolio_source"].eq("targeted_20260714")]
    numeric = targeted.iloc[0]
    incumbent = best_config.loc[
        best_config["portfolio_source"].eq("stage2_utrbasset_challenger")
    ].iloc[0]
    next_targeted = targeted.loc[
        ~targeted["base_config_id"].eq(numeric["base_config_id"])
    ].iloc[0]
    preferred = best_config.loc[best_config["base_config_id"].eq(preferred_id)].iloc[0]
    used = {
        numeric["base_config_id"],
        incumbent["base_config_id"],
        next_targeted["base_config_id"],
        preferred["base_config_id"],
    }
    secondary_pool = targeted.loc[
        targeted["within_combined_best_one_se_band"]
        & ~targeted["base_config_id"].isin(used)
    ].sort_values(
        ["pooled_oof_cod_r2", "pooled_oof_rmse", "fold_pearson_sd", "base_config_id"],
        ascending=[False, True, True, True],
    )
    secondary = secondary_pool.iloc[0]
    review = [
        (numeric, "targeted_numerical_winner"),
        (incumbent, "stage2_incumbent_error_calibration_anchor"),
        (preferred, "preregistered_targeted_one_se_stability_choice"),
        (secondary, "near_best_secondary_metric_and_fold_stability_choice"),
        (next_targeted, "next_highest_distinct_targeted_pooled_performer"),
    ]
    for row, reason in review:
        index = best_config.index[
            best_config["base_config_id"].eq(row["base_config_id"])
        ][0]
        best_config.loc[index, "balanced_review_portfolio"] = True
        best_config.loc[index, "balanced_review_reason"] = reason
    best_config["portfolio_decision_status"] = "requires_dated_full_id_freeze"
    best_config["stage3_weighted_cells_if_selected"] = 10
    best_config["unweighted_reuse_cells_if_selected"] = 10
    best_config["stage3_trains_both_rc_modes"] = True
    return combined, best_config


def flatten_predictions(
    targeted_arms: Mapping[tuple[str, str], pd.DataFrame],
    stage2_arms: Mapping[tuple[str, str], pd.DataFrame],
    stage2_metadata: Mapping[str, Mapping],
) -> pd.DataFrame:
    records = []
    for source, arms in (
        ("targeted_20260714", targeted_arms),
        ("stage2", stage2_arms),
    ):
        for (config_id, rc_mode), frame in arms.items():
            copy = frame.copy()
            if source == "stage2":
                source_label = (
                    "stage2_utrbasset_challenger"
                    if stage2_metadata[config_id]["architecture"] == "UTR_BassetVL"
                    else "stage2_resnet_core"
                )
                architecture = stage2_metadata[config_id]["architecture"]
            else:
                source_label = source
                architecture = "UTR_BassetVL"
            copy["portfolio_source"] = source_label
            copy["architecture"] = architecture
            copy["base_config_id"] = config_id
            copy["rc_mode"] = rc_mode
            records.append(copy)
    columns = [
        "portfolio_source",
        "architecture",
        "base_config_id",
        "rc_mode",
        "development_fold",
        "cell_id",
        "run_id",
        "construct_id",
        TARGET_COLUMN,
        PREDICTION_COLUMN,
    ]
    return pd.concat(records, ignore_index=True)[columns]


def write_frame(frame: pd.DataFrame, path: Path, sep: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, sep=sep)


def run_analysis(
    manifest_path: str | Path = TARGETED_MANIFEST,
    registry_path: str | Path = TARGETED_REGISTRY,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    require_complete: bool = True,
) -> dict:
    if not require_complete:
        raise ValueError("This frozen post-campaign analysis requires all 240 cells.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_jsonl(manifest_path)
    validate_targeted_manifest(manifest_rows, manifest_path)
    cells, targeted_arms, wandb = reconcile_targeted_cells(manifest_rows, registry_path)
    targeted_metadata = base_metadata(manifest_rows)
    targeted_metrics, targeted_folds = score_arms(
        targeted_arms, targeted_metadata, cells, "targeted_20260714"
    )
    stage2_metrics, stage2_folds, stage2_arms, stage2_metadata = stage2_comparators(
        next(iter(targeted_arms.values()))
    )
    targeted_ranked, bootstrap_samples, bootstrap_summary = bootstrap_selection(
        targeted_metrics, targeted_arms, stage2_metrics, stage2_arms
    )
    rc_pairs, rc_fold_pairs = targeted_rc_tables(targeted_arms, targeted_metadata)
    factors = factor_summary(targeted_ranked, cells)
    combined, config_review = combined_review(
        targeted_ranked,
        stage2_metrics,
        bootstrap_summary["targeted_one_se_threshold"],
        bootstrap_summary["preferred_targeted_base_config_id"],
    )
    combined_folds = pd.concat([targeted_folds, stage2_folds], ignore_index=True, sort=False)
    predictions = flatten_predictions(targeted_arms, stage2_arms, stage2_metadata)

    if len(cells) != EXPECTED_ROWS or not cells["status"].eq("completed").all():
        raise ValueError("Targeted campaign completion invariant failed.")
    if len(targeted_ranked) != EXPECTED_ARMS or len(rc_pairs) != EXPECTED_RC_PAIRS:
        raise ValueError("Targeted arm/RC accounting invariant failed.")
    if len(targeted_folds) != EXPECTED_ARMS * 5:
        raise ValueError("Targeted fold-metric accounting invariant failed.")
    if len(combined) != 88 or combined["base_config_id"].nunique() != 44:
        raise ValueError("Combined Stage 2/targeted portfolio accounting failed.")
    if len(predictions) != 88 * EXPECTED_OOF_CONSTRUCTS:
        raise ValueError("Combined OOF prediction accounting failed.")
    if int(cells["nonblank_test_metric_count"].sum()) != 0:
        raise ValueError("Targeted registry contains a test metric.")
    if not cells["provenance_n_test"].eq(0).all():
        raise ValueError("Targeted provenance contains a test set.")

    numeric = targeted_ranked.loc[
        targeted_ranked["base_config_id"].eq(
            bootstrap_summary["targeted_numerical_best_base_config_id"]
        )
        & targeted_ranked["rc_mode"].eq(
            bootstrap_summary["targeted_numerical_best_rc_mode"]
        )
    ].iloc[0]
    preferred = targeted_ranked.loc[targeted_ranked["preferred_targeted_one_se_arm"]].iloc[0]
    incumbent = stage2_metrics.loc[
        stage2_metrics["base_config_id"].eq(
            bootstrap_summary["stage2_incumbent_base_config_id"]
        )
        & stage2_metrics["rc_mode"].eq(bootstrap_summary["stage2_incumbent_rc_mode"])
    ].iloc[0]
    best_resnet = stage2_metrics.loc[
        stage2_metrics["portfolio_source"].eq("stage2_resnet_core")
    ].sort_values("pooled_oof_pearson", ascending=False).iloc[0]

    summary = {
        "analysis_status": "complete_development_only_stage3_selection_not_frozen",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "targeted_cells_completed": int(len(cells)),
        "completion_evidence": {
            "completed_registry_rows": int(len(cells)),
            "local_wandb_summaries": int(len(wandb)),
            "launcher_done_markers": int(
                len(list((TARGETED_STATUS_ROOT / "done").glob("row_*.done")))
            ),
            "pilot_row_1_done_marker_present": bool(
                (TARGETED_STATUS_ROOT / "done/row_1.done").is_file()
            ),
            "completion_source_of_truth": (
                "exact_manifest_registry_prediction_provenance_wandb_reconciliation"
            ),
        },
        "targeted_configs": int(targeted_ranked["base_config_id"].nunique()),
        "targeted_arms": int(len(targeted_ranked)),
        "targeted_fold_metric_rows": int(len(targeted_folds)),
        "targeted_rc_pairs": int(len(rc_pairs)),
        "oof_constructs_per_arm": EXPECTED_OOF_CONSTRUCTS,
        "combined_cells_represented": 440,
        "combined_configs": int(combined["base_config_id"].nunique()),
        "combined_arms": int(len(combined)),
        "combined_oof_prediction_rows": int(len(predictions)),
        "constant_prediction_cells": int(cells["constant_prediction"].sum()),
        "constant_prediction_cells_rc_on": int(
            cells.loc[cells["rc_mode"].eq("on"), "constant_prediction"].sum()
        ),
        "constant_prediction_cells_lr_0p006": int(
            cells.loc[cells["lr"].eq(0.006), "constant_prediction"].sum()
        ),
        "targeted_total_fit_wall_time_seconds": float(
            cells["fit_wall_time_seconds"].sum()
        ),
        "targeted_total_fit_wall_time_gpu_hours": float(
            cells["fit_wall_time_seconds"].sum() / 3600.0
        ),
        "targeted_median_cell_fit_wall_time_seconds": float(
            cells["fit_wall_time_seconds"].median()
        ),
        "targeted_model_parameter_count": int(cells["model_parameter_count"].iloc[0]),
        "retained_local_checkpoint_files": int(
            cells["local_checkpoint_file_count"].sum()
        ),
        "local_test_prediction_files": int(
            cells["local_test_prediction_file_count"].sum()
        ),
        "wandb_organization": {
            "entity": EXPECTED_ENTITY,
            "project": EXPECTED_PROJECT,
            "group": EXPECTED_WANDB_GROUP,
            "job_type": EXPECTED_WANDB_JOB_TYPE,
            "local_summaries_reconciled": int(len(wandb)),
        },
        "numerical_targeted_winner": {
            "base_config_id": numeric["base_config_id"],
            "rc_mode": numeric["rc_mode"],
            "lr": float(numeric["lr"]),
            "weight_decay": float(numeric["weight_decay"]),
            "linear_dropout_p": float(numeric["linear_dropout_p"]),
            "pooled_oof_pearson": float(numeric["pooled_oof_pearson"]),
            "pooled_oof_spearman": float(numeric["pooled_oof_spearman"]),
            "pooled_oof_rmse": float(numeric["pooled_oof_rmse"]),
            "pooled_oof_cod_r2": float(numeric["pooled_oof_cod_r2"]),
            "fold_pearson_min": float(numeric["fold_pearson_min"]),
        },
        "preferred_targeted_one_se_arm": {
            "base_config_id": preferred["base_config_id"],
            "rc_mode": preferred["rc_mode"],
            "pooled_oof_pearson": float(preferred["pooled_oof_pearson"]),
            "pooled_oof_rmse": float(preferred["pooled_oof_rmse"]),
            "pooled_oof_cod_r2": float(preferred["pooled_oof_cod_r2"]),
            "fold_pearson_min": float(preferred["fold_pearson_min"]),
        },
        "stage2_incumbent": {
            "base_config_id": incumbent["base_config_id"],
            "rc_mode": incumbent["rc_mode"],
            "pooled_oof_pearson": float(incumbent["pooled_oof_pearson"]),
            "pooled_oof_spearman": float(incumbent["pooled_oof_spearman"]),
            "pooled_oof_rmse": float(incumbent["pooled_oof_rmse"]),
            "pooled_oof_cod_r2": float(incumbent["pooled_oof_cod_r2"]),
            "fold_pearson_min": float(incumbent["fold_pearson_min"]),
        },
        "best_stage2_resnet": {
            "base_config_id": best_resnet["base_config_id"],
            "rc_mode": best_resnet["rc_mode"],
            "pooled_oof_pearson": float(best_resnet["pooled_oof_pearson"]),
        },
        "rc_evidence": {
            "configs_where_rc_on_has_higher_pooled_pearson": int(
                (
                    rc_pairs["delta_rc_on_minus_off_pooled_oof_pearson"] > 0
                ).sum()
            ),
            "mean_pooled_pearson_delta_on_minus_off": float(
                rc_pairs["delta_rc_on_minus_off_pooled_oof_pearson"].mean()
            ),
            "configs_passing_formal_pearson_gate": int(
                rc_pairs[
                    "formal_pearson_fold_gate_mean_ge_0p005_and_positive_ge_4"
                ].sum()
            ),
            "configs_passing_formal_gate_and_zero_tolerance_error_guard": int(
                rc_pairs[
                    "formal_pearson_gate_and_zero_tolerance_error_guard"
                ].sum()
            ),
            "interpretation": "rc_off_is_the_unweighted_default_but_stage3_still_crosses_both_rc_states",
        },
        "bootstrap": bootstrap_summary,
        "stage3_review": {
            "pure_pooled_top5_full_ids": config_review.loc[
                config_review["pure_pooled_top5"], "base_config_id"
            ].tolist(),
            "balanced_review_full_ids": config_review.loc[
                config_review["balanced_review_portfolio"], "base_config_id"
            ].tolist(),
            "selection_status": "requires_dated_full_id_freeze",
            "new_weighted_cells_after_five_are_frozen": 50,
            "unweighted_reuse_cells_after_five_are_frozen": 50,
            "stage3_manifest_generated": False,
        },
        "audit_isolation": {
            "audit_loader_instantiated": False,
            "audit_targets_loaded": False,
            "audit_predictions_scored": False,
            "trainer_test_called": False,
            "test_metrics_present": False,
        },
    }

    write_frame(cells, output_dir / "utr3_targeted_hpo_cell_completion.csv")
    write_frame(wandb, output_dir / "utr3_targeted_hpo_local_wandb_summaries.csv")
    write_frame(targeted_ranked, output_dir / "utr3_targeted_hpo_arm_metrics.csv")
    write_frame(targeted_folds, output_dir / "utr3_targeted_hpo_fold_metrics.csv")
    write_frame(rc_pairs, output_dir / "utr3_targeted_hpo_rc_pair_metrics.csv")
    write_frame(rc_fold_pairs, output_dir / "utr3_targeted_hpo_rc_fold_pair_metrics.csv")
    write_frame(factors, output_dir / "utr3_targeted_hpo_factor_summary.csv")
    write_frame(combined, output_dir / "utr3_targeted_hpo_combined_arm_metrics.csv")
    write_frame(combined_folds, output_dir / "utr3_targeted_hpo_combined_fold_metrics.csv")
    write_frame(config_review, output_dir / "utr3_targeted_hpo_stage3_config_review.csv")
    write_frame(
        predictions,
        output_dir / "utr3_targeted_hpo_combined_oof_predictions.tsv.gz",
        sep="\t",
    )
    write_frame(
        bootstrap_samples,
        output_dir / "utr3_targeted_hpo_bootstrap_samples.csv.gz",
    )
    (output_dir / "utr3_targeted_hpo_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(TARGETED_MANIFEST))
    parser.add_argument("--registry", default=str(TARGETED_REGISTRY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require all 240 cells (the only supported frozen post-campaign mode).",
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
