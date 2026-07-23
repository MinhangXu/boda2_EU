#!/usr/bin/env python3
"""One-time, allowlist-bound audit scorer for the five frozen Lib1 policies."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shlex
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

import boda
from boda.common import utils
from src.analysis.lib1_dedup_intron_sensitivity_reporting import (
    STRATUM,
    STRATUM_ORDER,
    intron_estimands,
)
from src.analysis.lib1_dedup_stage1_analysis import assign_inferred_intron_subsets


REPO = Path(__file__).resolve().parents[2]
LEARN = REPO / "src/learn"
OUT_DIR = LEARN / "outputs/audit/lib1_dedup_final_audit_july2026"
DEFAULT_ALLOWLIST = OUT_DIR / "lib1_dedup_final_refit_checkpoint_allowlist.json"
AMENDMENT = (
    REPO / "plan/phase1_lib1/dedup_phase1_rerun_july2026"
    / "lib1_dedup_final_refit_and_audit_protocol_amendment_july16_2026.md"
)
EXPECTED_AMENDMENT_SHA256 = "ff5ca5765f15c270ee33a7098dfb18646c426140be132c538aaff8ba003ec686"
RECONCILIATION = (
    REPO / "plan/phase1_lib1/dedup_phase1_rerun_july2026"
    / "lib1_dedup_final_refit_implementation_reconciliation_july16_2026.md"
)
EXPECTED_RECONCILIATION_SHA256 = (
    "07dc683d292a75cdef228af6065d6f14264f6588a4235f1b9c7f51ba72ee8620"
)
FINAL_REFIT_MANIFEST = (
    LEARN / "outputs/hpo_manifests"
    / "lib1_dedup_final_refit_july2026__dry_run_manifest.jsonl"
)
EXPECTED_FINAL_REFIT_MANIFEST_SHA256 = (
    "83ec532cf84e83d3477f2e6e8c716a04284fcc43b7d7c4426338a8b0f093582c"
)
EXPECTED_AUDIT_N = {"enhancer": 250, "promoter": 386, "intron": 265, "utr3": 250, "utr5": 359}
PREDICTOR_ORDER = ("ensemble_mean", "seed_1701", "seed_1702", "seed_1703")
PRODUCTS_DIR = OUT_DIR / "frozen_products"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def append_jsonl(path: Path, value: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def option_value(command: str, option: str) -> str:
    tokens = shlex.split(command)
    matches = [tokens[index + 1] for index, token in enumerate(tokens[:-1]) if token == option]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {option} in frozen command")
    return matches[0]


def safe_pearson(observed, predicted, suppress_below_n: int | None = None) -> tuple[float, str]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if suppress_below_n is not None and len(observed) < suppress_below_n:
        return math.nan, f"suppressed_n_lt_{suppress_below_n}"
    if len(observed) < 2:
        return math.nan, "undefined_n_lt_2"
    if np.ptp(observed) == 0:
        return math.nan, "undefined_zero_observed_variance"
    if np.ptp(predicted) == 0:
        return math.nan, "undefined_zero_prediction_variance"
    return float(np.corrcoef(observed, predicted)[0, 1]), ""


def metrics(observed, predicted, suppress_pearson_below_n: int | None = None) -> dict:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(observed) != len(predicted) or len(observed) == 0:
        raise ValueError("Observed/predicted arrays must have equal positive length")
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Observed/predicted arrays must contain only finite values")
    pearson, pearson_reason = safe_pearson(
        observed, predicted, suppress_below_n=suppress_pearson_below_n
    )
    spearman = (
        math.nan
        if len(observed) < 2 or np.ptp(observed) == 0 or np.ptp(predicted) == 0
        else float(spearmanr(observed, predicted).correlation)
    )
    residual = predicted - observed
    sst = float(np.sum((observed - observed.mean()) ** 2))
    cod = math.nan if sst == 0 else 1.0 - float(np.sum(residual ** 2)) / sst
    if len(observed) >= 2 and np.ptp(predicted) > 0:
        slope, intercept = np.polyfit(predicted, observed, 1)
    else:
        slope, intercept = math.nan, math.nan
    return {
        "n": int(len(observed)),
        "pearson": pearson,
        "pearson_reason": pearson_reason,
        "spearman": spearman,
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "cod_r2": cod,
        "bias_prediction_minus_observed": float(np.mean(residual)),
        "observed_mean": float(np.mean(observed)),
        "prediction_mean": float(np.mean(predicted)),
        "calibration_slope_observed_on_prediction": float(slope),
        "calibration_intercept_observed_on_prediction": float(intercept),
    }


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model_class = getattr(boda.model, checkpoint["model_module"])
    model = model_class(**vars(checkpoint["model_hparams"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def build_audit_data(allowlist_row: dict):
    checkpoint = torch.load(allowlist_row["checkpoint_path"], map_location="cpu")
    data_hparams = vars(checkpoint["data_hparams"]).copy()
    data_hparams.update(
        manifest_mode="audit_eval",
        num_workers=0,
        train_size_frac=1.0,
        train_size_n=None,
        train_max_barcodes=None,
    )
    data_class = getattr(boda.data, checkpoint["data_module"])
    data = data_class(**data_hparams)
    data.setup()
    return data


def predict(model, loader, device: torch.device) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            output = model(x.to(device))
            if isinstance(output, (tuple, list)):
                output = output[0]
            chunks.append(output.detach().cpu().reshape(len(x), -1))
    result = torch.cat(chunks, dim=0).numpy()
    if result.shape[1] != 1:
        raise ValueError(f"Expected one model output, found shape {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Model emitted a nonfinite audit prediction")
    return result[:, 0]


def checkpoint_preflight(row: dict) -> None:
    checkpoint_path = Path(row["checkpoint_path"])
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    for key in ("model_module", "graph_module", "data_module"):
        if checkpoint.get(key) != row.get(key):
            raise ValueError(f"Checkpoint {key} differs for {row['part_slug']}/seed{row['model_seed']}")
    data_hparams = vars(checkpoint["data_hparams"])
    expected_data = {
        "datafile_path": row["dataset_path"],
        "split_manifest_path": row["split_manifest_path"],
        "expected_data_sha256": row["dataset_sha256"],
        "expected_split_sha256": row["split_manifest_sha256"],
        "manifest_mode": "final_refit",
        "train_min_barcodes": 1,
        "train_max_barcodes": None,
        "train_size_frac": 1.0,
        "train_size_n": None,
        "use_reverse_complements": row["rc_mode"] == "on",
        "barcode_weighting": row["loss_mode"] == "barcode_weighted_mse",
    }
    for key, value in expected_data.items():
        if data_hparams.get(key) != value:
            raise ValueError(
                f"Checkpoint data_hparams.{key} differs for "
                f"{row['part_slug']}/seed{row['model_seed']}"
            )
    model = load_model(checkpoint_path, torch.device("cpu"))
    del model


def runtime_preflight(allowlist: dict, device_name: str, technical_retry: bool) -> None:
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA audit device requested but CUDA is unavailable")
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA audit device index {index} is unavailable")
        probe = torch.zeros(1, device=device)
        del probe
    for row in allowlist["rows"]:
        checkpoint_preflight(row)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    probe_path = OUT_DIR / f".audit_write_preflight_{os.getpid()}"
    probe_path.write_text("pre-audit write preflight\n")
    probe_path.unlink()
    if PRODUCTS_DIR.exists():
        raise RuntimeError(
            "A published frozen audit-products directory already exists; repeat scoring is forbidden"
        )


def verify_allowlist(path: Path, expected_sha: str) -> dict:
    if sha256_file(path) != expected_sha:
        raise ValueError("Checkpoint allowlist SHA256 mismatch")
    payload = json.loads(path.read_text())
    rows = payload.get("rows", [])
    if len(rows) != 15:
        raise ValueError(f"Expected 15 checkpoint rows, found {len(rows)}")
    if payload.get("audit_loader_instantiated") is not False:
        raise ValueError("Allowlist was not frozen pre-audit")
    if sha256_file(RECONCILIATION) != EXPECTED_RECONCILIATION_SHA256:
        raise ValueError("Frozen implementation reconciliation changed")
    if payload.get("implementation_reconciliation_sha256") != EXPECTED_RECONCILIATION_SHA256:
        raise ValueError("Allowlist implementation reconciliation hash mismatch")
    if sha256_file(FINAL_REFIT_MANIFEST) != EXPECTED_FINAL_REFIT_MANIFEST_SHA256:
        raise ValueError("Frozen final-refit manifest changed")
    if Path(payload.get("manifest_path", "")).resolve() != FINAL_REFIT_MANIFEST.resolve():
        raise ValueError("Allowlist points to an unexpected final-refit manifest")
    if payload.get("manifest_sha256") != EXPECTED_FINAL_REFIT_MANIFEST_SHA256:
        raise ValueError("Allowlist final-refit manifest hash mismatch")
    manifest_rows = [
        json.loads(line) for line in FINAL_REFIT_MANIFEST.read_text().splitlines() if line.strip()
    ]
    if len(manifest_rows) != 15:
        raise ValueError("Frozen final-refit manifest must contain exactly 15 rows")
    expected_by_pair = {
        (row["part_slug"], int(row["model_seed"])): row for row in manifest_rows
    }
    expected_pairs = {(part, seed) for part in EXPECTED_AUDIT_N for seed in (1701, 1702, 1703)}
    observed_pairs = {(row["part_slug"], int(row["model_seed"])) for row in rows}
    if observed_pairs != expected_pairs:
        raise ValueError("Allowlist does not contain the exact part/seed Cartesian product")
    for row in rows:
        pair = (row["part_slug"], int(row["model_seed"]))
        expected = expected_by_pair[pair]
        exact_fields = {
            "cell_id": "cell_id",
            "part_slug": "part_slug",
            "base_config_id": "base_config_id",
            "architecture": "architecture",
            "training_regime": "training_regime",
            "unfreeze_scope": "unfreeze_scope",
            "rc_mode": "rc_mode",
            "loss_mode": "loss_mode",
            "model_seed": "model_seed",
            "fixed_completed_epochs": "fixed_epochs",
            "wandb_project": "logger_project",
            "dataset_path": "dataset_path",
            "dataset_sha256": "dataset_sha256",
            "split_manifest_path": "split_manifest_path",
            "split_manifest_id": "split_manifest_id",
            "split_manifest_sha256": "split_manifest_sha256",
            "selection_manifest_sha256": "selection_manifest_sha256",
            "protocol_amendment_sha256": "protocol_amendment_sha256",
        }
        for allowlist_field, manifest_field in exact_fields.items():
            if row.get(allowlist_field) != expected.get(manifest_field):
                raise ValueError(
                    f"Allowlist {allowlist_field} differs from frozen manifest for {pair}"
                )
        if row.get("status") != "completed_reconciled_pre_audit":
            raise ValueError(f"Allowlist row is not reconciled pre-audit for {pair}")
        if row.get("implementation_reconciliation_sha256") != EXPECTED_RECONCILIATION_SHA256:
            raise ValueError(f"Allowlist reconciliation identity differs for {pair}")
        if row.get("isolation_method") != "stable_id_only_physical_row_exclusion":
            raise ValueError(f"Allowlist isolation method differs for {pair}")
        if row.get("non_audit_training_allowlist_hash") != row.get("train_row_id_hash"):
            raise ValueError(f"Allowlist non-audit training hash differs for {pair}")
        if row.get("model_module") != option_value(expected["train_command"], "--model_module"):
            raise ValueError(f"Allowlist model module differs for {pair}")
        if row.get("graph_module") != option_value(expected["train_command"], "--graph_module"):
            raise ValueError(f"Allowlist graph module differs for {pair}")
        if row.get("data_module") != option_value(expected["train_command"], "--data_module"):
            raise ValueError(f"Allowlist data module differs for {pair}")
        expected_checkpoint = Path(expected["default_root_dir"]) / "torch_checkpoint.pt"
        if Path(row.get("checkpoint_path", "")).resolve() != expected_checkpoint.resolve():
            raise ValueError(f"Allowlist checkpoint path differs for {pair}")
        for field in (
            "artifact_path", "checkpoint_path", "compact_provenance_path",
            "completion_marker_path", "completion_log_path",
        ):
            if not Path(row[field]).is_file():
                raise ValueError(f"Allowlisted file is missing: {row[field]}")
        if sha256_file(Path(row["artifact_path"])) != row["artifact_sha256"]:
            raise ValueError("Allowlisted artifact hash mismatch")
        if sha256_file(Path(row["checkpoint_path"])) != row["checkpoint_sha256"]:
            raise ValueError("Allowlisted checkpoint hash mismatch")
        if sha256_file(Path(row["compact_provenance_path"])) != row["compact_provenance_sha256"]:
            raise ValueError("Allowlisted compact-provenance hash mismatch")
        if sha256_file(Path(row["completion_marker_path"])) != row["completion_marker_sha256"]:
            raise ValueError("Allowlisted completion-marker hash mismatch")
        if sha256_file(Path(row["completion_log_path"])) != row["completion_log_sha256"]:
            raise ValueError("Allowlisted completion-log hash mismatch")
        for relative_path, expected_source_sha in row.get("implementation_source_sha256", {}).items():
            source_path = REPO / relative_path
            if not source_path.is_file() or sha256_file(source_path) != expected_source_sha:
                raise ValueError(f"Allowlisted implementation source changed: {relative_path}")
    return payload


def score(
    allowlist_path: Path,
    expected_allowlist_sha: str,
    device_name: str,
    technical_retry: bool = False,
    retry_reason: str | None = None,
) -> dict:
    if sha256_file(AMENDMENT) != EXPECTED_AMENDMENT_SHA256:
        raise ValueError("Frozen audit protocol amendment changed")
    if sha256_file(RECONCILIATION) != EXPECTED_RECONCILIATION_SHA256:
        raise ValueError("Frozen implementation reconciliation changed")
    allowlist = verify_allowlist(allowlist_path, expected_allowlist_sha)
    # Finish every checkpoint/device/filesystem check before recording audit
    # access or constructing the first audit DataModule.
    runtime_preflight(allowlist, device_name, technical_retry)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = OUT_DIR / "audit_access_started.json"
    completed = OUT_DIR / "audit_access_completed.json"
    attempts = OUT_DIR / "audit_access_attempts.jsonl"
    scorer_sha = sha256_file(Path(__file__))
    if completed.exists():
        raise RuntimeError(
            "The one-time audit completed marker exists; repeat scoring is forbidden"
        )
    if started.exists():
        if not technical_retry or not str(retry_reason or "").strip():
            raise RuntimeError(
                "An incomplete audit attempt exists; an exact technical retry requires "
                "--technical-retry and --retry-reason"
            )
        first = json.loads(started.read_text())
        invariant = {
            "allowlist_path": str(allowlist_path),
            "allowlist_sha256": expected_allowlist_sha,
            "protocol_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
            "implementation_reconciliation_sha256": EXPECTED_RECONCILIATION_SHA256,
            "final_refit_manifest_sha256": EXPECTED_FINAL_REFIT_MANIFEST_SHA256,
            "scorer_sha256": scorer_sha,
            "device": device_name,
        }
        for field, value in invariant.items():
            if first.get(field) != value:
                raise RuntimeError(f"Technical retry changed frozen field {field}")
        prior_attempts = (
            sum(1 for line in attempts.read_text().splitlines() if line.strip())
            if attempts.exists()
            else 1
        )
        attempt_number = prior_attempts + 1
        append_jsonl(attempts, {
            "attempt": attempt_number,
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "technical_retry_started_before_loader_instantiation",
            "reason": str(retry_reason).strip(),
            **invariant,
        })
    else:
        if technical_retry:
            raise RuntimeError("--technical-retry is invalid before an initial audit attempt")
        invariant = {
            "allowlist_path": str(allowlist_path),
            "allowlist_sha256": expected_allowlist_sha,
            "protocol_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
            "implementation_reconciliation_sha256": EXPECTED_RECONCILIATION_SHA256,
            "final_refit_manifest_sha256": EXPECTED_FINAL_REFIT_MANIFEST_SHA256,
            "scorer_sha256": scorer_sha,
            "device": device_name,
        }
        attempt_number = 1
        atomic_json(started, {
            "status": "started_before_loader_instantiation",
            "protocol_amendment_path": str(AMENDMENT),
            "audit_results_visible_before_start": False,
            **invariant,
        })
        append_jsonl(attempts, {
            "attempt": 1,
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "initial_attempt_started_before_loader_instantiation",
            "reason": "authorized_initial_one_time_audit",
            **invariant,
        })

    device = torch.device(device_name)
    rows_by_part = {
        part: sorted(
            [row for row in allowlist["rows"] if row["part_slug"] == part],
            key=lambda row: int(row["model_seed"]),
        )
        for part in EXPECTED_AUDIT_N
    }
    seed_prediction_frames = []
    ensemble_frames = []
    metric_rows = []
    intron_stratum_rows = []
    intron_estimand_rows = []
    intron_cutoff_rows = []

    for part, rows in rows_by_part.items():
        data = build_audit_data(rows[0])
        split = data.split_summary
        if split["split_mode"] != "manifest_audit_eval" or split["n_test"] != EXPECTED_AUDIT_N[part]:
            raise RuntimeError(f"Unexpected audit split for {part}: {split}")
        if split["audit_row_id_hash"] != rows[0]["audit_exclusion_row_id_hash"]:
            raise RuntimeError(f"Audit ID hash changed for {part}")
        if split["normalization_row_id_hash"] != rows[0]["normalization_row_id_hash"]:
            raise RuntimeError(f"Normalization row hash changed for {part}")
        if not np.isclose(float(data.target_mean), float(rows[0]["target_normalization_mean"]), rtol=0, atol=1e-12):
            raise RuntimeError(f"Target mean changed for {part}")
        if not np.isclose(float(data.target_std), float(rows[0]["target_normalization_std"]), rtol=0, atol=1e-12):
            raise RuntimeError(f"Target std changed for {part}")
        frame = data.df_test.copy().reset_index(drop=True)
        id_column = data.split_id_column
        target_column = data.target_column
        barcode_column = data.barcode_column
        base = frame[[id_column, barcode_column, target_column]].rename(
            columns={id_column: "construct_id", barcode_column: "n_barcodes", target_column: "observed_raw"}
        )
        if base["construct_id"].duplicated().any() or len(base) != EXPECTED_AUDIT_N[part]:
            raise RuntimeError(f"Audit construct identity mismatch for {part}")
        test_loader = data.test_dataloader()
        if test_loader is None:
            raise RuntimeError(f"Audit loader is unexpectedly absent for {part}")
        seed_columns = []
        for row in rows:
            if row["normalization_row_id_hash"] != rows[0]["normalization_row_id_hash"]:
                raise RuntimeError(f"Seed normalization hash mismatch for {part}")
            model = load_model(Path(row["checkpoint_path"]), device)
            processed = predict(model, test_loader, device)
            raw = processed * float(row["target_normalization_std"]) + float(row["target_normalization_mean"])
            if len(raw) != len(base) or not np.isfinite(raw).all():
                raise RuntimeError(f"Audit prediction alignment failed for {part}/{row['model_seed']}")
            predictor = f"seed_{row['model_seed']}"
            seed_frame = base.copy()
            seed_frame["part_slug"] = part
            seed_frame["predictor"] = predictor
            seed_frame["model_seed"] = int(row["model_seed"])
            seed_frame["base_config_id"] = row["base_config_id"]
            seed_frame["prediction_raw"] = raw
            if not seed_frame["construct_id"].equals(base["construct_id"]):
                raise RuntimeError(f"Seed construct order changed for {part}/{row['model_seed']}")
            seed_prediction_frames.append(seed_frame)
            seed_columns.append(raw)
            metric_rows.append({
                "part_slug": part,
                "predictor": predictor,
                "primary_predictor": False,
                **metrics(base["observed_raw"], raw),
            })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        ensemble = np.mean(np.vstack(seed_columns), axis=0)
        ensemble_frame = base.copy()
        ensemble_frame["part_slug"] = part
        ensemble_frame["predictor"] = "ensemble_mean"
        ensemble_frame["model_seed"] = pd.NA
        ensemble_frame["base_config_id"] = rows[0]["base_config_id"]
        ensemble_frame["prediction_raw"] = ensemble
        ensemble_frames.append(ensemble_frame)
        metric_rows.append({
            "part_slug": part,
            "predictor": "ensemble_mean",
            "primary_predictor": True,
            **metrics(base["observed_raw"], ensemble),
        })

        if part == "intron":
            intron_base = assign_inferred_intron_subsets(
                frame[[id_column, data.sequence_column, barcode_column, target_column]].copy(),
                sequence_column=data.sequence_column,
            ).rename(columns={
                id_column: "construct_id", barcode_column: "n_barcodes",
                target_column: "log2_RNA_DNA",
                "inferred_intron_subset": STRATUM,
            })
            prediction_map = {
                "ensemble_mean": ensemble,
                **{f"seed_{row['model_seed']}": values for row, values in zip(rows, seed_columns)},
            }
            for predictor in PREDICTOR_ORDER:
                values = prediction_map[predictor]
                analysis = intron_base.copy()
                analysis["prediction_raw"] = values
                analysis["development_fold"] = -1
                estimand = intron_estimands(analysis)
                suppressed_stratum_values = []
                for stratum in STRATUM_ORDER:
                    sub = analysis.loc[analysis[STRATUM].eq(stratum)]
                    stratum_metrics = metrics(
                        sub["log2_RNA_DNA"], sub["prediction_raw"],
                        suppress_pearson_below_n=30,
                    )
                    intron_stratum_rows.append({
                        "predictor": predictor,
                        "inferred_stratum": stratum,
                        **stratum_metrics,
                    })
                    estimand[f"{stratum}_pearson"] = stratum_metrics["pearson"]
                    estimand[f"{stratum}_pearson_reason"] = stratum_metrics["pearson_reason"]
                    suppressed_stratum_values.append(stratum_metrics["pearson"])
                if np.isfinite(suppressed_stratum_values).all():
                    estimand["macro_stratum_pearson"] = float(np.mean(suppressed_stratum_values))
                    estimand["minimum_stratum_pearson"] = float(np.min(suppressed_stratum_values))
                    estimand["macro_stratum_pearson_reason"] = ""
                    estimand["minimum_stratum_pearson_reason"] = ""
                else:
                    estimand["macro_stratum_pearson"] = math.nan
                    estimand["minimum_stratum_pearson"] = math.nan
                    reason = "suppressed_one_or_more_strata_ineligible"
                    estimand["macro_stratum_pearson_reason"] = reason
                    estimand["minimum_stratum_pearson_reason"] = reason
                intron_estimand_rows.append({"predictor": predictor, **estimand})
                for cutoff in (8, 10, 12):
                    sub = analysis.loc[analysis["n_barcodes"].ge(cutoff)]
                    intron_cutoff_rows.append({
                        "predictor": predictor,
                        "minimum_barcodes": cutoff,
                        **metrics(sub["log2_RNA_DNA"], sub["prediction_raw"]),
                    })

    seed_predictions = pd.concat(seed_prediction_frames, ignore_index=True)
    ensemble_predictions = pd.concat(ensemble_frames, ignore_index=True)
    audit_metrics = pd.DataFrame(metric_rows)
    audit_metrics["predictor_order"] = audit_metrics["predictor"].map(
        {name: index for index, name in enumerate(PREDICTOR_ORDER)}
    )
    audit_metrics = audit_metrics.sort_values(["part_slug", "predictor_order"]).drop(columns="predictor_order")
    products = {
        "audit_seed_predictions.tsv.gz": seed_predictions,
        "audit_ensemble_predictions.tsv.gz": ensemble_predictions,
        "audit_metrics.csv": audit_metrics,
        "audit_intron_stratum_metrics.csv": pd.DataFrame(intron_stratum_rows),
        "audit_intron_estimand_metrics.csv": pd.DataFrame(intron_estimand_rows),
        "audit_intron_barcode_cutoff_metrics.csv": pd.DataFrame(intron_cutoff_rows),
    }
    stage_dir = OUT_DIR / f".audit_products_attempt_{attempt_number}_{os.getpid()}"
    if stage_dir.exists():
        raise RuntimeError(f"Audit staging directory already exists: {stage_dir}")
    stage_dir.mkdir(parents=False, exist_ok=False)
    product_index = {}
    for filename, table in products.items():
        stage_path = stage_dir / filename
        final_path = PRODUCTS_DIR / filename
        if filename.endswith(".tsv.gz"):
            table.to_csv(stage_path, sep="\t", index=False, compression="gzip")
        else:
            table.to_csv(stage_path, index=False)
        product_index[filename] = {
            "path": str(final_path), "sha256": sha256_file(stage_path), "rows": int(len(table))
        }
    summary = {
        "schema_version": "lib1_dedup_final_audit_summary_v1",
        "allowlist_path": str(allowlist_path),
        "allowlist_sha256": expected_allowlist_sha,
        "protocol_amendment_sha256": EXPECTED_AMENDMENT_SHA256,
        "implementation_reconciliation_sha256": EXPECTED_RECONCILIATION_SHA256,
        "final_refit_manifest_sha256": EXPECTED_FINAL_REFIT_MANIFEST_SHA256,
        "primary_predictor": "arithmetic_mean_raw_predictions_seeds_1701_1702_1703",
        "canonical_checkpoint_seed": 1701,
        "audit_counts": EXPECTED_AUDIT_N,
        "audit_loader_instantiated": True,
        "audit_targets_loaded": True,
        "audit_predictions_generated": True,
        "audit_metrics_computed": True,
        "audit_used_for_model_selection": False,
        "raw_predictions_primary": True,
        "audit_fitted_calibration_applied": False,
        "products": product_index,
    }
    stage_summary_path = stage_dir / "audit_summary.json"
    final_summary_path = PRODUCTS_DIR / "audit_summary.json"
    atomic_json(stage_summary_path, summary)
    product_index["audit_summary.json"] = {
        "path": str(final_summary_path), "sha256": sha256_file(stage_summary_path), "rows": 1
    }
    stage_index_path = stage_dir / "audit_artifact_index.json"
    final_index_path = PRODUCTS_DIR / "audit_artifact_index.json"
    atomic_json(stage_index_path, {
        "schema_version": "lib1_dedup_final_audit_artifact_index_v1",
        "products": product_index,
    })
    if PRODUCTS_DIR.exists():
        raise RuntimeError("Frozen audit products appeared during scoring")
    os.replace(str(stage_dir), str(PRODUCTS_DIR))
    atomic_json(completed, {
        "status": "completed_one_time_audit",
        "allowlist_sha256": expected_allowlist_sha,
        "audit_summary_path": str(final_summary_path),
        "audit_summary_sha256": sha256_file(final_summary_path),
        "artifact_index_path": str(final_index_path),
        "artifact_index_sha256": sha256_file(final_index_path),
        "audit_used_for_model_selection": False,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--expected-allowlist-sha256", required=True)
    parser.add_argument("--confirm-one-time-audit", action="store_true")
    parser.add_argument("--technical-retry", action="store_true")
    parser.add_argument("--retry-reason")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.confirm_one_time_audit:
        raise ValueError("One-time audit scoring requires --confirm-one-time-audit")
    result = score(
        args.allowlist.resolve(),
        args.expected_allowlist_sha256,
        args.device,
        technical_retry=args.technical_retry,
        retry_reason=args.retry_reason,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
