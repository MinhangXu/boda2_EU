#!/usr/bin/env python3
"""Preview or execute the frozen Lib1 dedup Stage 4 downsampling campaign.

Preview is the default. Execution is resume-safe and fail-closed: row 1 is the
only authorized pilot, and every non-pilot launch is locked until that exact
cell completes and reconciles locally. The runner never imports a DataModule
or constructs a final-test loader.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence


HERE = Path(__file__).resolve().parent
PREFIX = HERE / "outputs/hpo_manifests/lib1_dedup_stage4_downsampling_july2026"
MANIFEST = Path(str(PREFIX) + "__dry_run_manifest.jsonl")
PORTFOLIO = Path(str(PREFIX) + "__portfolio.json")
SUMMARY = Path(str(PREFIX) + "__summary.json")
VALIDATION_REPORT = Path(str(PREFIX) + "__validation_report.json")
VERIFIER = HERE / "verify_lib1_dedup_stage4_downsampling_manifest.py"
STATUS_DIR = HERE / "outputs/hpo_runs/status/lib1_dedup_stage4_downsampling_july2026"
STAGE4_RUNS_CSV = STATUS_DIR / "stage4_runs.csv"
RUNS_CSV = STAGE4_RUNS_CSV
EXPECTED_ENTITY = "minhangxu1998-baylor-college-of-medicine"
EXPECTED_ROWS = 660
EXPECTED_MANIFEST_SHA256 = "dd6abda4726846f482536a235093b2ed9aa5a36b12591613c400601dcb27a84a"
REQUIRED_PILOT_ROW = 1
REQUIRED_PILOT_CELL = "stage4cell_1ebd9c906d22b299b8cb"
TEST_METRIC_FIELDS = (
    "test_loss", "test_r2", "test_pearson", "test_spearman",
    "test_pearson_r2", "test_cod_r2", "test_mse",
)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(values: Sequence[str]) -> str:
    payload = json.dumps(
        sorted(str(value) for value in values),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_hash(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def provenance_scalar_text(value) -> str:
    """Normalize a scalar without treating valid numeric zero as missing."""
    return "" if value is None else str(value)


def expected_runtime_argv(row: dict) -> List[str]:
    return shlex.split(row["train_command"])[1:]


def read_jsonl(path: Path) -> List[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_registry(path: Path = STAGE4_RUNS_CSV) -> Dict[str, List[dict]]:
    by_cell: Dict[str, List[dict]] = {}
    if path.expanduser().resolve() != STAGE4_RUNS_CSV.resolve():
        raise RuntimeError("Stage 4 refuses to read a non-Stage4 run registry")
    if not path.is_file():
        return by_cell
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            if (
                record.get("campaign_id", "") != "lib1_dedup_phase1_rerun_july2026"
                or record.get("campaign_stage", "") != "stage4_downsampling"
            ):
                raise RuntimeError("Stage4-only registry contains an out-of-scope row")
            populated_test = {
                field: record.get(field, "")
                for field in TEST_METRIC_FIELDS
                if str(record.get(field, "")).strip()
            }
            if populated_test:
                raise RuntimeError(
                    f"Stage4-only registry contains final-test metrics: {populated_test}"
                )
            cell_id = record.get("cell_id", "")
            if cell_id:
                by_cell.setdefault(cell_id, []).append(record)
    return by_cell


def marker_fields(path: Path) -> Dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def command_option(row: dict, name: str) -> str:
    tokens = shlex.split(row["train_command"])
    option = f"--{name}"
    positions = [index for index, token in enumerate(tokens) if token == option]
    if len(positions) != 1 or positions[0] + 1 >= len(tokens):
        raise RuntimeError(f"Row {row['row']} lacks one {option}")
    return tokens[positions[0] + 1]


def expected_registry_fields(row: dict) -> Dict[str, str]:
    fields = {
        "run_name": row["planned_run_name"],
        "wandb_entity": row["wandb_entity"],
        "wandb_project": row["logger_project"],
        "logger_project": row["logger_project"],
        "campaign_id": row["campaign_id"],
        "campaign_stage": row["campaign_stage"],
        "part_slug": row["part_slug"],
        "analysis_lane": row["analysis_lane"],
        "challenger_family": row["challenger_family"],
        "policy_id": row["policy_id"],
        "training_regime": row["training_regime"],
        "cell_id": row["cell_id"],
        "rc_mode": row["rc_mode"],
        "execution_disposition": "launch",
        "initialization": row["initialization"],
        "source_head": row["source_head"],
        "unfreeze_scope": row["unfreeze_scope"],
        "input_policy": row["input_policy"],
        "pretrained_artifact_sha256": row["pretrained_artifact_sha256"],
        "data_generation_id": row["data_generation_id"],
        "dataset_sha256": row["dataset_sha256"],
        "split_manifest_id": row["split_manifest_id"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "development_fold": str(row["outer_oof_fold"]),
        "base_config_id": row["base_config_id"],
        "architecture": row["architecture"],
        "model_seed": str(row["model_seed"]),
        "loss_mode": row["loss_mode"],
        "target_definition": row["target_definition"],
        "length_policy": row["length_policy"],
        "artifact_retention": "none",
        "graph_module": command_option(row, "graph_module"),
        "launch_script": "run_lib1_dedup_stage4_downsampling_campaign.py",
        "config_path": str(MANIFEST.resolve()),
        "config_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_row": str(row["row"]),
        "manifest_row_fingerprint": row["row_fingerprint"],
        "runtime_argv_sha256": canonical_json_hash(expected_runtime_argv(row)),
        "run_registry_path": str(STAGE4_RUNS_CSV.resolve()),
    }
    return {key: str(value or "") for key, value in fields.items()}


def _resolved_argument_value(arguments: dict, name: str):
    matches = [
        values[name]
        for values in arguments.values()
        if isinstance(values, dict) and name in values
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Compact provenance must resolve {name!r} exactly once; found {len(matches)}"
        )
    return matches[0]


def validate_resolved_arguments(row: dict, arguments: dict) -> None:
    if not isinstance(arguments, dict) or not arguments:
        raise RuntimeError(f"Completed {row['cell_id']} lacks resolved arguments")
    expected = {
        "campaign_id": row["campaign_id"],
        "campaign_stage": row["campaign_stage"],
        "part_slug": row["part_slug"],
        "analysis_lane": row["analysis_lane"],
        "training_regime": row["training_regime"],
        "cell_id": row["cell_id"],
        "rc_mode": row["rc_mode"],
        "execution_disposition": "launch",
        "dataset_sha256": row["dataset_sha256"],
        "split_manifest_id": row["split_manifest_id"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "development_fold": row["outer_oof_fold"],
        "base_config_id": row["base_config_id"],
        "architecture": row["architecture"],
        "model_seed": row["model_seed"],
        "loss_mode": row["loss_mode"],
        "run_name": row["planned_run_name"],
        "logger_project": row["logger_project"],
        "wandb_entity": row["wandb_entity"],
        "artifact_retention": "none",
        "evaluate_test_after_fit": False,
        "prediction_splits": ["oof"],
        "epoch_eval_splits": ["train", "val"],
        "manifest_mode": "development_inner_oof",
        "split_fold": row["outer_oof_fold"],
        "train_size_n": row["train_size_n"],
        "train_subsample_seed": row["train_subsample_seed"],
        "expected_data_sha256": row["dataset_sha256"],
        "expected_split_sha256": row["split_manifest_sha256"],
        "use_reverse_complements": row["rc_mode"] == "on",
        "barcode_weighting": row["loss_mode"] == "barcode_weighted_mse",
    }
    for name, wanted in expected.items():
        observed = _resolved_argument_value(arguments, name)
        if observed != wanted:
            raise RuntimeError(
                f"Resolved argument {name} mismatch for {row['cell_id']}: "
                f"{observed!r} != {wanted!r}"
            )
    observed_split_path = Path(
        str(_resolved_argument_value(arguments, "split_manifest_path"))
    ).expanduser().resolve()
    if observed_split_path != Path(row["split_manifest_path"]).resolve():
        raise RuntimeError(f"Resolved split manifest path mismatch for {row['cell_id']}")
    observed_root = Path(
        str(_resolved_argument_value(arguments, "default_root_dir"))
    ).expanduser().resolve()
    if observed_root != Path(row["default_root_dir"]).resolve():
        raise RuntimeError(f"Resolved output root mismatch for {row['cell_id']}")


def validate_completed_record(row: dict, record: dict) -> tuple[Path, Path]:
    populated_test = {
        field: record.get(field, "")
        for field in TEST_METRIC_FIELDS
        if str(record.get(field, "")).strip()
    }
    if populated_test:
        raise RuntimeError(f"Completed {row['cell_id']} contains final-test metrics: {populated_test}")
    expected = expected_registry_fields(row)
    mismatches = {
        field: {"observed": record.get(field, ""), "expected": value}
        for field, value in expected.items()
        if record.get(field, "") != value
    }
    if mismatches:
        raise RuntimeError(
            f"Registry identity mismatch for {row['cell_id']}:\n"
            + json.dumps(mismatches, indent=2, sort_keys=True)
        )
    try:
        optimizer_steps = int(record.get("optimizer_steps", ""))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Completed {row['cell_id']} lacks optimizer_steps") from exc
    if optimizer_steps <= 0:
        raise RuntimeError(f"Completed {row['cell_id']} has no optimizer updates")
    resolved_hash = str(record.get("resolved_arguments_sha256", "") or "")
    if not resolved_hash:
        raise RuntimeError(f"Completed {row['cell_id']} lacks resolved-argument evidence")
    expected_hashes = {
        "train_row_id_hash": row["expected_train_id_hash"],
        "val_row_id_hash": row["expected_inner_val_id_hash"],
        "normalization_row_id_hash": row["expected_normalization_id_hash"],
        "selected_row_hash": row["expected_train_id_hash"],
        "audit_row_id_hash": row["final_test_exclusion_id_hash"],
    }
    for field, value in expected_hashes.items():
        if record.get(field, "") != value:
            raise RuntimeError(f"Registry {field} mismatch for {row['cell_id']}")

    prediction_value = str(record.get("prediction_path", "")).strip()
    if not prediction_value:
        raise RuntimeError(f"Completed {row['cell_id']} lacks prediction_path")
    prediction = Path(prediction_value)
    expected_parent = (Path(row["default_root_dir"]) / "predictions").resolve()
    if not prediction.is_file() or prediction.resolve().parent != expected_parent:
        raise RuntimeError(f"Completed {row['cell_id']} has a missing/misplaced OOF prediction")
    if "__oof_predictions.tsv" not in prediction.name:
        raise RuntimeError(f"Completed {row['cell_id']} prediction is not an OOF export")
    with prediction.open(newline="") as handle:
        prediction_rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(prediction_rows) != int(row["expected_oof_n"]):
        raise RuntimeError(f"Completed {row['cell_id']} has the wrong OOF row count")
    required = {"construct_id", "prediction_raw", "log2_RNA_DNA"}
    if prediction_rows and not required.issubset(prediction_rows[0]):
        raise RuntimeError(f"Completed {row['cell_id']} OOF export lacks {sorted(required)}")
    ids = [item["construct_id"] for item in prediction_rows]
    if len(ids) != len(set(ids)) or canonical_hash(ids) != row["expected_oof_id_hash"]:
        raise RuntimeError(f"Completed {row['cell_id']} has the wrong OOF IDs")

    run_id = str(record.get("run_id", "")).strip()
    provenance = Path(row["default_root_dir"]) / "provenance" / f"{run_id}__run_provenance.json"
    if not run_id or not provenance.is_file():
        raise RuntimeError(f"Completed {row['cell_id']} lacks compact provenance")
    payload = json.loads(provenance.read_text())
    for field, value in expected.items():
        # JSON preserves numeric zero for fields such as development_fold.
        # Do not use ``or ""`` here: zero is valid provenance, not missing.
        if provenance_scalar_text(payload.get(field, "")) != value:
            raise RuntimeError(f"Provenance {field} mismatch for {row['cell_id']}")
    if str(payload.get("prediction_path", "")) != str(prediction):
        raise RuntimeError(f"Provenance prediction mismatch for {row['cell_id']}")
    if payload.get("optimizer_steps") != optimizer_steps:
        raise RuntimeError(f"Provenance optimizer_steps mismatch for {row['cell_id']}")
    arguments = payload.get("resolved_arguments")
    if canonical_json_hash(arguments) != resolved_hash:
        raise RuntimeError(f"Resolved-argument SHA mismatch for {row['cell_id']}")
    if str(payload.get("resolved_arguments_sha256", "") or "") != resolved_hash:
        raise RuntimeError(f"Provenance resolved-argument hash mismatch for {row['cell_id']}")
    validate_resolved_arguments(row, arguments)
    split = payload.get("data_split_summary", {})
    split_expected = {
        "manifest_mode": "development_inner_oof",
        "split_manifest_id": row["split_manifest_id"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "dataset_sha256": row["dataset_sha256"],
        "development_fold": row["outer_oof_fold"],
        "outer_development_fold": row["outer_oof_fold"],
        "inner_development_fold": row["inner_validation_fold"],
        "n_test": 0,
        "n_oof": row["expected_oof_n"],
        "n_val": row["expected_inner_val_n"],
        "n_train_pool_eligible": row["expected_pool_n"],
        "n_train_final": row["expected_train_n"],
        "train_subsample_seed": row["train_subsample_seed"],
        "train_size_n": row["train_size_n"],
        "train_pool_id_hash": row["expected_pool_id_hash"],
        "train_final_id_hash": row["expected_train_id_hash"],
        "normalization_id_hash": row["expected_normalization_id_hash"],
        "val_id_hash": row["expected_inner_val_id_hash"],
        "oof_id_hash": row["expected_oof_id_hash"],
        "audit_id_hash": row["final_test_exclusion_id_hash"],
        "target_normalization_row_count": row["expected_train_n"],
        "final_test_rows_physically_excluded": True,
        "audit_loader_authorized": False,
    }
    for field, value in split_expected.items():
        if split.get(field) != value:
            raise RuntimeError(
                f"Split provenance {field} mismatch for {row['cell_id']}: "
                f"{split.get(field)!r} != {value!r}"
            )
    run_url = str(record.get("run_url", ""))
    if f"/{row['wandb_entity']}/{row['logger_project']}/runs/{run_id}" not in run_url:
        raise RuntimeError(f"Completed {row['cell_id']} has an invalid W&B URL")
    return prediction, provenance


def row_completed(row: dict, registry: Dict[str, List[dict]], manifest_sha: str) -> bool:
    matching = []
    expected = expected_registry_fields(row)
    for record in registry.get(row["cell_id"], []):
        collision = {
            field: {"observed": record.get(field, ""), "expected": value}
            for field, value in expected.items()
            if record.get(field, "") != value
        }
        if collision:
            raise RuntimeError(
                f"runs.csv collision for {row['cell_id']}:\n"
                + json.dumps(collision, indent=2, sort_keys=True)
            )
        if record.get("status", "").lower() == "completed":
            validate_completed_record(row, record)
            matching.append(record)
    if len(matching) > 1:
        raise RuntimeError(f"Multiple completions resolve to {row['cell_id']}")

    marker = STATUS_DIR / "done" / f"row_{row['row']}.done"
    if marker.is_file():
        fields = marker_fields(marker)
        expected_marker = {
            "row": str(row["row"]),
            "manifest_sha256": manifest_sha,
            "row_fingerprint": row["row_fingerprint"],
            "cell_id": row["cell_id"],
        }
        if any(fields.get(key) != value for key, value in expected_marker.items()):
            raise RuntimeError(f"Completion marker changed for row {row['row']}")
        if not matching:
            raise RuntimeError(f"Completion marker lacks registry evidence for row {row['row']}")
    return bool(matching)


def parse_list(value: str) -> List[str]:
    return [token for token in value.replace(",", " ").split() if token]


def detect_idle_gpus(memory_threshold_mb: int = 512) -> List[str]:
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot detect GPUs; pass --gpus explicitly: {exc}") from exc
    busy = set()
    try:
        pmon = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1"], check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        for line in pmon.stdout.splitlines():
            values = line.split()
            if values and not values[0].startswith("#") and len(values) > 1 and values[1] != "-":
                busy.add(values[0])
    except OSError:
        pass
    idle = []
    for line in query.stdout.splitlines():
        index, used = (item.strip() for item in line.split(",", 1))
        if index not in busy and int(used) <= memory_threshold_mb:
            idle.append(index)
    return idle


def validate_frozen_inputs() -> tuple[List[dict], str]:
    required = (MANIFEST, PORTFOLIO, SUMMARY, VALIDATION_REPORT, VERIFIER)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing frozen Stage 4 inputs: {missing}")
    subprocess.run([sys.executable, str(VERIFIER)], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    rows = read_jsonl(MANIFEST)
    manifest_sha = sha256_file(MANIFEST)
    if len(rows) != EXPECTED_ROWS or manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Frozen Stage 4 manifest count or SHA changed")
    summary = json.loads(SUMMARY.read_text())
    report = json.loads(VALIDATION_REPORT.read_text())
    if summary.get("manifest_sha256") != manifest_sha or report.get("manifest_sha256") != manifest_sha:
        raise RuntimeError("Stage 4 summary/validation report no longer binds the manifest")
    if summary.get("commands_executed") != 0 or report.get("commands_executed") != 0:
        raise RuntimeError("Stage 4 dry-run products claim commands were executed")
    if report.get("final_test_loader_instantiated") is not False:
        raise RuntimeError("Stage 4 validation no longer proves final-test isolation")
    if rows[0]["cell_id"] != REQUIRED_PILOT_CELL:
        raise RuntimeError("Frozen Stage 4 pilot identity changed")
    return rows, manifest_sha


def select_rows(
    rows: Sequence[dict], args: argparse.Namespace,
    registry: Dict[str, List[dict]], manifest_sha: str,
) -> tuple[List[dict], int]:
    selected = []
    completed = 0
    wanted_parts = {value.lower() for value in parse_list(args.parts)}
    wanted_lanes = {value.lower() for value in parse_list(args.lanes)}
    wanted_folds = {int(value) for value in parse_list(args.folds)}
    for row in rows:
        is_complete = row_completed(row, registry, manifest_sha)
        completed += int(is_complete)
        number = int(row["row"])
        if args.pilot_row is not None and number != args.pilot_row:
            continue
        if args.row_start is not None and number < args.row_start:
            continue
        if args.row_end is not None and number > args.row_end:
            continue
        if wanted_parts and row["part_slug"].lower() not in wanted_parts:
            continue
        if wanted_lanes and row["stage4_lane"].lower() not in wanted_lanes:
            continue
        if wanted_folds and int(row["outer_oof_fold"]) not in wanted_folds:
            continue
        if is_complete and not args.include_completed:
            continue
        selected.append(row)
    if args.max_rows is not None:
        selected = selected[: args.max_rows]
    return selected, completed


def wandb_preflight() -> None:
    import wandb

    api = wandb.Api(timeout=15)
    if not getattr(api, "api_key", None):
        raise RuntimeError("No W&B API key resolved; run `wandb login` before launch")
    try:
        next(iter(api.projects(entity=EXPECTED_ENTITY, per_page=1)), None)
    except Exception as exc:
        raise RuntimeError(f"W&B access preflight failed: {exc}") from exc
    print(f"W&B preflight passed for {EXPECTED_ENTITY}")


def check_storage() -> List[dict]:
    evidence = []
    for path, minimum_free_gib, maximum_used_fraction in (
        (Path("/home"), 100, 0.85), (Path("/"), 20, 1.0),
    ):
        usage = shutil.disk_usage(str(path))
        free = usage.free // (1024**3)
        used = usage.used / usage.total
        print(f"Storage {path}: {free} GiB free, {100 * used:.1f}% used")
        if free < minimum_free_gib or used >= maximum_used_fraction:
            raise RuntimeError(f"Storage stop condition reached for {path}")
        evidence.append({
            "path": str(path), "free_gib": int(free), "used_fraction": float(used),
            "minimum_free_gib": minimum_free_gib,
            "maximum_used_fraction": maximum_used_fraction,
        })
    return evidence


def atomic_write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(body)
    os.replace(str(temporary), str(path))


class CampaignRunner:
    def __init__(self, rows: Sequence[dict], gpus: Sequence[str], manifest_sha: str, args) -> None:
        self.rows = list(rows)
        self.gpus = list(gpus)
        self.manifest_sha = manifest_sha
        self.args = args
        self.launch_id = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        self.log_dir = STATUS_DIR / "logs" / self.launch_id
        self.failure_dir = STATUS_DIR / "failures" / self.launch_id
        self.done_dir = STATUS_DIR / "done"
        self.work: "queue.Queue[dict]" = queue.Queue()
        for row in self.rows:
            self.work.put(row)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.active: Dict[int, subprocess.Popen] = {}
        self.failed_rows: List[int] = []

    def run_row(self, row: dict, gpu: str, worker_id: int) -> bool:
        tokens = shlex.split(row["train_command"])
        if tokens[:2] != ["python", "train_wandb_log.py"]:
            raise RuntimeError(f"Unexpected command prefix for row {row['row']}")
        command = [sys.executable] + tokens[1:]
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "WANDB_ENTITY": EXPECTED_ENTITY,
            "BODA_WANDB_ENTITY": EXPECTED_ENTITY,
            "WANDB_MODE": "online",
            "WANDB_DIR": str(HERE),
            "BODA_WANDB_PROJECT": row["logger_project"],
            "BODA_CONFIG_PATH": str(MANIFEST.resolve()),
            "BODA_CONFIG_MANIFEST_SHA256": self.manifest_sha,
            "BODA_MANIFEST_ROW": str(row["row"]),
            "BODA_MANIFEST_ROW_FINGERPRINT": row["row_fingerprint"],
            "BODA_RUNTIME_ARGV_SHA256": canonical_json_hash(command[1:]),
            "BODA_LAUNCH_SCRIPT": "run_lib1_dedup_stage4_downsampling_campaign.py",
            "BODA_RUNS_CSV": str(STAGE4_RUNS_CSV.resolve()),
            "BODA_LAUNCH_NOTES": f"{row['campaign_id']};{row['campaign_stage']}",
        })
        number = int(row["row"])
        log_path = self.log_dir / f"row_{number}.log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{now_iso()}] worker={worker_id} gpu={gpu} start row={number} "
            f"part={row['part_slug']} lane={row['stage4_lane']} n={row['downsample_n_label']}",
            flush=True,
        )
        with log_path.open("w") as handle:
            process = subprocess.Popen(
                command, cwd=str(HERE), env=env,
                stdout=handle, stderr=subprocess.STDOUT,
            )
            with self.lock:
                self.active[worker_id] = process
            status = process.wait()
            with self.lock:
                self.active.pop(worker_id, None)
        if status == 0:
            registry = read_registry()
            if not row_completed(row, registry, self.manifest_sha):
                status = 98
        if status == 0:
            marker = "\n".join((
                f"completed_at={now_iso()}", f"row={number}",
                f"manifest_sha256={self.manifest_sha}",
                f"row_fingerprint={row['row_fingerprint']}",
                f"cell_id={row['cell_id']}", f"log={log_path}", "",
            ))
            atomic_write(self.done_dir / f"row_{number}.done", marker)
            print(f"[{now_iso()}] worker={worker_id} gpu={gpu} done row={number}", flush=True)
            return True
        atomic_write(
            self.failure_dir / f"row_{number}.fail",
            "\n".join((
                f"failed_at={now_iso()}", f"row={number}",
                f"manifest_sha256={self.manifest_sha}",
                f"row_fingerprint={row['row_fingerprint']}",
                f"cell_id={row['cell_id']}", f"status={status}", f"log={log_path}", "",
            )),
        )
        with self.lock:
            self.failed_rows.append(number)
        print(f"[{now_iso()}] FAILED row={number}; {log_path}", file=sys.stderr, flush=True)
        if not self.args.continue_on_error:
            self.stop.set()
        return False

    def worker(self, gpu: str, worker_id: int) -> None:
        while not self.stop.is_set():
            try:
                row = self.work.get_nowait()
            except queue.Empty:
                return
            try:
                self.run_row(row, gpu, worker_id)
            except Exception as exc:
                print(f"Worker {worker_id} crashed on row {row['row']}: {exc}", file=sys.stderr)
                with self.lock:
                    self.failed_rows.append(int(row["row"]))
                if not self.args.continue_on_error:
                    self.stop.set()
            finally:
                self.work.task_done()

    def terminate_active(self) -> None:
        self.stop.set()
        with self.lock:
            processes = list(self.active.values())
        for process in processes:
            process.terminate()
        deadline = time.time() + 20
        for process in processes:
            try:
                process.wait(timeout=max(0.0, deadline - time.time()))
            except subprocess.TimeoutExpired:
                process.kill()

    def run(self) -> int:
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        workers = [
            threading.Thread(target=self.worker, args=(gpu, index + 1), name=f"gpu-{gpu}")
            for index, gpu in enumerate(self.gpus)
        ]
        for worker in workers:
            worker.start()
        try:
            for worker in workers:
                worker.join()
        except KeyboardInterrupt:
            self.terminate_active()
            for worker in workers:
                worker.join()
            return 130
        return 1 if self.failed_rows else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pilot-row", type=int)
    parser.add_argument("--confirm-pilot", action="store_true")
    parser.add_argument("--confirm-full-campaign", action="store_true")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--parts", default="")
    parser.add_argument("--lanes", default="")
    parser.add_argument("--folds", default="")
    parser.add_argument("--row-start", type=int)
    parser.add_argument("--row-end", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--include-completed", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--show-commands", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name in ("pilot_row", "row_start", "row_end"):
        value = getattr(args, name)
        if value is not None and not 1 <= value <= EXPECTED_ROWS:
            raise ValueError(f"--{name.replace('_', '-')} must be in 1..{EXPECTED_ROWS}")
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be positive")
    if args.include_completed and args.execute:
        raise ValueError("Completed cells are immutable")
    if args.execute and args.pilot_row is not None and not args.confirm_pilot:
        raise ValueError("Pilot execution requires --confirm-pilot")
    if args.execute and args.pilot_row is None and not args.confirm_full_campaign:
        raise ValueError("Non-pilot execution requires --confirm-full-campaign")
    if args.confirm_pilot and args.pilot_row is None:
        raise ValueError("--confirm-pilot requires --pilot-row")
    if args.confirm_full_campaign and args.pilot_row is not None:
        raise ValueError("Use --confirm-pilot for a one-row pilot")
    if args.execute and args.pilot_row not in (None, REQUIRED_PILOT_ROW):
        raise ValueError(f"Pilot execution is frozen to row {REQUIRED_PILOT_ROW}")

    rows, manifest_sha = validate_frozen_inputs()
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    with (STATUS_DIR / "launcher.lock").open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Stage 4 runner owns the launcher lock") from exc
        registry = read_registry()
        if args.execute and args.pilot_row is None:
            if not row_completed(rows[REQUIRED_PILOT_ROW - 1], registry, manifest_sha):
                raise RuntimeError("Full execution is locked until pilot row 1 reconciles")
        selected, completed = select_rows(rows, args, registry, manifest_sha)
        if args.pilot_row is not None and len(selected) != 1:
            raise RuntimeError(f"Pilot row {args.pilot_row} is not exactly one unfinished cell")

        print("Lib1 dedup Stage 4 downsampling preflight")
        print(f"  manifest: {MANIFEST}")
        print(f"  manifest SHA256: {manifest_sha}")
        print(f"  total cells: {len(rows)}")
        print(f"  completed cells: {completed}")
        print(f"  selected unfinished cells: {len(selected)}")
        print(f"  selected by part: {dict(sorted(Counter(row['part_slug'] for row in selected).items()))}")
        print(f"  selected by lane: {dict(sorted(Counter(row['stage4_lane'] for row in selected).items()))}")
        print(f"  W&B entity: {EXPECTED_ENTITY}")
        print(f"  Stage4-only registry: {STAGE4_RUNS_CSV}")
        print("  final-test evaluation: physically excluded and disabled")
        if selected:
            print(f"  selected row span: {selected[0]['row']}..{selected[-1]['row']}")
        if args.show_commands:
            for row in selected:
                print(
                    f"\nrow={row['row']} part={row['part_slug']} lane={row['stage4_lane']} "
                    f"n={row['downsample_n_label']}\n{row['train_command']}"
                )
        if not args.execute:
            print("Preview only; no GPU was claimed and no training command executed.")
            return 0
        if not selected:
            print("No unfinished selected Stage 4 cells remain.")
            return 0
        if Path(sys.prefix).name != "boda_env":
            raise RuntimeError(f"Use boda_env Python; current prefix is {str(sys.prefix)!r}")

        idle = detect_idle_gpus()
        gpus = parse_list(args.gpus) if args.gpus else idle
        if not gpus or len(gpus) != len(set(gpus)):
            raise RuntimeError("No unique idle GPUs selected")
        unavailable = sorted(set(gpus) - set(idle))
        if unavailable:
            raise RuntimeError(f"Requested GPUs are not idle: {unavailable}")
        maximum = args.max_parallel or len(gpus)
        if maximum < 1 or maximum > len(gpus):
            raise ValueError("--max-parallel is outside the selected GPU count")
        gpus = gpus[:maximum]
        wandb_preflight()
        storage = check_storage()
        preflight = {
            "recorded_at": now_iso(), "manifest_sha256": manifest_sha,
            "selected_rows": [row["row"] for row in selected], "gpus": gpus,
            "storage": storage, "wandb_entity": EXPECTED_ENTITY,
            "stage4_registry": str(STAGE4_RUNS_CSV.resolve()),
            "wandb_projects": sorted({row["logger_project"] for row in selected}),
            "final_test_loader_instantiated": False,
            "final_test_evaluation_enabled": False,
        }
        preflight_path = STATUS_DIR / "preflight" / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
        atomic_write(preflight_path, json.dumps(preflight, indent=2, sort_keys=True) + "\n")
        atomic_write(STATUS_DIR / "manifest.sha256", manifest_sha + "\n")
        print(f"Recorded launch preflight: {preflight_path}")
        runner = CampaignRunner(selected, gpus, manifest_sha, args)
        print(f"Launching {len(selected)} row(s) with {len(gpus)} worker(s).")
        status = runner.run()
        if status == 0:
            print("All selected Stage 4 rows completed and passed local reconciliation.")
        else:
            print(f"Run stopped with failures; inspect {runner.failure_dir}", file=sys.stderr)
        return status


if __name__ == "__main__":
    raise SystemExit(main())
