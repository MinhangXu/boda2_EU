#!/usr/bin/env python3
"""Preview or run the frozen Lib1 dedup Stage 3 weighted-loss manifest.

The default is a read-only preview. Execution is fail-closed: a one-row pilot
requires ``--confirm-pilot`` and every other execution requires
``--confirm-full-campaign``. Completion is accepted only when the exact
registry row, development validation predictions, provenance, and manifest
fingerprint agree. This program never imports a DataModule or constructs an
audit/test loader.
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
from typing import Dict, List, Sequence, Tuple


HERE = Path(__file__).resolve().parent
PREFIX = HERE / "outputs/hpo_manifests/lib1_dedup_stage3_weighted_loss_july2026"
MANIFEST = Path(str(PREFIX) + "__dry_run_manifest.jsonl")
ANALYSIS_MANIFEST = Path(str(PREFIX) + "__analysis_manifest.jsonl")
REUSE_MANIFEST = Path(str(PREFIX) + "__unweighted_reuse.jsonl")
PORTFOLIO = Path(str(PREFIX) + "__portfolio.json")
SUMMARY = Path(str(PREFIX) + "__summary.json")
VERIFIER = HERE / "verify_lib1_dedup_stage3_manifest.py"
RUNS_CSV = HERE / "run_registry/runs.csv"
STATUS_DIR = HERE / "outputs/hpo_runs/status/lib1_dedup_stage3_weighted_loss_july2026"
EXPECTED_ENTITY = "minhangxu1998-baylor-college-of-medicine"
EXPECTED_ROWS = 450
EXPECTED_MANIFEST_SHA256 = (
    "09de6182cf107c7b9485390fc9556ac48a92efe776bc35ab3ea6ca01a0ebca44"
)
REQUIRED_PILOT_ROWS = (1, 61)
REQUIRED_PILOT_CELLS = {
    1: "cell_fd3ec0f68e4c3b375e52",
    61: "cell_b448ff21b125f3e8cfc0",
}
TEST_METRIC_FIELDS = (
    "test_loss",
    "test_r2",
    "test_pearson",
    "test_spearman",
    "test_pearson_r2",
    "test_cod_r2",
    "test_mse",
)


def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id_hash(values: Sequence[str]) -> str:
    payload = json.dumps(
        sorted(str(value) for value in values),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> List[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_registry(path: Path) -> Dict[str, List[dict]]:
    by_cell: Dict[str, List[dict]] = {}
    if not path.is_file():
        return by_cell
    with path.open(newline="") as handle:
        for record in csv.DictReader(handle):
            cell_id = record.get("cell_id", "")
            if cell_id:
                by_cell.setdefault(cell_id, []).append(record)
    return by_cell


def marker_fields(path: Path) -> Dict[str, str]:
    fields = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


def expected_registry_fields(row: dict) -> Dict[str, str]:
    manifest_to_registry = {
        "planned_run_name": "run_name",
        "wandb_entity": "wandb_entity",
        "logger_project": "logger_project",
        "campaign_id": "campaign_id",
        "campaign_stage": "campaign_stage",
        "part_slug": "part_slug",
        "analysis_lane": "analysis_lane",
        "challenger_family": "challenger_family",
        "policy_id": "policy_id",
        "config_origin": "config_origin",
        "training_regime": "training_regime",
        "cell_id": "cell_id",
        "rc_pair_id": "rc_pair_id",
        "loss_pair_id": "loss_pair_id",
        "source_unweighted_cell_id": "source_unweighted_cell_id",
        "rc_mode": "rc_mode",
        "execution_disposition": "execution_disposition",
        "initialization": "initialization",
        "source_head": "source_head",
        "unfreeze_scope": "unfreeze_scope",
        "input_policy": "input_policy",
        "pretrained_artifact_sha256": "pretrained_artifact_sha256",
        "data_generation_id": "data_generation_id",
        "dataset_sha256": "dataset_sha256",
        "split_manifest_id": "split_manifest_id",
        "split_manifest_sha256": "split_manifest_sha256",
        "development_fold": "development_fold",
        "base_config_id": "base_config_id",
        "architecture": "architecture",
        "model_seed": "model_seed",
        "loss_mode": "loss_mode",
        "target_definition": "target_definition",
        "length_policy": "length_policy",
        "artifact_retention": "artifact_retention",
    }
    expected = {
        registry_field: str(row.get(manifest_field, ""))
        for manifest_field, registry_field in manifest_to_registry.items()
    }
    expected["wandb_project"] = str(row["logger_project"])
    expected["graph_module"] = (
        "CNNBassetBranchedScopedWeightedTransfer"
        if row["training_regime"] == "transfer"
        else "CNNWeightedRegressionTraining"
    )
    return expected


def validate_completed_record(row: dict, record: dict) -> None:
    nonblank_test = {
        field: record.get(field, "")
        for field in TEST_METRIC_FIELDS
        if record.get(field, "").strip()
    }
    if nonblank_test:
        raise RuntimeError(
            f"Completed record for {row['cell_id']} contains forbidden test metrics: "
            f"{nonblank_test}"
        )
    expected_val_hash = row["source_val_row_id_hash"]
    if not expected_val_hash or record.get("val_row_id_hash", "") != expected_val_hash:
        raise RuntimeError(f"Completed record for {row['cell_id']} has the wrong val-row hash")

    prediction_value = record.get("prediction_path", "").strip()
    if not prediction_value:
        raise RuntimeError(f"Completed record for {row['cell_id']} lacks prediction_path")
    prediction = Path(prediction_value)
    expected_parent = (Path(row["default_root_dir"]) / "predictions").resolve()
    if not prediction.is_file() or prediction.resolve().parent != expected_parent:
        raise RuntimeError(f"Completed record for {row['cell_id']} has a missing/misplaced prediction")
    with prediction.open(newline="") as handle:
        prediction_rows = list(csv.DictReader(handle, delimiter="\t"))
    header = list(prediction_rows[0]) if prediction_rows else []
    prediction_count = len(prediction_rows)
    required_columns = {"construct_id", "log2_RNA_DNA", "prediction_raw"}
    if not required_columns.issubset(header) or prediction_count != row["source_prediction_rows"]:
        raise RuntimeError(
            f"Completed record for {row['cell_id']} has invalid val predictions "
            f"(columns={header}, rows={prediction_count})"
        )
    construct_ids = [str(item["construct_id"]) for item in prediction_rows]
    if len(set(construct_ids)) != len(construct_ids):
        raise RuntimeError(f"Completed record for {row['cell_id']} has duplicate val IDs")
    if stable_id_hash(construct_ids) != expected_val_hash:
        raise RuntimeError(
            f"Completed record for {row['cell_id']} prediction IDs do not match "
            "the frozen held-out-row hash"
        )

    run_id = record.get("run_id", "").strip()
    provenance = Path(row["default_root_dir"]) / "provenance" / f"{run_id}__run_provenance.json"
    if not run_id or not provenance.is_file():
        raise RuntimeError(f"Completed record for {row['cell_id']} lacks local provenance")
    payload = json.loads(provenance.read_text())
    expected_provenance = expected_registry_fields(row)
    provenance_mismatches = {
        field: {
            "observed": str(payload.get(field, "")),
            "expected": expected,
        }
        for field, expected in expected_provenance.items()
        if str(payload.get(field, "")) != expected
    }
    provenance_mismatches.update(
        {
            field: {
                "observed": str(payload.get(field, "")),
                "expected": expected,
            }
            for field, expected in {
                "run_id": run_id,
                "status": "completed",
                "prediction_path": str(prediction),
            }.items()
            if str(payload.get(field, "")) != expected
        }
    )
    if provenance_mismatches:
        raise RuntimeError(
            f"Completed record for {row['cell_id']} has manifest/provenance "
            "identity mismatches: "
            + json.dumps(provenance_mismatches, indent=2, sort_keys=True)
        )

    run_url = str(record.get("run_url", "")).strip()
    expected_url_fragment = (
        f"/{row['wandb_entity']}/{row['logger_project']}/runs/{run_id}"
    )
    if not run_url or expected_url_fragment not in run_url:
        raise RuntimeError(
            f"Completed record for {row['cell_id']} has an invalid W&B run URL"
        )
    if str(payload.get("run_url", "")) != run_url:
        raise RuntimeError(
            f"Completed record for {row['cell_id']} registry/provenance run URL mismatch"
        )

    split = payload.get("data_split_summary", {})
    if split.get("n_test") != 0:
        raise RuntimeError(f"Completed record for {row['cell_id']} instantiated a test set")
    if split.get("n_val") != row["source_prediction_rows"]:
        raise RuntimeError(f"Completed record for {row['cell_id']} has unexpected n_val")
    if split.get("val_row_id_hash") != expected_val_hash:
        raise RuntimeError(f"Completed record for {row['cell_id']} provenance val hash mismatch")
    split_identity = {
        "data_generation_id": row["data_generation_id"],
        "dataset_sha256": row["dataset_sha256"],
        "split_manifest_id": row["split_manifest_id"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "development_fold": row["development_fold"],
    }
    split_mismatches = {
        field: {"observed": str(split.get(field, "")), "expected": str(expected)}
        for field, expected in split_identity.items()
        if str(split.get(field, "")) != str(expected)
    }
    if split_mismatches:
        raise RuntimeError(
            f"Completed record for {row['cell_id']} has split-provenance "
            "identity mismatches: "
            + json.dumps(split_mismatches, indent=2, sort_keys=True)
        )
    if str(record.get("audit_row_id_hash", "")) != str(
        split.get("audit_row_id_hash", "")
    ):
        raise RuntimeError(
            f"Completed record for {row['cell_id']} registry/provenance audit-exclusion "
            "hash mismatch"
        )
    if payload.get("prediction_path") != str(prediction):
        raise RuntimeError(f"Completed record for {row['cell_id']} provenance prediction mismatch")


def row_completed(
    row: dict, registry: Dict[str, List[dict]], status_dir: Path, manifest_sha: str
) -> bool:
    completed_records = []
    expected = expected_registry_fields(row)
    for record in registry.get(row["cell_id"], []):
        mismatches = {
            field: {"observed": record.get(field, ""), "expected": value}
            for field, value in expected.items()
            if record.get(field, "") != value
        }
        if mismatches:
            raise RuntimeError(
                f"runs.csv provenance collision for cell_id={row['cell_id']}:\n"
                + json.dumps(mismatches, indent=2, sort_keys=True)
            )
        if record.get("status", "").lower() != "completed":
            continue
        validate_completed_record(row, record)
        completed_records.append(record)
    if len(completed_records) > 1:
        raise RuntimeError(f"Multiple completed registry records resolve to {row['cell_id']}")

    marker = status_dir / "done" / f"row_{row['manifest_row']}.done"
    if marker.is_file():
        fields = marker_fields(marker)
        expected_marker = {
            "manifest_row": str(row["manifest_row"]),
            "manifest_sha256": manifest_sha,
            "row_fingerprint": row["row_fingerprint"],
            "cell_id": row["cell_id"],
        }
        mismatches = {
            key: {"observed": fields.get(key), "expected": value}
            for key, value in expected_marker.items()
            if fields.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Completion-marker mismatch for {marker}: {mismatches}")
        if not completed_records:
            raise RuntimeError(f"Completion marker has no validated registry evidence: {marker}")
    return bool(completed_records)


def parse_list(value: str) -> List[str]:
    return [token for token in value.replace(",", " ").split() if token]


def detect_idle_gpus(memory_threshold_mb: int = 512) -> List[str]:
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot detect GPUs; pass --gpus explicitly: {exc}") from exc
    busy = set()
    try:
        pmon = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        for line in pmon.stdout.splitlines():
            fields = line.split()
            if fields and not fields[0].startswith("#") and len(fields) > 1 and fields[1] != "-":
                busy.add(fields[0])
    except OSError:
        pass
    idle = []
    for line in query.stdout.splitlines():
        index, used = (field.strip() for field in line.split(",", 1))
        if index not in busy and int(used) <= memory_threshold_mb:
            idle.append(index)
    return idle


def validate_frozen_inputs() -> Tuple[List[dict], str]:
    required = (MANIFEST, ANALYSIS_MANIFEST, REUSE_MANIFEST, PORTFOLIO, SUMMARY, VERIFIER)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing frozen Stage 3 inputs: {missing}")
    subprocess.run([sys.executable, str(VERIFIER)], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    rows = read_jsonl(MANIFEST)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} manifest rows; found {len(rows)}")
    manifest_sha = sha256_file(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Frozen Stage 3 manifest SHA changed: expected {EXPECTED_MANIFEST_SHA256}, "
            f"observed {manifest_sha}"
        )
    summary = json.loads(SUMMARY.read_text())
    if summary.get("dry_run_manifest_sha256") != manifest_sha:
        raise RuntimeError("Frozen manifest SHA does not match its summary")
    if summary.get("commands_executed") != 0 or summary.get("audit_loader_instantiated") is not False:
        raise RuntimeError("Frozen summary no longer proves a zero-command, audit-isolated dry run")
    return rows, manifest_sha


def select_rows(
    rows: Sequence[dict], args: argparse.Namespace, registry: Dict[str, List[dict]], manifest_sha: str
) -> Tuple[List[dict], int]:
    selected = []
    completed = 0
    wanted_rc = {value.lower() for value in parse_list(args.rc_modes)}
    wanted_folds = {int(value) for value in parse_list(args.folds)}
    wanted_parts = {value.lower() for value in parse_list(args.parts)}
    for row in rows:
        is_complete = row_completed(row, registry, STATUS_DIR, manifest_sha)
        if is_complete:
            completed += 1
        number = int(row["manifest_row"])
        if args.pilot_row is not None and number != args.pilot_row:
            continue
        if args.row_start is not None and number < args.row_start:
            continue
        if args.row_end is not None and number > args.row_end:
            continue
        if wanted_parts and row["part_slug"].lower() not in wanted_parts:
            continue
        if wanted_rc and row["rc_mode"].lower() not in wanted_rc:
            continue
        if wanted_folds and int(row["development_fold"]) not in wanted_folds:
            continue
        if is_complete and not args.include_completed:
            continue
        selected.append(row)
    if args.max_rows is not None:
        selected = selected[: args.max_rows]
    return selected, completed


def check_storage() -> List[dict]:
    evidence = []
    for path, minimum_free_gib, maximum_used_fraction in (
        (Path("/home"), 150, 0.80),
        (Path("/"), 20, 1.00),
    ):
        usage = shutil.disk_usage(str(path))
        free_gib = usage.free // (1024**3)
        used_fraction = usage.used / usage.total
        print(f"Storage {path}: {free_gib} GiB free, {100 * used_fraction:.1f}% used")
        evidence.append(
            {
                "path": str(path),
                "free_gib": int(free_gib),
                "used_fraction": float(used_fraction),
                "minimum_free_gib": int(minimum_free_gib),
                "maximum_used_fraction": float(maximum_used_fraction),
            }
        )
        if free_gib < minimum_free_gib or used_fraction >= maximum_used_fraction:
            raise RuntimeError(f"Storage stop condition reached for {path}")
    return evidence


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


def atomic_write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(body)
    os.replace(str(temporary), str(path))


class CampaignRunner:
    def __init__(
        self, rows: Sequence[dict], gpus: Sequence[str], manifest_sha: str, args: argparse.Namespace
    ) -> None:
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
        self.state_lock = threading.Lock()
        self.active: Dict[int, subprocess.Popen] = {}
        self.completed_rows: List[int] = []
        self.failed_rows: List[int] = []

    def run_row(self, row: dict, gpu: str, worker_id: int) -> bool:
        tokens = shlex.split(row["train_command"])
        if tokens[:2] != ["python", "train_wandb_log.py"]:
            raise RuntimeError(f"Unexpected command prefix for row {row['manifest_row']}")
        command = [sys.executable] + tokens[1:]
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "WANDB_ENTITY": EXPECTED_ENTITY,
                "BODA_WANDB_ENTITY": EXPECTED_ENTITY,
                "WANDB_MODE": "online",
                "WANDB_DIR": str(HERE),
                "BODA_WANDB_PROJECT": row["logger_project"],
                "BODA_CONFIG_PATH": str(MANIFEST),
                "BODA_COMPARISON_GROUP": row["loss_pair_id"],
                "BODA_LAUNCH_SCRIPT": "run_lib1_dedup_stage3_campaign.py",
                "BODA_RUNS_CSV": str(RUNS_CSV),
                "BODA_LAUNCH_NOTES": f"{row['campaign_id']};{row['campaign_stage']}",
            }
        )
        row_number = int(row["manifest_row"])
        log_path = self.log_dir / f"row_{row_number}.log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[{now_iso()}] worker={worker_id} gpu={gpu} start row={row_number} "
            f"cell={row['cell_id']}",
            flush=True,
        )
        with log_path.open("w") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=str(HERE),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            with self.state_lock:
                self.active[worker_id] = process
            status = process.wait()
            with self.state_lock:
                self.active.pop(worker_id, None)

        if status == 0:
            registry = read_registry(RUNS_CSV)
            if not row_completed(row, registry, STATUS_DIR, self.manifest_sha):
                status = 98
        if status == 0:
            run_url = ""
            for line in log_path.read_text(errors="replace").splitlines():
                if line.startswith("Resolved W&B run URL: "):
                    run_url = line.split(": ", 1)[1]
            marker = "\n".join(
                (
                    f"completed_at={now_iso()}",
                    f"manifest_row={row_number}",
                    f"manifest_sha256={self.manifest_sha}",
                    f"row_fingerprint={row['row_fingerprint']}",
                    f"base_config_id={row['base_config_id']}",
                    f"cell_id={row['cell_id']}",
                    f"loss_pair_id={row['loss_pair_id']}",
                    f"rc_pair_id={row['rc_pair_id']}",
                    f"planned_run_name={row['planned_run_name']}",
                    f"wandb_url={run_url}",
                    f"log={log_path}",
                    "",
                )
            )
            atomic_write(self.done_dir / f"row_{row_number}.done", marker)
            with self.state_lock:
                self.completed_rows.append(row_number)
            print(f"[{now_iso()}] worker={worker_id} gpu={gpu} done row={row_number}", flush=True)
            return True

        failure_path = self.failure_dir / f"row_{row_number}.fail"
        failure = "\n".join(
            (
                f"failed_at={now_iso()}",
                f"manifest_row={row_number}",
                f"manifest_sha256={self.manifest_sha}",
                f"row_fingerprint={row['row_fingerprint']}",
                f"cell_id={row['cell_id']}",
                f"status={status}",
                f"log={log_path}",
                "",
            )
        )
        atomic_write(failure_path, failure)
        with self.state_lock:
            self.failed_rows.append(row_number)
        print(
            f"[{now_iso()}] worker={worker_id} gpu={gpu} FAILED row={row_number}; {log_path}",
            file=sys.stderr,
            flush=True,
        )
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
                print(
                    f"Worker {worker_id} crashed on row {row.get('manifest_row')}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                with self.state_lock:
                    self.failed_rows.append(int(row["manifest_row"]))
                if not self.args.continue_on_error:
                    self.stop.set()
            finally:
                self.work.task_done()

    def terminate_active(self) -> None:
        self.stop.set()
        with self.state_lock:
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
            print("Interrupt received; terminating active rows.", file=sys.stderr)
            self.terminate_active()
            for worker in workers:
                worker.join()
            return 130
        if self.failed_rows:
            print(f"Failed rows: {sorted(set(self.failed_rows))}", file=sys.stderr)
            return 1
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Launch selected rows; preview is default")
    parser.add_argument("--pilot-row", type=int, help="Select exactly one manifest row")
    parser.add_argument("--confirm-pilot", action="store_true", help="Required with --execute --pilot-row")
    parser.add_argument(
        "--confirm-full-campaign",
        action="store_true",
        help="Required for any non-pilot execution, including a filtered batch",
    )
    parser.add_argument("--gpus", default="", help="Comma/space-separated physical GPU IDs")
    parser.add_argument("--max-parallel", type=int, help="Maximum workers")
    parser.add_argument("--parts", default="", help="Optional comma-separated part slugs")
    parser.add_argument("--row-start", type=int)
    parser.add_argument("--row-end", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--folds", default="", help="Optional comma-separated development folds")
    parser.add_argument("--rc-modes", default="", help="Optional comma-separated off/on modes")
    parser.add_argument("--include-completed", action="store_true", help="Preview completed cells")
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
        raise ValueError("--max-rows must be >= 1")
    if args.include_completed and args.execute:
        raise ValueError("Refusing --include-completed with --execute; completed cells are immutable")
    if args.execute and args.pilot_row is not None and not args.confirm_pilot:
        raise ValueError("Pilot execution requires --confirm-pilot")
    if args.execute and args.pilot_row is None and not args.confirm_full_campaign:
        raise ValueError("Non-pilot execution requires --confirm-full-campaign")
    if args.confirm_pilot and args.pilot_row is None:
        raise ValueError("--confirm-pilot is valid only with --pilot-row")
    if args.confirm_full_campaign and args.pilot_row is not None:
        raise ValueError("Use --confirm-pilot, not --confirm-full-campaign, for a one-row pilot")
    if args.execute and args.pilot_row is not None and args.pilot_row not in REQUIRED_PILOT_ROWS:
        raise ValueError(
            f"Pilot execution is frozen to manifest rows {REQUIRED_PILOT_ROWS}; "
            "other rows require the non-pilot gate"
        )

    rows, manifest_sha = validate_frozen_inputs()
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = STATUS_DIR / "launcher.lock"
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another Stage 3 runner owns {lock_path}") from exc

        registry = read_registry(RUNS_CSV)
        if args.execute and args.pilot_row == 61:
            transfer_pilot = rows[0]
            if (
                transfer_pilot["cell_id"] != REQUIRED_PILOT_CELLS[1]
                or not row_completed(
                    transfer_pilot, registry, STATUS_DIR, manifest_sha
                )
            ):
                raise RuntimeError(
                    "Scratch pilot row 61 is locked until transfer pilot row 1 "
                    "passes local reconciliation"
                )
        if args.execute and args.pilot_row is None:
            by_row = {int(row["manifest_row"]): row for row in rows}
            pilot_failures = []
            for pilot_row in REQUIRED_PILOT_ROWS:
                pilot = by_row[pilot_row]
                if pilot["cell_id"] != REQUIRED_PILOT_CELLS[pilot_row]:
                    raise RuntimeError(f"Frozen pilot identity changed for row {pilot_row}")
                if not row_completed(pilot, registry, STATUS_DIR, manifest_sha):
                    pilot_failures.append(pilot_row)
            if pilot_failures:
                raise RuntimeError(
                    "Non-pilot execution is locked until the exact transfer and scratch "
                    f"pilots pass local reconciliation; missing rows={pilot_failures}"
                )
        selected, completed = select_rows(rows, args, registry, manifest_sha)
        if args.pilot_row is not None and len(selected) != 1:
            raise RuntimeError(
                f"Pilot row {args.pilot_row} did not resolve to exactly one unfinished row"
            )

        counts_by_part = Counter(row["part_slug"] for row in selected)
        counts_by_route = Counter(row["training_regime"] for row in selected)
        print("Lib1 dedup Stage 3 weighted-loss preflight")
        print(f"  manifest: {MANIFEST}")
        print(f"  manifest SHA256: {manifest_sha}")
        print(f"  total weighted cells: {len(rows)}")
        print(f"  completed cells: {completed}")
        print(f"  selected unfinished cells: {len(selected)}")
        print(f"  selected by part: {dict(sorted(counts_by_part.items()))}")
        print(f"  selected by route: {dict(sorted(counts_by_route.items()))}")
        print(f"  W&B entity: {EXPECTED_ENTITY}")
        print("  audit/test evaluation: disabled by independently validated manifest")
        if selected:
            print(f"  selected row span: {selected[0]['manifest_row']}..{selected[-1]['manifest_row']}")
        if args.show_commands:
            for row in selected:
                print(
                    f"\nrow={row['manifest_row']} part={row['part_slug']} "
                    f"route={row['training_regime']} run={row['planned_run_name']}\n"
                    f"{row['train_command']}"
                )
        if not args.execute:
            print("Preview only; no GPU was claimed and no training command executed.")
            return 0
        if not selected:
            print("No unfinished selected Stage 3 cells remain.")
            return 0
        if Path(sys.prefix).name != "boda_env":
            raise RuntimeError(f"Use boda_env Python; current prefix is {str(sys.prefix)!r}")

        idle_gpus = detect_idle_gpus()
        gpus = parse_list(args.gpus) if args.gpus else idle_gpus
        if not gpus:
            raise RuntimeError("No GPU passed the launch-time idle check")
        if len(gpus) != len(set(gpus)):
            raise ValueError("GPU IDs must be unique")
        unavailable = sorted(set(gpus) - set(idle_gpus))
        if unavailable:
            raise RuntimeError(
                f"Requested GPUs did not pass the launch-time idle check: {unavailable}"
            )
        max_parallel = args.max_parallel or len(gpus)
        if max_parallel < 1 or max_parallel > len(gpus):
            raise ValueError("--max-parallel must be between 1 and the selected GPU count")
        gpus = gpus[:max_parallel]

        wandb_preflight()
        storage_evidence = check_storage()
        preflight = {
            "recorded_at": now_iso(),
            "manifest_sha256": manifest_sha,
            "selected_manifest_rows": [int(row["manifest_row"]) for row in selected],
            "required_pilot_rows": list(REQUIRED_PILOT_ROWS),
            "required_pilots_reconciled": all(
                row_completed(rows[number - 1], registry, STATUS_DIR, manifest_sha)
                for number in REQUIRED_PILOT_ROWS
            ),
            "gpus": list(gpus),
            "storage": storage_evidence,
            "wandb_entity": EXPECTED_ENTITY,
            "wandb_projects": sorted({row["logger_project"] for row in selected}),
            "audit_loader_instantiated": False,
            "test_evaluation_enabled": False,
        }
        preflight_path = (
            STATUS_DIR
            / "preflight"
            / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json"
        )
        atomic_write(preflight_path, json.dumps(preflight, indent=2, sort_keys=True) + "\n")
        print(f"Recorded launch preflight: {preflight_path}")
        atomic_write(STATUS_DIR / "manifest.sha256", manifest_sha + "\n")
        runner = CampaignRunner(selected, gpus, manifest_sha, args)
        print(f"Launching {len(selected)} row(s) with {len(gpus)} worker(s).")
        status = runner.run()
        if status == 0:
            print("All selected Stage 3 rows completed and passed local reconciliation.")
        else:
            print(f"Run stopped with failures; inspect {runner.failure_dir}", file=sys.stderr)
        return status


if __name__ == "__main__":
    raise SystemExit(main())
