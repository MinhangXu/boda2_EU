#!/usr/bin/env python3
"""Validate and run the frozen 240-cell targeted 3'UTR HPO manifest.

The default is a read-only queue preview.  Training requires an explicit
``--execute`` flag.  Completion is reconciled from exact-provenance registry
records and fingerprinted local markers, so the successful one-row pilot is
skipped automatically and interrupted launches are safely resumable.
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
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PREFIX = HERE / "outputs/hpo_manifests/lib1_dedup_utr3_targeted_hpo_july2026"
MANIFEST = Path(str(PREFIX) + "__dry_run_manifest.jsonl")
SEARCH_CONFIGS = Path(str(PREFIX) + "__search_configs.jsonl")
SUMMARY = Path(str(PREFIX) + "__summary.json")
STAGE2_MANIFEST = HERE / "outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl"
VERIFIER = HERE / "verify_lib1_dedup_utr3_targeted_hpo_manifest.py"
RUNS_CSV = HERE / "run_registry/runs.csv"
STATUS_DIR = HERE / "outputs/hpo_runs/status/lib1_dedup_utr3_targeted_hpo_july2026"
EXPECTED_ENTITY = "minhangxu1998-baylor-college-of-medicine"
EXPECTED_PROJECT = "utr3__bashor_in_house__dedup_exact_v1__targeted_hpo_development"
EXPECTED_ROWS = 240
EXPECTED_MANIFEST_SHA256 = (
    "8ea6b205816a55b80221ce44393fa488f1641c4f1df8be8590c27a4a67ea1f4a"
)
EXPECTED_VAL_COUNT = 105
EXPECTED_VAL_HASHES = {
    0: "3815a9763c386ed3aa3105e6914e86d3b740c1d125cb63e36a85f602ad11ead2",
    1: "59da104761cf4ba9bbab8b1f74b161db670e9b44e2055793143fd9bd747e7993",
    2: "7ffcc5f4e0e45e6c83fc05ac2048c85a8c4ee1cffb08230f11fce26315697c4c",
    3: "1b90ff7726c7f64241c06e4887e12f5b5a8e697ee31c5e7c2de5cf75266b5629",
    4: "db32461ccf98c667eb732da0c431bd5157a10615d566779db993f0a334f8bc05",
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
    fields = (
        "planned_run_name",
        "wandb_entity",
        "logger_project",
        "campaign_id",
        "campaign_stage",
        "part_slug",
        "analysis_lane",
        "challenger_family",
        "policy_id",
        "config_origin",
        "training_regime",
        "cell_id",
        "rc_pair_id",
        "rc_mode",
        "execution_disposition",
        "initialization",
        "input_policy",
        "data_generation_id",
        "dataset_sha256",
        "split_manifest_id",
        "split_manifest_sha256",
        "development_fold",
        "base_config_id",
        "architecture",
        "model_seed",
        "loss_mode",
        "target_definition",
        "length_policy",
        "artifact_retention",
    )
    expected = {field: str(row.get(field, "")) for field in fields}
    expected["run_name"] = expected.pop("planned_run_name")
    expected["wandb_project"] = row["logger_project"]
    return expected


def validate_completed_record(row: dict, record: dict) -> None:
    fold = int(row["development_fold"])
    expected_val_hash = EXPECTED_VAL_HASHES[fold]
    if record.get("val_row_id_hash", "") != expected_val_hash:
        raise RuntimeError(
            "Completed record for {} has the wrong fold-{} validation hash".format(
                row["cell_id"], fold
            )
        )
    nonblank_test = {
        field: record.get(field, "")
        for field in TEST_METRIC_FIELDS
        if record.get(field, "").strip()
    }
    if nonblank_test:
        raise RuntimeError(
            "Completed record for {} contains forbidden test metrics: {}".format(
                row["cell_id"], nonblank_test
            )
        )
    prediction_value = record.get("prediction_path", "").strip()
    if not prediction_value:
        raise RuntimeError("Completed record for {} lacks prediction_path".format(row["cell_id"]))
    prediction = Path(prediction_value)
    expected_parent = (Path(row["default_root_dir"]) / "predictions").resolve()
    if not prediction.is_file() or prediction.resolve().parent != expected_parent:
        raise RuntimeError(
            "Completed record for {} has a missing/misplaced prediction".format(row["cell_id"])
        )
    with prediction.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        prediction_count = sum(1 for line in handle if line.strip())
    required_columns = {"construct_id", "log2_RNA_DNA", "prediction_raw"}
    if not required_columns.issubset(header) or prediction_count != EXPECTED_VAL_COUNT:
        raise RuntimeError(
            "Completed record for {} has an invalid val prediction export "
            "(columns={}, rows={})".format(row["cell_id"], header, prediction_count)
        )

    run_id = record.get("run_id", "").strip()
    provenance = Path(row["default_root_dir"]) / "provenance" / "{}__run_provenance.json".format(run_id)
    if not run_id or not provenance.is_file():
        raise RuntimeError("Completed record for {} lacks local provenance".format(row["cell_id"]))
    payload = json.loads(provenance.read_text())
    split = payload.get("data_split_summary", {})
    if split.get("n_test") != 0:
        raise RuntimeError("Completed record for {} instantiated a test set".format(row["cell_id"]))
    if split.get("n_val") != EXPECTED_VAL_COUNT:
        raise RuntimeError("Completed record for {} has an unexpected n_val".format(row["cell_id"]))
    if split.get("val_row_id_hash") != expected_val_hash:
        raise RuntimeError("Completed record for {} provenance val hash mismatch".format(row["cell_id"]))
    if payload.get("prediction_path") != str(prediction):
        raise RuntimeError("Completed record for {} provenance prediction mismatch".format(row["cell_id"]))


def row_completed(row: dict, registry: Dict[str, List[dict]], status_dir: Path) -> bool:
    completed_records = []
    for record in registry.get(row["cell_id"], []):
        expected = expected_registry_fields(row)
        mismatches = {
            field: {"observed": record.get(field, ""), "expected": value}
            for field, value in expected.items()
            if record.get(field, "") != value
        }
        if mismatches:
            raise RuntimeError(
                "runs.csv provenance collision for cell_id={}:\n{}".format(
                    row["cell_id"], json.dumps(mismatches, indent=2, sort_keys=True)
                )
            )
        if record.get("status", "").lower() != "completed":
            continue
        validate_completed_record(row, record)
        completed_records.append(record)
    if len(completed_records) > 1:
        raise RuntimeError(
            "Multiple completed registry records resolve to cell_id={}".format(row["cell_id"])
        )

    marker = status_dir / "done" / "row_{}.done".format(row["manifest_row"])
    if marker.is_file():
        fields = marker_fields(marker)
        if fields.get("row_fingerprint") != row["row_fingerprint"]:
            raise RuntimeError("Completion-marker fingerprint mismatch: {}".format(marker))
        if fields.get("manifest_row") != str(row["manifest_row"]):
            raise RuntimeError("Completion-marker row mismatch: {}".format(marker))
        if fields.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256:
            raise RuntimeError("Completion-marker manifest SHA mismatch: {}".format(marker))
        if not completed_records:
            raise RuntimeError(
                "Completion marker has no validated registry/prediction evidence: {}".format(marker)
            )
    return bool(completed_records)


def parse_gpu_list(value: str) -> List[str]:
    return [token for token in value.replace(",", " ").split() if token]


def detect_idle_gpus(memory_threshold_mb: int = 512) -> List[str]:
    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Cannot detect GPUs; pass --gpus explicitly: {}".format(exc))
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
    required = (MANIFEST, SEARCH_CONFIGS, SUMMARY, STAGE2_MANIFEST, VERIFIER)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing frozen targeted-HPO inputs: {}".format(missing))
    subprocess.run(
        [
            sys.executable,
            str(VERIFIER),
            "--manifest",
            str(MANIFEST),
            "--search-configs",
            str(SEARCH_CONFIGS),
            "--summary",
            str(SUMMARY),
            "--stage2-analysis-manifest",
            str(STAGE2_MANIFEST),
        ],
        cwd=HERE,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    rows = read_jsonl(MANIFEST)
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError("Expected {} manifest rows; found {}".format(EXPECTED_ROWS, len(rows)))
    manifest_sha = sha256_file(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen targeted-HPO manifest SHA changed: expected {}, observed {}".format(
                EXPECTED_MANIFEST_SHA256, manifest_sha
            )
        )
    summary = json.loads(SUMMARY.read_text())
    if summary.get("dry_run_manifest_sha256") != manifest_sha:
        raise RuntimeError("Frozen manifest SHA does not match its summary")
    return rows, manifest_sha


def select_rows(rows: Sequence[dict], args: argparse.Namespace, registry) -> Tuple[List[dict], int]:
    selected = []
    completed = 0
    wanted_rc = set(parse_gpu_list(args.rc_modes.lower())) if args.rc_modes else set()
    wanted_folds = {int(value) for value in parse_gpu_list(args.folds)} if args.folds else set()
    for row in rows:
        is_complete = row_completed(row, registry, STATUS_DIR)
        if is_complete:
            completed += 1
        number = int(row["manifest_row"])
        if args.row_start is not None and number < args.row_start:
            continue
        if args.row_end is not None and number > args.row_end:
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


def check_storage() -> None:
    for path, minimum_free_gib, maximum_used_fraction in (
        (Path("/home"), 150, 0.80),
        (Path("/"), 20, 1.00),
    ):
        usage = shutil.disk_usage(str(path))
        free_gib = usage.free // (1024 ** 3)
        used_fraction = usage.used / usage.total
        print(
            "Storage {}: {} GiB free, {:.1f}% used".format(
                path, free_gib, 100 * used_fraction
            )
        )
        if free_gib < minimum_free_gib or used_fraction >= maximum_used_fraction:
            raise RuntimeError("Storage stop condition reached for {}".format(path))


def wandb_preflight() -> None:
    import wandb

    api = wandb.Api(timeout=15)
    if not getattr(api, "api_key", None):
        raise RuntimeError("No W&B API key resolved; run `wandb login` before launch")
    try:
        next(iter(api.projects(entity=EXPECTED_ENTITY, per_page=1)), None)
    except Exception as exc:
        raise RuntimeError("W&B access preflight failed: {}".format(exc))
    print("W&B preflight passed for {}".format(EXPECTED_ENTITY))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    temporary.write_text(text)
    os.replace(str(temporary), str(path))


class CampaignRunner:
    def __init__(
        self,
        rows: Sequence[dict],
        gpus: Sequence[str],
        manifest_sha: str,
        args: argparse.Namespace,
    ) -> None:
        self.rows = list(rows)
        self.gpus = list(gpus)
        self.manifest_sha = manifest_sha
        self.args = args
        self.launch_id = "{}_{}".format(dt.datetime.now().strftime("%Y%m%d_%H%M%S"), os.getpid())
        self.log_dir = STATUS_DIR / "logs" / self.launch_id
        self.failure_dir = STATUS_DIR / "failures" / self.launch_id
        self.done_dir = STATUS_DIR / "done"
        self.monitor_path = STATUS_DIR / "monitor.tsv"
        self.work: "queue.Queue[dict]" = queue.Queue()
        for row in self.rows:
            self.work.put(row)
        self.stop = threading.Event()
        self.monitor_stop = threading.Event()
        self.state_lock = threading.Lock()
        self.active: Dict[int, subprocess.Popen] = {}
        self.completed_rows: List[int] = []
        self.failed_rows: List[int] = []

    def run_row(self, row: dict, gpu: str, worker_id: int) -> bool:
        tokens = shlex.split(row["train_command"])
        if tokens[:2] != ["python", "train_wandb_log.py"]:
            raise RuntimeError("Unexpected command prefix for row {}".format(row["manifest_row"]))
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
                "BODA_COMPARISON_GROUP": row["analysis_lane"],
                "BODA_LAUNCH_SCRIPT": "run_lib1_dedup_utr3_targeted_hpo_campaign.py",
                "BODA_RUNS_CSV": str(RUNS_CSV),
                "BODA_LAUNCH_NOTES": "{};{}".format(
                    row["campaign_id"], row["campaign_stage"]
                ),
            }
        )
        row_number = int(row["manifest_row"])
        log_path = self.log_dir / "row_{}.log".format(row_number)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[{}] worker={} gpu={} start row={} cell={}".format(
                now_iso(), worker_id, gpu, row_number, row["cell_id"]
            ),
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
            # The training process appends runs.csv before returning. Reconcile
            # its exact provenance and prediction path before marking the row.
            registry = read_registry(RUNS_CSV)
            if not row_completed(row, registry, STATUS_DIR):
                status = 98
        if status == 0:
            run_url = ""
            for line in log_path.read_text(errors="replace").splitlines():
                if line.startswith("Resolved W&B run URL: "):
                    run_url = line.split(": ", 1)[1]
            marker = "\n".join(
                (
                    "completed_at={}".format(now_iso()),
                    "manifest_row={}".format(row_number),
                    "manifest_sha256={}".format(self.manifest_sha),
                    "row_fingerprint={}".format(row["row_fingerprint"]),
                    "base_config_id={}".format(row["base_config_id"]),
                    "cell_id={}".format(row["cell_id"]),
                    "rc_pair_id={}".format(row["rc_pair_id"]),
                    "planned_run_name={}".format(row["planned_run_name"]),
                    "wandb_url={}".format(run_url),
                    "log={}".format(log_path),
                    "",
                )
            )
            atomic_write(self.done_dir / "row_{}.done".format(row_number), marker)
            with self.state_lock:
                self.completed_rows.append(row_number)
            print("[{}] worker={} gpu={} done row={}".format(now_iso(), worker_id, gpu, row_number), flush=True)
            return True

        failure_path = self.failure_dir / "row_{}.fail".format(row_number)
        failure = "\n".join(
            (
                "failed_at={}".format(now_iso()),
                "manifest_row={}".format(row_number),
                "manifest_sha256={}".format(self.manifest_sha),
                "row_fingerprint={}".format(row["row_fingerprint"]),
                "cell_id={}".format(row["cell_id"]),
                "status={}".format(status),
                "log={}".format(log_path),
                "",
            )
        )
        atomic_write(failure_path, failure)
        with self.state_lock:
            self.failed_rows.append(row_number)
        print(
            "[{}] worker={} gpu={} FAILED row={}; {}".format(
                now_iso(), worker_id, gpu, row_number, log_path
            ),
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
                    "Worker {} crashed on row {}: {}".format(
                        worker_id, row.get("manifest_row"), exc
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                with self.state_lock:
                    self.failed_rows.append(int(row["manifest_row"]))
                if not self.args.continue_on_error:
                    self.stop.set()
            finally:
                self.work.task_done()

    def monitor(self) -> None:
        self.monitor_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.monitor_path.is_file():
            self.monitor_path.write_text(
                "timestamp\tlaunch_id\tselected\tcompleted_launch\tactive\tfailed_launch\tqueued\n"
            )
        while not self.monitor_stop.wait(self.args.monitor_interval):
            with self.state_lock:
                line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                    now_iso(),
                    self.launch_id,
                    len(self.rows),
                    len(self.completed_rows),
                    len(self.active),
                    len(self.failed_rows),
                    self.work.qsize(),
                )
            with self.monitor_path.open("a") as handle:
                handle.write(line)
            print("Monitor: " + line.rstrip(), flush=True)

    def terminate_active(self) -> None:
        self.stop.set()
        with self.state_lock:
            processes = list(self.active.values())
        for process in processes:
            process.terminate()
        deadline = time.time() + 20
        for process in processes:
            remaining = max(0.0, deadline - time.time())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()

    def run(self) -> int:
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        monitor = threading.Thread(target=self.monitor, name="monitor", daemon=True)
        monitor.start()
        workers = [
            threading.Thread(
                target=self.worker,
                args=(gpu, index + 1),
                name="gpu-{}".format(gpu),
            )
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
        finally:
            self.monitor_stop.set()
            monitor.join(timeout=5)
        if self.failed_rows:
            print("Failed rows: {}".format(sorted(set(self.failed_rows))), file=sys.stderr)
            return 1
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Launch selected rows; default is preview only")
    parser.add_argument(
        "--confirm-full-campaign",
        action="store_true",
        help="Required acknowledgement for any campaign execution",
    )
    parser.add_argument("--gpus", default="", help="Comma/space-separated physical GPU IDs; auto-detect if omitted")
    parser.add_argument("--max-parallel", type=int, default=None, help="Maximum workers; defaults to selected GPU count")
    parser.add_argument("--row-start", type=int)
    parser.add_argument("--row-end", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--folds", default="", help="Optional comma-separated development folds")
    parser.add_argument("--rc-modes", default="", help="Optional comma-separated off/on modes")
    parser.add_argument("--include-completed", action="store_true", help="Include completed cells (normally unsafe)")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep claiming rows after a failure")
    parser.add_argument("--skip-storage-check", action="store_true")
    parser.add_argument("--allow-manifest-change", action="store_true")
    parser.add_argument("--show-commands", action="store_true", help="Print every selected training command in preview")
    parser.add_argument("--monitor-interval", type=int, default=60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.row_start is not None and args.row_start < 1:
        raise ValueError("--row-start must be >= 1")
    if args.row_end is not None and args.row_end > EXPECTED_ROWS:
        raise ValueError("--row-end must be <= {}".format(EXPECTED_ROWS))
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be >= 1")
    if args.include_completed and args.execute:
        raise ValueError("Refusing --include-completed with --execute; completed cells are immutable")
    if args.execute and not args.confirm_full_campaign:
        raise ValueError("--execute requires --confirm-full-campaign")

    rows, manifest_sha = validate_frozen_inputs()
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = STATUS_DIR / "launcher.lock"
    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another targeted-HPO campaign runner owns {}".format(lock_path))

        registry = read_registry(RUNS_CSV)
        selected, completed = select_rows(rows, args, registry)
        sha_path = STATUS_DIR / "manifest.sha256"
        if sha_path.is_file():
            previous = sha_path.read_text().strip()
            if previous != manifest_sha and completed and not args.allow_manifest_change:
                raise RuntimeError(
                    "Manifest SHA changed after completed rows: {} -> {}".format(previous, manifest_sha)
                )

        gpus = parse_gpu_list(args.gpus) if args.gpus else detect_idle_gpus()
        if not gpus:
            raise RuntimeError("No idle GPUs found; pass --gpus explicitly when appropriate")
        if len(gpus) != len(set(gpus)):
            raise ValueError("GPU IDs must be unique")
        max_parallel = args.max_parallel or len(gpus)
        if max_parallel < 1 or max_parallel > len(gpus):
            raise ValueError("--max-parallel must be between 1 and the selected GPU count")
        gpus = gpus[:max_parallel]

        counts = Counter(row["rc_mode"] for row in selected)
        print("Targeted 3'UTR HPO campaign preflight")
        print("  manifest: {}".format(MANIFEST))
        print("  manifest SHA256: {}".format(manifest_sha))
        print("  total cells: {}".format(len(rows)))
        print("  completed cells: {}".format(completed))
        print("  selected remaining cells: {} ({})".format(len(selected), dict(sorted(counts.items()))))
        print("  W&B: {}/{}".format(EXPECTED_ENTITY, EXPECTED_PROJECT))
        print("  GPUs/workers: {}".format(" ".join(gpus)))
        print("  audit/test evaluation: disabled by validated manifest")

        if selected:
            print(
                "  selected row span: {}..{}".format(
                    selected[0]["manifest_row"], selected[-1]["manifest_row"]
                )
            )
        if args.show_commands:
            for row, gpu in zip(selected, (gpus * (len(selected) // len(gpus) + 1))):
                print(
                    "\nrow={} gpu={} run={}\nCUDA_VISIBLE_DEVICES={} {}".format(
                        row["manifest_row"], gpu, row["planned_run_name"], gpu, row["train_command"]
                    )
                )
        if not args.execute:
            print("Preview only; no training command executed. Add --execute to launch.")
            return 0
        if not selected:
            print("All 240 targeted-HPO cells are already complete.")
            return 0
        if Path(sys.prefix).name != "boda_env":
            raise RuntimeError(
                "Use boda_env Python; current prefix is {!r}".format(str(sys.prefix))
            )
        wandb_preflight()
        if not args.skip_storage_check:
            check_storage()
        atomic_write(sha_path, manifest_sha + "\n")

        runner = CampaignRunner(selected, gpus, manifest_sha, args)
        print("Launching {} row(s) with {} worker(s).".format(len(selected), len(gpus)))
        status = runner.run()
        if status == 0:
            print("All selected targeted-HPO rows completed.")
        else:
            print(
                "Campaign stopped with failures; inspect {}".format(runner.failure_dir),
                file=sys.stderr,
            )
        return status


if __name__ == "__main__":
    raise SystemExit(main())
