#!/usr/bin/env python3
"""Preview or execute the frozen 15-cell Lib1 dedup final-refit campaign.

Training is strictly pre-audit: every row uses the physically audit-excluded
``final_refit`` DataModule mode, has no validation/test loader, and retains one
portable final-epoch model artifact.  This runner never invokes the later
audit scorer.
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


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "outputs/hpo_manifests/lib1_dedup_final_refit_july2026__dry_run_manifest.jsonl"
VERIFIER = HERE / "verify_lib1_dedup_final_refit_manifest.py"
RUNS_CSV = HERE / "run_registry/runs.csv"
STATUS_DIR = HERE / "outputs/hpo_runs/status/lib1_dedup_final_refit_july2026"
EXPECTED_MANIFEST_SHA256 = "83ec532cf84e83d3477f2e6e8c716a04284fcc43b7d7c4426338a8b0f093582c"
EXPECTED_ENTITY = "minhangxu1998-baylor-college-of-medicine"
EXPECTED_TRAIN_N = {
    "enhancer": 4537,
    "promoter": 7507,
    "intron": 7583,
    "utr3": 6595,
    "utr5": 7972,
}
TEST_FIELDS = (
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


def read_rows() -> list[dict]:
    with MANIFEST.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_registry() -> dict[str, list[dict]]:
    by_cell: dict[str, list[dict]] = {}
    if not RUNS_CSV.is_file():
        return by_cell
    with RUNS_CSV.open(newline="") as handle:
        for record in csv.DictReader(handle):
            cell = record.get("cell_id", "")
            if cell:
                by_cell.setdefault(cell, []).append(record)
    return by_cell


def marker_fields(path: Path) -> dict[str, str]:
    fields = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key] = value
    return fields


def validate_record(row: dict, record: dict) -> tuple[Path, Path]:
    expected = {
        "status": "completed",
        "run_name": row["planned_run_name"],
        "wandb_entity": row["wandb_entity"],
        "wandb_project": row["logger_project"],
        "logger_project": row["logger_project"],
        "campaign_id": "lib1_dedup_phase1_rerun_july2026",
        "campaign_stage": "final_refit",
        "part_slug": row["part_slug"],
        "cell_id": row["cell_id"],
        "base_config_id": row["base_config_id"],
        "architecture": row["architecture"],
        "model_seed": str(row["model_seed"]),
        "loss_mode": row["loss_mode"],
        "rc_mode": row["rc_mode"],
        "artifact_retention": "selected",
    }
    mismatches = {
        key: {"observed": record.get(key, ""), "expected": value}
        for key, value in expected.items()
        if str(record.get(key, "")) != str(value)
    }
    if mismatches:
        raise RuntimeError(
            f"Registry mismatch for {row['cell_id']}: "
            + json.dumps(mismatches, sort_keys=True)
        )
    nonblank_test = {
        key: record.get(key, "")
        for key in TEST_FIELDS
        if record.get(key, "").strip()
    }
    if nonblank_test or record.get("prediction_path", "").strip():
        raise RuntimeError(f"Refit {row['cell_id']} contains test/prediction output")
    run_id = record.get("run_id", "").strip()
    provenance = Path(row["provenance_dir"]) / f"{run_id}__run_provenance.json"
    if not run_id or not provenance.is_file():
        raise RuntimeError(f"Refit {row['cell_id']} lacks compact provenance")
    payload = json.loads(provenance.read_text())
    split = payload.get("data_split_summary", {})
    split_expected = {
        "split_mode": "manifest_final_refit",
        "n_test": 0,
        "n_val": 0,
        "n_train_final": EXPECTED_TRAIN_N[row["part_slug"]],
        "n_source_rows_loaded": EXPECTED_TRAIN_N[row["part_slug"]],
        "train_min_barcodes": 1,
        "train_size_frac": 1.0,
        "train_size_n": None,
        "audit_loader_authorized": False,
        "dataset_sha256": row["dataset_sha256"],
        "split_manifest_sha256": row["split_manifest_sha256"],
    }
    split_mismatches = {
        key: {"observed": split.get(key), "expected": value}
        for key, value in split_expected.items()
        if split.get(key) != value
    }
    if split_mismatches:
        raise RuntimeError(
            f"Split mismatch for {row['cell_id']}: "
            + json.dumps(split_mismatches, sort_keys=True)
        )
    for key in (
        "train_row_id_hash", "normalization_row_id_hash",
        "audit_row_id_hash", "target_normalization_mean",
        "target_normalization_std", "target_normalization_row_count",
    ):
        if split.get(key) in (None, ""):
            raise RuntimeError(f"Refit {row['cell_id']} lacks split field {key}")
    if split["train_row_id_hash"] != split["normalization_row_id_hash"]:
        raise RuntimeError(f"Refit {row['cell_id']} normalization rows differ from train rows")
    if int(split["target_normalization_row_count"]) != EXPECTED_TRAIN_N[row["part_slug"]]:
        raise RuntimeError(f"Refit {row['cell_id']} normalization count mismatch")
    artifact_value = str(payload.get("artifact_path", "") or record.get("artifact_path", "")).strip()
    artifact = Path(artifact_value)
    if not artifact.is_file() or artifact.resolve().parent != Path(row["artifact_dir"]).resolve():
        raise RuntimeError(f"Refit {row['cell_id']} lacks its portable model artifact")
    if str(payload.get("run_id", "")) != run_id:
        raise RuntimeError(f"Refit {row['cell_id']} provenance run ID mismatch")
    return provenance, artifact


def completed(row: dict, registry: dict[str, list[dict]], manifest_sha: str) -> bool:
    candidates = [
        record for record in registry.get(row["cell_id"], [])
        if record.get("status", "").lower() == "completed"
    ]
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple completions for immutable {row['cell_id']}")
    if candidates:
        validate_record(row, candidates[0])
    marker = STATUS_DIR / "done" / f"row_{row['row']}.done"
    if marker.is_file():
        fields = marker_fields(marker)
        expected = {
            "manifest_sha256": manifest_sha,
            "row_fingerprint": row["row_fingerprint"],
            "cell_id": row["cell_id"],
        }
        if any(fields.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Completion marker mismatch for row {row['row']}")
        if not candidates:
            raise RuntimeError(f"Completion marker for row {row['row']} lacks registry evidence")
    return bool(candidates)


def validate_inputs() -> tuple[list[dict], str]:
    subprocess.run([sys.executable, str(VERIFIER)], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    manifest_sha = sha256_file(MANIFEST)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"Frozen manifest SHA changed: {manifest_sha}")
    rows = read_rows()
    if len(rows) != 15:
        raise RuntimeError(f"Expected 15 rows, found {len(rows)}")
    return rows, manifest_sha


def parse_list(value: str) -> list[str]:
    return [token for token in value.replace(",", " ").split() if token]


def idle_gpus() -> list[str]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    result = []
    for line in query.stdout.splitlines():
        index, used = (field.strip() for field in line.split(",", 1))
        if int(used) <= 512:
            result.append(index)
    return result


def wandb_preflight() -> None:
    import wandb
    api = wandb.Api(timeout=15)
    if not getattr(api, "api_key", None):
        raise RuntimeError("No W&B API key resolved")
    next(iter(api.projects(entity=EXPECTED_ENTITY, per_page=1)), None)
    print(f"W&B preflight passed for {EXPECTED_ENTITY}")


def storage_preflight() -> None:
    for path, minimum in ((Path("/home"), 100), (Path("/"), 20)):
        usage = shutil.disk_usage(str(path))
        free = usage.free // (1024 ** 3)
        print(f"Storage {path}: {free} GiB free")
        if free < minimum:
            raise RuntimeError(f"Storage stop condition reached for {path}")


def atomic_write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(body)
    os.replace(str(temporary), str(path))


class Runner:
    def __init__(self, rows: list[dict], gpus: list[str], manifest_sha: str, continue_on_error: bool):
        self.rows = rows
        self.gpus = gpus
        self.manifest_sha = manifest_sha
        self.continue_on_error = continue_on_error
        self.launch_id = f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        self.log_dir = STATUS_DIR / "logs" / self.launch_id
        self.failure_dir = STATUS_DIR / "failures" / self.launch_id
        self.work = queue.Queue()
        for row in rows:
            self.work.put(row)
        self.stop = threading.Event()
        self.failed: list[int] = []
        self.lock = threading.Lock()

    def run_one(self, row: dict, gpu: str, worker: int) -> None:
        tokens = shlex.split(row["train_command"])
        command = [sys.executable] + tokens[1:]
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": gpu,
            "WANDB_ENTITY": EXPECTED_ENTITY,
            "BODA_WANDB_ENTITY": EXPECTED_ENTITY,
            "WANDB_MODE": "online",
            "WANDB_DIR": str(HERE),
            "BODA_WANDB_PROJECT": row["logger_project"],
            "BODA_CONFIG_PATH": str(MANIFEST),
            "BODA_COMPARISON_GROUP": "final_refit",
            "BODA_LAUNCH_SCRIPT": "run_lib1_dedup_final_refit_campaign.py",
            "BODA_RUNS_CSV": str(RUNS_CSV),
            "BODA_LAUNCH_NOTES": "locked 15-cell development-only final refit",
        })
        number = int(row["row"])
        log_path = self.log_dir / f"row_{number}.log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{now_iso()}] worker={worker} gpu={gpu} start row={number} {row['part_slug']} seed={row['model_seed']}", flush=True)
        with log_path.open("w") as handle:
            status = subprocess.run(command, cwd=HERE, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        if status == 0:
            registry = read_registry()
            records = [r for r in registry.get(row["cell_id"], []) if r.get("status", "").lower() == "completed"]
            if len(records) != 1:
                status = 98
            else:
                provenance, artifact = validate_record(row, records[0])
        if status == 0:
            marker = "\n".join((
                f"completed_at={now_iso()}",
                f"manifest_row={number}",
                f"manifest_sha256={self.manifest_sha}",
                f"row_fingerprint={row['row_fingerprint']}",
                f"cell_id={row['cell_id']}",
                f"artifact={artifact}",
                f"artifact_sha256={sha256_file(artifact)}",
                f"provenance={provenance}",
                f"log={log_path}", "",
            ))
            atomic_write(STATUS_DIR / "done" / f"row_{number}.done", marker)
            print(f"[{now_iso()}] worker={worker} gpu={gpu} done row={number}", flush=True)
            return
        failure = self.failure_dir / f"row_{number}.fail"
        atomic_write(failure, f"failed_at={now_iso()}\nstatus={status}\nlog={log_path}\n")
        with self.lock:
            self.failed.append(number)
        print(f"[{now_iso()}] worker={worker} gpu={gpu} FAILED row={number}; {log_path}", file=sys.stderr, flush=True)
        if not self.continue_on_error:
            self.stop.set()

    def worker(self, gpu: str, worker: int) -> None:
        while not self.stop.is_set():
            try:
                row = self.work.get_nowait()
            except queue.Empty:
                return
            try:
                self.run_one(row, gpu, worker)
            finally:
                self.work.task_done()

    def run(self) -> int:
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        threads = [threading.Thread(target=self.worker, args=(gpu, i + 1)) for i, gpu in enumerate(self.gpus)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return 1 if self.failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pilot-row", type=int)
    parser.add_argument("--confirm-pilot", action="store_true")
    parser.add_argument("--confirm-full-campaign", action="store_true")
    parser.add_argument("--gpus", default="")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--parts", default="")
    parser.add_argument("--row-start", type=int)
    parser.add_argument("--row-end", type=int)
    parser.add_argument("--show-commands", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.execute and args.pilot_row is not None and not args.confirm_pilot:
        raise ValueError("Pilot execution requires --confirm-pilot")
    if args.execute and args.pilot_row is None and not args.confirm_full_campaign:
        raise ValueError("Full execution requires --confirm-full-campaign")
    if args.confirm_pilot and args.pilot_row is None:
        raise ValueError("--confirm-pilot requires --pilot-row")
    if args.pilot_row is not None and args.pilot_row != 1:
        raise ValueError("The frozen final-refit pilot is row 1")

    rows, manifest_sha = validate_inputs()
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    with (STATUS_DIR / "launcher.lock").open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another final-refit launcher is active") from exc
        registry = read_registry()
        finished = sum(completed(row, registry, manifest_sha) for row in rows)
        selected = []
        wanted_parts = set(parse_list(args.parts))
        for row in rows:
            if completed(row, registry, manifest_sha):
                continue
            if args.pilot_row is not None and int(row["row"]) != args.pilot_row:
                continue
            if args.row_start is not None and int(row["row"]) < args.row_start:
                continue
            if args.row_end is not None and int(row["row"]) > args.row_end:
                continue
            if wanted_parts and row["part_slug"] not in wanted_parts:
                continue
            selected.append(row)
        if args.execute and args.pilot_row is None and not completed(rows[0], registry, manifest_sha):
            raise RuntimeError("Full execution is locked until pilot row 1 reconciles")
        print("Lib1 dedup final-refit preflight")
        print(f"  manifest: {MANIFEST}")
        print(f"  manifest SHA256: {manifest_sha}")
        print(f"  completed cells: {finished}/15")
        print(f"  selected unfinished cells: {len(selected)}")
        print(f"  selected by part: {dict(Counter(row['part_slug'] for row in selected))}")
        print("  validation loader: disabled")
        print("  audit/test loader: disabled")
        if args.show_commands:
            for row in selected:
                print(f"\nrow={row['row']} {row['part_slug']} seed={row['model_seed']}\n{row['train_command']}")
        if not args.execute:
            print("Preview only; no training command executed.")
            return 0
        if not selected:
            print("No unfinished selected refits remain.")
            return 0
        if Path(sys.prefix).name != "boda_env":
            raise RuntimeError(f"Use boda_env Python; current prefix is {sys.prefix}")
        available = idle_gpus()
        gpus = parse_list(args.gpus) if args.gpus else available
        unavailable = sorted(set(gpus) - set(available))
        if unavailable:
            raise RuntimeError(f"Requested GPUs are not idle: {unavailable}")
        if not gpus or len(gpus) != len(set(gpus)):
            raise RuntimeError("At least one unique idle GPU is required")
        max_parallel = args.max_parallel or len(gpus)
        gpus = gpus[:max_parallel]
        wandb_preflight()
        storage_preflight()
        preflight = {
            "recorded_at": now_iso(), "manifest_sha256": manifest_sha,
            "selected_rows": [row["row"] for row in selected], "gpus": gpus,
            "audit_loader_instantiated": False, "test_evaluation_enabled": False,
        }
        atomic_write(
            STATUS_DIR / "preflight" / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.json",
            json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        )
        runner = Runner(selected, gpus, manifest_sha, args.continue_on_error)
        print(f"Launching {len(selected)} row(s) with {len(gpus)} worker(s).")
        status = runner.run()
        if status == 0:
            print("All selected final refits completed and reconciled.")
        return status


if __name__ == "__main__":
    raise SystemExit(main())
