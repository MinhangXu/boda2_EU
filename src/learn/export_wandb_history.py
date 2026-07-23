#!/usr/bin/env python3
"""
Export chartable history rows from local W&B run files.

Use this when W&B cloud has run summaries/configs but the Charts tab or
`scan_history` endpoint cannot return history rows. The exporter reads the
local `run-*.wandb` protobuf file directly and writes merged history rows to
TSV files for notebooks and sweep triage.
"""

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore
except Exception as exc:  # pragma: no cover - import failure is user/environment-facing
    print(
        "ERROR: could not import W&B internal readers. Run inside the training "
        f"environment with wandb installed. Details: {exc}",
        file=sys.stderr,
    )
    raise


LEARN_ROOT = Path(__file__).resolve().parent
DEFAULT_WANDB_DIR = LEARN_ROOT / "wandb"
DEFAULT_OUTPUT_DIR = LEARN_ROOT / "run_registry" / "wandb_history_exports"


def parse_value_json(value_json: str) -> Any:
    if value_json == "":
        return ""
    try:
        return json.loads(value_json)
    except Exception:
        return value_json


def cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def history_item_key(item: Any) -> str:
    """Return the metric key for W&B history records across SDK encodings."""
    if item.key:
        return item.key
    nested_key = list(getattr(item, "nested_key", []))
    if nested_key:
        return "/".join(str(part) for part in nested_key)
    return ""


def safe_stem(value: str) -> str:
    value = value or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def discover_run_files(wandb_dir: Path, run_dirs: Iterable[Path]) -> List[Path]:
    explicit_dirs = [Path(p) for p in run_dirs]
    candidates = explicit_dirs if explicit_dirs else sorted(wandb_dir.glob("*run-*"))
    run_files: List[Path] = []
    for run_dir in candidates:
        if run_dir.is_file() and run_dir.suffix == ".wandb":
            run_files.append(run_dir)
            continue
        matches = sorted(run_dir.glob("run-*.wandb"))
        run_files.extend(matches)
    return run_files


def maybe_update_run_metadata(record: Any, metadata: Dict[str, Any], config: Dict[str, Any]) -> None:
    run = record.run
    if run.run_id:
        metadata["run_id"] = run.run_id
    if run.entity:
        metadata["entity"] = run.entity
    if run.project:
        metadata["project"] = run.project
    if run.display_name:
        metadata["run_name"] = run.display_name
    if run.sweep_id:
        metadata["sweep_id"] = run.sweep_id
    if run.host:
        metadata["host"] = run.host
    if run.start_time.seconds:
        metadata["start_time_unix"] = run.start_time.seconds

    for update in run.config.update:
        config[update.key] = parse_value_json(update.value_json)


def fallback_run_id(run_file: Path) -> str:
    return run_file.stem.replace("run-", "", 1)


def read_wandb_metadata(run_file: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    datastore = DataStore()
    datastore.open_for_scan(str(run_file))

    metadata: Dict[str, Any] = {"local_run_file": str(run_file)}
    config: Dict[str, Any] = {}

    while True:
        data = datastore.scan_data()
        if data is None:
            break

        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        maybe_update_run_metadata(record, metadata, config)

        if metadata.get("run_id") and metadata.get("project"):
            break

    metadata.setdefault("run_id", fallback_run_id(run_file))
    metadata.setdefault("project", config.get("logger_project", ""))
    metadata.setdefault("run_name", config.get("run_name", ""))
    metadata.setdefault("sweep_id", config.get("sweep_id", ""))
    metadata["config_model_module"] = config.get("model_module", "")
    metadata["config_data_module"] = config.get("data_module", "")
    metadata["config_use_reverse_complements"] = config.get("use_reverse_complements", "")
    return metadata, config


def read_wandb_file(
    run_file: Path, tolerate_truncated_tail: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], List[str]]:
    datastore = DataStore()
    datastore.open_for_scan(str(run_file))

    metadata: Dict[str, Any] = {"local_run_file": str(run_file)}
    config: Dict[str, Any] = {}
    rows_by_step: "OrderedDict[str, OrderedDict[str, Any]]" = OrderedDict()
    column_order: "OrderedDict[str, None]" = OrderedDict()
    record_index = 0

    while True:
        try:
            data = datastore.scan_data()
        except AssertionError as exc:
            if tolerate_truncated_tail and rows_by_step:
                metadata["history_scan_warning"] = (
                    "truncated_or_invalid_local_wandb_tail: " + str(exc)
                )
                break
            raise
        if data is None:
            break

        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)

        maybe_update_run_metadata(record, metadata, config)

        if record.history.item:
            parsed = OrderedDict()
            for item in record.history.item:
                key = history_item_key(item)
                if not key:
                    continue
                parsed[key] = parse_value_json(item.value_json)
                column_order.setdefault(key, None)

            step_key = parsed.get("_step")
            if step_key is None:
                step_key = f"record_{record_index}"
            else:
                step_key = str(step_key)

            row = rows_by_step.setdefault(step_key, OrderedDict())
            row["_source_record_index"] = record_index
            for key, value in parsed.items():
                row[key] = value

        record_index += 1

    metadata.setdefault("run_id", fallback_run_id(run_file))
    metadata.setdefault("project", config.get("logger_project", ""))
    metadata.setdefault("run_name", config.get("run_name", ""))
    metadata.setdefault("sweep_id", config.get("sweep_id", ""))
    metadata["config_model_module"] = config.get("model_module", "")
    metadata["config_data_module"] = config.get("data_module", "")
    metadata["config_use_reverse_complements"] = config.get("use_reverse_complements", "")

    columns = ["_source_record_index"] + [
        key for key in column_order.keys() if key != "_source_record_index"
    ]
    return metadata, config, [dict(row) for row in rows_by_step.values()], columns


def selected(
    metadata: Dict[str, Any],
    projects: Optional[Iterable[str]],
    run_ids: Optional[Iterable[str]],
    sweep_ids: Optional[Iterable[str]],
) -> bool:
    project_set = set(projects or [])
    run_id_set = set(run_ids or [])
    sweep_id_set = set(sweep_ids or [])

    if project_set and str(metadata.get("project", "")) not in project_set:
        return False
    if run_id_set and str(metadata.get("run_id", "")) not in run_id_set:
        return False
    if sweep_id_set and str(metadata.get("sweep_id", "")) not in sweep_id_set:
        return False
    return True


def write_history(
    output_dir: Path,
    metadata: Dict[str, Any],
    rows: List[Dict[str, Any]],
    columns: List[str],
) -> Optional[Path]:
    if not rows:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    project = safe_stem(str(metadata.get("project", "")))
    run_id = safe_stem(str(metadata.get("run_id", "")))
    output_path = output_dir / f"{project}__{run_id}__history.tsv"

    base_columns = ["run_id", "run_name", "project", "sweep_id"]
    fieldnames = base_columns + [col for col in columns if col not in base_columns]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {
                "run_id": metadata.get("run_id", ""),
                "run_name": metadata.get("run_name", ""),
                "project": metadata.get("project", ""),
                "sweep_id": metadata.get("sweep_id", ""),
            }
            out.update({key: cell(value) for key, value in row.items()})
            writer.writerow(out)

    return output_path


def write_manifest(output_dir: Path, manifest_rows: List[Dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "manifest.tsv"
    fieldnames = [
        "run_id",
        "run_name",
        "project",
        "sweep_id",
        "history_rows",
        "output_path",
        "local_run_file",
        "config_model_module",
        "config_data_module",
        "config_use_reverse_complements",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow({key: cell(row.get(key, "")) for key in fieldnames})
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-dir", type=Path, default=DEFAULT_WANDB_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument("--project", action="append", default=[])
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--sweep-id", action="append", default=[])
    parser.add_argument(
        "--fail-if-empty",
        action="store_true",
        help="Exit non-zero when no selected local W&B histories are exported.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_files = discover_run_files(args.wandb_dir, args.run_dir)
    manifest_rows: List[Dict[str, Any]] = []

    for run_file in run_files:
        file_run_id = fallback_run_id(run_file)
        if args.run_id and file_run_id not in set(args.run_id):
            continue

        try:
            metadata, _metadata_config = read_wandb_metadata(run_file)
        except Exception as exc:
            print(f"WARN: failed to read metadata from {run_file}: {exc}", file=sys.stderr)
            continue

        if not selected(metadata, args.project, args.run_id, args.sweep_id):
            continue

        try:
            metadata, _config, rows, columns = read_wandb_file(run_file)
        except Exception as exc:
            print(f"WARN: failed to parse history from {run_file}: {exc}", file=sys.stderr)
            continue

        output_path = write_history(args.output_dir, metadata, rows, columns)
        if output_path is None:
            continue

        manifest_rows.append(
            {
                **metadata,
                "history_rows": len(rows),
                "output_path": str(output_path),
            }
        )

    manifest_path = write_manifest(args.output_dir, manifest_rows)
    print(f"Exported {len(manifest_rows)} run history file(s).")
    print(f"Manifest: {manifest_path}")
    if args.fail_if_empty and not manifest_rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
