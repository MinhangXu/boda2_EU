#!/usr/bin/env python3
"""Verify that a W&B cloud run has canonical metric history rows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import wandb


LEARN_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_CSV = LEARN_ROOT / "run_registry" / "runs.csv"
DEFAULT_KEYS = ["val_pearson", "val_loss", "trainer/global_step"]
CANARY_KEYS = ["wandb_history_canary", "trainer/global_step"]


def _row_matches(row: Dict[str, str], args: argparse.Namespace) -> bool:
    if not row.get("run_id"):
        return False
    if args.project and row.get("wandb_project") != args.project:
        return False
    if args.entity and row.get("wandb_entity") != args.entity:
        return False
    if args.sweep_id and row.get("wandb_sweep_id") != args.sweep_id:
        return False
    if args.notes and args.notes not in row.get("notes", ""):
        return False
    return bool(row.get("wandb_entity") and row.get("wandb_project"))


def _latest_run_path(args: argparse.Namespace) -> Tuple[str, Dict[str, str]]:
    rows_path = Path(args.runs_csv)
    if not rows_path.exists():
        raise FileNotFoundError(f"runs.csv not found: {rows_path}")

    with rows_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in reversed(rows):
        if _row_matches(row, args):
            return f"{row['wandb_entity']}/{row['wandb_project']}/{row['run_id']}", row
    raise RuntimeError(
        "No matching run found in runs.csv. Pass --run-path directly or relax "
        "--entity/--project/--sweep-id/--notes filters."
    )


def resolve_run_path(args: argparse.Namespace) -> Tuple[str, Optional[Dict[str, str]]]:
    if args.run_path:
        return args.run_path, None
    if args.entity and args.project and args.run_id:
        return f"{args.entity}/{args.project}/{args.run_id}", None
    if args.latest:
        return _latest_run_path(args)
    raise RuntimeError("Use --run-path, --entity/--project/--run-id, or --latest.")


def scan_rows(run: wandb.apis.public.Run, keys: Iterable[str], max_rows: int) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    key_list = list(keys)
    for row in run.scan_history(keys=key_list, page_size=1000):
        rows.append({key: row.get(key) for key in key_list})
        if len(rows) >= max_rows:
            break
    return rows


def complete_rows(rows: Iterable[Dict[str, object]], keys: Iterable[str]) -> List[Dict[str, object]]:
    key_list = list(keys)
    return [
        row
        for row in rows
        if all(row.get(key) is not None for key in key_list)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", help="Full W&B run path: entity/project/run_id.")
    parser.add_argument("--entity", help="W&B entity.")
    parser.add_argument("--project", help="W&B project.")
    parser.add_argument("--run-id", help="W&B run id.")
    parser.add_argument("--latest", action="store_true", help="Use the latest matching row from runs.csv.")
    parser.add_argument("--runs-csv", default=str(DEFAULT_RUNS_CSV), help="runs.csv path for --latest.")
    parser.add_argument("--sweep-id", help="Optional runs.csv sweep id filter for --latest.")
    parser.add_argument("--notes", help="Optional substring filter on runs.csv notes for --latest.")
    parser.add_argument("--keys", nargs="+", default=DEFAULT_KEYS, help="History keys that must co-occur.")
    parser.add_argument("--min-rows", type=int, default=1, help="Minimum complete history rows required.")
    parser.add_argument("--max-rows", type=int, default=1000, help="Maximum history rows to scan.")
    parser.add_argument("--skip-canary", action="store_true", help="Do not also report the W&B canary row.")
    args = parser.parse_args()

    try:
        run_path, registry_row = resolve_run_path(args)
        run = wandb.Api().run(run_path)
        rows = scan_rows(run, args.keys, args.max_rows)
        matches = complete_rows(rows, args.keys)
    except Exception as exc:
        print(f"ERROR: W&B cloud history verification failed: {exc}", file=sys.stderr)
        return 2

    print(f"Run: {run_path}")
    if registry_row:
        print(
            "Registry: "
            f"sweep_id={registry_row.get('wandb_sweep_id', '')} "
            f"notes={registry_row.get('notes', '')}"
        )
    print(f"Keys: {', '.join(args.keys)}")
    print(f"Complete rows found: {len(matches)}")
    if matches:
        print("First complete row:")
        print(json.dumps(matches[0], indent=2, sort_keys=True))

    if not args.skip_canary:
        canary_rows = complete_rows(scan_rows(run, CANARY_KEYS, args.max_rows), CANARY_KEYS)
        print(f"Canary rows found: {len(canary_rows)}")

    if len(matches) < args.min_rows:
        print(
            "ERROR: required metric rows were not found in W&B cloud history. "
            "Check that the pilot finished online and that Charts can see the same run.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
