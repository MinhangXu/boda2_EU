#!/usr/bin/env python3
"""Validate and print or execute exactly one targeted 3'UTR HPO pilot row."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PREFIX = HERE / "outputs/hpo_manifests/lib1_dedup_utr3_targeted_hpo_july2026"
DEFAULT_MANIFEST = Path(str(PREFIX) + "__dry_run_manifest.jsonl")
EXPECTED_ENTITY = "minhangxu1998-baylor-college-of-medicine"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=int, default=1, help="One-based manifest row; default 1")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the single row after validation; otherwise only print it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subprocess.run(
        [sys.executable, str(HERE / "verify_lib1_dedup_utr3_targeted_hpo_manifest.py")],
        cwd=HERE,
        check=True,
    )
    with DEFAULT_MANIFEST.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    matches = [row for row in rows if row.get("manifest_row") == args.row]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one manifest row {args.row}; found {len(matches)}")
    row = matches[0]
    if row["wandb_entity"] != EXPECTED_ENTITY:
        raise ValueError("Pilot row targets an unexpected W&B entity")
    if row["evaluate_test_after_fit"] is not False or row["prediction_splits"] != ["val"]:
        raise ValueError("Pilot row violates audit isolation")
    tokens = shlex.split(row["train_command"])
    if tokens[:2] != ["python", "train_wandb_log.py"]:
        raise ValueError("Pilot command has an unexpected entry point")

    print(f"Validated one-row pilot: row={args.row} cell_id={row['cell_id']}")
    print(f"run_name={row['planned_run_name']}")
    print(row["train_command"])
    if not args.execute:
        print("Dry run only. Add --execute to launch this one row.")
        return

    env = os.environ.copy()
    env["WANDB_ENTITY"] = EXPECTED_ENTITY
    env["BODA_WANDB_ENTITY"] = EXPECTED_ENTITY
    env["WANDB_MODE"] = "online"
    env["WANDB_DIR"] = str(HERE)
    execution_tokens = [sys.executable, *tokens[1:]]
    subprocess.run(execution_tokens, cwd=HERE, env=env, check=True)


if __name__ == "__main__":
    main()
