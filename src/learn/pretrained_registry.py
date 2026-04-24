#!/usr/bin/env python3
"""
Pretrained-model registry helper for `boda2_EU/src/learn`.

This module is the single entry point used by downstream active-learning code
and notebooks whenever they need "the best known model for region X". It
deliberately hides where a model came from (historical recovery, curated
best run, or a freshly finished pilot) behind one `resolve_pretrained` API.

Data sources, in resolution order:
    1. `run_registry/best_runs.csv` — manually curated manifest of the
       canonical model per task_family/target_family/comparison_group.
    2. `run_registry/runs.csv`      — append-only manifest written by
       `train_wandb_log.py` after every run. Used as a fallback when
       best_runs.csv does not yet have an entry for the region, or when
       `prefer=latest` is requested.

Nothing in this module imports torch, so it can be used from lightweight
dashboards. Use `load_model_from_artifact(path)` from
`boda.common.utils.model_fn` to actually materialize a model.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BEST_RUNS_CSV = os.path.join(_HERE, "run_registry", "best_runs.csv")
DEFAULT_RUNS_CSV = os.path.join(_HERE, "run_registry", "runs.csv")


@dataclass
class PretrainedRecord:
    """Normalized view over a single registry row."""

    source: str  # "best_runs" | "runs"
    task_family: str
    target_family: str
    comparison_group: str = ""
    wandb_entity: str = ""
    wandb_project: str = ""
    wandb_sweep_id: str = ""
    run_id: str = ""
    run_name: str = ""
    run_url: str = ""
    config_path: str = ""
    launch_script: str = ""
    model_module: str = ""
    graph_module: str = ""
    data_module: str = ""
    metric_name: str = ""
    metric_value: Optional[float] = None
    artifact_path: str = ""
    model_saved_path: str = ""
    timestamp: str = ""
    notes: str = ""
    raw: Dict[str, str] = field(default_factory=dict)

    def best_artifact(self) -> str:
        """Prefer `model_saved_path` (points at a .tar.gz), fall back to `artifact_path`."""
        return self.model_saved_path or self.artifact_path


def _coerce_float(value: Optional[str]) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _from_best_runs_row(row: Dict[str, str]) -> PretrainedRecord:
    return PretrainedRecord(
        source="best_runs",
        task_family=row.get("task_family", ""),
        target_family=row.get("target_family", ""),
        comparison_group=row.get("comparison_group", ""),
        wandb_entity=row.get("wandb_entity", ""),
        wandb_project=row.get("wandb_project", ""),
        wandb_sweep_id=row.get("sweep_id", ""),
        run_id=row.get("run_id", ""),
        config_path=row.get("config_path", ""),
        launch_script=row.get("launch_script", ""),
        model_module=row.get("model_module", ""),
        graph_module=row.get("graph_module", ""),
        data_module="",
        metric_name=row.get("metric_name", ""),
        metric_value=_coerce_float(row.get("metric_value")),
        artifact_path=row.get("artifact_path", ""),
        model_saved_path=row.get("model_saved_path", ""),
        timestamp=row.get("timestamp", ""),
        notes=row.get("notes", ""),
        raw=dict(row),
    )


def _from_runs_row(row: Dict[str, str]) -> PretrainedRecord:
    metric_name = row.get("best_metric_name") or "epoch_end_val_pearson_r2"
    metric_value = _coerce_float(row.get("best_metric_value"))
    if metric_value is None:
        metric_value = _coerce_float(row.get("val_r2"))
    return PretrainedRecord(
        source="runs",
        task_family=row.get("task_family", ""),
        target_family=row.get("target_family", ""),
        comparison_group=row.get("comparison_group", ""),
        wandb_entity=row.get("wandb_entity", ""),
        wandb_project=row.get("wandb_project", ""),
        wandb_sweep_id=row.get("wandb_sweep_id", ""),
        run_id=row.get("run_id", ""),
        run_name=row.get("run_name", ""),
        run_url=row.get("run_url", ""),
        config_path=row.get("config_path", ""),
        launch_script=row.get("launch_script", ""),
        model_module=row.get("model_module", ""),
        graph_module=row.get("graph_module", ""),
        data_module=row.get("data_module", ""),
        metric_name=metric_name,
        metric_value=metric_value,
        artifact_path=row.get("artifact_path", ""),
        model_saved_path=row.get("artifact_path", ""),  # runs.csv stores the tarball in artifact_path
        timestamp=row.get("timestamp", ""),
        notes=row.get("notes", ""),
        raw=dict(row),
    )


def load_best_runs(path: str = DEFAULT_BEST_RUNS_CSV) -> List[PretrainedRecord]:
    return [_from_best_runs_row(r) for r in _read_csv(path)]


def load_runs(path: str = DEFAULT_RUNS_CSV) -> List[PretrainedRecord]:
    return [_from_runs_row(r) for r in _read_csv(path)]


def list_regions(best_runs_csv: str = DEFAULT_BEST_RUNS_CSV,
                 runs_csv: str = DEFAULT_RUNS_CSV) -> List[str]:
    """Return a sorted set of known `task_family` values across both manifests."""
    seen = set()
    for rec in load_best_runs(best_runs_csv) + load_runs(runs_csv):
        if rec.task_family:
            seen.add(rec.task_family)
    return sorted(seen)


def resolve_pretrained(
    task_family: str,
    target_family: Optional[str] = None,
    comparison_group: Optional[str] = None,
    prefer: str = "best",
    best_runs_csv: str = DEFAULT_BEST_RUNS_CSV,
    runs_csv: str = DEFAULT_RUNS_CSV,
) -> Optional[PretrainedRecord]:
    """
    Pick a single `PretrainedRecord` matching the requested region slice.

    Args:
        task_family: e.g. "enhancer", "promoter", "utr3", "utr5".
        target_family: Optional dataset family filter (e.g. "bashor_in_house").
        comparison_group: Optional exact match on `comparison_group`.
        prefer: "best" (default) consults best_runs.csv first and falls back to
                the best-metric row in runs.csv; "latest" takes the most recent
                row from runs.csv regardless of best_runs.csv.

    Returns:
        A `PretrainedRecord` or None when no registry row matches.
    """
    assert prefer in {"best", "latest"}, prefer

    def _matches(rec: PretrainedRecord) -> bool:
        if rec.task_family != task_family:
            return False
        if target_family and rec.target_family != target_family:
            return False
        if comparison_group and rec.comparison_group != comparison_group:
            return False
        return True

    if prefer == "best":
        for rec in load_best_runs(best_runs_csv):
            if _matches(rec):
                return rec

    runs_matches = [r for r in load_runs(runs_csv) if _matches(r)]
    if not runs_matches:
        return None

    if prefer == "latest":
        return sorted(runs_matches, key=lambda r: r.timestamp or "", reverse=True)[0]

    scored = [r for r in runs_matches if r.metric_value is not None]
    if not scored:
        return sorted(runs_matches, key=lambda r: r.timestamp or "", reverse=True)[0]
    # Higher is better for R2 / Pearson / Spearman (all metrics we track today).
    return sorted(scored, key=lambda r: r.metric_value, reverse=True)[0]


def iter_region_summaries(
    best_runs_csv: str = DEFAULT_BEST_RUNS_CSV,
    runs_csv: str = DEFAULT_RUNS_CSV,
) -> Iterable[Dict[str, str]]:
    """
    Yield one summary dict per known (task_family, target_family) pair, with
    the best metric across both manifests. Intended for READMEs and status
    tables rather than programmatic access.
    """
    best = {}  # (task, target) -> PretrainedRecord
    for rec in load_best_runs(best_runs_csv) + load_runs(runs_csv):
        if not rec.task_family:
            continue
        key = (rec.task_family, rec.target_family)
        cur = best.get(key)
        if cur is None:
            best[key] = rec
            continue
        cur_val = cur.metric_value if cur.metric_value is not None else float("-inf")
        new_val = rec.metric_value if rec.metric_value is not None else float("-inf")
        if new_val > cur_val:
            best[key] = rec

    for (task, target), rec in sorted(best.items()):
        yield {
            "task_family": task,
            "target_family": target,
            "source": rec.source,
            "metric_name": rec.metric_name,
            "metric_value": "" if rec.metric_value is None else f"{rec.metric_value:.4f}",
            "run_id": rec.run_id,
            "artifact": rec.best_artifact(),
            "timestamp": rec.timestamp,
        }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-family", help="e.g. enhancer | promoter | utr3 | utr5")
    parser.add_argument("--target-family", default=None)
    parser.add_argument("--comparison-group", default=None)
    parser.add_argument("--prefer", default="best", choices=["best", "latest"])
    parser.add_argument("--list-regions", action="store_true",
                        help="Print every known task_family and exit.")
    parser.add_argument("--summary", action="store_true",
                        help="Print a status table across all regions.")
    args = parser.parse_args()

    if args.list_regions:
        for region in list_regions():
            print(region)
    elif args.summary:
        rows = list(iter_region_summaries())
        if not rows:
            print("(no registry rows found)")
        else:
            cols = ["task_family", "target_family", "source", "metric_name",
                    "metric_value", "run_id", "timestamp", "artifact"]
            print("\t".join(cols))
            for row in rows:
                print("\t".join(row.get(c, "") for c in cols))
    elif args.task_family:
        rec = resolve_pretrained(
            args.task_family,
            target_family=args.target_family,
            comparison_group=args.comparison_group,
            prefer=args.prefer,
        )
        if rec is None:
            raise SystemExit(f"No registry match for task_family={args.task_family}")
        print(json.dumps({
            "source": rec.source,
            "task_family": rec.task_family,
            "target_family": rec.target_family,
            "comparison_group": rec.comparison_group,
            "wandb_entity": rec.wandb_entity,
            "wandb_project": rec.wandb_project,
            "run_id": rec.run_id,
            "metric_name": rec.metric_name,
            "metric_value": rec.metric_value,
            "config_path": rec.config_path,
            "artifact_path": rec.best_artifact(),
            "timestamp": rec.timestamp,
            "notes": rec.notes,
        }, indent=2))
    else:
        parser.print_help()
