"""Utilities for notebook-friendly HPO result recovery and comparison."""

# pyright: reportMissingImports=false

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


DEFAULT_WANDB_ROOT = Path(__file__).resolve().parents[1] / "learn" / "wandb"


def _as_path(path_like: str | Path) -> Path:
    return Path(path_like).expanduser().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data or {}


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for notebook-side HPO result evaluation utilities."
        ) from exc
    return pd


def _parameter_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _scalar_summary_items(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in summary.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _find_first_key(tree: Any, keys: Iterable[str]) -> Any:
    key_set = set(keys)
    if isinstance(tree, Mapping):
        for key in key_set:
            if key in tree:
                return tree[key]
        for value in tree.values():
            found = _find_first_key(value, key_set)
            if found is not None:
                return found
    elif isinstance(tree, list):
        for value in tree:
            found = _find_first_key(value, key_set)
            if found is not None:
                return found
    return None


def iter_wandb_run_dirs(wandb_root: str | Path = DEFAULT_WANDB_ROOT) -> list[Path]:
    """Return local cached W&B run directories in sorted order."""
    root = _as_path(wandb_root)
    if not root.exists():
        return []
    return sorted(path for path in root.glob("run-*") if path.is_dir())


def load_run_config(run_dir: str | Path) -> dict[str, Any]:
    """Load the cached W&B config for one run."""
    run_path = _as_path(run_dir)
    return _load_yaml(run_path / "files" / "config.yaml")


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    """Load the cached W&B summary for one run."""
    run_path = _as_path(run_dir)
    return _load_json(run_path / "files" / "wandb-summary.json")


def extract_model_saved_path(
    config: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> str | None:
    """Recover the saved model path from either config or summary payloads."""
    config = config or {}
    summary = summary or {}

    for tree in (summary, config):
        value = _find_first_key(
            tree,
            keys=("model_saved_path", "artifact_path", "best_model_path"),
        )
        if value is not None:
            return str(value)
    return None


def build_run_record(run_dir: str | Path) -> dict[str, Any]:
    """Build a flat notebook-friendly record for one cached run."""
    run_path = _as_path(run_dir)
    config = load_run_config(run_path)
    summary = load_run_summary(run_path)
    parameters = config.get("parameters", {})

    parts = run_path.name.split("-")
    timestamp = parts[1] if len(parts) > 1 else None
    run_id = parts[-1] if parts else None

    record = {
        "run_dir": str(run_path),
        "run_name_local": run_path.name,
        "timestamp": timestamp,
        "run_id": run_id,
        "logger_project": _parameter_value(parameters.get("logger_project")),
        "run_name": _parameter_value(parameters.get("run_name")),
        "data_module": _parameter_value(parameters.get("data_module")),
        "model_module": _parameter_value(parameters.get("model_module")),
        "graph_module": _parameter_value(parameters.get("graph_module")),
        "artifact_path": _parameter_value(parameters.get("artifact_path")),
        "model_saved_path": extract_model_saved_path(config=config, summary=summary),
    }
    record.update(_scalar_summary_items(summary))
    return record


def load_wandb_runs_dataframe(
    wandb_root: str | Path = DEFAULT_WANDB_ROOT,
) -> Any:
    """Load the local W&B cache into a dataframe for notebook analysis."""
    pd = _require_pandas()
    records = [build_run_record(run_dir) for run_dir in iter_wandb_run_dirs(wandb_root)]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


def select_best_run(
    runs_df: Any,
    metric: str,
    maximize: bool = True,
    filters: Mapping[str, Any] | None = None,
) -> Any:
    """Return the best run after optional exact-match filtering."""
    if runs_df.empty:
        raise ValueError("No runs available.")
    if metric not in runs_df.columns:
        raise KeyError(f"Metric '{metric}' is not present in the dataframe.")

    filtered_df = runs_df.copy()
    for key, value in (filters or {}).items():
        if key not in filtered_df.columns:
            raise KeyError(f"Filter column '{key}' is not present in the dataframe.")
        filtered_df = filtered_df[filtered_df[key] == value]

    filtered_df = filtered_df.dropna(subset=[metric])
    if filtered_df.empty:
        raise ValueError("No runs matched the requested filters/metric.")

    best_index = filtered_df[metric].idxmax() if maximize else filtered_df[metric].idxmin()
    return filtered_df.loc[best_index]


def normalize_artifact_path(
    artifact_path: str | None,
    old_roots: Iterable[str | Path] | None = None,
    new_root: str | Path | None = None,
) -> str | None:
    """Rewrite historical artifact roots into a canonical local root."""
    if artifact_path is None:
        return None
    if new_root is None:
        return artifact_path

    normalized = str(artifact_path)
    for old_root in old_roots or []:
        old_root_str = str(_as_path(old_root))
        if normalized.startswith(old_root_str):
            suffix = normalized[len(old_root_str) :].lstrip("/")
            return str(_as_path(new_root) / suffix)
    return normalized


def add_normalized_artifact_path(
    runs_df: Any,
    old_roots: Iterable[str | Path] | None,
    new_root: str | Path,
    source_column: str = "artifact_path",
    target_column: str = "artifact_path_normalized",
) -> Any:
    """Add a normalized artifact-path column without mutating the input frame."""
    if source_column not in runs_df.columns:
        raise KeyError(f"Column '{source_column}' is not present in the dataframe.")

    updated_df = runs_df.copy()
    updated_df[target_column] = updated_df[source_column].map(
        lambda value: normalize_artifact_path(
            value,
            old_roots=old_roots,
            new_root=new_root,
        )
    )
    return updated_df
