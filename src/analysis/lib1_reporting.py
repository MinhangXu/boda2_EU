"""Small, repo-native helpers for reproducible Lib1 analysis notebooks.

Scientific metric computation belongs in tested analysis programs such as
``lib1_dedup_stage2_analysis.py``.  This module keeps the notebook/reporting
layer consistent: it validates already-produced tables, makes comparison-panel
axis sharing explicit, and records the files behind saved figures.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PathLike = Union[str, Path]
AxisMember = Union[int, Tuple[int, int]]
AxisGroups = Union[str, Sequence[Sequence[AxisMember]]]


def find_repo_root(start: Optional[PathLike] = None) -> Path:
    """Find the ``boda2_EU`` root without relying on the current directory.

    A valid root contains both ``src`` and ``tutorials``.  ``start`` may point
    to either a file or directory.
    """

    candidate = Path(start or __file__).expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate,) + tuple(candidate.parents):
        if (directory / "src").is_dir() and (directory / "tutorials").is_dir():
            return directory
    raise FileNotFoundError(
        "Could not find a repository root containing both 'src' and 'tutorials' "
        f"from {candidate}."
    )


def sha256_file(path: PathLike) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(
    frame: pd.DataFrame, columns: Iterable[str], table_name: str = "table"
) -> None:
    """Fail when a dataframe does not contain every required column."""

    required = list(columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def assert_unique_keys(
    frame: pd.DataFrame, keys: Sequence[str], table_name: str = "table"
) -> None:
    """Assert that ``keys`` identify at most one row."""

    if not keys:
        raise ValueError("keys must contain at least one column")
    require_columns(frame, keys, table_name)
    duplicate = frame.duplicated(list(keys), keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, list(keys)].drop_duplicates().head(5)
        raise ValueError(
            f"{table_name} has duplicate rows for keys {list(keys)}; examples: "
            f"{examples.to_dict(orient='records')}"
        )


def assert_exact_levels(
    frame: pd.DataFrame,
    column: str,
    expected: Iterable[Any],
    table_name: str = "table",
) -> None:
    """Assert that a categorical column has exactly the declared levels."""

    require_columns(frame, [column], table_name)
    expected_levels = set(expected)
    observed_levels = set(frame[column].drop_duplicates().tolist())
    if observed_levels != expected_levels:
        raise ValueError(
            f"{table_name}.{column} levels differ: "
            f"missing={sorted(expected_levels - observed_levels, key=str)}, "
            f"extra={sorted(observed_levels - expected_levels, key=str)}"
        )


def assert_paired_keys(
    frame: pd.DataFrame,
    pair_keys: Sequence[str],
    side_column: str,
    expected_sides: Sequence[Any] = ("off", "on"),
    table_name: str = "table",
) -> None:
    """Assert one row per expected side for every declared comparison key."""

    if not pair_keys:
        raise ValueError("pair_keys must contain at least one column")
    require_columns(frame, list(pair_keys) + [side_column], table_name)
    assert_unique_keys(frame, list(pair_keys) + [side_column], table_name)
    expected = set(expected_sides)
    failures = []
    grouped = frame.groupby(list(pair_keys), dropna=False, sort=False)
    for key, group in grouped:
        observed = set(group[side_column].tolist())
        if observed != expected:
            if not isinstance(key, tuple):
                key = (key,)
            failures.append(
                {
                    "key": dict(zip(pair_keys, key)),
                    "missing": sorted(expected - observed, key=str),
                    "extra": sorted(observed - expected, key=str),
                }
            )
        if len(failures) == 5:
            break
    if failures:
        raise ValueError(
            f"{table_name} does not contain exactly {list(expected_sides)} for every "
            f"{list(pair_keys)} key; examples: {failures}"
        )


@dataclass(frozen=True)
class AnalysisBundle:
    """Validated summary and tables loaded from one analysis output root."""

    root: Path
    summary: Mapping[str, Any]
    tables: Mapping[str, Any]
    paths: Mapping[str, Path]
    sha256: Mapping[str, str]

    def table(self, name: str) -> Any:
        if name not in self.tables:
            raise KeyError(f"Analysis bundle has no table named {name!r}.")
        return self.tables[name]

    def provenance_sources(self) -> Dict[str, Path]:
        """Return named source paths suitable for :func:`save_figure`."""

        return dict(self.paths)


def _read_bundle_file(path: Path, options: Optional[Mapping[str, Any]] = None) -> Any:
    options = dict(options or {})
    lower_name = path.name.lower()
    if lower_name.endswith(".json"):
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    if lower_name.endswith(".jsonl"):
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if lower_name.endswith(".tsv") or lower_name.endswith(".tsv.gz"):
        options.setdefault("sep", "\t")
    elif lower_name.endswith(".csv") or lower_name.endswith(".csv.gz"):
        options.setdefault("sep", ",")
    else:
        raise ValueError(f"Unsupported analysis bundle file type: {path}")
    return pd.read_csv(path, **options)


def load_analysis_bundle(
    root: PathLike,
    required_files: Mapping[str, PathLike],
    expected_summary: Optional[Mapping[str, Any]] = None,
    summary_file: PathLike = "stage2_analysis_summary.json",
    read_options: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> AnalysisBundle:
    """Load analysis outputs and validate their declared summary contract.

    ``required_files`` maps short notebook-facing names to paths relative to
    ``root``.  Every loaded file and the summary receive a SHA-256 digest for
    downstream figure provenance.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Analysis output root does not exist: {root_path}")

    summary_path = Path(summary_file).expanduser()
    if not summary_path.is_absolute():
        summary_path = root_path / summary_path
    summary_path = summary_path.resolve()
    if not summary_path.is_file():
        raise FileNotFoundError(f"Analysis summary does not exist: {summary_path}")
    summary = _read_bundle_file(summary_path)
    if not isinstance(summary, Mapping):
        raise ValueError(f"Analysis summary must be a JSON object: {summary_path}")

    mismatches = {}
    for key, expected in (expected_summary or {}).items():
        if key not in summary:
            mismatches[key] = {"expected": expected, "observed": "<missing>"}
        elif summary[key] != expected:
            mismatches[key] = {"expected": expected, "observed": summary[key]}
    if mismatches:
        raise ValueError(f"Analysis summary contract differs: {mismatches}")

    tables: Dict[str, Any] = {}
    paths: Dict[str, Path] = {"summary": summary_path}
    digests: Dict[str, str] = {"summary": sha256_file(summary_path)}
    options_by_name = dict(read_options or {})
    for name, relative_path in required_files.items():
        if name == "summary":
            raise ValueError("'summary' is reserved for the analysis summary file")
        path = Path(relative_path).expanduser()
        if not path.is_absolute():
            path = root_path / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Required analysis file {name!r} is missing: {path}")
        tables[name] = _read_bundle_file(path, options_by_name.get(name))
        paths[name] = path
        digests[name] = sha256_file(path)

    return AnalysisBundle(
        root=root_path,
        summary=dict(summary),
        tables=tables,
        paths=paths,
        sha256=digests,
    )


def _axes_2d(axes: Any) -> np.ndarray:
    array = np.asarray(axes, dtype=object)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("axes must be a scalar, one-dimensional, or two-dimensional grid")
    return array


def _axis_groups(axes: Any, y_groups: Optional[AxisGroups]) -> Sequence[Sequence[int]]:
    array = _axes_2d(axes)
    nrows, ncols = array.shape
    size = array.size
    if y_groups is None:
        if size > 1:
            raise ValueError(
                "Multi-panel comparisons require an explicit y_groups policy: "
                "'row', 'column', 'all', 'independent', or explicit index groups."
            )
        return ((0,),)
    if isinstance(y_groups, str):
        policy = y_groups.lower()
        if policy == "row":
            return tuple(
                tuple(row * ncols + column for column in range(ncols))
                for row in range(nrows)
            )
        if policy == "column":
            return tuple(
                tuple(row * ncols + column for row in range(nrows))
                for column in range(ncols)
            )
        if policy == "all":
            return (tuple(range(size)),)
        if policy == "independent":
            return tuple((index,) for index in range(size))
        raise ValueError(f"Unknown y_groups policy: {y_groups!r}")

    normalized = []
    seen = set()
    for declared_group in y_groups:
        group = []
        for member in declared_group:
            if isinstance(member, tuple):
                if len(member) != 2:
                    raise ValueError(f"Axis coordinate must be (row, column): {member}")
                row, column = member
                if not (0 <= row < nrows and 0 <= column < ncols):
                    raise IndexError(f"Axis coordinate is outside {array.shape}: {member}")
                index = row * ncols + column
            else:
                index = int(member)
                if not 0 <= index < size:
                    raise IndexError(f"Flat axis index is outside 0..{size - 1}: {index}")
            if index in seen:
                raise ValueError(f"Axis {index} occurs in more than one y group")
            seen.add(index)
            group.append(index)
        if not group:
            raise ValueError("Explicit y groups may not be empty")
        normalized.append(tuple(group))
    normalized.extend((index,) for index in range(size) if index not in seen)
    return tuple(normalized)


def comparison_subplots(
    nrows: int = 1,
    ncols: int = 1,
    *,
    y_groups: Optional[AxisGroups] = None,
    **kwargs: Any,
) -> Tuple[Any, np.ndarray]:
    """Create panels with y sharing limited to declared comparable groups.

    For a multi-panel figure, callers must declare which axes have the same
    metric, units, and estimand.  ``'row'``, ``'column'``, and ``'all'`` share
    within those groups; ``'independent'`` explicitly keeps every panel apart.
    Explicit groups use flattened indices or ``(row, column)`` coordinates.
    """

    if "sharey" in kwargs:
        raise ValueError("Use y_groups instead of passing sharey directly")
    if "squeeze" in kwargs:
        raise ValueError("comparison_subplots always returns a two-dimensional axes grid")
    if nrows < 1 or ncols < 1:
        raise ValueError("nrows and ncols must be positive")
    # Validate before constructing a figure so a missing policy does not leak a figure.
    placeholder = np.empty((nrows, ncols), dtype=object)
    groups = _axis_groups(placeholder, y_groups)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
        sharey=False,
        **kwargs,
    )
    flat = axes.ravel()
    for group in groups:
        reference = flat[group[0]]
        for index in group[1:]:
            flat[index].sharey(reference)
    return fig, axes


def _shared_axis_groups(axes: np.ndarray) -> Sequence[Sequence[int]]:
    flat = list(axes.ravel())
    index_by_identity = {id(axis): index for index, axis in enumerate(flat)}
    groups = []
    seen = set()
    for index, axis in enumerate(flat):
        if index in seen:
            continue
        siblings = axis.get_shared_y_axes().get_siblings(axis)
        group = sorted(
            index_by_identity[id(sibling)]
            for sibling in siblings
            if id(sibling) in index_by_identity
        )
        if not group:
            group = [index]
        groups.append(tuple(group))
        seen.update(group)
    return tuple(groups)


def harmonize_y_limits(
    axes: Any,
    *,
    y_groups: Optional[AxisGroups] = None,
    pad_fraction: float = 0.08,
    include_zero: bool = False,
) -> Sequence[Optional[Tuple[float, float]]]:
    """Apply identical y limits within each comparable axis group.

    Call this after plotting.  When ``y_groups`` is omitted, the sharing
    declared by :func:`comparison_subplots` (or Matplotlib ``sharey``) is used.
    Empty groups are left unchanged.  Log-scale groups are padded in log space
    and cannot include zero.
    """

    if pad_fraction < 0:
        raise ValueError("pad_fraction must be non-negative")
    array = _axes_2d(axes)
    groups = (
        _shared_axis_groups(array)
        if y_groups is None
        else _axis_groups(array, y_groups)
    )
    flat = array.ravel()
    applied = []
    for group in groups:
        group_axes = [flat[index] for index in group]
        scales = {axis.get_yscale() for axis in group_axes}
        if len(scales) != 1:
            raise ValueError(f"Comparable y group {group} mixes axis scales: {scales}")
        scale = next(iter(scales))
        if include_zero and scale == "log":
            raise ValueError("include_zero cannot be used for log-scale y groups")

        bounds = []
        for axis in group_axes:
            if not axis.has_data():
                continue
            lower = float(axis.dataLim.ymin)
            upper = float(axis.dataLim.ymax)
            if math.isfinite(lower) and math.isfinite(upper) and lower <= upper:
                bounds.append((lower, upper))
            else:
                lower, upper = map(float, axis.get_ylim())
                if math.isfinite(lower) and math.isfinite(upper):
                    bounds.append((min(lower, upper), max(lower, upper)))
        if not bounds:
            applied.append(None)
            continue

        lower = min(value[0] for value in bounds)
        upper = max(value[1] for value in bounds)
        if include_zero:
            lower = min(lower, 0.0)
            upper = max(upper, 0.0)
        if scale == "log":
            if lower <= 0 or upper <= 0:
                raise ValueError(f"Log-scale y group {group} contains non-positive data")
            log_lower, log_upper = math.log(lower), math.log(upper)
            span = log_upper - log_lower
            if span == 0:
                span = 1e-6
            limits = (
                math.exp(log_lower - pad_fraction * span),
                math.exp(log_upper + pad_fraction * span),
            )
        else:
            span = upper - lower
            if span == 0:
                span = max(abs(lower), 1.0) * 1e-6
            pad = pad_fraction * span
            limits = (lower - pad, upper + pad)
        for axis in group_axes:
            axis.set_ylim(*limits)
        applied.append(limits)
    return tuple(applied)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_figure(
    fig: Any,
    output_stem: PathLike,
    *,
    source_paths: Optional[Mapping[str, PathLike]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 180,
    close: bool = False,
) -> Mapping[str, Path]:
    """Save a figure and a JSON sidecar that hashes every declared source.

    ``output_stem`` must not have a suffix.  The returned mapping contains one
    path per requested format plus ``'provenance'``.
    """

    stem = Path(output_stem).expanduser()
    if stem.suffix:
        raise ValueError("output_stem must not include a file extension")
    stem = stem.resolve()
    stem.parent.mkdir(parents=True, exist_ok=True)
    normalized_formats = []
    for value in formats:
        fmt = str(value).lower().lstrip(".")
        if not fmt or not fmt.replace("-", "").isalnum():
            raise ValueError(f"Invalid figure format: {value!r}")
        if fmt not in normalized_formats:
            normalized_formats.append(fmt)
    if not normalized_formats:
        raise ValueError("formats must contain at least one output format")

    sources = {}
    for name, source in (source_paths or {}).items():
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Figure source {name!r} does not exist: {path}")
        sources[str(name)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    written: Dict[str, Path] = {}
    for fmt in normalized_formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", format=fmt)
        written[fmt] = path

    figure_files = {
        fmt: {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for fmt, path in written.items()
    }
    sidecar = stem.with_suffix(".provenance.json")
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure_files": figure_files,
        "source_files": sources,
        "metadata": _jsonable(dict(metadata or {})),
    }
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    written["provenance"] = sidecar
    if close:
        plt.close(fig)
    return written
