"""Position-wise sequence-composition summaries and presentation plots.

The helpers in this module are intentionally independent of a particular Lib1
part or model.  They can, for example, compare Intron inferred mask categories,
development folds, or prediction-error groups, provided that every sequence in
one call has the same aligned length.

The computation is strict by design: every symbol must belong to the declared
alphabet and every group must have positive total weight.  This avoids silently
turning ambiguous or padded bases into a changing denominator across positions.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DNA_ALPHABET = ("A", "C", "G", "T")


def _as_list(values: Iterable[Any], name: str) -> list:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a collection, not one string")
    try:
        result = list(values)
    except TypeError:
        raise TypeError(f"{name} must be an iterable collection")
    return result


def _normalize_alphabet(
    alphabet: Sequence[str], normalize_case: bool
) -> Tuple[str, ...]:
    values = list(alphabet)
    if not values:
        raise ValueError("alphabet must contain at least one symbol")
    if any(not isinstance(symbol, str) or len(symbol) != 1 for symbol in values):
        raise ValueError("every alphabet symbol must be a one-character string")
    if normalize_case:
        values = [symbol.upper() for symbol in values]
    if len(set(values)) != len(values):
        raise ValueError("alphabet symbols must be unique after case normalization")
    return tuple(values)


def _validate_group_labels(groups: Sequence[Any]) -> None:
    for index, label in enumerate(groups):
        try:
            hash(label)
        except TypeError:
            raise ValueError(f"groups[{index}] must be a hashable scalar")
        missing = pd.isna(label)
        if not isinstance(missing, (bool, np.bool_)):
            raise ValueError(f"groups[{index}] must be a scalar label")
        if bool(missing):
            raise ValueError(f"groups[{index}] is missing")


def positional_base_distribution(
    sequences: Iterable[str],
    *,
    groups: Optional[Iterable[Any]] = None,
    weights: Optional[Iterable[float]] = None,
    alphabet: Sequence[str] = DNA_ALPHABET,
    position_start: int = 1,
    normalize_case: bool = True,
) -> pd.DataFrame:
    """Return a long-form positional base-composition table.

    Parameters
    ----------
    sequences:
        Aligned, equal-length sequences.  Empty input and empty sequences are
        rejected.
    groups:
        Optional group label for every sequence.  Frequencies are normalized
        independently inside each group.  With no groups, the label is
        ``"all"``.
    weights:
        Optional non-negative sequence weights.  ``frequency`` uses weighted
        counts when supplied, while ``count`` always reports the unweighted
        number of sequences carrying that base.  A group whose weights sum to
        zero is rejected.
    alphabet:
        Allowed one-character symbols in the desired output order.  The
        default is strict DNA (A/C/G/T); pass a custom alphabet explicitly if
        ambiguous or gap symbols have a scientific meaning.
    position_start:
        Coordinate assigned to the first aligned position.  Biological plots
        usually use the default one-based coordinates.
    normalize_case:
        Convert sequences and alphabet symbols to uppercase before validation.

    Returns
    -------
    pandas.DataFrame
        One row per ``group x position x base`` with columns ``count``,
        ``weighted_count``, ``frequency``, ``n_sequences``, and
        ``total_weight``.  Every alphabet base is present, including zero-count
        cells, so groups can be plotted on identical grids.

    Notes
    -----
    A weight applies to an entire sequence.  Position-specific coverage or
    uncertainty requires a different estimator and must not be passed here as
    if it were a sequence weight.
    """

    sequence_values = _as_list(sequences, "sequences")
    if not sequence_values:
        raise ValueError("sequences must contain at least one sequence")
    if any(not isinstance(sequence, str) for sequence in sequence_values):
        bad_index = next(
            index
            for index, sequence in enumerate(sequence_values)
            if not isinstance(sequence, str)
        )
        raise TypeError(f"sequences[{bad_index}] is not a string")

    normalized_alphabet = _normalize_alphabet(alphabet, normalize_case)
    if normalize_case:
        sequence_values = [sequence.upper() for sequence in sequence_values]

    lengths = [len(sequence) for sequence in sequence_values]
    if lengths[0] == 0:
        raise ValueError("sequences may not be empty strings")
    if any(length != lengths[0] for length in lengths):
        distinct = sorted(set(lengths))
        raise ValueError(f"sequences must have equal length; observed lengths={distinct}")

    allowed = set(normalized_alphabet)
    for sequence_index, sequence in enumerate(sequence_values):
        invalid = sorted(set(sequence) - allowed)
        if invalid:
            positions = [
                position_start + index
                for index, symbol in enumerate(sequence)
                if symbol not in allowed
            ][:5]
            raise ValueError(
                f"sequences[{sequence_index}] contains symbols outside the declared "
                f"alphabet: {invalid}; first positions={positions}"
            )

    n_sequences = len(sequence_values)
    if groups is None:
        group_values = ["all"] * n_sequences
    else:
        group_values = _as_list(groups, "groups")
        if len(group_values) != n_sequences:
            raise ValueError(
                "groups must contain exactly one label per sequence; "
                f"got {len(group_values)} labels for {n_sequences} sequences"
            )
        _validate_group_labels(group_values)

    if weights is None:
        weight_values = np.ones(n_sequences, dtype=float)
    else:
        raw_weights = _as_list(weights, "weights")
        if len(raw_weights) != n_sequences:
            raise ValueError(
                "weights must contain exactly one value per sequence; "
                f"got {len(raw_weights)} weights for {n_sequences} sequences"
            )
        try:
            weight_values = np.asarray(raw_weights, dtype=float)
        except (TypeError, ValueError):
            raise ValueError("weights must contain only numeric values")
        if weight_values.ndim != 1:
            raise ValueError("weights must be a one-dimensional collection")
        if not np.isfinite(weight_values).all():
            raise ValueError("weights must contain only finite values")
        if (weight_values < 0).any():
            raise ValueError("weights must be non-negative")

    group_order = []
    indices_by_group = {}
    for index, label in enumerate(group_values):
        if label not in indices_by_group:
            group_order.append(label)
            indices_by_group[label] = []
        indices_by_group[label].append(index)

    sequence_matrix = np.asarray([list(sequence) for sequence in sequence_values])
    rows = []
    for label in group_order:
        indices = np.asarray(indices_by_group[label], dtype=int)
        group_matrix = sequence_matrix[indices, :]
        group_weights = weight_values[indices]
        total_weight = float(group_weights.sum())
        if total_weight <= 0:
            raise ValueError(f"group {label!r} has zero total weight")
        group_size = int(indices.size)
        for position_index in range(group_matrix.shape[1]):
            column = group_matrix[:, position_index]
            for base in normalized_alphabet:
                matches = column == base
                count = int(matches.sum())
                weighted_count = float(group_weights[matches].sum())
                rows.append(
                    {
                        "group": label,
                        "position": int(position_start + position_index),
                        "base": base,
                        "count": count,
                        "weighted_count": weighted_count,
                        "frequency": weighted_count / total_weight,
                        "n_sequences": group_size,
                        "total_weight": total_weight,
                    }
                )

    return pd.DataFrame.from_records(
        rows,
        columns=[
            "group",
            "position",
            "base",
            "count",
            "weighted_count",
            "frequency",
            "n_sequences",
            "total_weight",
        ],
    )


def _ordered_levels(
    observed: Sequence[Any], requested: Optional[Sequence[Any]], name: str
) -> list:
    observed_list = list(observed)
    if requested is None:
        return observed_list
    requested_list = list(requested)
    if len(set(requested_list)) != len(requested_list):
        raise ValueError(f"{name} must not contain duplicates")
    observed_set = set(observed_list)
    requested_set = set(requested_list)
    if requested_set != observed_set:
        raise ValueError(
            f"{name} must contain every observed level exactly once; "
            f"missing={list(observed_set - requested_set)}, "
            f"extra={list(requested_set - observed_set)}"
        )
    return requested_list


def _position_ticks(n_positions: int, maximum: int = 12) -> np.ndarray:
    if n_positions <= maximum:
        return np.arange(n_positions, dtype=int)
    return np.unique(np.linspace(0, n_positions - 1, maximum).round().astype(int))


def plot_positional_base_distribution(
    distribution: pd.DataFrame,
    *,
    value: str = "frequency",
    group_order: Optional[Sequence[Any]] = None,
    base_order: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[Any, np.ndarray]:
    """Plot one positional-composition heatmap per group on one color scale.

    The input is normally produced by :func:`positional_base_distribution`.
    Each panel has bases on rows and aligned sequence positions on columns.
    Frequency plots always default to the absolute 0--1 scale; count plots use
    one global scale across groups.  This makes color differences comparable
    without describing axis mechanics in the presentation title.
    """

    required = {"group", "position", "base", value}
    missing = sorted(required - set(distribution.columns))
    if missing:
        raise ValueError(f"distribution is missing required columns: {missing}")
    if distribution.empty:
        raise ValueError("distribution may not be empty")
    if distribution.duplicated(["group", "position", "base"]).any():
        raise ValueError("distribution has duplicate group-position-base rows")

    values = pd.to_numeric(distribution[value], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"distribution.{value} must contain only finite numbers")
    if (values < 0).any():
        raise ValueError(f"distribution.{value} must be non-negative")
    if value == "frequency" and (values > 1 + 1e-12).any():
        raise ValueError("distribution.frequency values must lie between zero and one")

    observed_groups = list(pd.unique(distribution["group"]))
    observed_bases = list(pd.unique(distribution["base"]))
    groups = _ordered_levels(observed_groups, group_order, "group_order")
    bases = _ordered_levels(observed_bases, base_order, "base_order")
    positions = sorted(pd.unique(distribution["position"]).tolist())
    if not positions:
        raise ValueError("distribution contains no positions")

    expected_cells = len(positions) * len(bases)
    for group in groups:
        subset = distribution.loc[distribution["group"] == group]
        if len(subset) != expected_cells:
            raise ValueError(
                f"group {group!r} does not have a complete position x base grid"
            )
        if set(subset["position"]) != set(positions) or set(subset["base"]) != set(bases):
            raise ValueError(
                f"group {group!r} does not use the same positions and bases"
            )
    if value == "frequency":
        frequency_sums = distribution.groupby(
            ["group", "position"], dropna=False
        )[value].sum()
        if not np.allclose(frequency_sums.to_numpy(dtype=float), 1.0):
            raise ValueError(
                "distribution.frequency must sum to one in every group-position cell"
            )

    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = 1.0 if value == "frequency" else float(values.max())
        if vmax <= vmin:
            vmax = vmin + 1.0
    if not np.isfinite([vmin, vmax]).all() or vmax <= vmin:
        raise ValueError("vmin and vmax must be finite with vmax greater than vmin")

    if figsize is None:
        width = max(8.0, min(18.0, 0.12 * len(positions)))
        height = max(2.8, 2.25 * len(groups) + 0.8)
        figsize = (width, height)
    fig, axes_grid = plt.subplots(
        nrows=len(groups),
        ncols=1,
        squeeze=False,
        sharex=True,
        figsize=figsize,
    )
    axes = axes_grid[:, 0]
    image = None
    for axis, group in zip(axes, groups):
        subset = distribution.loc[distribution["group"] == group]
        matrix = (
            subset.pivot(index="base", columns="position", values=value)
            .reindex(index=bases, columns=positions)
            .to_numpy(dtype=float)
        )
        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=float(vmin),
            vmax=float(vmax),
        )
        axis.set_yticks(np.arange(len(bases)))
        axis.set_yticklabels([str(base) for base in bases])
        axis.set_ylabel("Base")
        axis.set_title(str(group), loc="left")

    tick_indices = _position_ticks(len(positions))
    axes[-1].set_xticks(tick_indices)
    axes[-1].set_xticklabels([str(positions[index]) for index in tick_indices])
    axes[-1].set_xlabel("Aligned sequence position")
    if title:
        fig.suptitle(title)

    if colorbar_label is None:
        colorbar_label = {
            "frequency": "Base frequency",
            "count": "Sequence count",
            "weighted_count": "Weighted sequence count",
        }.get(value, value.replace("_", " ").title())
    colorbar = fig.colorbar(image, ax=list(axes), pad=0.02, fraction=0.025)
    colorbar.set_label(colorbar_label)
    fig.subplots_adjust(
        left=0.08,
        right=0.90,
        bottom=0.10,
        top=0.90 if title else 0.96,
        hspace=0.38,
    )
    return fig, axes
