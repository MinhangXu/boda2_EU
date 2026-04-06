#!/usr/bin/env python3
"""Build a derived Malinois enhancer table for single-head pan-cell training."""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("/home/minhang/synBio_AL/opt_EU_learn_n_design/CRE/MPRA_ALL_HD_v2.txt")
DEFAULT_OUTPUT = Path(
    "/home/minhang/synBio_AL/boda2_EU/src/learn/derived_data/enhancer/malinois_mpra/"
    "MPRA_ALL_HD_v2__single_head_combined.tsv"
)

DEFAULT_ACTIVITY_COLUMNS = ["K562_mean", "HepG2_mean", "SKNSH_mean"]
DEFAULT_STDERR_COLUMNS = ["lfcSE_k562", "lfcSE_hepg2", "lfcSE_sknsh"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a derived enhancer table with single-head combined targets."
    )
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--activity_columns",
        nargs="+",
        default=DEFAULT_ACTIVITY_COLUMNS,
        help="Columns to combine into a single pan-cell activity target.",
    )
    parser.add_argument(
        "--stderr_columns",
        nargs="+",
        default=DEFAULT_STDERR_COLUMNS,
        help="Columns used to derive a conservative combined stderr value.",
    )
    return parser.parse_args()


def validate_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing {label} columns: {missing}")

# z-score a series
# if the series has zero variance, raise a ValueError
# otherwise, return the z-scored series
def zscore_series(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0:
        raise ValueError(f"Column '{series.name}' has zero variance; cannot z-score.")
    return (series - mean) / std

# main function
def main() -> None:
    args = parse_args()

    frame = pd.read_csv(args.input_path, sep=r"\s+", engine="python")
    validate_columns(frame, args.activity_columns, "activity")
    validate_columns(frame, args.stderr_columns, "stderr")

    zscore_columns = []
    for column in args.activity_columns:
        z_column = f"{column}__zscore_global"
        frame[z_column] = zscore_series(pd.to_numeric(frame[column], errors="coerce"))
        zscore_columns.append(z_column)

    frame["combined_activity_mean"] = (
        frame[args.activity_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    )
    frame["combined_activity_zmean"] = frame[zscore_columns].mean(axis=1)
    frame["combined_stderr_max"] = (
        frame[args.stderr_columns].apply(pd.to_numeric, errors="coerce").max(axis=1)
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_path, sep="\t", index=False)

    print(f"Wrote derived enhancer dataset to: {args.output_path}")
    print(f"Rows: {len(frame)}")
    print(f"Combined activity column: combined_activity_zmean")
    print(f"Combined stderr column: combined_stderr_max")


if __name__ == "__main__":
    main()
