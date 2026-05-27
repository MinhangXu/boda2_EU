#!/usr/bin/env python3
"""Prepare Hani UTR Library 1 observed-head tables for branched models."""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_PROCESSED_DIR = Path(
    "/home/minhang/synBio_AL/opt_EU_learn_n_design/utr_hani_2025/processed_utr_data"
)

LIBRARY_SPECS = {
    "3UTR_lib1": {
        "input_name": "3UTR_lib1_processed.csv",
        "output_name": "3UTR_lib1_branched_observed_heads.csv",
        "manifest_name": "3UTR_lib1_branched_observed_heads_manifest.json",
        "heads": ["c1", "c2", "c4", "c6", "c13", "c17"],
    },
    "5UTR_lib1": {
        "input_name": "5UTR_lib1_processed.csv",
        "output_name": "5UTR_lib1_branched_observed_heads.csv",
        "manifest_name": "5UTR_lib1_branched_observed_heads_manifest.json",
        "heads": ["c1", "c2", "c4", "c6", "c17"],
    },
}

BIN_COLUMNS = ["1", "2", "3", "4"]
REQUIRED_COLUMNS = ["seq", "cell_type", "fold", *BIN_COLUMNS]


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Aggregate Hani UTR Lib1 replicate rows into wide observed-head targets."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing existing *_processed.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory where wide branched CSVs and manifests will be written.",
    )
    parser.add_argument(
        "--libraries",
        nargs="+",
        choices=sorted(LIBRARY_SPECS),
        default=sorted(LIBRARY_SPECS),
        help="Library specs to prepare.",
    )
    return parser


def validate_columns(df, input_path):
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {missing}")


def validate_fold_consistency(df, input_path):
    seq_fold_counts = df.groupby("seq", observed=True)["fold"].nunique()
    if (seq_fold_counts > 1).any():
        examples = seq_fold_counts[seq_fold_counts > 1].head().index.tolist()
        raise ValueError(
            f"{input_path} has sequences assigned to multiple folds; examples: {examples}"
        )

    seq_cell_fold_counts = df.groupby(["seq", "cell_type"], observed=True)["fold"].nunique()
    if (seq_cell_fold_counts > 1).any():
        examples = seq_cell_fold_counts[seq_cell_fold_counts > 1].head().index.tolist()
        raise ValueError(
            f"{input_path} has seq/cell pairs assigned to multiple folds; examples: {examples}"
        )


def aggregate_library(input_path, output_path, manifest_path, library_name, heads):
    df = pd.read_csv(input_path)
    validate_columns(df, input_path)

    present_heads = sorted(df["cell_type"].dropna().unique())
    missing_heads = [head for head in heads if head not in present_heads]
    if missing_heads:
        raise ValueError(f"{input_path} is missing expected observed heads: {missing_heads}")

    df = df[df["cell_type"].isin(heads)].copy()
    validate_fold_consistency(df, input_path)

    for column in BIN_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["seq", "cell_type", "fold", *BIN_COLUMNS])

    grouped = (
        df.groupby(["seq", "cell_type", "fold"], observed=True)[BIN_COLUMNS]
        .sum()
        .reset_index()
    )
    denominator = grouped[BIN_COLUMNS].sum(axis=1)
    grouped = grouped[denominator > 0].copy()
    denominator = denominator[denominator > 0]
    grouped["rna_activity"] = (
        grouped["1"] * 1.0
        + grouped["2"] * 2.0
        + grouped["3"] * 3.0
        + grouped["4"] * 4.0
    ) / denominator

    wide = grouped.pivot_table(
        index=["seq", "fold"],
        columns="cell_type",
        values="rna_activity",
        aggfunc="first",
    )
    wide = wide.reindex(columns=heads)
    incomplete = int(wide.isna().any(axis=1).sum())
    if incomplete:
        raise ValueError(
            f"{input_path} produced {incomplete} sequences missing at least one observed head"
        )

    wide = wide.reset_index()
    wide.insert(2, "library", library_name)
    wide = wide[["seq", "fold", "library", *heads]].sort_values(["fold", "seq"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_path, index=False)

    manifest = {
        "library": library_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "heads": heads,
        "n_input_rows": int(len(df)),
        "n_grouped_seq_cell_rows": int(len(grouped)),
        "n_output_sequences": int(len(wide)),
        "fold_counts": {k: int(v) for k, v in wide["fold"].value_counts().sort_index().items()},
        "cell_type_input_rows": {
            k: int(v) for k, v in df["cell_type"].value_counts().sort_index().items()
        },
        "target_means": {k: float(v) for k, v in wide[heads].mean().items()},
        "target_stds": {k: float(v) for k, v in wide[heads].std().items()},
    }
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


def main():
    args = build_argparser().parse_args()
    for library_name in args.libraries:
        spec = LIBRARY_SPECS[library_name]
        input_path = args.input_dir / spec["input_name"]
        output_path = args.output_dir / spec["output_name"]
        manifest_path = args.output_dir / spec["manifest_name"]
        manifest = aggregate_library(
            input_path=input_path,
            output_path=output_path,
            manifest_path=manifest_path,
            library_name=library_name,
            heads=spec["heads"],
        )
        print(
            f"Wrote {output_path} with {manifest['n_output_sequences']} sequences "
            f"and heads {','.join(manifest['heads'])}"
        )
        print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
