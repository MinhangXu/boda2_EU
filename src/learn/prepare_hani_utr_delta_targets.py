#!/usr/bin/env python3
"""Add PARADE-style delta-from-cell-mean targets to Hani wide UTR tables."""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_PROCESSED_DIR = Path(
    "/home/minhang/synBio_AL/opt_EU_learn_n_design/utr_hani_2025/processed_utr_data"
)

LIBRARY_SPECS = {
    "3UTR_lib1": {
        "input_name": "3UTR_lib1_branched_observed_heads.csv",
        "output_name": "3UTR_lib1_branched_observed_heads_with_deltas.csv",
        "manifest_name": "3UTR_lib1_branched_observed_heads_with_deltas_manifest.json",
        "heads": ["c1", "c2", "c4", "c6", "c13", "c17"],
    },
    "5UTR_lib1": {
        "input_name": "5UTR_lib1_branched_observed_heads.csv",
        "output_name": "5UTR_lib1_branched_observed_heads_with_deltas.csv",
        "manifest_name": "5UTR_lib1_branched_observed_heads_with_deltas_manifest.json",
        "heads": ["c1", "c2", "c4", "c6", "c17"],
    },
}


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Read Hani observed-head wide activity tables and add per-cell "
            "delta targets: delta_c = activity_c - mean(activity across heads)."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing existing *_branched_observed_heads.csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory where delta-augmented CSVs and manifests will be written.",
    )
    parser.add_argument(
        "--libraries",
        nargs="+",
        choices=sorted(LIBRARY_SPECS),
        default=sorted(LIBRARY_SPECS),
        help="Library specs to prepare.",
    )
    return parser


def add_delta_targets(input_path, output_path, manifest_path, library_name, heads):
    df = pd.read_csv(input_path)
    required = ["seq", "fold", *heads]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {missing}")

    out = df.copy()
    for column in heads:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["seq", "fold", *heads]).copy()

    delta_columns = [f"delta_{head}" for head in heads]
    out["mean_activity"] = out[heads].mean(axis=1)
    for head, delta_column in zip(heads, delta_columns):
        out[delta_column] = out[head] - out["mean_activity"]

    front_columns = ["seq", "fold"]
    if "library" in out.columns:
        front_columns.append("library")
    ordered_columns = front_columns + heads + ["mean_activity"] + delta_columns
    out = out[ordered_columns].sort_values(["fold", "seq"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    manifest = {
        "library": library_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "heads": heads,
        "delta_columns": delta_columns,
        "n_output_sequences": int(len(out)),
        "fold_counts": {k: int(v) for k, v in out["fold"].value_counts().sort_index().items()},
        "activity_columns_for_hpo": [*heads, *delta_columns],
        "target_means": {k: float(v) for k, v in out[[*heads, *delta_columns]].mean().items()},
        "target_stds": {k: float(v) for k, v in out[[*heads, *delta_columns]].std().items()},
    }
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest


def main():
    args = build_argparser().parse_args()
    for library_name in args.libraries:
        spec = LIBRARY_SPECS[library_name]
        manifest = add_delta_targets(
            input_path=args.input_dir / spec["input_name"],
            output_path=args.output_dir / spec["output_name"],
            manifest_path=args.output_dir / spec["manifest_name"],
            library_name=library_name,
            heads=spec["heads"],
        )
        print(
            f"Wrote {manifest['output_path']} with {manifest['n_output_sequences']} "
            f"sequences and targets {','.join(manifest['activity_columns_for_hpo'])}"
        )


if __name__ == "__main__":
    main()
