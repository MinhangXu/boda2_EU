#!/usr/bin/env python3
"""Create a learn-ready Lib1 enhancer table from the fastqs1-5 subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/single_part_variant_level/enhancers/"
    "L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.csv"
)
DEFAULT_OUTPUT = Path(
    "/home/minhang/synBio_AL/boda2_EU/src/learn/derived_data/enhancer/bashor_in_house/"
    "lib1_fastqs1_5_0filtered_out__learn_ready.tsv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize column names/targets for Lib1 enhancer scratch HPO."
    )
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input_path)

    required = {
        "Enhancer",
        "number_of_barcodes",
        "DNA_bc_counts_sum",
        "RNA_bc_counts_sum",
        "RNA/DNA",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing required columns in {args.input_path}: {missing}")

    out = pd.DataFrame(
        {
            "Enhancers": frame["Enhancer"].astype(str).str.upper(),
            "DNA_Counts_Sum": pd.to_numeric(frame["DNA_bc_counts_sum"], errors="coerce"),
            "RNA_Counts_Sum": pd.to_numeric(frame["RNA_bc_counts_sum"], errors="coerce"),
            "n_barcodes": pd.to_numeric(frame["number_of_barcodes"], errors="coerce"),
            "RNA_DNA_Ratio_raw": pd.to_numeric(frame["RNA/DNA"], errors="coerce"),
        }
    )

    valid = (
        out["Enhancers"].notna()
        & out["n_barcodes"].notna()
        & np.isfinite(out["RNA_DNA_Ratio_raw"])
        & (out["RNA_DNA_Ratio_raw"] > 0)
    )
    out = out.loc[valid].reset_index(drop=True)

    out["RNA_DNA"] = out["RNA_DNA_Ratio_raw"]
    out["log2_RNA_DNA"] = np.log2(out["RNA_DNA"])
    out["log10_RNA_DNA"] = np.log10(out["RNA_DNA"])

    # The earlier Lib1 table used an approximately log10(raw_ratio) + 2 scale.
    # Recreating that target keeps this larger table closer to the old training setup.
    out["RNA_DNA_Ratio_log10_scaled"] = np.log10(out["RNA_DNA_Ratio_raw"]) + 2.0
    target = out["RNA_DNA_Ratio_log10_scaled"]
    target_std = target.std(ddof=0)
    if target_std == 0 or not np.isfinite(target_std):
        raise ValueError("Cannot z-score a constant/non-finite target column.")
    out["RNA_DNA_Ratio_log10_scaled_zscore"] = (target - target.mean()) / target_std

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, sep="\t", index=False)

    print(f"Wrote learn-ready Lib1 enhancer dataset to: {args.output_path}")
    print(f"Rows: {len(out)}")
    print(f"HQ rows with n_barcodes >= 4: {int((out['n_barcodes'] >= 4).sum())}")
    print("Default target column: RNA_DNA_Ratio_log10_scaled")
    print("Alternative target column: log2_RNA_DNA")


if __name__ == "__main__":
    main()
