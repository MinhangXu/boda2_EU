#!/usr/bin/env python3
"""Prepare in-house Lib1 promoter data for single-output scratch HPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORK_ROOT = Path("/home/minhang/synBio_AL")
REPO_ROOT = WORK_ROOT / "boda2_EU"
DEFAULT_INPUT = (
    WORK_ROOT
    / "opt_EU_learn_n_design"
    / "MattLee_lib1"
    / "single_part_variant_level"
    / "promoters"
    / "L1_final_fastqs1-5_sublibrary_Promoter_subset.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "src"
    / "learn"
    / "derived_data"
    / "promoter"
    / "bashor_in_house"
    / "lib1_promoter_allvalid_fastqs1_5__learn_ready.tsv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence-column", default="Promoter")
    parser.add_argument("--target-column", default="RNA/DNA")
    parser.add_argument("--barcode-column", default="number_of_barcodes")
    parser.add_argument("--heldout-min-barcodes", type=int, default=8)
    parser.add_argument("--val-frac-within-hq", type=float, default=0.1295)
    parser.add_argument("--test-frac-within-hq", type=float, default=0.1295)
    parser.add_argument("--val-size-within-hq", type=int, default=250)
    parser.add_argument("--test-size-within-hq", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = pd.read_csv(args.input_path)
    required = [args.sequence_column, args.target_column, args.barcode_column]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"{args.input_path} is missing required columns: {missing}")

    seq = raw[args.sequence_column].astype(str).str.strip().str.upper()
    target = pd.to_numeric(raw[args.target_column], errors="coerce")
    barcode = pd.to_numeric(raw[args.barcode_column], errors="coerce")

    valid_dna = seq.str.fullmatch(r"[ACGTN]+").fillna(False)
    finite_positive_target = np.isfinite(target) & (target > 0)
    finite_barcode = np.isfinite(barcode)
    usable_mask = valid_dna & finite_positive_target & finite_barcode

    usable = raw.loc[usable_mask].copy()
    usable["source_row_id"] = raw.index[usable_mask].to_numpy(dtype=int)
    usable["Promoter"] = seq.loc[usable_mask].to_numpy()
    usable["sequence_len"] = usable["Promoter"].str.len()
    usable["RNA_DNA"] = target.loc[usable_mask].to_numpy(dtype=float)
    usable["log2_RNA_DNA"] = np.log2(usable["RNA_DNA"])
    usable["log10_RNA_DNA"] = np.log10(usable["RNA_DNA"])
    usable["n_barcodes"] = barcode.loc[usable_mask].astype(int).to_numpy()

    output_columns = [
        "source_row_id",
        "Promoter",
        "sequence_len",
        "n_barcodes",
        "RNA_DNA",
        "log2_RNA_DNA",
        "log10_RNA_DNA",
    ]
    for optional in ["DNA_bc_counts_sum", "RNA_bc_counts_sum", "parts_concatenated"]:
        if optional in usable.columns:
            output_columns.append(optional)
    out = usable[output_columns].reset_index(drop=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, sep="\t", index=False)

    n_hq = int(out["n_barcodes"].ge(args.heldout_min_barcodes).sum())
    expected_test = int(args.test_size_within_hq)
    expected_val = int(args.val_size_within_hq)
    if expected_test < 1 or expected_val < 1:
        raise ValueError(f"Val/test sizes must be positive: val={expected_val}, test={expected_test}")
    if expected_test + expected_val >= n_hq:
        raise ValueError(
            "Requested val/test sizes "
            f"({expected_val}/{expected_test}) exhaust HQ rows ({n_hq}) at "
            f"n_barcodes >= {args.heldout_min_barcodes}."
        )
    length_counts = {
        str(int(length)): int(count)
        for length, count in out["sequence_len"].value_counts().sort_index().items()
    }
    metadata = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "raw_rows": int(len(raw)),
        "output_rows": int(len(out)),
        "dropped_rows": int(len(raw) - len(out)),
        "dropped_invalid_dna_rows": int((~valid_dna).sum()),
        "sequence_length_counts": length_counts,
        "heldout_min_barcodes": int(args.heldout_min_barcodes),
        "hq_rows_at_heldout_threshold": n_hq,
        "val_frac_within_hq": float(args.val_frac_within_hq),
        "test_frac_within_hq": float(args.test_frac_within_hq),
        "val_size_within_hq": expected_val,
        "test_size_within_hq": expected_test,
        "expected_val_rows": expected_val,
        "expected_test_rows": expected_test,
        "expected_train_rows": int(len(out) - expected_val - expected_test),
        "target_column": "log2_RNA_DNA",
        "barcode_column": "n_barcodes",
        "sequence_column": "Promoter",
    }
    metadata_path = args.output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {len(out)} rows to {args.output_path}")
    print(f"Wrote metadata to {metadata_path}")
    print(f"Length counts: {length_counts}")
    print(
        "HQ rows at barcode >= "
        f"{args.heldout_min_barcodes}: {n_hq}; "
        f"expected train/val/test: {metadata['expected_train_rows']}/{expected_val}/{expected_test}"
    )


if __name__ == "__main__":
    main()
