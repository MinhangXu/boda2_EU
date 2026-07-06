#!/usr/bin/env python3
"""Prepare in-house Lib1 ThreePrime data for single-output scratch HPO.

The first 3'UTR in-house branch uses a dedicated modal-length model rather
than padding/truncating into the public Hani/PARADE 240 nt geometry. By default
this script keeps valid finite-positive ThreePrime rows at the modal insert
length, computes log-transformed RNA/DNA targets, and writes a small learn-ready
TSV plus metadata JSON under `src/learn/derived_data/`.
"""

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
    / "ThreePrimes"
    / "L1_final_fastqs1-5_sublibrary_ThreePrime_subset.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "src"
    / "learn"
    / "derived_data"
    / "utr3"
    / "bashor_in_house"
    / "lib1_threeprime_modal100_fastqs1_5__learn_ready.tsv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sequence-column", default="ThreePrime")
    parser.add_argument("--target-column", default="RNA/DNA")
    parser.add_argument("--barcode-column", default="number_of_barcodes")
    parser.add_argument(
        "--length-policy",
        choices=["modal", "exact", "all"],
        default="modal",
        help="Default modal keeps the in-house 100 nt ThreePrime branch.",
    )
    parser.add_argument(
        "--exact-length",
        type=int,
        default=100,
        help="Used only when --length-policy exact.",
    )
    parser.add_argument("--heldout-min-barcodes", type=int, default=8)
    parser.add_argument("--val-frac-within-hq", type=float, default=0.25)
    parser.add_argument("--test-frac-within-hq", type=float, default=0.25)
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
    usable["ThreePrime"] = seq.loc[usable_mask].to_numpy()
    usable["sequence_len"] = usable["ThreePrime"].str.len()
    usable["RNA_DNA"] = target.loc[usable_mask].to_numpy(dtype=float)
    usable["log2_RNA_DNA"] = np.log2(usable["RNA_DNA"])
    usable["log10_RNA_DNA"] = np.log10(usable["RNA_DNA"])
    usable["n_barcodes"] = barcode.loc[usable_mask].astype(int).to_numpy()

    length_policy = args.length_policy
    selected_length = None
    if length_policy == "modal":
        selected_length = int(usable["sequence_len"].mode().iloc[0])
        out = usable.loc[usable["sequence_len"].eq(selected_length)].copy()
    elif length_policy == "exact":
        selected_length = int(args.exact_length)
        out = usable.loc[usable["sequence_len"].eq(selected_length)].copy()
    else:
        out = usable.copy()

    output_columns = [
        "source_row_id",
        "ThreePrime",
        "sequence_len",
        "n_barcodes",
        "RNA_DNA",
        "log2_RNA_DNA",
        "log10_RNA_DNA",
    ]
    for optional in ["DNA_bc_counts_sum", "RNA_bc_counts_sum", "parts_concatenated"]:
        if optional in usable.columns:
            output_columns.append(optional)
    out = out[output_columns].reset_index(drop=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_path, sep="\t", index=False)

    n_hq = int(out["n_barcodes"].ge(args.heldout_min_barcodes).sum())
    expected_test = int(round(n_hq * args.test_frac_within_hq))
    expected_val = int(round(n_hq * args.val_frac_within_hq))
    metadata = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "raw_rows": int(len(raw)),
        "usable_valid_finite_positive_rows": int(len(usable)),
        "output_rows": int(len(out)),
        "length_policy": length_policy,
        "selected_length": selected_length,
        "heldout_min_barcodes": int(args.heldout_min_barcodes),
        "hq_rows_at_heldout_threshold": n_hq,
        "expected_val_rows_at_fraction": expected_val,
        "expected_test_rows_at_fraction": expected_test,
        "expected_leftover_hq_train_rows": int(max(0, n_hq - expected_val - expected_test)),
        "target_column": "log2_RNA_DNA",
        "barcode_column": "n_barcodes",
        "sequence_column": "ThreePrime",
    }
    metadata_path = args.output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote {len(out)} rows to {args.output_path}")
    print(f"Wrote metadata to {metadata_path}")
    print(
        "HQ rows at barcode >= "
        f"{args.heldout_min_barcodes}: {n_hq}; "
        f"expected val/test: {expected_val}/{expected_test}"
    )


if __name__ == "__main__":
    main()
