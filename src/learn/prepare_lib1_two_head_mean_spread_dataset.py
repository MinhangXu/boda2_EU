#!/usr/bin/env python3
"""Prepare deduplicated Lib1 single-part data for two-head mean/spread training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(os.environ.get("BODA_WORK_ROOT", REPO_ROOT.parent)).expanduser()
LIB1_ROOT = WORK_ROOT / "opt_EU_learn_n_design" / "MattLee_lib1"
VARIANT_ROOT = LIB1_ROOT / "single_part_variant_level"
BARCODE_ROOT = LIB1_ROOT / "barcode_level" / "by_library"
OUTPUT_ROOT = REPO_ROOT / "src" / "learn" / "derived_data"

PARTS_COL = "parts_concatenated"
DNA_COL = "DNA_bc_counts"
RNA_COL = "RNA_bc_counts"
BARCODE_COL = "bba1_ddc1_concat"


PART_SPECS = {
    "promoter": {
        "library_name": "Promoter",
        "sequence_column": "Promoter",
        "required_sequence_len": None,
        "variant_path": VARIANT_ROOT / "L1_final_fastqs1-5_sublibrary_Promoter_subset.dedup_exact.csv",
        "barcode_path": BARCODE_ROOT / "single_part__Promoter_subset.dedup_exact.barcode_level.csv",
        "output_path": OUTPUT_ROOT
        / "promoter"
        / "bashor_in_house"
        / "lib1_promoter_allvalid_fastqs1_5_dedup_exact_two_head_mean_spread__learn_ready.tsv",
        "heldout_min_barcodes": 8,
        "val_frac_within_hq": 0.1295,
        "test_frac_within_hq": 0.1295,
        "val_size_within_hq": 250,
        "test_size_within_hq": 250,
    },
    "utr5": {
        "library_name": "FivePrime",
        "sequence_column": "FivePrime",
        "required_sequence_len": 50,
        "variant_path": VARIANT_ROOT / "L1_final_fastqs1-5_sublibrary_FivePrime_subset.dedup_exact.csv",
        "barcode_path": BARCODE_ROOT / "single_part__FivePrime_subset.dedup_exact.barcode_level.csv",
        "output_path": OUTPUT_ROOT
        / "utr5"
        / "bashor_in_house"
        / "lib1_fiveprime_modal50_fastqs1_5_dedup_exact_two_head_mean_spread__learn_ready.tsv",
        "heldout_min_barcodes": 8,
        "val_frac_within_hq": 0.2,
        "test_frac_within_hq": 0.2,
        "val_size_within_hq": 250,
        "test_size_within_hq": 250,
    },
    "intron": {
        "library_name": "Intron",
        "sequence_column": "Intron",
        "required_sequence_len": 80,
        "variant_path": VARIANT_ROOT / "L1_final_fastqs1-5_sublibrary_Intron_subset.dedup_exact.csv",
        "barcode_path": BARCODE_ROOT / "single_part__Intron_subset.dedup_exact.barcode_level.csv",
        "output_path": OUTPUT_ROOT
        / "introns"
        / "bashor_in_house"
        / "lib1_intron_modal80_fastqs1_5_dedup_exact_two_head_mean_spread__learn_ready.tsv",
        "heldout_min_barcodes": 8,
        "val_frac_within_hq": 0.2,
        "test_frac_within_hq": 0.2,
        "val_size_within_hq": 250,
        "test_size_within_hq": 250,
    },
    "utr3": {
        "library_name": "ThreePrime",
        "sequence_column": "ThreePrime",
        "required_sequence_len": 100,
        "variant_path": VARIANT_ROOT / "L1_final_fastqs1-5_sublibrary_ThreePrime_subset.dedup_exact.csv",
        "barcode_path": BARCODE_ROOT / "single_part__ThreePrime_subset.dedup_exact.barcode_level.csv",
        "output_path": OUTPUT_ROOT
        / "utr3"
        / "bashor_in_house"
        / "lib1_threeprime_modal100_fastqs1_5_dedup_exact_two_head_mean_spread__learn_ready.tsv",
        "heldout_min_barcodes": 8,
        "val_frac_within_hq": 0.25,
        "test_frac_within_hq": 0.25,
        "val_size_within_hq": None,
        "test_size_within_hq": None,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=sorted(PART_SPECS),
        default=sorted(PART_SPECS),
        help="Single-part libraries to prepare.",
    )
    parser.add_argument("--min-dna-count", type=float, default=1.0)
    parser.add_argument("--alpha-rna", type=float, default=0.5)
    parser.add_argument("--alpha-dna", type=float, default=0.5)
    parser.add_argument("--spread-epsilon", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def read_barcode_counts(path: Path) -> pd.DataFrame:
    requested = [PARTS_COL, BARCODE_COL, DNA_COL, RNA_COL]
    header = pd.read_csv(path, nrows=0)
    usecols = [column for column in requested if column in header.columns]
    require_columns(header, path, [PARTS_COL, DNA_COL, RNA_COL])
    frame = pd.read_csv(path, usecols=usecols)
    frame[PARTS_COL] = frame[PARTS_COL].astype("string")
    frame[DNA_COL] = pd.to_numeric(frame[DNA_COL], errors="coerce")
    frame[RNA_COL] = pd.to_numeric(frame[RNA_COL], errors="coerce")
    if BARCODE_COL not in frame.columns:
        frame[BARCODE_COL] = pd.NA
    frame[BARCODE_COL] = frame[BARCODE_COL].astype("string")
    return frame


def count_barcodes(frame: pd.DataFrame) -> int:
    if BARCODE_COL not in frame.columns:
        return int(len(frame))
    n_unique = int(frame[BARCODE_COL].dropna().nunique())
    return n_unique if n_unique > 0 else int(len(frame))


def summarize_barcode_targets(
    barcode: pd.DataFrame,
    min_dna_count: float,
    alpha_rna: float,
    alpha_dna: float,
    spread_epsilon: float,
) -> pd.DataFrame:
    raw_records = []
    for construct, group in barcode.groupby(PARTS_COL, dropna=True):
        dna = group[DNA_COL]
        raw_records.append(
            {
                PARTS_COL: construct,
                "n_barcode_rows_total_raw": int(len(group)),
                "n_barcodes_total_raw": count_barcodes(group),
                "zero_DNA_barcode_count_raw": int(dna.eq(0).sum()),
                "zero_DNA_barcode_frac_raw": float(dna.eq(0).mean()) if len(group) else np.nan,
                "n_barcodes_dropped_dna_low_or_zero": int((~np.isfinite(dna) | dna.lt(min_dna_count)).sum()),
            }
        )
    raw_stats = pd.DataFrame(raw_records)

    retained = barcode.loc[
        np.isfinite(barcode[DNA_COL])
        & np.isfinite(barcode[RNA_COL])
        & barcode[DNA_COL].ge(min_dna_count)
    ].copy()
    records = []
    for construct, group in retained.groupby(PARTS_COL, dropna=True):
        dna = group[DNA_COL].to_numpy(dtype=float)
        rna = group[RNA_COL].to_numpy(dtype=float)
        barcode_logratio = np.log2((rna + alpha_rna) / (dna + alpha_dna))

        total_dna = float(dna.sum())
        total_rna = float(rna.sum())
        dna_sq_sum = float(np.square(dna).sum())
        equal_mean = float(barcode_logratio.mean())
        equal_var = float(np.square(barcode_logratio - equal_mean).mean())
        dna_weights = dna / total_dna
        dna_weighted_mean = float(np.sum(dna_weights * barcode_logratio))
        dna_weighted_var = float(np.sum(dna_weights * np.square(barcode_logratio - dna_weighted_mean)))
        n_eff = float(total_dna * total_dna / dna_sq_sum) if dna_sq_sum > 0 else np.nan
        mean_se_var = float(dna_weighted_var / n_eff) if n_eff and np.isfinite(n_eff) and n_eff > 0 else np.nan

        records.append(
            {
                PARTS_COL: construct,
                "n_barcode_rows_retained": int(len(group)),
                "n_barcodes_retained": count_barcodes(group),
                "n_barcodes": count_barcodes(group),
                "total_RNA_retained": total_rna,
                "total_DNA_retained": total_dna,
                "dna_sq_sum": dna_sq_sum,
                "zero_RNA_barcode_count_retained": int(np.sum(rna == 0)),
                "zero_RNA_barcode_frac_retained": float(np.mean(rna == 0)),
                "aggregate_zero_RNA_retained": int(total_rna == 0 and total_dna > 0),
                "mean_expr": float(np.log2((total_rna + alpha_rna) / (total_dna + alpha_dna))),
                "mean_expr_no_pc": float(np.log2(total_rna / total_dna)) if total_rna > 0 and total_dna > 0 else np.nan,
                "barcode_logratio_mean_equal_weight": equal_mean,
                "barcode_logratio_var_equal_weight": equal_var,
                "barcode_logratio_mean_dna_weighted": dna_weighted_mean,
                "barcode_var": dna_weighted_var,
                "log_barcode_var": float(np.log(dna_weighted_var + spread_epsilon)),
                "n_eff": n_eff,
                "mean_se_var": mean_se_var,
                "log_mean_se_var": float(np.log(mean_se_var + spread_epsilon)) if np.isfinite(mean_se_var) else np.nan,
                "alpha_R": float(alpha_rna),
                "alpha_D": float(alpha_dna),
                "spread_epsilon": float(spread_epsilon),
                "min_dna_count_for_modeling": float(min_dna_count),
            }
        )

    target_stats = pd.DataFrame(records)
    if target_stats.empty:
        raise ValueError("No retained barcode groups after DNA filtering.")
    return target_stats.merge(raw_stats, on=PARTS_COL, how="left", validate="one_to_one")


def read_variant_sequences(path: Path, sequence_column: str, required_sequence_len: int | None = None) -> pd.DataFrame:
    variant = pd.read_csv(path)
    require_columns(variant, path, [PARTS_COL, sequence_column])
    variant = variant.copy()
    variant["source_row_id"] = np.arange(len(variant), dtype=int)
    if variant[PARTS_COL].duplicated().any():
        duplicated = int(variant[PARTS_COL].duplicated().sum())
        raise ValueError(f"{path} has {duplicated} duplicate {PARTS_COL} rows.")
    sequence = variant[sequence_column].astype("string").str.strip().str.upper()
    valid_dna = sequence.str.fullmatch(r"[ACGTN]+").fillna(False)
    variant = variant.loc[valid_dna].copy()
    variant[sequence_column] = sequence.loc[valid_dna].to_numpy()
    variant["sequence_len"] = variant[sequence_column].str.len()
    if required_sequence_len is not None:
        variant = variant.loc[variant["sequence_len"].eq(int(required_sequence_len))].copy()
    columns = ["source_row_id", PARTS_COL, sequence_column, "sequence_len"]
    for optional in ["Enhancer", "Promoter", "FivePrime", "Intron", "ThreePrime", "number_of_barcodes", "DNA_bc_counts_sum", "RNA_bc_counts_sum", "RNA/DNA"]:
        if optional in variant.columns and optional not in columns:
            columns.append(optional)
    out = variant[columns].reset_index(drop=True)
    rename = {
        "number_of_barcodes": "variant_number_of_barcodes",
        "DNA_bc_counts_sum": "variant_DNA_bc_counts_sum",
        "RNA_bc_counts_sum": "variant_RNA_bc_counts_sum",
        "RNA/DNA": "variant_RNA_DNA",
    }
    return out.rename(columns=rename)


def prepare_part(part: str, args: argparse.Namespace) -> dict:
    spec = PART_SPECS[part]
    output_path = Path(spec["output_path"])
    metadata_path = output_path.with_suffix(".metadata.json")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Use --overwrite to replace it.")

    barcode = read_barcode_counts(Path(spec["barcode_path"]))
    targets = summarize_barcode_targets(
        barcode,
        min_dna_count=args.min_dna_count,
        alpha_rna=args.alpha_rna,
        alpha_dna=args.alpha_dna,
        spread_epsilon=args.spread_epsilon,
    )
    variants = read_variant_sequences(
        Path(spec["variant_path"]),
        spec["sequence_column"],
        spec["required_sequence_len"],
    )
    merged = variants.merge(targets, on=PARTS_COL, how="inner", validate="one_to_one")
    merged["library_name"] = spec["library_name"]
    merged["library_class"] = "single_part"
    merged["RNA_DNA_pc"] = np.power(2.0, merged["mean_expr"])
    merged["log2_RNA_DNA_pc"] = merged["mean_expr"]

    first_columns = [
        "source_row_id",
        "library_class",
        "library_name",
        spec["sequence_column"],
        "sequence_len",
        "n_barcodes",
        "mean_expr",
        "log_barcode_var",
        "barcode_var",
        "n_eff",
        PARTS_COL,
    ]
    remaining_columns = [column for column in merged.columns if column not in first_columns]
    merged = merged[first_columns + remaining_columns].reset_index(drop=True)

    n_hq = int(merged["n_barcodes"].ge(spec["heldout_min_barcodes"]).sum())
    if spec["val_size_within_hq"] is None:
        val_size = max(1, int(round(n_hq * float(spec["val_frac_within_hq"]))))
    else:
        val_size = int(spec["val_size_within_hq"])
    if spec["test_size_within_hq"] is None:
        test_size = max(1, int(round(n_hq * float(spec["test_frac_within_hq"]))))
    else:
        test_size = int(spec["test_size_within_hq"])
    if val_size + test_size >= n_hq:
        raise ValueError(
            f"{part}: requested val/test sizes ({val_size}/{test_size}) exhaust "
            f"HQ rows ({n_hq}) at n_barcodes >= {spec['heldout_min_barcodes']}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, sep="\t", index=False)

    metadata = {
        "part": part,
        "library_name": spec["library_name"],
        "sequence_column": spec["sequence_column"],
        "required_sequence_len": spec["required_sequence_len"],
        "variant_path": str(spec["variant_path"]),
        "barcode_path": str(spec["barcode_path"]),
        "output_path": str(output_path),
        "raw_barcode_rows": int(len(barcode)),
        "target_constructs_after_dna_filter": int(len(targets)),
        "variant_constructs_with_valid_sequence": int(len(variants)),
        "output_rows": int(len(merged)),
        "heldout_min_barcodes": int(spec["heldout_min_barcodes"]),
        "hq_rows_at_heldout_threshold": n_hq,
        "val_frac_within_hq": float(spec["val_frac_within_hq"]),
        "test_frac_within_hq": float(spec["test_frac_within_hq"]),
        "val_size_within_hq": None if spec["val_size_within_hq"] is None else val_size,
        "test_size_within_hq": None if spec["test_size_within_hq"] is None else test_size,
        "expected_val_rows": val_size,
        "expected_test_rows": test_size,
        "expected_train_rows": int(len(merged) - val_size - test_size),
        "target_columns": ["mean_expr", "log_barcode_var"],
        "mean_expr_definition": "log2((sum retained RNA + alpha_R) / (sum retained DNA + alpha_D))",
        "spread_definition": "log(DNA-weighted barcode-level log2-ratio variance + spread_epsilon)",
        "min_dna_count_for_modeling": float(args.min_dna_count),
        "alpha_R": float(args.alpha_rna),
        "alpha_D": float(args.alpha_dna),
        "spread_epsilon": float(args.spread_epsilon),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    args = parse_args()
    all_metadata = []
    for part in args.parts:
        metadata = prepare_part(part, args)
        all_metadata.append(metadata)
        print(
            f"{part:>8}: wrote {metadata['output_rows']:,} rows to {metadata['output_path']} "
            f"(HQ rows >= {metadata['heldout_min_barcodes']}: {metadata['hq_rows_at_heldout_threshold']:,})"
        )

    summary_path = OUTPUT_ROOT / "lib1_two_head_mean_spread_dedup_exact_manifest.json"
    summary_path.write_text(json.dumps(all_metadata, indent=2) + "\n")
    print(f"Wrote manifest to {summary_path}")


if __name__ == "__main__":
    main()
