#!/usr/bin/env python3
"""Prepare the Phase 3 Hani 5'UTR Lib1+Lib2 scratch-training table.

The output is a wide observed-head CSV compatible with
`UTR5_Branched_RNA_Activity_DataModule`: one row per unique 50-nt sequence,
`fold` in {train,val,test}, `library`, and the shared observed activity heads.

Policy:
- Preserve the existing Lib1 train/val/test folds.
- Aggregate Lib2 replicate rows by uppercased sequence and cell type.
- Assign Lib2 folds by deterministic sequence hash, reserving Lib2 test.
- Drop Lib1 rows whose sequence appears in Lib2 by default, preventing
  cross-library sequence leakage into the reserved Lib2 test surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = REPO_ROOT.parent
DEFAULT_PROCESSED_DIR = (
    WORK_ROOT / "opt_EU_learn_n_design" / "utr_hani_2025" / "processed_utr_data"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "derived_data" / "utr5" / "hani_rna_activity"
)
DEFAULT_OUTPUT_PATH = (
    DEFAULT_OUTPUT_DIR / "5UTR_lib1_lib2_phase3_branched_observed_heads.csv"
)
DEFAULT_MANIFEST_PATH = (
    DEFAULT_OUTPUT_DIR / "5UTR_lib1_lib2_phase3_branched_observed_heads_manifest.json"
)
DEFAULT_SPLIT_MANIFEST_PATH = (
    DEFAULT_OUTPUT_DIR / "5UTR_lib1_lib2_phase3_sequence_split_manifest.csv"
)
HEADS = ["c1", "c2", "c4", "c6", "c17"]
VALID_DNA = set("ACGT")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a learn-ready Hani 5'UTR Lib1+Lib2 Phase 3 table."
    )
    parser.add_argument(
        "--lib1_path",
        type=Path,
        default=DEFAULT_PROCESSED_DIR / "5UTR_lib1_branched_observed_heads.csv",
        help="Wide Lib1 observed-head table.",
    )
    parser.add_argument(
        "--lib2_path",
        type=Path,
        default=DEFAULT_PROCESSED_DIR / "5UTR_lib2_processed.csv",
        help="Long Lib2 processed table with replicate rows.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination combined CSV.",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Destination JSON audit manifest.",
    )
    parser.add_argument(
        "--split_manifest_path",
        type=Path,
        default=DEFAULT_SPLIT_MANIFEST_PATH,
        help="Destination per-sequence split manifest CSV.",
    )
    parser.add_argument("--heads", nargs="+", default=HEADS)
    parser.add_argument("--lib2_val_frac", type=float, default=0.10)
    parser.add_argument("--lib2_test_frac", type=float, default=0.10)
    parser.add_argument("--lib2_split_seed", type=int, default=42)
    parser.add_argument(
        "--overlap_policy",
        choices=["drop_lib1", "drop_lib2", "error", "keep"],
        default="drop_lib1",
        help="How to handle uppercased sequences present in both Lib1 and Lib2.",
    )
    return parser


def clean_sequence(value: Any) -> str:
    return str(value).strip().upper()


def is_valid_exact_dna(seq: str, length: int = 50) -> bool:
    return len(seq) == length and set(seq).issubset(VALID_DNA)


def gc_fraction(seq: str) -> float:
    if not seq:
        return float("nan")
    return float((seq.count("G") + seq.count("C")) / len(seq))


def hash_float(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def validate_split_fractions(val_frac: float, test_frac: float) -> None:
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"lib2_val_frac must be in (0, 1), got {val_frac}")
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"lib2_test_frac must be in (0, 1), got {test_frac}")
    if val_frac + test_frac >= 1.0:
        raise ValueError("lib2_val_frac + lib2_test_frac must be < 1")


def assign_lib2_folds(df: pd.DataFrame, seed: int, val_frac: float, test_frac: float) -> pd.DataFrame:
    validate_split_fractions(val_frac, test_frac)
    if len(df) < 10:
        raise ValueError("Need at least 10 Lib2 sequences for train/val/test splitting.")

    out = df.copy()
    out["split_hash"] = out["seq"].map(lambda seq: hash_float(seq, seed))
    out = out.sort_values(["split_hash", "seq"]).reset_index(drop=True)

    n_total = len(out)
    n_test = max(1, int(round(n_total * test_frac)))
    n_val = max(1, int(round(n_total * val_frac)))
    if n_test + n_val >= n_total:
        raise ValueError("Requested Lib2 val/test fractions leave no training sequences.")

    folds = np.full(n_total, "train", dtype=object)
    folds[:n_test] = "test"
    folds[n_test : n_test + n_val] = "val"
    out["fold"] = folds
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_lib1_wide(path: Path, heads: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_csv(path)
    required = ["seq", "fold", *heads]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    out = df.copy()
    out["seq_original_example"] = out["seq"].astype(str)
    out["seq"] = out["seq_original_example"].map(clean_sequence)
    out["sequence_len"] = out["seq"].str.len()
    out["is_valid_exact_50nt"] = out["seq"].map(is_valid_exact_dna)
    for head in heads:
        out[head] = pd.to_numeric(out[head], errors="coerce")

    input_rows = len(out)
    out = out.loc[out["is_valid_exact_50nt"]].dropna(subset=["fold", *heads]).copy()
    valid_complete_rows = len(out)
    out["fold"] = out["fold"].astype(str)
    bad_folds = sorted(set(out["fold"]) - {"train", "val", "test"})
    if bad_folds:
        raise ValueError(f"{path} has unsupported Lib1 fold labels: {bad_folds}")

    seq_fold_counts = out.groupby("seq", observed=True)["fold"].nunique()
    multi_fold = seq_fold_counts[seq_fold_counts > 1]
    if len(multi_fold):
        examples = multi_fold.head().index.tolist()
        raise ValueError(f"Lib1 has sequences assigned to multiple folds: {examples}")

    out = (
        out.groupby(["seq", "fold"], observed=True)
        .agg(
            **{head: (head, "mean") for head in heads},
            seq_original_example=("seq_original_example", "first"),
        )
        .reset_index()
    )
    out["library"] = "5UTR_lib1"
    out["source_fold"] = out["fold"]
    out["split_policy"] = "preserve_lib1_fold"
    out["split_hash"] = np.nan
    out["gc_fraction"] = out["seq"].map(gc_fraction)
    out["sequence_len"] = out["seq"].str.len()

    audit = {
        "input_rows": int(input_rows),
        "valid_exact_complete_rows_before_duplicate_collapse": int(valid_complete_rows),
        "output_sequences": int(len(out)),
        "fold_counts": {k: int(v) for k, v in out["fold"].value_counts().sort_index().items()},
    }
    return out, audit


def load_lib2_wide(
    path: Path,
    heads: list[str],
    split_seed: int,
    val_frac: float,
    test_frac: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path)
    required = ["seq", "cell_type", "rna_activity"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = raw.copy()
    df["seq_original"] = df["seq"].astype(str)
    df["seq"] = df["seq_original"].map(clean_sequence)
    df["cell_type"] = df["cell_type"].astype(str)
    df["rna_activity"] = pd.to_numeric(df["rna_activity"], errors="coerce")
    df["is_valid_exact_50nt"] = df["seq"].map(is_valid_exact_dna)

    usable = df.loc[
        df["is_valid_exact_50nt"]
        & df["cell_type"].isin(heads)
        & np.isfinite(df["rna_activity"])
    ].copy()
    if usable.empty:
        raise ValueError(f"No usable exact 50-nt Lib2 rows found in {path}")

    agg = (
        usable.groupby(["seq", "cell_type"], observed=True)
        .agg(
            rna_activity=("rna_activity", "mean"),
            n_observations=("rna_activity", "size"),
            seq_original_example=("seq_original", "first"),
        )
        .reset_index()
    )
    wide = agg.pivot(index="seq", columns="cell_type", values="rna_activity")
    counts = agg.pivot(index="seq", columns="cell_type", values="n_observations")
    meta = (
        usable.groupby("seq", observed=True)
        .agg(
            seq_original_example=("seq_original", "first"),
            n_raw_rows=("rna_activity", "size"),
        )
        .reset_index()
        .set_index("seq")
    )
    wide = wide.join(meta, how="left")

    for head in heads:
        if head not in wide.columns:
            wide[head] = np.nan
        wide[f"n_obs_{head}"] = counts[head] if head in counts.columns else 0
        wide[f"n_obs_{head}"] = wide[f"n_obs_{head}"].fillna(0).astype(int)

    wide = wide.reset_index().dropna(subset=heads).copy()
    wide = assign_lib2_folds(
        wide,
        seed=split_seed,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    wide["library"] = "5UTR_lib2"
    wide["source_fold"] = wide["fold"]
    wide["split_policy"] = f"hash_seed_{split_seed}_val_{val_frac:g}_test_{test_frac:g}"
    wide["gc_fraction"] = wide["seq"].map(gc_fraction)
    wide["sequence_len"] = wide["seq"].str.len()

    audit = {
        "input_rows": int(len(raw)),
        "usable_long_rows": int(len(usable)),
        "wide_complete_sequences": int(len(wide)),
        "fold_counts": {k: int(v) for k, v in wide["fold"].value_counts().sort_index().items()},
        "raw_rows_by_head": {
            head: int((usable["cell_type"] == head).sum())
            for head in heads
        },
        "unique_sequences_by_head": {
            head: int(usable.loc[usable["cell_type"].eq(head), "seq"].nunique())
            for head in heads
        },
        "aggregation_policy": "mean rna_activity by uppercased sequence and cell_type",
    }
    return wide, audit


def apply_overlap_policy(
    lib1: pd.DataFrame,
    lib2: pd.DataFrame,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    overlap = sorted(set(lib1["seq"]) & set(lib2["seq"]))
    audit = {
        "overlap_policy": policy,
        "n_overlapping_sequences_before_policy": int(len(overlap)),
        "overlap_examples": overlap[:10],
    }
    if not overlap or policy == "keep":
        return lib1, lib2, audit
    if policy == "error":
        raise ValueError(f"Found {len(overlap)} Lib1/Lib2 overlapping sequences; examples: {overlap[:10]}")
    if policy == "drop_lib1":
        lib1 = lib1.loc[~lib1["seq"].isin(overlap)].copy()
        audit["dropped_from_lib1"] = int(len(overlap))
        audit["dropped_from_lib2"] = 0
        return lib1, lib2, audit
    if policy == "drop_lib2":
        lib2 = lib2.loc[~lib2["seq"].isin(overlap)].copy()
        audit["dropped_from_lib1"] = 0
        audit["dropped_from_lib2"] = int(len(overlap))
        return lib1, lib2, audit
    raise ValueError(f"Unsupported overlap policy: {policy}")


def summarize_targets(df: pd.DataFrame, heads: list[str]) -> dict[str, Any]:
    summary = {}
    for library, lib_df in df.groupby("library", observed=True):
        summary[library] = {}
        for fold, sub in lib_df.groupby("fold", observed=True):
            summary[library][fold] = {
                "n_sequences": int(len(sub)),
                "target_means": {head: float(sub[head].mean()) for head in heads},
                "target_stds": {head: float(sub[head].std()) for head in heads},
                "average_activity_mean": float(sub[heads].mean(axis=1).mean()),
                "average_activity_std": float(sub[heads].mean(axis=1).std()),
                "gc_mean": float(sub["gc_fraction"].mean()),
                "gc_std": float(sub["gc_fraction"].std()),
            }
    return summary


def main() -> None:
    args = build_argparser().parse_args()
    heads = list(args.heads)

    lib1, lib1_audit = load_lib1_wide(args.lib1_path, heads)
    lib2, lib2_audit = load_lib2_wide(
        args.lib2_path,
        heads,
        split_seed=args.lib2_split_seed,
        val_frac=args.lib2_val_frac,
        test_frac=args.lib2_test_frac,
    )
    lib1, lib2, overlap_audit = apply_overlap_policy(lib1, lib2, args.overlap_policy)

    ordered_extra_cols = [
        "seq_original_example",
        "source_fold",
        "split_policy",
        "split_hash",
        "sequence_len",
        "gc_fraction",
        "n_raw_rows",
        *[f"n_obs_{head}" for head in heads],
    ]
    for frame in (lib1, lib2):
        for column in ordered_extra_cols:
            if column not in frame.columns:
                frame[column] = np.nan

    combined_columns = ["seq", "fold", "library", *heads, *ordered_extra_cols]
    combined = pd.concat(
        [lib1[combined_columns], lib2[combined_columns]],
        ignore_index=True,
    ).sort_values(["fold", "library", "seq"]).reset_index(drop=True)

    seq_library_counts = combined.groupby("seq", observed=True)["library"].nunique()
    leaked = seq_library_counts[seq_library_counts > 1]
    if len(leaked):
        raise ValueError(f"Combined table still has cross-library duplicate sequences: {leaked.head().index.tolist()}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    args.split_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(args.output_path, index=False)
    split_manifest_cols = [
        "seq",
        "library",
        "fold",
        "source_fold",
        "split_policy",
        "split_hash",
        "sequence_len",
        "gc_fraction",
        "seq_original_example",
        *[f"n_obs_{head}" for head in heads],
    ]
    combined[split_manifest_cols].to_csv(args.split_manifest_path, index=False)

    manifest = {
        "description": "Phase 3 Hani 5'UTR Lib1+Lib2 scratch-training dataset",
        "heads": heads,
        "lib1_path": str(args.lib1_path),
        "lib2_path": str(args.lib2_path),
        "output_path": str(args.output_path),
        "split_manifest_path": str(args.split_manifest_path),
        "lib2_split_seed": int(args.lib2_split_seed),
        "lib2_val_frac": float(args.lib2_val_frac),
        "lib2_test_frac": float(args.lib2_test_frac),
        "lib1_audit": lib1_audit,
        "lib2_audit": lib2_audit,
        "overlap_audit": overlap_audit,
        "combined_rows": int(len(combined)),
        "combined_counts_by_library_fold": {
            f"{library}/{fold}": int(n)
            for (library, fold), n in combined.groupby(["library", "fold"], observed=True).size().sort_index().items()
        },
        "target_summary_by_library_fold": summarize_targets(combined, heads),
    }
    with args.manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote {args.output_path} with {len(combined)} rows")
    print(f"Wrote {args.split_manifest_path}")
    print(f"Wrote {args.manifest_path}")
    print("Counts by library/fold:")
    print(combined.groupby(["library", "fold"], observed=True).size().sort_index().to_string())


if __name__ == "__main__":
    main()
