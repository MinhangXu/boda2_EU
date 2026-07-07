#!/usr/bin/env python3
"""Generate exact-deduplicated Lib1 single-part data products.

This script implements the data-level update described in
plan/repo_hygiene/barcode_level_dedup_update_july6_2026.md.
It intentionally writes new `.dedup_exact` files and leaves the original
source CSV paths in place.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATA_ROOT = Path(
    "/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1"
)
BARCODE_REL = Path("barcode_level/L1_variant_bc_expr_combined_20251107_np_fastq1-5.csv")
ARCHIVE_REL = Path("archive_pre_dedup_20260706")


@dataclass(frozen=True)
class SinglePartLibrary:
    library_name: str
    part_pattern: str
    source_rel: Path
    output_name: str
    barcode_output_name: str


SINGLE_PART_LIBRARIES = (
    SinglePartLibrary(
        library_name="enhancer_subset_0filtered_out",
        part_pattern="enhancer",
        source_rel=Path(
            "single_part_variant_level/enhancers/"
            "L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.csv"
        ),
        output_name="L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.dedup_exact.csv",
        barcode_output_name=(
            "single_part__enhancer_subset_0filtered_out.dedup_exact.barcode_level.csv"
        ),
    ),
    SinglePartLibrary(
        library_name="Promoter_subset",
        part_pattern="promoter",
        source_rel=Path(
            "single_part_variant_level/promoters/"
            "L1_final_fastqs1-5_sublibrary_Promoter_subset.csv"
        ),
        output_name="L1_final_fastqs1-5_sublibrary_Promoter_subset.dedup_exact.csv",
        barcode_output_name="single_part__Promoter_subset.dedup_exact.barcode_level.csv",
    ),
    SinglePartLibrary(
        library_name="FivePrime_subset",
        part_pattern="five_prime",
        source_rel=Path(
            "single_part_variant_level/FivePrimes/"
            "L1_final_fastqs1-5_sublibrary_FivePrime_subset.csv"
        ),
        output_name="L1_final_fastqs1-5_sublibrary_FivePrime_subset.dedup_exact.csv",
        barcode_output_name="single_part__FivePrime_subset.dedup_exact.barcode_level.csv",
    ),
    SinglePartLibrary(
        library_name="Intron_subset",
        part_pattern="intron",
        source_rel=Path(
            "single_part_variant_level/introns/"
            "L1_final_fastqs1-5_sublibrary_Intron_subset.csv"
        ),
        output_name="L1_final_fastqs1-5_sublibrary_Intron_subset.dedup_exact.csv",
        barcode_output_name="single_part__Intron_subset.dedup_exact.barcode_level.csv",
    ),
    SinglePartLibrary(
        library_name="ThreePrime_subset",
        part_pattern="three_prime",
        source_rel=Path(
            "single_part_variant_level/ThreePrimes/"
            "L1_final_fastqs1-5_sublibrary_ThreePrime_subset.csv"
        ),
        output_name="L1_final_fastqs1-5_sublibrary_ThreePrime_subset.dedup_exact.csv",
        barcode_output_name="single_part__ThreePrime_subset.dedup_exact.barcode_level.csv",
    ),
)


UNFILTERED_ENHANCER_REL = Path(
    "single_part_variant_level/enhancers/"
    "L1_final_fastqs1-5_sublibrary_enhancer_subset.csv"
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def prepare_output(path: Path, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; rerun with --force to overwrite")


def write_json(path: Path, payload: dict[str, Any], *, force: bool) -> None:
    prepare_output(path, force=force)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def finite_max_abs(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    finite = finite.dropna()
    if finite.empty:
        return None
    return float(finite.abs().max())


def count_distinct_nonblank(values: pd.Series) -> int:
    as_str = values.dropna().astype(str).str.strip()
    return int(as_str[as_str != ""].nunique())


def audit_variable_key_groups(barcode_df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    key_cols = ["parts_concatenated", "bba1_ddc1_concat"]
    check_cols = [
        "bba1_barcode",
        "bba1_score",
        "ddc1_barcode",
        "ddc1_score",
        "DNA_bc_counts",
        "RNA_bc_counts",
        "RNA/DNA",
    ]
    nunique = barcode_df.groupby(key_cols, dropna=False)[check_cols].nunique(dropna=False)
    variable_groups = nunique[(nunique > 1).any(axis=1)]
    if variable_groups.empty:
        return 0, pd.DataFrame()

    rows = []
    for group_index, (parts_concatenated, concat_barcode) in enumerate(
        variable_groups.index, start=1
    ):
        mask = (
            barcode_df["parts_concatenated"].eq(parts_concatenated)
            & barcode_df["bba1_ddc1_concat"].eq(concat_barcode)
        )
        if pd.isna(concat_barcode):
            mask = barcode_df["parts_concatenated"].eq(parts_concatenated) & barcode_df[
                "bba1_ddc1_concat"
            ].isna()
        unique_rows = barcode_df.loc[mask].drop_duplicates().copy()
        variable_columns = [
            col for col in check_cols if variable_groups.loc[(parts_concatenated, concat_barcode), col] > 1
        ]
        unique_rows.insert(0, "variable_columns", ",".join(variable_columns))
        unique_rows.insert(0, "exact_unique_row_count", len(unique_rows))
        unique_rows.insert(0, "raw_row_count", int(mask.sum()))
        unique_rows.insert(0, "variable_group_index", group_index)
        rows.append(unique_rows)

    return len(variable_groups), pd.concat(rows, ignore_index=True)


def make_barcode_dedup(
    data_root: Path, *, force: bool, script_path: Path, timestamp: str
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    source_path = data_root / BARCODE_REL
    require_path(source_path)
    output_path = source_path.with_name(source_path.stem + ".dedup_exact.csv")
    manifest_path = output_path.with_suffix(".manifest.json")
    audit_path = output_path.with_suffix(".variable_key_audit.csv")

    prepare_output(output_path, force=force)

    barcode_df = pd.read_csv(source_path)
    row_count_before = int(len(barcode_df))
    exact_duplicate_rows = int(barcode_df.duplicated(keep="first").sum())
    dedup_df = barcode_df.drop_duplicates(keep="first").copy()

    variable_key_group_count, variable_key_audit = audit_variable_key_groups(barcode_df)
    if not variable_key_audit.empty:
        prepare_output(audit_path, force=force)
        variable_key_audit.to_csv(audit_path, index=False)

    dedup_df.to_csv(output_path, index=False)

    manifest = {
        "created_at_utc": timestamp,
        "script": str(script_path),
        "deduplication_policy": "drop exact duplicate rows; keep first representative",
        "deduplication_subset": list(barcode_df.columns),
        "source_file": str(source_path),
        "source_sha256": sha256_file(source_path),
        "output_file": str(output_path),
        "output_sha256": sha256_file(output_path),
        "row_count_before": row_count_before,
        "row_count_after": int(len(dedup_df)),
        "duplicate_rows_removed": exact_duplicate_rows,
        "same_construct_barcode_variable_group_count": variable_key_group_count,
        "same_construct_barcode_variable_group_audit_file": (
            str(audit_path) if not variable_key_audit.empty else None
        ),
    }
    write_json(manifest_path, manifest, force=force)
    return dedup_df, output_path, manifest


def aggregate_barcode_rows(barcode_subset: pd.DataFrame) -> pd.DataFrame:
    aggregated = (
        barcode_subset.groupby("parts_concatenated", dropna=False)
        .agg(
            number_of_barcodes=("bba1_ddc1_concat", count_distinct_nonblank),
            DNA_bc_counts_sum=("DNA_bc_counts", "sum"),
            RNA_bc_counts_sum=("RNA_bc_counts", "sum"),
        )
        .reset_index()
    )
    aggregated["RNA/DNA"] = np.where(
        aggregated["DNA_bc_counts_sum"] > 0,
        aggregated["RNA_bc_counts_sum"] / aggregated["DNA_bc_counts_sum"],
        np.nan,
    )
    return aggregated


def validate_variant_output(
    old_df: pd.DataFrame, new_df: pd.DataFrame, library: SinglePartLibrary
) -> dict[str, Any]:
    merged = old_df.merge(
        new_df[
            [
                "parts_concatenated",
                "number_of_barcodes",
                "DNA_bc_counts_sum",
                "RNA_bc_counts_sum",
                "RNA/DNA",
            ]
        ],
        on="parts_concatenated",
        how="outer",
        suffixes=("_old", "_new"),
        indicator=True,
    )
    row_set_matches = bool((merged["_merge"] == "both").all() and len(merged) == len(old_df))
    both = merged[merged["_merge"] == "both"].copy()

    ratio_expected = both["RNA_bc_counts_sum_new"] / both["DNA_bc_counts_sum_new"]
    ratio_expected = ratio_expected.replace([np.inf, -np.inf], np.nan)
    ratio_delta = both["RNA/DNA_new"] - ratio_expected

    old_ratio = pd.to_numeric(both["RNA/DNA_old"], errors="coerce")
    new_ratio = pd.to_numeric(both["RNA/DNA_new"], errors="coerce")

    validation = {
        "library_name": library.library_name,
        "row_count_old": int(len(old_df)),
        "row_count_new": int(len(new_df)),
        "row_set_matches": row_set_matches,
        "rows_missing_from_new": int((merged["_merge"] == "left_only").sum()),
        "rows_added_to_new": int((merged["_merge"] == "right_only").sum()),
        "number_of_barcodes_mismatch_count": int(
            (
                both["number_of_barcodes_old"].astype("int64")
                != both["number_of_barcodes_new"].astype("int64")
            ).sum()
        ),
        "dna_sum_increase_count": int(
            (both["DNA_bc_counts_sum_new"] > both["DNA_bc_counts_sum_old"]).sum()
        ),
        "rna_sum_increase_count": int(
            (both["RNA_bc_counts_sum_new"] > both["RNA_bc_counts_sum_old"]).sum()
        ),
        "ratio_recompute_max_abs_delta": finite_max_abs(ratio_delta),
        "dna_sum_changed_count": int(
            (both["DNA_bc_counts_sum_new"] != both["DNA_bc_counts_sum_old"]).sum()
        ),
        "rna_sum_changed_count": int(
            (both["RNA_bc_counts_sum_new"] != both["RNA_bc_counts_sum_old"]).sum()
        ),
        "rna_dna_changed_count": int((~np.isclose(old_ratio, new_ratio, equal_nan=True)).sum()),
        "old_rna_dna_min": float(old_ratio.min(skipna=True)),
        "old_rna_dna_max": float(old_ratio.max(skipna=True)),
        "new_rna_dna_min": float(new_ratio.min(skipna=True)),
        "new_rna_dna_max": float(new_ratio.max(skipna=True)),
    }
    validation["passed"] = bool(
        validation["row_set_matches"]
        and validation["number_of_barcodes_mismatch_count"] == 0
        and validation["dna_sum_increase_count"] == 0
        and validation["rna_sum_increase_count"] == 0
        and (
            validation["ratio_recompute_max_abs_delta"] is None
            or validation["ratio_recompute_max_abs_delta"] <= 1e-12
        )
    )
    return validation


def make_variant_and_barcode_library_outputs(
    data_root: Path,
    barcode_dedup: pd.DataFrame,
    barcode_dedup_path: Path,
    barcode_manifest: dict[str, Any],
    *,
    force: bool,
    script_path: Path,
    timestamp: str,
) -> list[dict[str, Any]]:
    output_root = data_root / "single_part_variant_level"
    barcode_library_root = data_root / "barcode_level/by_library"
    validations = []

    for library in SINGLE_PART_LIBRARIES:
        source_path = data_root / library.source_rel
        require_path(source_path)
        old_df = pd.read_csv(source_path)
        output_path = output_root / library.output_name
        manifest_path = output_path.with_suffix(".manifest.json")
        barcode_output_path = barcode_library_root / library.barcode_output_name
        barcode_manifest_path = barcode_output_path.with_suffix(".manifest.json")

        part_set = set(old_df["parts_concatenated"])
        barcode_subset = barcode_dedup[
            barcode_dedup["parts_concatenated"].isin(part_set)
        ].copy()

        missing_parts = sorted(part_set - set(barcode_subset["parts_concatenated"]))
        if missing_parts:
            raise ValueError(
                f"{library.library_name}: {len(missing_parts)} parts are missing "
                "from the deduplicated barcode table"
            )

        aggregated = aggregate_barcode_rows(barcode_subset)
        new_df = old_df.drop(
            columns=[
                "number_of_barcodes",
                "DNA_bc_counts_sum",
                "RNA_bc_counts_sum",
                "RNA/DNA",
            ]
        ).merge(aggregated, on="parts_concatenated", how="left")
        new_df = new_df.loc[:, old_df.columns]

        validation = validate_variant_output(old_df, new_df, library)
        validations.append(validation)
        if not validation["passed"]:
            raise RuntimeError(
                f"{library.library_name} validation failed: "
                + json.dumps(validation, sort_keys=True)
            )

        prepare_output(output_path, force=force)
        new_df.to_csv(output_path, index=False)

        variant_manifest = {
            "created_at_utc": timestamp,
            "script": str(script_path),
            "library_layer": "single_part",
            "library_name": library.library_name,
            "part_pattern": library.part_pattern,
            "source_variant_file": str(source_path),
            "source_variant_sha256": sha256_file(source_path),
            "barcode_dedup_file": str(barcode_dedup_path),
            "barcode_dedup_sha256": barcode_manifest["output_sha256"],
            "output_file": str(output_path),
            "output_sha256": sha256_file(output_path),
            "aggregation_policy": {
                "barcode_filter": (
                    "deduplicated barcode rows with parts_concatenated in the "
                    "source variant row set"
                ),
                "number_of_barcodes": "distinct nonblank bba1_ddc1_concat values",
                "DNA_bc_counts_sum": "sum DNA_bc_counts over exact-deduplicated barcode rows",
                "RNA_bc_counts_sum": "sum RNA_bc_counts over exact-deduplicated barcode rows",
                "RNA/DNA": "RNA_bc_counts_sum / DNA_bc_counts_sum",
            },
            "validation": validation,
        }
        write_json(manifest_path, variant_manifest, force=force)

        barcode_subset["library_layer"] = "single_part"
        barcode_subset["library_name"] = library.library_name
        barcode_subset["variant_file"] = str(output_path)
        barcode_subset["part_pattern"] = library.part_pattern
        prepare_output(barcode_output_path, force=force)
        barcode_subset.to_csv(barcode_output_path, index=False)

        barcode_library_manifest = {
            "created_at_utc": timestamp,
            "script": str(script_path),
            "library_layer": "single_part",
            "library_name": library.library_name,
            "part_pattern": library.part_pattern,
            "source_variant_file": str(source_path),
            "dedup_variant_file": str(output_path),
            "barcode_dedup_file": str(barcode_dedup_path),
            "output_file": str(barcode_output_path),
            "output_sha256": sha256_file(barcode_output_path),
            "row_count": int(len(barcode_subset)),
            "unique_parts_concatenated": int(barcode_subset["parts_concatenated"].nunique()),
            "variant_row_count": int(len(old_df)),
            "filter_policy": (
                "deduplicated barcode rows whose parts_concatenated is in the "
                "single-part variant table row set"
            ),
            "split_policy_note": (
                "For sequence-to-expression modeling, split by parts_concatenated "
                "rather than individual barcode rows."
            ),
            "per_barcode_log_target_columns": "not added; pseudocount policy is unresolved",
        }
        write_json(barcode_manifest_path, barcode_library_manifest, force=force)

    return validations


def archive_source_files(
    data_root: Path, *, force: bool, timestamp: str, script_path: Path
) -> dict[str, Any]:
    archive_root = data_root / ARCHIVE_REL
    files_to_archive = [BARCODE_REL, UNFILTERED_ENHANCER_REL] + [
        library.source_rel for library in SINGLE_PART_LIBRARIES
    ]

    entries = []
    for rel_path in files_to_archive:
        source_path = data_root / rel_path
        require_path(source_path)
        archive_path = archive_root / rel_path.with_suffix(rel_path.suffix + ".gz")
        prepare_output(archive_path, force=force)
        with source_path.open("rb") as source_handle, archive_path.open(
            "wb"
        ) as compressed_handle, gzip.GzipFile(
            fileobj=compressed_handle, mode="wb", compresslevel=6, mtime=0
        ) as archive_handle:
            shutil.copyfileobj(source_handle, archive_handle, length=1024 * 1024)

        entries.append(
            {
                "source_file": str(source_path),
                "source_sha256": sha256_file(source_path),
                "archive_file": str(archive_path),
                "archive_sha256": sha256_file(archive_path),
                "original_left_in_place": True,
            }
        )

    manifest = {
        "created_at_utc": timestamp,
        "script": str(script_path),
        "archive_policy": (
            "gzip copies were written under archive_pre_dedup_20260706; original "
            "CSV paths were left in place pending repo code default updates"
        ),
        "entries": entries,
    }

    manifest_json_path = archive_root / "MANIFEST.json"
    write_json(manifest_json_path, manifest, force=force)

    manifest_md_path = archive_root / "MANIFEST.md"
    prepare_output(manifest_md_path, force=force)
    lines = [
        "# Lib1 Pre-Dedup Archive",
        "",
        f"Created: {timestamp}",
        "",
        "The files below are gzip copies of the pre-deduplication CSVs. The",
        "original CSV paths were intentionally left in place so existing notebooks",
        "and scripts keep working until the repo defaults are updated.",
        "",
        "| Source | Archive | Source SHA256 |",
        "|---|---|---|",
    ]
    for entry in entries:
        lines.append(
            f"| `{entry['source_file']}` | `{entry['archive_file']}` | "
            f"`{entry['source_sha256']}` |"
        )
    lines.append("")
    manifest_md_path.write_text("\n".join(lines))
    return manifest


def summarize_validations(validations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all_passed": all(validation["passed"] for validation in validations),
        "libraries": validations,
        "total_rows_old": int(sum(validation["row_count_old"] for validation in validations)),
        "total_rows_new": int(sum(validation["row_count_new"] for validation in validations)),
        "total_dna_sum_changed_rows": int(
            sum(validation["dna_sum_changed_count"] for validation in validations)
        ),
        "total_rna_sum_changed_rows": int(
            sum(validation["rna_sum_changed_count"] for validation in validations)
        ),
        "total_rna_dna_changed_rows": int(
            sum(validation["rna_dna_changed_count"] for validation in validations)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exact-deduplicated Lib1 single-part data products."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"MattLee_lib1 data root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--archive-old",
        action="store_true",
        help="Write gzip archive copies of old barcode and single-part CSVs after validation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing generated outputs and manifests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    script_path = Path(__file__).resolve()
    timestamp = utc_timestamp()

    barcode_dedup, barcode_dedup_path, barcode_manifest = make_barcode_dedup(
        data_root, force=args.force, script_path=script_path, timestamp=timestamp
    )
    validations = make_variant_and_barcode_library_outputs(
        data_root,
        barcode_dedup,
        barcode_dedup_path,
        barcode_manifest,
        force=args.force,
        script_path=script_path,
        timestamp=timestamp,
    )
    validation_summary = summarize_validations(validations)
    summary_path = data_root / "single_part_variant_level/dedup_exact.validation_summary.json"
    write_json(
        summary_path,
        {
            "created_at_utc": timestamp,
            "script": str(script_path),
            "barcode_manifest": barcode_manifest,
            "validation_summary": validation_summary,
        },
        force=args.force,
    )

    archive_manifest = None
    if args.archive_old:
        archive_manifest = archive_source_files(
            data_root, force=args.force, timestamp=timestamp, script_path=script_path
        )

    print(json.dumps(
        {
            "barcode_output": str(barcode_dedup_path),
            "barcode_rows_before": barcode_manifest["row_count_before"],
            "barcode_rows_after": barcode_manifest["row_count_after"],
            "duplicate_rows_removed": barcode_manifest["duplicate_rows_removed"],
            "same_construct_barcode_variable_group_count": barcode_manifest[
                "same_construct_barcode_variable_group_count"
            ],
            "validation_summary": validation_summary,
            "archive_manifest": archive_manifest,
        },
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
