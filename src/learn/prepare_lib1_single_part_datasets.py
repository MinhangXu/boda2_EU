#!/usr/bin/env python3
"""Prepare the canonical five-part Lib1 single-output data products.

This module is the shared implementation behind the five historical
``prepare_lib1_*`` entry points.  It can produce both the exact-barcode-dedup
data used by the July 2026 campaign and byte-stable pre-dedup calibration
mates with the same schema and length policies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = Path(os.environ.get("BODA_WORK_ROOT", REPO_ROOT.parent)).expanduser()
VARIANT_ROOT = (
    WORK_ROOT
    / "opt_EU_learn_n_design"
    / "MattLee_lib1"
    / "single_part_variant_level"
)
DERIVED_ROOT = REPO_ROOT / "src" / "learn" / "derived_data"
DEFAULT_DATA_MANIFEST = (
    REPO_ROOT
    / "src"
    / "learn"
    / "data_manifests"
    / "lib1_single_part_dedup_exact_v1.json"
)

DATA_GENERATION_IDS = {
    "dedup_exact": "lib1_single_part_dedup_exact_v1",
    "pre_dedup": "lib1_single_part_pre_dedup_v0",
}
DEDUP_POLICY = "exact_barcode_row_dedup_v1"
PRE_DEDUP_POLICY = "pre_dedup_source_v0"
TARGET_COLUMN = "log2_RNA_DNA"
TARGET_FORMULA = "log2(RNA_bc_counts_sum / DNA_bc_counts_sum)"
SPLIT_ID_COLUMN = "construct_id"
STABLE_ID_SOURCE_COLUMN = "parts_concatenated"
STABLE_ID_ALGORITHM = "sha256_utf8(parts_concatenated)"


@dataclass(frozen=True)
class PartSpec:
    part: str
    part_slug: str
    sequence_column: str
    dedup_source_path: Path
    dedup_manifest_path: Path
    dedup_output_path: Path
    pre_dedup_output_path: Path
    length_policy: str
    selected_length: int | None
    padded_seq_len: int
    padding_mode: str
    expected_row_count: int
    expected_high_barcode_count: int


def _variant_path(filename: str) -> Path:
    return VARIANT_ROOT / filename


PART_SPECS: Mapping[str, PartSpec] = {
    "enhancer": PartSpec(
        part="Enhancer",
        part_slug="enhancer",
        sequence_column="Enhancer",
        dedup_source_path=_variant_path(
            "L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.dedup_exact.csv"
        ),
        dedup_manifest_path=_variant_path(
            "L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.dedup_exact.manifest.json"
        ),
        dedup_output_path=(
            DERIVED_ROOT
            / "enhancer"
            / "bashor_in_house"
            / "lib1_enhancer_allvalid_pad216_fastqs1_5_dedup_exact__learn_ready.tsv"
        ),
        pre_dedup_output_path=(
            DERIVED_ROOT
            / "enhancer"
            / "bashor_in_house"
            / "lib1_enhancer_allvalid_pad216_fastqs1_5_pre_dedup_v0__learn_ready.tsv"
        ),
        length_policy="all_valid_76_211_neutral_pad_to_216",
        selected_length=None,
        padded_seq_len=216,
        padding_mode="neutral",
        expected_row_count=4787,
        expected_high_barcode_count=1229,
    ),
    "promoter": PartSpec(
        part="Promoter",
        part_slug="promoter",
        sequence_column="Promoter",
        dedup_source_path=_variant_path(
            "L1_final_fastqs1-5_sublibrary_Promoter_subset.dedup_exact.csv"
        ),
        dedup_manifest_path=_variant_path(
            "L1_final_fastqs1-5_sublibrary_Promoter_subset.dedup_exact.manifest.json"
        ),
        dedup_output_path=(
            DERIVED_ROOT
            / "promoter"
            / "bashor_in_house"
            / "lib1_promoter_allvalid_fastqs1_5_dedup_exact__learn_ready.tsv"
        ),
        pre_dedup_output_path=(
            DERIVED_ROOT
            / "promoter"
            / "bashor_in_house"
            / "lib1_promoter_allvalid_fastqs1_5_pre_dedup_v0__learn_ready.tsv"
        ),
        length_policy="all_valid_41_51_neutral_pad_to_51",
        selected_length=None,
        padded_seq_len=51,
        padding_mode="neutral",
        expected_row_count=7893,
        expected_high_barcode_count=1931,
    ),
    "intron": PartSpec(
        part="Intron",
        part_slug="intron",
        sequence_column="Intron",
        dedup_source_path=_variant_path(
            "L1_final_fastqs1-5_sublibrary_Intron_subset.dedup_exact.csv"
        ),
        dedup_manifest_path=_variant_path(
            "L1_final_fastqs1-5_sublibrary_Intron_subset.dedup_exact.manifest.json"
        ),
        dedup_output_path=(
            DERIVED_ROOT
            / "introns"
            / "bashor_in_house"
            / "lib1_intron_modal80_fastqs1_5_dedup_exact__learn_ready.tsv"
        ),
        pre_dedup_output_path=(
            DERIVED_ROOT
            / "introns"
            / "bashor_in_house"
            / "lib1_intron_modal80_fastqs1_5_pre_dedup_v0__learn_ready.tsv"
        ),
        length_policy="modal_exact_80",
        selected_length=80,
        padded_seq_len=80,
        padding_mode="none",
        expected_row_count=7848,
        expected_high_barcode_count=1326,
    ),
    "utr3": PartSpec(
        part="3UTR",
        part_slug="utr3",
        sequence_column="ThreePrime",
        dedup_source_path=_variant_path(
            "L1_final_fastqs1-5_sublibrary_ThreePrime_subset.dedup_exact.csv"
        ),
        dedup_manifest_path=_variant_path(
            "L1_final_fastqs1-5_sublibrary_ThreePrime_subset.dedup_exact.manifest.json"
        ),
        dedup_output_path=(
            DERIVED_ROOT
            / "utr3"
            / "bashor_in_house"
            / "lib1_threeprime_modal100_fastqs1_5_dedup_exact__learn_ready.tsv"
        ),
        pre_dedup_output_path=(
            DERIVED_ROOT
            / "utr3"
            / "bashor_in_house"
            / "lib1_threeprime_modal100_fastqs1_5_pre_dedup_v0__learn_ready.tsv"
        ),
        length_policy="modal_exact_100",
        selected_length=100,
        padded_seq_len=100,
        padding_mode="none",
        expected_row_count=6845,
        expected_high_barcode_count=775,
    ),
    "utr5": PartSpec(
        part="5UTR",
        part_slug="utr5",
        sequence_column="FivePrime",
        dedup_source_path=_variant_path(
            "L1_final_fastqs1-5_sublibrary_FivePrime_subset.dedup_exact.csv"
        ),
        dedup_manifest_path=_variant_path(
            "L1_final_fastqs1-5_sublibrary_FivePrime_subset.dedup_exact.manifest.json"
        ),
        dedup_output_path=(
            DERIVED_ROOT
            / "utr5"
            / "bashor_in_house"
            / "lib1_fiveprime_modal50_fastqs1_5_dedup_exact__learn_ready.tsv"
        ),
        pre_dedup_output_path=(
            DERIVED_ROOT
            / "utr5"
            / "bashor_in_house"
            / "lib1_fiveprime_modal50_fastqs1_5_pre_dedup_v0__learn_ready.tsv"
        ),
        length_policy="modal_exact_50",
        selected_length=50,
        padded_seq_len=50,
        padding_mode="none",
        expected_row_count=8331,
        expected_high_barcode_count=1797,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_construct_id(parts_concatenated: object) -> str:
    value = str(parts_concatenated).strip()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text)


def _source_for_generation(spec: PartSpec, generation: str) -> Path:
    if generation == "dedup_exact":
        return spec.dedup_source_path
    manifest = _read_json(spec.dedup_manifest_path)
    source_path = manifest.get("source_variant_file")
    if not source_path:
        raise ValueError(
            f"{spec.dedup_manifest_path} does not record source_variant_file"
        )
    return Path(source_path)


def _output_for_generation(spec: PartSpec, generation: str) -> Path:
    if generation == "dedup_exact":
        return spec.dedup_output_path
    return spec.pre_dedup_output_path


def _validate_source_manifest(
    spec: PartSpec, source_path: Path, generation: str
) -> tuple[dict, str]:
    manifest = _read_json(spec.dedup_manifest_path)
    manifest_sha256 = sha256_file(spec.dedup_manifest_path)
    if generation == "dedup_exact":
        expected_path = manifest.get("output_file")
        expected_sha = manifest.get("output_sha256")
    else:
        expected_path = manifest.get("source_variant_file")
        expected_sha = manifest.get("source_variant_sha256")
    if expected_path and Path(expected_path).resolve() != source_path.resolve():
        raise ValueError(
            f"{spec.part}: source path {source_path} does not match dedup manifest "
            f"record {expected_path}"
        )
    observed_sha = sha256_file(source_path)
    if expected_sha and observed_sha != expected_sha:
        raise ValueError(
            f"{spec.part}: source SHA256 mismatch for {source_path}: "
            f"expected {expected_sha}, observed {observed_sha}"
        )
    return manifest, manifest_sha256


def prepare_frame(
    raw: pd.DataFrame,
    spec: PartSpec,
    *,
    heldout_min_barcodes: int = 8,
) -> tuple[pd.DataFrame, dict]:
    """Apply the shared schema, target, ID, and length contract to one part."""

    required = [
        STABLE_ID_SOURCE_COLUMN,
        spec.sequence_column,
        "number_of_barcodes",
        "DNA_bc_counts_sum",
        "RNA_bc_counts_sum",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns for {spec.part}: {missing}")

    parts = raw[STABLE_ID_SOURCE_COLUMN].astype("string").str.strip()
    sequence = raw[spec.sequence_column].astype("string").str.strip().str.upper()
    barcodes = pd.to_numeric(raw["number_of_barcodes"], errors="coerce")
    dna = pd.to_numeric(raw["DNA_bc_counts_sum"], errors="coerce")
    rna = pd.to_numeric(raw["RNA_bc_counts_sum"], errors="coerce")
    valid_dna = sequence.str.fullmatch(r"[ACGTN]+").fillna(False)
    valid_counts = (
        np.isfinite(barcodes)
        & np.isfinite(dna)
        & np.isfinite(rna)
        & barcodes.ge(0)
        & dna.gt(0)
        & rna.gt(0)
    )
    valid_id = parts.notna() & parts.ne("")
    usable_mask = valid_dna & valid_counts & valid_id

    frame = pd.DataFrame(
        {
            "source_row_id": raw.index.to_numpy(dtype=int),
            STABLE_ID_SOURCE_COLUMN: parts,
            spec.sequence_column: sequence,
            "n_barcodes": barcodes,
            "DNA_bc_counts_sum": dna,
            "RNA_bc_counts_sum": rna,
        }
    ).loc[usable_mask].copy()
    frame["sequence_len"] = frame[spec.sequence_column].str.len().astype(int)
    if spec.selected_length is not None:
        frame = frame.loc[
            frame["sequence_len"].eq(int(spec.selected_length))
        ].copy()

    if frame.empty:
        raise ValueError(f"No usable rows remain for {spec.part}")
    if frame[STABLE_ID_SOURCE_COLUMN].duplicated().any():
        duplicate_count = int(frame[STABLE_ID_SOURCE_COLUMN].duplicated().sum())
        raise ValueError(
            f"{spec.part}: {duplicate_count} duplicate {STABLE_ID_SOURCE_COLUMN} values"
        )

    frame.insert(
        1,
        SPLIT_ID_COLUMN,
        frame[STABLE_ID_SOURCE_COLUMN].map(stable_construct_id),
    )
    if frame[SPLIT_ID_COLUMN].duplicated().any():
        raise ValueError(f"{spec.part}: stable construct-ID collision detected")

    frame["n_barcodes"] = frame["n_barcodes"].astype(np.int64)
    for column in ["DNA_bc_counts_sum", "RNA_bc_counts_sum"]:
        values = frame[column]
        if np.all(np.equal(values.to_numpy(), np.floor(values.to_numpy()))):
            frame[column] = values.astype(np.int64)
    ratio = frame["RNA_bc_counts_sum"] / frame["DNA_bc_counts_sum"]
    frame["RNA_DNA"] = ratio.astype(float)
    frame[TARGET_COLUMN] = np.log2(ratio.astype(float))
    frame["log10_RNA_DNA"] = np.log10(ratio.astype(float))

    output_columns = [
        "source_row_id",
        SPLIT_ID_COLUMN,
        STABLE_ID_SOURCE_COLUMN,
        spec.sequence_column,
        "sequence_len",
        "n_barcodes",
        "DNA_bc_counts_sum",
        "RNA_bc_counts_sum",
        "RNA_DNA",
        TARGET_COLUMN,
        "log10_RNA_DNA",
    ]
    frame = frame[output_columns].reset_index(drop=True)
    diagnostics = {
        "raw_rows": int(len(raw)),
        "output_rows": int(len(frame)),
        "dropped_rows": int(len(raw) - len(frame)),
        "dropped_invalid_dna_rows": int((~valid_dna).sum()),
        "dropped_invalid_count_rows": int((~valid_counts).sum()),
        "dropped_invalid_id_rows": int((~valid_id).sum()),
        "high_barcode_rows": int(
            frame["n_barcodes"].ge(int(heldout_min_barcodes)).sum()
        ),
        "sequence_length_counts": {
            str(int(length)): int(count)
            for length, count in frame["sequence_len"]
            .value_counts()
            .sort_index()
            .items()
        },
    }
    return frame, diagnostics


def prepare_part(
    part_slug: str,
    *,
    generation: str = "dedup_exact",
    input_path: Path | None = None,
    output_path: Path | None = None,
    dedup_manifest_path: Path | None = None,
    heldout_min_barcodes: int = 8,
    validate_expected_counts: bool = True,
) -> dict:
    if generation not in DATA_GENERATION_IDS:
        raise ValueError(f"Unknown generation {generation!r}")
    if part_slug not in PART_SPECS:
        raise ValueError(f"Unknown Lib1 part {part_slug!r}")

    spec = PART_SPECS[part_slug]
    if dedup_manifest_path is not None:
        spec = replace(spec, dedup_manifest_path=Path(dedup_manifest_path))
    canonical_source_path = _source_for_generation(spec, generation)
    source_path = Path(input_path) if input_path is not None else canonical_source_path
    destination = (
        Path(output_path)
        if output_path is not None
        else _output_for_generation(spec, generation)
    )
    source_manifest, source_manifest_sha256 = _validate_source_manifest(
        spec, source_path, generation
    )
    raw = pd.read_csv(source_path)
    frame, diagnostics = prepare_frame(
        raw, spec, heldout_min_barcodes=heldout_min_barcodes
    )

    using_canonical_source = source_path.resolve() == canonical_source_path.resolve()
    if validate_expected_counts and using_canonical_source:
        if diagnostics["output_rows"] != spec.expected_row_count:
            raise ValueError(
                f"{spec.part}: expected {spec.expected_row_count} rows, observed "
                f"{diagnostics['output_rows']}"
            )
        if diagnostics["high_barcode_rows"] != spec.expected_high_barcode_count:
            raise ValueError(
                f"{spec.part}: expected {spec.expected_high_barcode_count} rows at "
                f"n_barcodes >= {heldout_min_barcodes}, observed "
                f"{diagnostics['high_barcode_rows']}"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        destination,
        sep="\t",
        index=False,
        line_terminator="\n",
        float_format="%.17g",
    )
    output_sha256 = sha256_file(destination)
    source_sha256 = sha256_file(source_path)
    data_generation_id = DATA_GENERATION_IDS[generation]
    dedup_policy = DEDUP_POLICY if generation == "dedup_exact" else PRE_DEDUP_POLICY
    metadata = {
        "schema_version": "1.0",
        "dataset_id": f"{data_generation_id}:{part_slug}",
        "data_generation_id": data_generation_id,
        "generation": generation,
        "dedup_policy": dedup_policy,
        "part": spec.part,
        "part_slug": part_slug,
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha256,
        "external_source_path": str(source_path.resolve()),
        "external_source_sha256": source_sha256,
        "dedup_manifest_path": str(spec.dedup_manifest_path.resolve()),
        "dedup_manifest_sha256": source_manifest_sha256,
        "dedup_manifest_declared_output_sha256": source_manifest.get(
            "output_sha256"
        ),
        "output_path": str(destination.resolve()),
        "output_sha256": output_sha256,
        "dataset_sha256": output_sha256,
        "sequence_column": spec.sequence_column,
        "split_id_column": SPLIT_ID_COLUMN,
        "stable_id_source_column": STABLE_ID_SOURCE_COLUMN,
        "stable_id_algorithm": STABLE_ID_ALGORITHM,
        "length_policy": spec.length_policy,
        "selected_length": spec.selected_length,
        "padded_seq_len": spec.padded_seq_len,
        "padding_mode": spec.padding_mode,
        "target_column": TARGET_COLUMN,
        "target_formula": TARGET_FORMULA,
        "target_pseudocount": None,
        "barcode_column": "n_barcodes",
        "heldout_min_barcodes": int(heldout_min_barcodes),
        "row_count": diagnostics["output_rows"],
        "high_barcode_count": diagnostics["high_barcode_rows"],
        **diagnostics,
    }
    metadata_path = destination.with_suffix(".metadata.json")
    _write_json(metadata_path, metadata)
    metadata["metadata_path"] = str(metadata_path.resolve())
    metadata["metadata_sha256"] = sha256_file(metadata_path)
    return metadata


def _flat_dataset_entry(dedup: dict, pre_dedup: dict | None = None) -> dict:
    keys = [
        "part",
        "part_slug",
        "data_generation_id",
        "dedup_policy",
        "source_path",
        "source_sha256",
        "dedup_manifest_path",
        "dedup_manifest_sha256",
        "output_path",
        "output_sha256",
        "dataset_sha256",
        "metadata_path",
        "metadata_sha256",
        "sequence_column",
        "split_id_column",
        "stable_id_source_column",
        "stable_id_algorithm",
        "length_policy",
        "selected_length",
        "padded_seq_len",
        "padding_mode",
        "target_column",
        "target_formula",
        "barcode_column",
        "heldout_min_barcodes",
        "row_count",
        "high_barcode_count",
        "sequence_length_counts",
    ]
    entry = {key: dedup[key] for key in keys}
    entry["dedup"] = dedup
    if pre_dedup is not None:
        entry["pre_dedup_output_path"] = pre_dedup["output_path"]
        entry["pre_dedup_output_sha256"] = pre_dedup["output_sha256"]
        entry["pre_dedup"] = pre_dedup
    return entry


def write_data_manifest(
    dedup_metadata: Mapping[str, dict],
    *,
    pre_dedup_metadata: Mapping[str, dict] | None = None,
    output_path: Path = DEFAULT_DATA_MANIFEST,
) -> dict:
    datasets = {}
    for part_slug in PART_SPECS:
        if part_slug not in dedup_metadata:
            continue
        pre = None
        if pre_dedup_metadata is not None:
            pre = pre_dedup_metadata.get(part_slug)
        datasets[part_slug] = _flat_dataset_entry(dedup_metadata[part_slug], pre)
    payload = {
        "schema_version": "1.0",
        "manifest_id": DATA_GENERATION_IDS["dedup_exact"],
        "data_generation_id": DATA_GENERATION_IDS["dedup_exact"],
        "dedup_policy": DEDUP_POLICY,
        "target": {
            "column": TARGET_COLUMN,
            "formula": TARGET_FORMULA,
            "pseudocount": None,
            "normalize_during_training": True,
        },
        "stable_id": {
            "column": SPLIT_ID_COLUMN,
            "source_column": STABLE_ID_SOURCE_COLUMN,
            "algorithm": STABLE_ID_ALGORITHM,
        },
        "datasets": datasets,
    }
    _write_json(Path(output_path), payload)
    return payload


def prepare_many(
    parts: Iterable[str],
    *,
    include_pre_dedup: bool = False,
    data_manifest_path: Path = DEFAULT_DATA_MANIFEST,
) -> dict:
    dedup_metadata = {
        part_slug: prepare_part(part_slug, generation="dedup_exact")
        for part_slug in parts
    }
    pre_metadata = None
    if include_pre_dedup:
        pre_metadata = {
            part_slug: prepare_part(part_slug, generation="pre_dedup")
            for part_slug in parts
        }
    return write_data_manifest(
        dedup_metadata,
        pre_dedup_metadata=pre_metadata,
        output_path=data_manifest_path,
    )


def _build_parser(fixed_part: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    if fixed_part is None:
        parser.add_argument(
            "--parts",
            nargs="+",
            choices=list(PART_SPECS),
            default=list(PART_SPECS),
        )
    parser.add_argument(
        "--generation",
        choices=list(DATA_GENERATION_IDS),
        default="dedup_exact",
    )
    parser.add_argument(
        "--include-pre-dedup",
        action="store_true",
        help="With the five-part entry point, also write calibration mates.",
    )
    parser.add_argument("--input-path", "--input_path", type=Path, default=None)
    parser.add_argument("--output-path", "--output_path", type=Path, default=None)
    parser.add_argument("--dedup-manifest-path", type=Path, default=None)
    parser.add_argument("--heldout-min-barcodes", type=int, default=8)
    parser.add_argument(
        "--data-manifest-path", type=Path, default=DEFAULT_DATA_MANIFEST
    )
    # Retained for CLI compatibility with the historical per-part wrappers.
    parser.add_argument("--sequence-column", default=None)
    parser.add_argument("--target-column", default="RNA/DNA")
    parser.add_argument("--barcode-column", default="number_of_barcodes")
    parser.add_argument("--length-policy", choices=["modal", "exact", "all"], default=None)
    parser.add_argument("--exact-length", type=int, default=None)
    parser.add_argument("--val-frac-within-hq", type=float, default=None)
    parser.add_argument("--test-frac-within-hq", type=float, default=None)
    parser.add_argument("--val-size-within-hq", type=int, default=None)
    parser.add_argument("--test-size-within-hq", type=int, default=None)
    return parser


def main_for_part(part_slug: str) -> None:
    args = _build_parser(fixed_part=part_slug).parse_args()
    spec = PART_SPECS[part_slug]
    if args.sequence_column not in (None, spec.sequence_column):
        raise ValueError(
            f"Canonical {part_slug} sequence column is {spec.sequence_column!r}, not "
            f"{args.sequence_column!r}"
        )
    if args.target_column != "RNA/DNA" or args.barcode_column != "number_of_barcodes":
        raise ValueError(
            "Canonical preparation requires RNA/DNA aggregate inputs and "
            "number_of_barcodes"
        )
    metadata = prepare_part(
        part_slug,
        generation=args.generation,
        input_path=args.input_path,
        output_path=args.output_path,
        dedup_manifest_path=args.dedup_manifest_path,
        heldout_min_barcodes=args.heldout_min_barcodes,
    )
    print(
        f"Wrote {metadata['row_count']} {metadata['part']} rows to "
        f"{metadata['output_path']} (SHA256 {metadata['output_sha256']})"
    )
    print(
        f"HQ rows at n_barcodes >= {metadata['heldout_min_barcodes']}: "
        f"{metadata['high_barcode_count']}"
    )


def main() -> None:
    args = _build_parser().parse_args()
    if args.input_path is not None or args.output_path is not None:
        if len(args.parts) != 1:
            raise ValueError("Input/output overrides require exactly one --parts value")
        metadata = prepare_part(
            args.parts[0],
            generation=args.generation,
            input_path=args.input_path,
            output_path=args.output_path,
            dedup_manifest_path=args.dedup_manifest_path,
            heldout_min_barcodes=args.heldout_min_barcodes,
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return

    if args.generation == "pre_dedup":
        metadata = {
            slug: prepare_part(slug, generation="pre_dedup")
            for slug in args.parts
        }
        print(
            "Prepared pre-dedup calibration mates: "
            + ", ".join(
                f"{slug}={entry['row_count']}" for slug, entry in metadata.items()
            )
        )
        return

    manifest = prepare_many(
        args.parts,
        include_pre_dedup=args.include_pre_dedup,
        data_manifest_path=args.data_manifest_path,
    )
    print(f"Wrote data manifest to {args.data_manifest_path.resolve()}")
    for slug, entry in manifest["datasets"].items():
        print(
            f"{slug}: rows={entry['row_count']} HQ={entry['high_barcode_count']} "
            f"SHA256={entry['output_sha256']}"
        )


if __name__ == "__main__":
    main()
