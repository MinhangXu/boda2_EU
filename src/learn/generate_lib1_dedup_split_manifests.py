#!/usr/bin/env python3
"""Generate and verify frozen Lib1 audit/development split manifests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


LEARN_DIR = Path(__file__).resolve().parent
REPO_ROOT = LEARN_DIR.parent.parent
DEFAULT_DATA_MANIFEST = (
    LEARN_DIR / "data_manifests" / "lib1_single_part_dedup_exact_v1.json"
)
DEFAULT_OUTPUT_DIR = LEARN_DIR / "data_manifests" / "splits"
DEFAULT_INDEX_PATH = (
    LEARN_DIR / "data_manifests" / "lib1_dedup_exact_v1_split_manifests.json"
)
DEFAULT_SPLIT_SEED = 20260709
DEFAULT_N_FOLDS = 5
SCHEMA_VERSION = "lib1_dedup_split_v1"
ASSIGNMENT_ALGORITHM = "sha256_rank_round_robin_v1"
ID_HASH_ALGORITHM = "sha256_canonical_sorted_json_v1"
STABLE_ID_ALGORITHM = "sha256_utf8(parts_concatenated)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_id_hash(ids: Iterable[object]) -> str:
    values = sorted(str(value) for value in ids)
    return canonical_json_sha256(values)


def assignment_hash(assignments: Iterable[Mapping[str, object]]) -> str:
    membership = sorted(
        (
            {
                "construct_id": str(row["construct_id"]),
                "partition": str(row["partition"]),
                "development_fold": (
                    None
                    if row.get("development_fold") is None
                    else int(row["development_fold"])
                ),
            }
            for row in assignments
        ),
        key=lambda row: row["construct_id"],
    )
    return canonical_json_sha256(membership)


def _rank(seed: int, namespace: str, stable_id: str) -> tuple[str, str]:
    payload = f"{int(seed)}|{namespace}|{stable_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), stable_id


def audit_test_size(n_high_barcode: int) -> int:
    return min(400, max(250, int(round(0.20 * int(n_high_barcode)))))


def _generation_slug(data_generation_id: str) -> str:
    if data_generation_id == "lib1_single_part_dedup_exact_v1":
        return "dedup_exact_v1"
    if data_generation_id == "lib1_single_part_pre_dedup_v0":
        return "pre_dedup_v0"
    return str(data_generation_id)


def _stable_construct_id(value: object) -> str:
    return hashlib.sha256(str(value).strip().encode("utf-8")).hexdigest()


def _validate_input_frame(
    frame: pd.DataFrame,
    *,
    id_column: str,
    sequence_column: str,
    barcode_column: str,
) -> pd.DataFrame:
    required = [id_column, sequence_column, barcode_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing split columns: {missing}")
    out = frame.copy()
    ids = out[id_column].astype("string").str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise ValueError(f"Dataset contains blank {id_column} values")
    if ids.duplicated().any():
        raise ValueError(
            f"Dataset contains {int(ids.duplicated().sum())} duplicate {id_column} values"
        )
    out[id_column] = ids.astype(str)
    sequence = out[sequence_column].astype("string").str.strip().str.upper()
    if sequence.isna().any() or sequence.eq("").any():
        raise ValueError(f"Dataset contains blank {sequence_column} values")
    out[sequence_column] = sequence.astype(str)
    barcode = pd.to_numeric(out[barcode_column], errors="coerce")
    if not np.isfinite(barcode).all():
        raise ValueError(f"Dataset contains non-finite {barcode_column} values")
    out[barcode_column] = barcode.astype(int)
    return out


def _fold_records(
    assignments: list[dict], n_folds: int
) -> tuple[dict[str, dict], dict[str, str]]:
    fold_records: dict[str, dict] = {}
    development_hashes: dict[str, str] = {}
    for fold in range(int(n_folds)):
        val_ids = [
            row["construct_id"]
            for row in assignments
            if row["partition"] == "development"
            and row["development_fold"] == fold
        ]
        train_ids = [
            row["construct_id"]
            for row in assignments
            if row["partition"] == "train_only"
            or (
                row["partition"] == "development"
                and row["development_fold"] != fold
            )
        ]
        development_hashes[str(fold)] = stable_id_hash(val_ids)
        fold_records[str(fold)] = {
            "development_fold": fold,
            "train_pool_count": int(len(train_ids)),
            "validation_count": int(len(val_ids)),
            "train_pool_ids_sha256": stable_id_hash(train_ids),
            "validation_ids_sha256": stable_id_hash(val_ids),
            # Aliases consumed by manifest/replay utilities.
            "train_count": int(len(train_ids)),
            "val_count": int(len(val_ids)),
            "train_ids_sha256": stable_id_hash(train_ids),
            "val_ids_sha256": stable_id_hash(val_ids),
        }
    return fold_records, development_hashes


def build_split_manifest(
    frame: pd.DataFrame,
    *,
    part: str,
    part_slug: str,
    dataset_path: Path,
    dataset_sha256: str,
    data_generation_id: str,
    id_column: str,
    stable_id_source_column: str = "parts_concatenated",
    sequence_column: str,
    barcode_column: str = "n_barcodes",
    target_column: str = "log2_RNA_DNA",
    padded_seq_len: int | None = None,
    padding_mode: str | None = None,
    neutral_pad_char: str = "N",
    normalize: bool = True,
    heldout_min_barcodes: int = 8,
    train_min_barcodes: int = 1,
    split_seed: int = DEFAULT_SPLIT_SEED,
    n_folds: int = DEFAULT_N_FOLDS,
    audit_size: int | None = None,
) -> dict:
    frame = _validate_input_frame(
        frame,
        id_column=id_column,
        sequence_column=sequence_column,
        barcode_column=barcode_column,
    )
    if stable_id_source_column not in frame.columns:
        raise ValueError(
            f"Dataset is missing stable-ID audit column {stable_id_source_column!r}"
        )
    derived_ids = frame[stable_id_source_column].map(_stable_construct_id)
    if not derived_ids.eq(frame[id_column]).all():
        raise ValueError(
            f"{id_column} does not equal {STABLE_ID_ALGORITHM} for every row"
        )
    if int(n_folds) < 2:
        raise ValueError("n_folds must be at least 2")
    high_mask = frame[barcode_column].ge(int(heldout_min_barcodes))
    high_ids = frame.loc[high_mask, id_column].astype(str).tolist()
    if audit_size is None:
        audit_size = audit_test_size(len(high_ids))
    audit_size = int(audit_size)
    if audit_size < 1 or audit_size + int(n_folds) > len(high_ids):
        raise ValueError(
            f"audit_size={audit_size} leaves too few of {len(high_ids)} HQ rows "
            f"for {n_folds} development folds"
        )

    ranked_hq = sorted(
        high_ids, key=lambda stable_id: _rank(split_seed, "audit", stable_id)
    )
    audit_ids = set(ranked_hq[:audit_size])
    development_ids = sorted(
        (stable_id for stable_id in high_ids if stable_id not in audit_ids),
        key=lambda stable_id: _rank(split_seed, "development", stable_id),
    )
    development_folds = {
        stable_id: index % int(n_folds)
        for index, stable_id in enumerate(development_ids)
    }

    assignments = []
    for _, row in frame.iterrows():
        stable_id = str(row[id_column])
        if stable_id in audit_ids:
            partition = "audit_test"
            development_fold = None
        elif stable_id in development_folds:
            partition = "development"
            development_fold = int(development_folds[stable_id])
        else:
            partition = "train_only"
            development_fold = None
        assignments.append(
            {
                "construct_id": stable_id,
                "sequence": str(row[sequence_column]),
                "n_barcodes": int(row[barcode_column]),
                "partition": partition,
                "development_fold": development_fold,
            }
        )
    assignments.sort(key=lambda row: row["construct_id"])

    fold_records, development_hashes = _fold_records(assignments, int(n_folds))
    low_count = int((~high_mask).sum())
    development_count = int(len(development_ids))
    counts = {
        "total": int(len(frame)),
        "high_barcode": int(high_mask.sum()),
        "train_only": low_count,
        "development": development_count,
        "audit_test": audit_size,
        "development_folds": {
            str(fold): int(fold_records[str(fold)]["validation_count"])
            for fold in range(int(n_folds))
        },
    }
    expected = {
        "id_hash_algorithm": ID_HASH_ALGORITHM,
        "counts": counts,
        "all_ids_sha256": stable_id_hash(frame[id_column]),
        "assignment_sha256": assignment_hash(assignments),
        "audit_ids_sha256": stable_id_hash(audit_ids),
        "development_ids_sha256": stable_id_hash(development_ids),
        "train_only_ids_sha256": stable_id_hash(
            row["construct_id"]
            for row in assignments
            if row["partition"] == "train_only"
        ),
        "development_fold_ids_sha256": development_hashes,
        "per_fold": fold_records,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_id": (
            f"lib1_{part_slug}_{_generation_slug(data_generation_id)}_split_seed{int(split_seed)}"
        ),
        "data_generation_id": data_generation_id,
        "part": part,
        "part_slug": part_slug,
        "split_seed": int(split_seed),
        "n_development_folds": int(n_folds),
        "dataset": {
            "data_generation_id": data_generation_id,
            "path": str(Path(dataset_path).resolve()),
            "path_hint": str(Path(dataset_path).resolve()),
            "sha256": str(dataset_sha256),
            "row_count": int(len(frame)),
            "id_column": id_column,
            "stable_id_source_column": stable_id_source_column,
            "stable_id_algorithm": STABLE_ID_ALGORITHM,
            "sequence_column": sequence_column,
            "barcode_column": barcode_column,
            "target_column": target_column,
            "padded_seq_len": (
                None if padded_seq_len is None else int(padded_seq_len)
            ),
            "padding_mode": padding_mode,
            "neutral_pad_char": neutral_pad_char,
            "normalize": bool(normalize),
            "high_barcode_threshold": int(heldout_min_barcodes),
        },
        "policy": {
            "train_min_barcodes": int(train_min_barcodes),
            "heldout_min_barcodes": int(heldout_min_barcodes),
            "n_development_folds": int(n_folds),
            "audit_test_n": audit_size,
            "assignment_seed": int(split_seed),
            "algorithm": ASSIGNMENT_ALGORITHM,
            "assignment_reference_data_generation_id": data_generation_id,
        },
        "expected_counts": counts,
        "expected": expected,
        "folds": fold_records,
        "assignments": assignments,
    }


def rebind_split_manifest(
    base_manifest: dict,
    frame: pd.DataFrame,
    *,
    dataset_path: Path,
    dataset_sha256: str,
    data_generation_id: str,
) -> dict:
    """Bind frozen memberships to a calibration dataset with identical IDs."""

    dataset = base_manifest["dataset"]
    frame = _validate_input_frame(
        frame,
        id_column=dataset["id_column"],
        sequence_column=dataset["sequence_column"],
        barcode_column=dataset["barcode_column"],
    )
    stable_id_source_column = dataset.get("stable_id_source_column")
    if stable_id_source_column:
        if stable_id_source_column not in frame.columns:
            raise ValueError(
                f"Calibration dataset is missing {stable_id_source_column!r}"
            )
        derived_ids = frame[stable_id_source_column].map(_stable_construct_id)
        if not derived_ids.eq(frame[dataset["id_column"]]).all():
            raise ValueError(
                "Calibration construct IDs do not match parts_concatenated audit values"
            )
    rows = frame.set_index(dataset["id_column"], drop=False)
    base_ids = {row["construct_id"] for row in base_manifest["assignments"]}
    observed_ids = set(rows.index.astype(str))
    if base_ids != observed_ids:
        raise ValueError(
            "Calibration dataset stable-ID coverage differs from the frozen dedup split: "
            f"missing={len(base_ids - observed_ids)}, extra={len(observed_ids - base_ids)}"
        )

    rebound = copy.deepcopy(base_manifest)
    rebound["manifest_id"] = (
        f"lib1_{base_manifest['part_slug']}_{_generation_slug(data_generation_id)}_split_seed"
        f"{base_manifest['split_seed']}"
    )
    rebound["data_generation_id"] = data_generation_id
    rebound["dataset"]["data_generation_id"] = data_generation_id
    rebound["dataset"]["path"] = str(Path(dataset_path).resolve())
    rebound["dataset"]["path_hint"] = str(Path(dataset_path).resolve())
    rebound["dataset"]["sha256"] = str(dataset_sha256)
    rebound["dataset"]["row_count"] = int(len(frame))
    rebound["policy"]["assignment_reference_data_generation_id"] = (
        base_manifest["data_generation_id"]
    )
    for assignment in rebound["assignments"]:
        row = rows.loc[assignment["construct_id"]]
        sequence = str(row[dataset["sequence_column"]]).upper()
        if sequence != assignment["sequence"]:
            raise ValueError(
                f"Calibration sequence mismatch for {assignment['construct_id']}"
            )
        assignment["sequence"] = sequence
        assignment["n_barcodes"] = int(row[dataset["barcode_column"]])
    rebound["expected"]["all_ids_sha256"] = stable_id_hash(observed_ids)
    # This hash deliberately excludes barcode audit fields, so it must remain
    # identical across the dedup/pre-dedup calibration pair.
    rebound["expected"]["assignment_sha256"] = assignment_hash(
        rebound["assignments"]
    )
    if (
        rebound["expected"]["assignment_sha256"]
        != base_manifest["expected"]["assignment_sha256"]
    ):
        raise ValueError("Calibration rebinding changed frozen split membership")
    return rebound


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )


def _load_json(path: Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def _assert_dataset_hash(path: Path, expected_sha256: str) -> None:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(
            f"Dataset SHA256 mismatch for {path}: expected {expected_sha256}, "
            f"observed {observed}"
        )


def generate_all(
    *,
    data_manifest_path: Path = DEFAULT_DATA_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    index_path: Path = DEFAULT_INDEX_PATH,
    parts: Iterable[str] | None = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
    n_folds: int = DEFAULT_N_FOLDS,
    include_pre_dedup: bool = True,
    verify_only: bool = False,
) -> dict:
    data_manifest = _load_json(data_manifest_path)
    datasets = data_manifest.get("datasets", {})
    selected_parts = list(parts) if parts is not None else list(datasets)
    index_entries = {}
    for part_slug in selected_parts:
        entry = datasets[part_slug]
        dataset_path = Path(entry["output_path"])
        _assert_dataset_hash(dataset_path, entry["output_sha256"])
        frame = pd.read_csv(dataset_path, sep="\t")
        payload = build_split_manifest(
            frame,
            part=entry["part"],
            part_slug=part_slug,
            dataset_path=dataset_path,
            dataset_sha256=entry["output_sha256"],
            data_generation_id=entry["data_generation_id"],
            id_column=entry["split_id_column"],
            stable_id_source_column=entry["stable_id_source_column"],
            sequence_column=entry["sequence_column"],
            barcode_column=entry["barcode_column"],
            target_column=entry["target_column"],
            padded_seq_len=entry["padded_seq_len"],
            padding_mode=entry["padding_mode"],
            neutral_pad_char="N",
            normalize=True,
            heldout_min_barcodes=entry["heldout_min_barcodes"],
            split_seed=split_seed,
            n_folds=n_folds,
        )
        manifest_path = output_dir / f"lib1_{part_slug}_dedup_exact_v1_split.json"
        if verify_only:
            if _load_json(manifest_path) != payload:
                raise ValueError(f"Regenerated payload differs from {manifest_path}")
        else:
            _write_json(manifest_path, payload)
        manifest_sha256 = sha256_file(manifest_path)
        index_entry = {
            "part": entry["part"],
            "part_slug": part_slug,
            "manifest_id": payload["manifest_id"],
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": manifest_sha256,
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": entry["output_sha256"],
            "split_id_column": entry["split_id_column"],
            "assignment_sha256": payload["expected"]["assignment_sha256"],
            "audit_count": payload["expected_counts"]["audit_test"],
            "development_fold_counts": payload["expected_counts"][
                "development_folds"
            ],
        }

        pre_path_value = entry.get("pre_dedup_output_path")
        pre_sha = entry.get("pre_dedup_output_sha256")
        if include_pre_dedup and pre_path_value and pre_sha:
            pre_path = Path(pre_path_value)
            _assert_dataset_hash(pre_path, pre_sha)
            pre_frame = pd.read_csv(pre_path, sep="\t")
            pre_generation_id = entry.get("pre_dedup", {}).get(
                "data_generation_id", "lib1_single_part_pre_dedup_v0"
            )
            pre_payload = rebind_split_manifest(
                payload,
                pre_frame,
                dataset_path=pre_path,
                dataset_sha256=pre_sha,
                data_generation_id=pre_generation_id,
            )
            pre_manifest_path = (
                output_dir / f"lib1_{part_slug}_pre_dedup_v0_split.json"
            )
            if verify_only:
                if _load_json(pre_manifest_path) != pre_payload:
                    raise ValueError(
                        f"Regenerated payload differs from {pre_manifest_path}"
                    )
            else:
                _write_json(pre_manifest_path, pre_payload)
            index_entry.update(
                {
                    "pre_dedup_manifest_id": pre_payload["manifest_id"],
                    "pre_dedup_manifest_path": str(pre_manifest_path.resolve()),
                    "pre_dedup_manifest_sha256": sha256_file(pre_manifest_path),
                    "pre_dedup_dataset_path": str(pre_path.resolve()),
                    "pre_dedup_dataset_sha256": pre_sha,
                }
            )
            if (
                pre_payload["expected"]["assignment_sha256"]
                != payload["expected"]["assignment_sha256"]
            ):
                raise ValueError(
                    f"{part_slug}: pre-dedup and dedup assignments are not paired"
                )
        index_entries[part_slug] = index_entry

    index = {
        "schema_version": "lib1_dedup_split_index_v1",
        "manifest_id": "lib1_dedup_exact_v1_split_manifests",
        "data_manifest_path": str(Path(data_manifest_path).resolve()),
        "data_manifest_sha256": sha256_file(data_manifest_path),
        "split_seed": int(split_seed),
        "n_development_folds": int(n_folds),
        "parts": index_entries,
    }
    if verify_only:
        if _load_json(index_path) != index:
            raise ValueError(f"Regenerated split index differs from {index_path}")
    else:
        _write_json(index_path, index)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest-path", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--parts", nargs="+", default=None)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    pre_group = parser.add_mutually_exclusive_group()
    pre_group.add_argument(
        "--include-pre-dedup",
        dest="include_pre_dedup",
        action="store_true",
        default=True,
    )
    pre_group.add_argument(
        "--no-include-pre-dedup",
        dest="include_pre_dedup",
        action="store_false",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Regenerate in memory and require byte-addressed outputs to match.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = generate_all(
        data_manifest_path=args.data_manifest_path,
        output_dir=args.output_dir,
        index_path=args.index_path,
        parts=args.parts,
        split_seed=args.split_seed,
        n_folds=args.n_folds,
        include_pre_dedup=args.include_pre_dedup,
        verify_only=args.verify_only,
    )
    action = "Verified" if args.verify_only else "Wrote"
    print(f"{action} split index: {args.index_path.resolve()}")
    for part_slug, entry in index["parts"].items():
        print(
            f"{part_slug}: audit={entry['audit_count']} "
            f"folds={entry['development_fold_counts']} "
            f"SHA256={entry['manifest_sha256']}"
        )


if __name__ == "__main__":
    main()
