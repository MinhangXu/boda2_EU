#!/usr/bin/env python3
"""Prepare Rosenberg/Seelig 2015 processed splicing data for BODA."""

import argparse
import gzip
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse


DEFAULT_OUTPUT_DIR = Path(
    "/home/minhang/synBio_AL/opt_EU_learn_n_design/introns/seelig_2015"
)
DEFAULT_SOURCE_BASE = (
    "https://raw.githubusercontent.com/Alex-Rosenberg/cell-2015/master/data_gz"
)

SOURCE_FILES = {
    "A5SS_Seqs.csv.gz": f"{DEFAULT_SOURCE_BASE}/A5SS_Seqs.csv.gz",
    "A3SS_Seqs.csv.gz": f"{DEFAULT_SOURCE_BASE}/A3SS_Seqs.csv.gz",
    "Reads.mat.gz": f"{DEFAULT_SOURCE_BASE}/Reads.mat.gz",
}


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Download/read the processed Cell 2015 splicing data and materialize "
            "BODA-ready scalar target tables."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing A5SS_Seqs.csv.gz, A3SS_Seqs.csv.gz, "
            "and Reads.mat or Reads.mat.gz. If omitted, missing files are "
            "downloaded into <output_dir>/raw."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for prepared CSV/NPZ files and manifest.",
    )
    parser.add_argument(
        "--libraries",
        nargs="+",
        choices=["A5SS", "A3SS"],
        default=["A5SS"],
        help="Libraries to prepare. A5SS is the Stage-1 pretraining target.",
    )
    parser.add_argument("--min_reads", type=int, default=1)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--write_a5ss_distribution",
        action="store_true",
        help="Also write the paper-comparable A5SS 81-output donor distribution NPZ.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Redownload source files when using the default raw directory.",
    )
    return parser


def download_file(url, output_path, force=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        return output_path

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(output_path)
    return output_path


def resolve_source_files(input_dir, output_dir, libraries, force_download=False):
    wanted = ["Reads.mat.gz"]
    if "A5SS" in libraries:
        wanted.append("A5SS_Seqs.csv.gz")
    if "A3SS" in libraries:
        wanted.append("A3SS_Seqs.csv.gz")

    if input_dir is None:
        source_dir = output_dir / "raw"
        for filename in wanted:
            url = SOURCE_FILES[filename]
            download_file(url, source_dir / filename, force=force_download)
    else:
        source_dir = input_dir

    reads_mat = source_dir / "Reads.mat"
    reads_gz = source_dir / "Reads.mat.gz"
    if not reads_mat.exists():
        if not reads_gz.exists():
            raise FileNotFoundError(f"Missing {reads_mat} or {reads_gz}")
        with gzip.open(reads_gz, "rb") as source, reads_mat.open("wb") as dest:
            shutil.copyfileobj(source, dest)

    required = {"Reads.mat": reads_mat}
    if "A5SS" in libraries:
        required["A5SS_Seqs.csv.gz"] = source_dir / "A5SS_Seqs.csv.gz"
    if "A3SS" in libraries:
        required["A3SS_Seqs.csv.gz"] = source_dir / "A3SS_Seqs.csv.gz"
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required source files: {missing}")
    return required


def assign_folds(n_rows, val_frac, test_frac, seed):
    if n_rows <= 0:
        return np.array([], dtype=object)
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("Require val_frac >= 0, test_frac >= 0, and val_frac + test_frac < 1.")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_rows)
    n_test = int(round(n_rows * test_frac))
    n_val = int(round(n_rows * val_frac))
    if test_frac > 0:
        n_test = max(1, n_test)
    if val_frac > 0:
        n_val = max(1, n_val)
    if n_test + n_val >= n_rows:
        raise ValueError(f"Not enough rows ({n_rows}) for requested val/test fractions.")

    folds = np.full(n_rows, "train", dtype=object)
    folds[order[:n_test]] = "test"
    folds[order[n_test:n_test + n_val]] = "val"
    return folds


def _as_csr(matrix, library):
    if not scipy.sparse.issparse(matrix):
        raise TypeError(f"Expected scipy sparse matrix for {library}; got {type(matrix)!r}")
    return matrix.tocsr()


def _column_fraction(counts, numerator_col, denominator):
    numerator = np.asarray(counts[:, numerator_col].toarray()).ravel().astype(np.float64)
    out = np.full_like(denominator, np.nan, dtype=np.float64)
    valid = denominator > 0
    out[valid] = numerator[valid] / denominator[valid]
    return out


def prepare_a5ss(seq_path, counts, output_dir, min_reads, val_frac, test_frac, seed,
                 write_distribution=False):
    seqs = pd.read_csv(seq_path)
    counts = _as_csr(counts, "A5SS")
    if counts.shape[0] != len(seqs):
        raise ValueError(f"A5SS sequence/count row mismatch: {len(seqs)} vs {counts.shape[0]}")

    read_totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    valid = read_totals >= min_reads
    source_index = np.flatnonzero(valid)

    p_sd1 = _column_fraction(counts, 0, read_totals)
    sd1_counts = np.asarray(counts[:, 0].toarray()).ravel().astype(np.float64)
    sd2_counts = np.asarray(counts[:, 44].toarray()).ravel().astype(np.float64)
    sd12_counts = sd1_counts + sd2_counts
    p_sd2_conditional = np.full_like(read_totals, np.nan, dtype=np.float64)
    sd12_valid = sd12_counts > 0
    p_sd2_conditional[sd12_valid] = sd2_counts[sd12_valid] / sd12_counts[sd12_valid]

    out = pd.DataFrame(
        {
            "library": "A5SS",
            "source_index": source_index,
            "tag": seqs.loc[valid, "Tag"].to_numpy(),
            "seq": seqs.loc[valid, "Seq"].str.upper().to_numpy(),
            "seq_len": seqs.loc[valid, "Seq"].str.len().to_numpy(),
            "read_count_total": read_totals[valid].astype(np.int64),
            "p_sd1": p_sd1[valid],
            "p_sd2_conditional": p_sd2_conditional[valid],
        }
    )
    out["fold"] = assign_folds(len(out), val_frac, test_frac, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "A5SS_scalar_targets.csv"
    out.to_csv(output_path, index=False)

    distribution_path = None
    if write_distribution:
        selected = counts[valid]
        dist = scipy.sparse.hstack([selected[:, :80], selected[:, -1:]], format="csr")
        dist = dist.toarray().astype(np.float32)
        denom = read_totals[valid].astype(np.float32)
        dist = dist / denom[:, None]
        distribution_path = output_dir / "A5SS_donor_distribution_targets.npz"
        np.savez_compressed(
            distribution_path,
            source_index=source_index.astype(np.int64),
            donor_distribution=dist,
        )

    return output_path, distribution_path, {
        "library": "A5SS",
        "n_rows": int(len(out)),
        "source_rows": int(len(seqs)),
        "min_reads": int(min_reads),
        "target_columns": ["p_sd1", "p_sd2_conditional"],
        "fold_counts": {k: int(v) for k, v in out["fold"].value_counts().sort_index().items()},
        "output_path": str(output_path),
        "distribution_path": str(distribution_path) if distribution_path else "",
    }


def prepare_a3ss(seq_path, counts, output_dir, min_reads, val_frac, test_frac, seed):
    seqs = pd.read_csv(seq_path)
    counts = _as_csr(counts, "A3SS")
    if counts.shape[0] != len(seqs):
        raise ValueError(f"A3SS sequence/count row mismatch: {len(seqs)} vs {counts.shape[0]}")

    read_totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
    valid = read_totals >= min_reads
    source_index = np.flatnonzero(valid)
    p_sa1 = _column_fraction(counts, 235, read_totals)

    out = pd.DataFrame(
        {
            "library": "A3SS",
            "source_index": source_index,
            "tag": seqs.loc[valid, "Tag"].to_numpy(),
            "seq": seqs.loc[valid, "Seq"].str.upper().to_numpy(),
            "seq_len": seqs.loc[valid, "Seq"].str.len().to_numpy(),
            "read_count_total": read_totals[valid].astype(np.int64),
            "p_sa1": p_sa1[valid],
        }
    )
    out["fold"] = assign_folds(len(out), val_frac, test_frac, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "A3SS_scalar_targets.csv"
    out.to_csv(output_path, index=False)

    return output_path, {
        "library": "A3SS",
        "n_rows": int(len(out)),
        "source_rows": int(len(seqs)),
        "min_reads": int(min_reads),
        "target_columns": ["p_sa1"],
        "fold_counts": {k: int(v) for k, v in out["fold"].value_counts().sort_index().items()},
        "output_path": str(output_path),
    }


def main():
    args = build_argparser().parse_args()
    source_files = resolve_source_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        libraries=args.libraries,
        force_download=args.force_download,
    )
    reads = sio.loadmat(source_files["Reads.mat"])

    manifests = []
    combined_frames = []

    if "A5SS" in args.libraries:
        output_path, distribution_path, manifest = prepare_a5ss(
            seq_path=source_files["A5SS_Seqs.csv.gz"],
            counts=reads["A5SS"],
            output_dir=args.output_dir,
            min_reads=args.min_reads,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.split_seed,
            write_distribution=args.write_a5ss_distribution,
        )
        manifests.append(manifest)
        combined_frames.append(pd.read_csv(output_path))
        print(f"Wrote {output_path}")
        if distribution_path:
            print(f"Wrote {distribution_path}")

    if "A3SS" in args.libraries:
        output_path, manifest = prepare_a3ss(
            seq_path=source_files["A3SS_Seqs.csv.gz"],
            counts=reads["A3SS"],
            output_dir=args.output_dir,
            min_reads=args.min_reads,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.split_seed,
        )
        manifests.append(manifest)
        combined_frames.append(pd.read_csv(output_path))
        print(f"Wrote {output_path}")

    if combined_frames:
        combined_path = args.output_dir / "seelig_splicing_scalar_targets.csv"
        pd.concat(combined_frames, axis=0, ignore_index=True, sort=False).to_csv(
            combined_path, index=False
        )
        print(f"Wrote {combined_path}")
    else:
        combined_path = None

    manifest = {
        "source": "Rosenberg/Patwardhan/Shendure/Seelig Cell 2015 processed data",
        "source_base_url": DEFAULT_SOURCE_BASE,
        "input_dir": str(args.input_dir) if args.input_dir else str(args.output_dir / "raw"),
        "output_dir": str(args.output_dir),
        "combined_scalar_path": str(combined_path) if combined_path else "",
        "libraries": manifests,
        "split_seed": int(args.split_seed),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
