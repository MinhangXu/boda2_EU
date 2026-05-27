#!/usr/bin/env python3
"""Combine per-job Hani 5'UTR Lib2 fine-tuning outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_FILES = [
    "model_comparison_summary.csv",
    "per_head_metrics.csv",
    "inhouse_fiveprime_metrics.csv",
    "per_epoch_diagnostics.csv",
    "lib2_test_model_ranking.csv",
    "lib2_validation_model_ranking.csv",
    "lib2_final_test_model_ranking.csv",
]


def read_frames(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine Hani UTR5 Lib2 fine-tuning CSVs.")
    parser.add_argument("root", type=Path, help="Sweep output root containing per_job/ or legacy per_seed/ directories.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional legacy or v2 training seed filter.")
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def discover_job_dirs(root: Path, seeds: list[int] | None) -> list[Path]:
    per_job_root = root / "per_job"
    if per_job_root.exists():
        job_dirs = sorted(path for path in per_job_root.glob("*/*/training_seed_*") if path.is_dir())
        if seeds:
            wanted = {f"training_seed_{seed}" for seed in seeds}
            job_dirs = [path for path in job_dirs if path.name in wanted]
        return job_dirs

    per_seed_root = root / "per_seed"
    if seeds:
        return [per_seed_root / f"seed_{seed}" for seed in seeds]
    return sorted(path for path in per_seed_root.glob("seed_*") if path.is_dir())


def sort_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [
        col
        for col in ["average_activity_pearson", "mean_per_head_pearson"]
        if col in frame.columns
    ]
    if not sort_cols:
        return frame
    return frame.sort_values(sort_cols, ascending=[False] * len(sort_cols))


def copy_split_artifacts(root: Path, outdir: Path) -> None:
    split_dir = root / "split_manifests"
    if split_dir.exists():
        for file_name in [
            "outer_final_test_manifest.csv",
            "split_policy.json",
            "split_audit.csv",
        ]:
            source = split_dir / file_name
            if source.exists():
                target = outdir / file_name
                target.write_bytes(source.read_bytes())
                print(f"Wrote {target}")
        for source in sorted(split_dir.glob("inner_split_manifest_*.csv")):
            target = outdir / source.name
            target.write_bytes(source.read_bytes())
            print(f"Wrote {target}")
        return

    manifest_paths = sorted((root / "per_seed").glob("seed_*/lib2_sequence_split_manifest.csv"))
    if manifest_paths:
        split_manifest = pd.read_csv(manifest_paths[0])
        split_manifest.to_csv(outdir / "lib2_sequence_split_manifest.csv", index=False)
        print(f"Wrote {outdir / 'lib2_sequence_split_manifest.csv'}")


def main() -> None:
    args = parse_args()
    job_dirs = discover_job_dirs(args.root, args.seeds)
    outdir = args.outdir or (args.root / "combined")
    outdir.mkdir(parents=True, exist_ok=True)

    for file_name in OUTPUT_FILES:
        paths = [job_dir / file_name for job_dir in job_dirs]
        frame = read_frames(paths)
        if frame.empty:
            continue
        if "ranking" in file_name:
            frame = sort_ranking(frame)
        frame.to_csv(outdir / file_name, index=False)
        print(f"Wrote {outdir / file_name}")

    copy_split_artifacts(args.root, outdir)


if __name__ == "__main__":
    main()
