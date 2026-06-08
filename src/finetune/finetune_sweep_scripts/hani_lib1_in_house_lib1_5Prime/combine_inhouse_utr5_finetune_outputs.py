#!/usr/bin/env python3
"""Combine per-job in-house 5'UTR fine-tuning outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CSV_NAMES = [
    "model_comparison_summary.csv",
    "per_epoch_diagnostics.csv",
    "planned_finetune_specs.csv",
    "split_membership_summary.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Sweep output root containing per_job/ outputs.")
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def read_parts(root: Path, csv_name: str) -> pd.DataFrame:
    parts = []
    for path in sorted((root / "per_job").glob(f"**/{csv_name}")):
        part = pd.read_csv(path)
        part["source_job_dir"] = str(path.parent.relative_to(root))
        parts.append(part)
    direct = root / csv_name
    if direct.exists():
        part = pd.read_csv(direct)
        part["source_job_dir"] = "."
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def write_rankings(outdir: Path, summary: pd.DataFrame) -> None:
    if summary.empty or "split" not in summary.columns:
        return
    val = summary.loc[
        summary["split"].eq("val") & summary["model_label"].astype(str).str.startswith("finetuned__")
    ].copy()
    if val.empty:
        return
    sort_cols = [col for col in ["spearman", "pearson", "cod_r2"] if col in val.columns]
    if sort_cols:
        val = val.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    val.to_csv(outdir / "validation_model_ranking.csv", index=False)
    print(f"Wrote {outdir / 'validation_model_ranking.csv'}")

    test = summary.loc[
        summary["split"].eq("test") & summary["model_label"].astype(str).str.startswith("finetuned__")
    ].copy()
    if not test.empty and sort_cols:
        test = test.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        test.to_csv(outdir / "diagnostic_test_model_ranking.csv", index=False)
        print(f"Wrote {outdir / 'diagnostic_test_model_ranking.csv'}")


def copy_first(root: Path, outdir: Path, name: str) -> None:
    candidates = [root / name, *sorted((root / "per_job").glob(f"**/{name}"))]
    for source in candidates:
        if source.exists():
            target = outdir / name
            target.write_bytes(source.read_bytes())
            print(f"Wrote {target}")
            return


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    outdir = args.outdir or (root / "combined")
    outdir.mkdir(parents=True, exist_ok=True)

    combined: dict[str, pd.DataFrame] = {}
    for csv_name in CSV_NAMES:
        frame = read_parts(root, csv_name)
        if frame.empty:
            continue
        target = outdir / csv_name
        frame.to_csv(target, index=False)
        combined[csv_name] = frame
        print(f"Wrote {target} ({len(frame)} rows)")

    summary = combined.get("model_comparison_summary.csv", pd.DataFrame())
    write_rankings(outdir, summary)
    copy_first(root, outdir, "split_membership_rows.csv")
    copy_first(root, outdir, "run_manifest.json")
    copy_first(root, outdir, "data_audit.json")


if __name__ == "__main__":
    main()
