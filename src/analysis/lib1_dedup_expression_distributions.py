#!/usr/bin/env python3
"""Plot post-dedup Lib1 expression targets for the five single-part libraries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT / "src/learn/data_manifests/lib1_single_part_dedup_exact_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "src/learn/outputs/analysis/lib1_dedup_data_summary_july2026/reporting"
)

PARTS = (
    ("enhancer", "Enhancer", "#2F66D0"),
    ("promoter", "Promoter", "#137F77"),
    ("utr5", "5\N{PRIME}UTR", "#C95B08"),
    ("intron", "Intron", "#7437E8"),
    ("utr3", "3\N{PRIME}UTR", "#2E8B57"),
)
TARGET_COLUMN = "log2_RNA_DNA"
BARCODE_COLUMN = "n_barcodes"


def load_datasets(manifest_path: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("data_generation_id") != "lib1_single_part_dedup_exact_v1":
        raise ValueError("Expected the canonical post-dedup exact-v1 manifest.")

    frames: dict[str, pd.DataFrame] = {}
    for slug, _, _ in PARTS:
        spec = manifest["datasets"][slug]
        path = Path(spec["output_path"])
        frame = pd.read_csv(path, sep="\t")
        missing = {TARGET_COLUMN, BARCODE_COLUMN} - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if len(frame) != int(spec["row_count"]):
            raise ValueError(
                f"{slug} row-count mismatch: observed {len(frame)}, "
                f"manifest declares {spec['row_count']}"
            )
        frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="raise")
        frame[BARCODE_COLUMN] = pd.to_numeric(frame[BARCODE_COLUMN], errors="raise")
        if not np.isfinite(frame[TARGET_COLUMN]).all():
            raise ValueError(f"{slug} contains non-finite target values.")
        frames[slug] = frame
    return frames, manifest


def summarize(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for slug, label, _ in PARTS:
        frame = frames[slug]
        for subset, selected in (
            ("all_modeled", frame),
            ("n_barcodes_ge_8", frame.loc[frame[BARCODE_COLUMN] >= 8]),
        ):
            values = selected[TARGET_COLUMN]
            quantiles = values.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
            records.append(
                {
                    "part_slug": slug,
                    "part": label,
                    "subset": subset,
                    "n": len(values),
                    "min": values.min(),
                    "q05": quantiles.loc[0.05],
                    "q10": quantiles.loc[0.10],
                    "q25": quantiles.loc[0.25],
                    "median": quantiles.loc[0.50],
                    "q75": quantiles.loc[0.75],
                    "q90": quantiles.loc[0.90],
                    "q95": quantiles.loc[0.95],
                    "max": values.max(),
                    "mean": values.mean(),
                    "sd": values.std(),
                }
            )
    return pd.DataFrame.from_records(records)


def draw_violin_panel(
    ax: plt.Axes,
    frames: dict[str, pd.DataFrame],
    *,
    minimum_barcodes: int | None,
) -> None:
    values: list[np.ndarray] = []
    labels: list[str] = []
    colors: list[str] = []

    for slug, label, color in PARTS:
        frame = frames[slug]
        if minimum_barcodes is not None:
            frame = frame.loc[frame[BARCODE_COLUMN] >= minimum_barcodes]
        target = frame[TARGET_COLUMN].to_numpy(dtype=float)
        values.append(target)
        labels.append(f"{label}\n(n={len(target):,})")
        colors.append(color)

    violin = ax.violinplot(
        values,
        positions=np.arange(len(values)),
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        points=300,
        bw_method="scott",
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#24364B")
        body.set_linewidth(0.9)
        body.set_alpha(0.82)

    for position, target in enumerate(values):
        q25, median, q75 = np.quantile(target, [0.25, 0.50, 0.75])
        ax.vlines(position, q25, q75, color="#132A44", linewidth=4.5, zorder=4)
        ax.hlines(
            median,
            position - 0.16,
            position + 0.16,
            color="white",
            linewidth=3.0,
            zorder=5,
        )
        ax.hlines(
            median,
            position - 0.16,
            position + 0.16,
            color="#132A44",
            linewidth=1.2,
            zorder=6,
        )

    ax.set_xticks(np.arange(len(labels)), labels)
    ax.tick_params(axis="x", labelrotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", color="#DCE4EE", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#64758B")
    ax.spines["bottom"].set_color("#64758B")


def make_figure(frames: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "axes.titlesize": 14,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2), sharey=True)

    draw_violin_panel(axes[0], frames, minimum_barcodes=None)
    axes[0].set_title("All modeled constructs", pad=13)
    axes[0].set_ylabel(
        "Construct expression target\nlog\N{SUBSCRIPT TWO}(total RNA / total DNA)",
        fontsize=12,
        fontweight="semibold",
    )

    draw_violin_panel(axes[1], frames, minimum_barcodes=8)
    axes[1].set_title(
        "Higher-support constructs (\N{GREATER-THAN OR EQUAL TO}8 barcode identities)",
        pad=13,
    )
    axes[1].tick_params(axis="y", labelleft=True)

    all_values = np.concatenate(
        [frames[slug][TARGET_COLUMN].to_numpy(dtype=float) for slug, _, _ in PARTS]
    )
    padding = 0.04 * (all_values.max() - all_values.min())
    axes[0].set_ylim(all_values.min() - padding, all_values.max() + padding)

    fig.suptitle(
        "Post-dedup Lib1 single-part expression distributions",
        fontsize=18,
        fontweight="bold",
        color="#132A44",
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        "Raw construct-level target; no pseudocount or display standardization. "
        "Thick line = interquartile range; horizontal line = median.",
        ha="center",
        va="bottom",
        fontsize=10.5,
        color="#4A5E73",
    )
    fig.tight_layout(rect=(0.025, 0.06, 0.995, 0.92), w_pad=3.2)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "lib1_dedup_expression_target_distributions"
    paths = [stem.with_suffix(".png"), stem.with_suffix(".svg")]
    fig.savefig(paths[0], dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames, manifest = load_datasets(args.manifest)
    summary = summarize(frames)
    figure_paths = make_figure(frames, args.output_dir)
    summary_path = args.output_dir / "lib1_dedup_expression_target_summary.csv"
    summary.to_csv(summary_path, index=False)

    provenance = {
        "data_generation_id": manifest["data_generation_id"],
        "manifest": str(args.manifest.resolve()),
        "target": "log2(total RNA barcode counts / total DNA barcode counts)",
        "target_pseudocount": None,
        "display_standardization": None,
        "panels": ["all modeled constructs", "n_barcodes >= 8"],
        "outputs": [str(path.resolve()) for path in [*figure_paths, summary_path]],
    }
    provenance_path = args.output_dir / "lib1_dedup_expression_target_figure.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    for path in [*figure_paths, summary_path, provenance_path]:
        print(path)


if __name__ == "__main__":
    main()
