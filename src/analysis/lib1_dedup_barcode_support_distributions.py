#!/usr/bin/env python3
"""Plot post-dedup barcode-support distributions for Lib1 single-part data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


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
BARCODE_COLUMN = "n_barcodes"
DISPLAY_MAX_BARCODES = 64
SUPPORT_BINS = (
    ("1", lambda x: x == 1, "#EDF2F7"),
    ("2-3", lambda x: (x >= 2) & (x <= 3), "#CCD9E7"),
    ("4-7", lambda x: (x >= 4) & (x <= 7), "#87B2D5"),
    ("\N{GREATER-THAN OR EQUAL TO}8", lambda x: x >= 8, "#235A9F"),
)


def load_datasets(manifest_path: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("data_generation_id") != "lib1_single_part_dedup_exact_v1":
        raise ValueError("Expected the canonical post-dedup exact-v1 manifest.")

    frames: dict[str, pd.DataFrame] = {}
    for slug, _, _ in PARTS:
        spec = manifest["datasets"][slug]
        path = Path(spec["output_path"])
        frame = pd.read_csv(path, sep="\t")
        if BARCODE_COLUMN not in frame:
            raise ValueError(f"{path} is missing {BARCODE_COLUMN}.")
        if len(frame) != int(spec["row_count"]):
            raise ValueError(
                f"{slug} row-count mismatch: observed {len(frame)}, "
                f"manifest declares {spec['row_count']}"
            )
        frame[BARCODE_COLUMN] = pd.to_numeric(frame[BARCODE_COLUMN], errors="raise")
        if not np.isfinite(frame[BARCODE_COLUMN]).all():
            raise ValueError(f"{slug} contains non-finite barcode counts.")
        if (frame[BARCODE_COLUMN] < 1).any():
            raise ValueError(f"{slug} contains modeled rows with fewer than one barcode.")
        frames[slug] = frame
    return frames, manifest


def summarize(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for slug, label, _ in PARTS:
        values = frames[slug][BARCODE_COLUMN]
        quantiles = values.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        record: dict[str, float | int | str] = {
            "part_slug": slug,
            "part": label,
            "n_constructs": len(values),
            "min": values.min(),
            "q25": quantiles.loc[0.25],
            "median": quantiles.loc[0.50],
            "q75": quantiles.loc[0.75],
            "q90": quantiles.loc[0.90],
            "q95": quantiles.loc[0.95],
            "q99": quantiles.loc[0.99],
            "max": values.max(),
        }
        for bin_label, selector, _ in SUPPORT_BINS:
            count = int(selector(values).sum())
            safe_label = bin_label.replace("\N{GREATER-THAN OR EQUAL TO}", "ge_").replace("-", "_to_")
            record[f"n_{safe_label}"] = count
            record[f"percent_{safe_label}"] = 100.0 * count / len(values)
        records.append(record)
    return pd.DataFrame.from_records(records)


def plot_survival(ax: plt.Axes, frames: dict[str, pd.DataFrame]) -> None:
    thresholds = np.arange(1, DISPLAY_MAX_BARCODES + 1)
    for slug, label, color in PARTS:
        values = frames[slug][BARCODE_COLUMN].to_numpy(dtype=float)
        survival = np.array([(values >= threshold).mean() * 100 for threshold in thresholds])
        hq8 = 100.0 * (values >= 8).mean()
        ax.step(
            thresholds,
            survival,
            where="post",
            color=color,
            linewidth=2.8,
            alpha=0.62,
            label=f"{label} ({hq8:.1f}% HQ8)",
        )

    ax.axvline(8, color="#132A44", linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(
        8.45,
        47,
        "HQ8 cutoff",
        color="#132A44",
        fontsize=10,
        fontweight="semibold",
        rotation=90,
        ha="left",
        va="center",
    )
    ax.set_xscale("log", base=2)
    ticks = [1, 2, 4, 8, 16, 32, 64]
    ax.set_xticks(ticks, [str(tick) for tick in ticks])
    ax.set_xlim(1, DISPLAY_MAX_BARCODES)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("distinct barcode per construct", fontweight="semibold")
    ax.set_ylabel(
        "Constructs with \N{GREATER-THAN OR EQUAL TO} x distinct barcodes (%)",
        fontweight="semibold",
    )
    ax.set_title("Barcode count distribution", pad=13)
    ax.legend(frameon=False, fontsize=9.4, loc="upper right")
    ax.grid(color="#DCE4EE", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def plot_composition(ax: plt.Axes, frames: dict[str, pd.DataFrame]) -> None:
    y = np.arange(len(PARTS))
    left = np.zeros(len(PARTS))
    labels = []

    for slug, label, _ in PARTS:
        labels.append(f"{label}\n(n={len(frames[slug]):,})")

    for bin_label, selector, color in SUPPORT_BINS:
        percentages = np.array(
            [
                100.0 * selector(frames[slug][BARCODE_COLUMN]).mean()
                for slug, _, _ in PARTS
            ]
        )
        if bin_label == "\N{GREATER-THAN OR EQUAL TO}8":
            legend_label = "barcode-count \N{GREATER-THAN OR EQUAL TO} 8"
        else:
            display_bin_label = bin_label.replace("-", "\N{EN DASH}")
            legend_label = f"barcode-count = {display_bin_label}"
        bars = ax.barh(
            y,
            percentages,
            left=left,
            height=0.62,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            label=legend_label,
        )
        for bar, percentage in zip(bars, percentages):
            if percentage >= 8:
                text_color = "white" if color == "#235A9F" else "#132A44"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{percentage:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9.4,
                    fontweight="semibold",
                    color=text_color,
                )
        left += percentages

    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlabel("Modeled constructs", fontweight="semibold")
    ax.set_title("Support composition by library", pad=13)
    ax.legend(
        frameon=False,
        fontsize=8.8,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=4,
    )
    ax.grid(axis="x", color="#DCE4EE", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def style_axes(axes: np.ndarray) -> None:
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#64758B")
        ax.spines["bottom"].set_color("#64758B")
        ax.tick_params(labelsize=10)


def make_figure(frames: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "axes.titlesize": 14,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2), gridspec_kw={"width_ratios": [1.08, 0.92]})
    plot_survival(axes[0], frames)
    plot_composition(axes[1], frames)
    style_axes(axes)

    off_scale = []
    for slug, label, _ in PARTS:
        values = frames[slug][BARCODE_COLUMN]
        if values.max() > DISPLAY_MAX_BARCODES:
            off_scale.append(f"{label}: max {int(values.max()):,}")
    off_scale_note = "; ".join(off_scale)

    fig.suptitle(
        "Lib1 barcode support by single-part library",
        fontsize=18,
        fontweight="bold",
        color="#132A44",
        y=0.98,
    )
    fig.text(
        0.5,
        0.012,
        "n_barcodes counts distinct nonblank barcode identities and is a measurement-support proxy, "
        "not a count of independent biological replicates. "
        f"Survival x-axis ends at {DISPLAY_MAX_BARCODES}; off-scale maxima: {off_scale_note}.",
        ha="center",
        va="bottom",
        fontsize=9.6,
        color="#4A5E73",
    )
    fig.tight_layout(rect=(0.025, 0.075, 0.995, 0.92), w_pad=3.6)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "lib1_dedup_barcode_support_distributions"
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
    summary_path = args.output_dir / "lib1_dedup_barcode_support_summary.csv"
    summary.to_csv(summary_path, index=False)

    provenance = {
        "data_generation_id": manifest["data_generation_id"],
        "manifest": str(args.manifest.resolve()),
        "barcode_definition": "distinct nonblank bba1_ddc1_concat identities",
        "hq_threshold": 8,
        "survival_display_max": DISPLAY_MAX_BARCODES,
        "outputs": [str(path.resolve()) for path in [*figure_paths, summary_path]],
    }
    provenance_path = args.output_dir / "lib1_dedup_barcode_support_figure.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    for path in [*figure_paths, summary_path, provenance_path]:
        print(path)


if __name__ == "__main__":
    main()
