#!/usr/bin/env python3
"""Comparable-bin barcode-range fine-tuning for filtered lib1 enhancer data.

This is a thin configuration wrapper around
`lib1_enhancer_barcode_range_finetuning.py`.

The Apr 30 exact-bin run used uneven pools (`1`, `2-3`, `4-10`, `>10`, `>=4`),
which makes full-fraction comparisons hard to interpret. This runner defaults
to non-overlapping, better-balanced barcode bins and caps each eligible training
pool to the same size before building learning curves:

  - 1-3 barcodes
  - 4-6 barcodes
  - >=7 barcodes

The default cap is 1000 constructs per bin/split, so the 1.0 train-size
fraction compares matched-size training pools across bins.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
BASE_RUNNER_PATH = THIS_DIR / "lib1_enhancer_barcode_range_finetuning.py"
DEFAULT_OUTDIR = (
    Path("/home/minhang/synBio_AL/boda2_EU")
    / "src"
    / "finetune"
    / "learning_curve"
    / "lib1_enhancer_barcode_range_comparable_bins_hq4_hq8_b2_b3_may2026"
)


def load_base_runner():
    spec = importlib.util.spec_from_file_location("lib1_barcode_range_base_runner", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base runner from {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class ComparableBarcodeBinSpec:
    name: str
    label: str
    query: str
    sort_order: int
    min_barcodes: int
    max_barcodes: int | None = None

    def mask(self, values: pd.Series) -> pd.Series:
        bc = pd.to_numeric(values, errors="coerce")
        mask = bc >= self.min_barcodes
        if self.max_barcodes is not None:
            mask &= bc <= self.max_barcodes
        return mask


COMPARABLE_BIN_SPECS = {
    "bc_1_3": ComparableBarcodeBinSpec(
        "bc_1_3",
        "1-3 barcodes",
        "1 <= number_of_barcodes <= 3",
        10,
        min_barcodes=1,
        max_barcodes=3,
    ),
    "bc_4_6": ComparableBarcodeBinSpec(
        "bc_4_6",
        "4-6 barcodes",
        "4 <= number_of_barcodes <= 6",
        20,
        min_barcodes=4,
        max_barcodes=6,
    ),
    "bc_ge7": ComparableBarcodeBinSpec(
        "bc_ge7",
        ">=7 barcodes",
        "number_of_barcodes >= 7",
        30,
        min_barcodes=7,
        max_barcodes=None,
    ),
}


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--train_pool_cap",
        type=int,
        default=1000,
        help=(
            "Cap each eligible barcode-bin training pool before learning-curve "
            "sampling. Use <=0 to disable matched-size capping."
        ),
    )
    parser.add_argument(
        "--train_pool_cap_seed",
        type=int,
        default=104729,
        help="Base random seed for deterministic per-bin training-pool caps.",
    )
    parser.add_argument(
        "--allow_all_pretrained_heads",
        action="store_true",
        help="Do not default to K562-only when --pretrained_heads is omitted.",
    )
    parser.add_argument(
        "--no_default_b2_b3",
        action="store_true",
        help="Do not inject --include_b2 --include_b3 when no setting flags are provided.",
    )
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def configure_base_runner(base_runner, wrapper_args: argparse.Namespace, remaining_argv: list[str]) -> list[str]:
    train_pool_cap = int(wrapper_args.train_pool_cap)
    cap_seed = int(wrapper_args.train_pool_cap_seed)

    base_runner.DEFAULT_OUTDIR = DEFAULT_OUTDIR
    base_runner.DEFAULT_SEEDS = [23, 19, 31, 37, 43]
    base_runner.DEFAULT_HELDOUT_MIN_BARCODES = [4, 8]
    base_runner.DEFAULT_TRAIN_BARCODE_BINS = ["bc_1_3", "bc_4_6", "bc_ge7"]
    base_runner.DEFAULT_MAX_EPOCHS = 90
    base_runner.DEFAULT_PATIENCE = 20
    base_runner.DEFAULT_TRAIN_SIZE_FRACS = [0.25, 0.50, 0.75, 1.0]
    cap_tag = f"cap{train_pool_cap}" if train_pool_cap > 0 else "uncapped"
    base_runner.CACHE_LAYOUT_VERSION = f"filtered_raw_ratio_comparable_barcode_bins_{cap_tag}_per_epoch_metrics_v1"
    base_runner.BARCODE_BIN_SPECS = COMPARABLE_BIN_SPECS
    base_runner.BARCODE_BIN_CHOICES = sorted(COMPARABLE_BIN_SPECS)

    def build_comparable_bin_train_pool_components(
        train_rest_df: pd.DataFrame,
        train_bin: ComparableBarcodeBinSpec,
        barcode_column: str,
        heldout_min_barcodes: int,
    ) -> dict[str, Any]:
        eligible_uncapped = (
            train_rest_df.loc[train_bin.mask(train_rest_df[barcode_column])]
            .copy()
            .reset_index(drop=True)
        )
        if len(eligible_uncapped) == 0:
            raise ValueError(f"No train rows available for barcode bin {train_bin.name}.")

        cap_enabled = train_pool_cap > 0 and len(eligible_uncapped) > train_pool_cap
        if cap_enabled:
            sample_seed = cap_seed + int(heldout_min_barcodes) * 10_007 + train_bin.sort_order * 1_009
            eligible = (
                eligible_uncapped.sample(n=train_pool_cap, random_state=sample_seed, replace=False)
                .sort_values("row_id" if "row_id" in eligible_uncapped.columns else barcode_column)
                .reset_index(drop=True)
            )
        else:
            eligible = eligible_uncapped

        leftover_hq = eligible.loc[eligible[barcode_column] >= heldout_min_barcodes].copy().reset_index(drop=True)
        lower_quality = eligible.loc[eligible[barcode_column] < heldout_min_barcodes].copy().reset_index(drop=True)
        return {
            "eligible": eligible,
            "leftover_hq": leftover_hq,
            "lower_quality": lower_quality,
            "test_min_barcodes": int(heldout_min_barcodes),
            "train_barcode_bin": train_bin.name,
            "train_barcode_bin_label": train_bin.label,
            "train_barcode_bin_query": train_bin.query,
            "train_pool_uncapped_size": int(len(eligible_uncapped)),
            "train_pool_cap": int(train_pool_cap) if train_pool_cap > 0 else None,
        }

    base_runner.build_barcode_bin_train_pool_components = build_comparable_bin_train_pool_components

    patched_argv = list(remaining_argv)
    if "--seeds" not in patched_argv:
        patched_argv.extend(["--seeds", "23", "19", "31", "37", "43"])
    if "--patience" not in patched_argv:
        patched_argv.extend(["--patience", "20"])
    if "--b3_bcaps" not in patched_argv:
        patched_argv.extend(["--b3_bcaps", "10", "30"])
    if "--pretrained_heads" not in patched_argv and not wrapper_args.allow_all_pretrained_heads:
        patched_argv.extend(["--pretrained_heads", "K562"])

    has_setting_flag = any(flag in patched_argv for flag in ("--include_b1", "--include_b2", "--include_b3"))
    if not has_setting_flag and not wrapper_args.no_default_b2_b3:
        patched_argv.extend(["--include_b2", "--include_b3"])

    return patched_argv


def main() -> None:
    wrapper_args, remaining_argv = parse_wrapper_args(sys.argv[1:])
    base_runner = load_base_runner()
    patched_argv = configure_base_runner(base_runner, wrapper_args, remaining_argv)

    # Add wrapper-level details to the manifest produced by the base runner.
    original_main = base_runner.main
    sys.argv = [sys.argv[0], *patched_argv]
    original_main()

    outdir = base_runner.DEFAULT_OUTDIR
    if "--outdir" in patched_argv:
        outdir_idx = patched_argv.index("--outdir")
        if outdir_idx + 1 < len(patched_argv):
            outdir = Path(patched_argv[outdir_idx + 1])
    manifest_path = outdir / "run_manifest.json"
    if manifest_path.exists():
        # The base runner writes the manifest before training. Append wrapper metadata
        # after the run starts so analysis notebooks can recover the cap settings.
        import json

        with manifest_path.open() as handle:
            manifest = json.load(handle)
        manifest["wrapper_script"] = str(Path(__file__).resolve())
        manifest["comparable_bin_specs"] = {name: asdict(spec) for name, spec in COMPARABLE_BIN_SPECS.items()}
        manifest["matched_train_pool_cap"] = int(wrapper_args.train_pool_cap)
        manifest["train_pool_cap_seed"] = int(wrapper_args.train_pool_cap_seed)
        manifest["default_pretrained_head"] = "K562" if not wrapper_args.allow_all_pretrained_heads else "all"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
