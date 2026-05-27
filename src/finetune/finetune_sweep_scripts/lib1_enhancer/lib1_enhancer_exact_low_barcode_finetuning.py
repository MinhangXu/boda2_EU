#!/usr/bin/env python3
"""Exact low-barcode-count fine-tuning wrapper for filtered lib1 enhancers.

This narrows the comparable-bin barcode runner to answer the follow-up question:
is exactly one barcode worse than nearby exact counts when training N is matched?

Default bins are:

  - exactly 1 barcode
  - exactly 2 barcodes
  - exactly 3 barcodes
  - 4-6 barcodes
  - >=7 barcodes

The default training-pool cap is 500 constructs because the filtered dataset has
590 exact-1 constructs before heldout removal.
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
    / "lib1_enhancer_exact_low_barcode_hq4_hq8_cap500_b1_b2_may2026"
)


def load_base_runner():
    spec = importlib.util.spec_from_file_location("lib1_barcode_range_base_runner_exact_low", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base runner from {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class ExactBarcodeBinSpec:
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


EXACT_LOW_BIN_SPECS = {
    "bc_eq1": ExactBarcodeBinSpec(
        "bc_eq1",
        "1 barcode",
        "number_of_barcodes == 1",
        10,
        min_barcodes=1,
        max_barcodes=1,
    ),
    "bc_eq2": ExactBarcodeBinSpec(
        "bc_eq2",
        "2 barcodes",
        "number_of_barcodes == 2",
        20,
        min_barcodes=2,
        max_barcodes=2,
    ),
    "bc_eq3": ExactBarcodeBinSpec(
        "bc_eq3",
        "3 barcodes",
        "number_of_barcodes == 3",
        30,
        min_barcodes=3,
        max_barcodes=3,
    ),
    "bc_2_3": ExactBarcodeBinSpec(
        "bc_2_3",
        "2-3 barcodes",
        "2 <= number_of_barcodes <= 3",
        35,
        min_barcodes=2,
        max_barcodes=3,
    ),
    "bc_4_6": ExactBarcodeBinSpec(
        "bc_4_6",
        "4-6 barcodes",
        "4 <= number_of_barcodes <= 6",
        40,
        min_barcodes=4,
        max_barcodes=6,
    ),
    "bc_ge7": ExactBarcodeBinSpec(
        "bc_ge7",
        ">=7 barcodes",
        "number_of_barcodes >= 7",
        50,
        min_barcodes=7,
        max_barcodes=None,
    ),
}

DEFAULT_EXACT_LOW_BINS = ["bc_eq1", "bc_eq2", "bc_eq3", "bc_4_6", "bc_ge7"]


def parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--train_pool_cap",
        type=int,
        default=500,
        help="Cap each eligible barcode-bin training pool before learning-curve sampling.",
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
        "--no_default_b1_b2",
        action="store_true",
        help="Do not inject --include_b1 --include_b2 when no setting flags are provided.",
    )
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def configure_base_runner(base_runner, wrapper_args: argparse.Namespace, remaining_argv: list[str]) -> list[str]:
    train_pool_cap = int(wrapper_args.train_pool_cap)
    cap_seed = int(wrapper_args.train_pool_cap_seed)

    base_runner.DEFAULT_OUTDIR = DEFAULT_OUTDIR
    base_runner.DEFAULT_SEEDS = [23, 19, 31]
    base_runner.DEFAULT_HELDOUT_MIN_BARCODES = [4, 8]
    base_runner.DEFAULT_TRAIN_BARCODE_BINS = list(DEFAULT_EXACT_LOW_BINS)
    base_runner.DEFAULT_MAX_EPOCHS = 70
    base_runner.DEFAULT_PATIENCE = 10
    base_runner.DEFAULT_TRAIN_SIZE_FRACS = [0.02, 0.05, 0.10, 0.20, 0.50, 1.0]
    cap_tag = f"cap{train_pool_cap}" if train_pool_cap > 0 else "uncapped"
    base_runner.CACHE_LAYOUT_VERSION = f"filtered_raw_ratio_exact_low_barcode_bins_{cap_tag}_v1"
    base_runner.BARCODE_BIN_SPECS = EXACT_LOW_BIN_SPECS
    base_runner.BARCODE_BIN_CHOICES = sorted(EXACT_LOW_BIN_SPECS)

    def build_exact_low_train_pool_components(
        train_rest_df: pd.DataFrame,
        train_bin: ExactBarcodeBinSpec,
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

    base_runner.build_barcode_bin_train_pool_components = build_exact_low_train_pool_components

    patched_argv = list(remaining_argv)
    if "--seeds" not in patched_argv:
        patched_argv.extend(["--seeds", "23", "19", "31"])
    if "--heldout_min_barcodes" not in patched_argv:
        patched_argv.extend(["--heldout_min_barcodes", "4", "8"])
    if "--train_barcode_bins" not in patched_argv:
        patched_argv.extend(["--train_barcode_bins", *DEFAULT_EXACT_LOW_BINS])
    if "--train_size_fracs" not in patched_argv:
        patched_argv.extend(["--train_size_fracs", "0.02", "0.05", "0.1", "0.2", "0.5", "1.0"])
    if "--min_train_size" not in patched_argv:
        patched_argv.extend(["--min_train_size", "10"])
    if "--max_epochs" not in patched_argv:
        patched_argv.extend(["--max_epochs", "70"])
    if "--patience" not in patched_argv:
        patched_argv.extend(["--patience", "10"])
    if "--unfreeze_scopes" not in patched_argv:
        patched_argv.extend(["--unfreeze_scopes", "branched_only"])
    if "--pretrained_heads" not in patched_argv and not wrapper_args.allow_all_pretrained_heads:
        patched_argv.extend(["--pretrained_heads", "K562"])

    has_setting_flag = any(flag in patched_argv for flag in ("--include_b1", "--include_b2", "--include_b3"))
    if not has_setting_flag and not wrapper_args.no_default_b1_b2:
        patched_argv.extend(["--include_b1", "--include_b2"])

    return patched_argv


def main() -> None:
    wrapper_args, remaining_argv = parse_wrapper_args(sys.argv[1:])
    base_runner = load_base_runner()
    patched_argv = configure_base_runner(base_runner, wrapper_args, remaining_argv)

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
        import json

        with manifest_path.open() as handle:
            manifest = json.load(handle)
        manifest["wrapper_script"] = str(Path(__file__).resolve())
        manifest["exact_low_bin_specs"] = {name: asdict(spec) for name, spec in EXACT_LOW_BIN_SPECS.items()}
        manifest["matched_train_pool_cap"] = int(wrapper_args.train_pool_cap)
        manifest["train_pool_cap_seed"] = int(wrapper_args.train_pool_cap_seed)
        manifest["default_pretrained_head"] = "K562" if not wrapper_args.allow_all_pretrained_heads else "all"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
