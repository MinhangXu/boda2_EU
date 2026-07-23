#!/usr/bin/env python3
"""Prepare canonical Lib1 Intron single-output data."""

try:
    from .prepare_lib1_single_part_datasets import main_for_part
except ImportError:  # Direct script execution.
    from prepare_lib1_single_part_datasets import main_for_part


if __name__ == "__main__":
    main_for_part("intron")
