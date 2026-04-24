# Introns — HPO configs (placeholder)

This directory is a reservation for future intronic-element MPRA data. No
intron MPRA dataset has landed in `opt_EU_learn_n_design/` yet, so every
config in here is a **non-runnable template** that serves three purposes:

1. It mirrors the `<region>/<target_family>/<model_family>/` directory
   layout used for enhancer / promoter / utr3 / utr5, so the introns
   campaign can be added without inventing a new scheme.
2. It documents the W&B naming convention this repo now enforces:
   `introns__<target_family>__<mode>__<model>`. Until a concrete dataset
   arrives we freeze the `target_family` as `placeholder`.
3. It flags what must be filled in before the first real pilot runs —
   the `TODO(introns_data)` markers in the template should be grep-able.

## When real intron data arrives

1. Replace `target_family=placeholder` with the canonical source
   (e.g. `bcm_intron_mpra_2026` or `author_library_name`) and rename the
   subdirectory accordingly.
2. Add a data module in `boda/data/` (likely a subclass of
   `MPRA_DataModule` or `UTR_Polysome_MPRA_DataModule`, depending on
   barcode semantics). Register it in `boda/data/__init__.py`.
3. Point `datafile_path` at the materialized table under
   `opt_EU_learn_n_design/` and fill in `sequence_column` /
   `activity_column` / `fold_column` (or splitting strategy).
4. Add a launcher in `src/learn/launch/introns_*.sh` modeled after
   `utr3_hani_utr_bassetvl_sweep.sh`.
5. Drop the placeholder template once a real sweep config exists.

## Why the placeholder exists at all

The repo's registry (`run_registry/runs.csv`) and the pretrained-model
lookup helper (`pretrained_registry.py`) iterate over the `configs/`
tree to surface "every CRE region the agent knows about". Having a
stub here means `introns` shows up as explicitly *pending* rather than
silently missing — which is the same signal that motivated this
reorganization.
