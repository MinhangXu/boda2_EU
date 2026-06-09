# Promoter Phase 1 Plan

Generated: 2026-06-08

This plan corrects the promoter framing for Phase 1. The e7 and e30 core
promoter libraries are legacy in-house promoter libraries from an older
experiment, not public data. For cross-part bookkeeping, they can still serve
the same role that public pretraining data serves for enhancer, UTR, and intron:
a larger, older, non-Lib1 source used to initialize a promoter encoder before
adapting to the new Library 1 promoter data.

## Decision To Resolve

Promoter should enter the Phase 2 combinatorial work only after we know whether
a useful encoder comes from:

- legacy e7/e30 in-house pretraining with a proper validation/test split,
- new Lib1 promoter training from scratch,
- legacy e7/e30 pretraining followed by Lib1 fine-tuning,
- or a documented decision to keep promoter random/scratch-initialized.

The earlier deBoer-era work used a heavy architecture and train/validation-only
splits. Phase 1 should replace that with split-safe, logged, test-evaluated
promoter runs.

## Current Data Inventory

### Legacy e7/e30 core promoter data

Primary existing BODA2 training config points at:

`/home/minhang/synBio_AL/Core_Promoter_Model/deBoer_model/preprocess_data/core_promoter_data_with_rc_df_comb_apr1.csv`

That file has 211,908 rows including reverse-complement rows, 105,954 original
rows, and columns `sequence`, `expression`, `complexity`, `set`, `RC_bool`.
It contains `e7`, `e30`, and an `unknown` bucket:

| Bucket | Original rows | Rows with RC copies | Current split |
|---|---:|---:|---|
| e7 | 51,217 | 102,434 | mostly train/val |
| e30 | 41,617 | 83,234 | mostly train/val |
| unknown | 13,120 | 26,240 | almost entirely val |
| total | 105,954 | 211,908 | 80% train, 20% val, 0% test |

There is also an e7/e30-only derived table:

`/home/minhang/synBio_AL/Core_Promoter_Model/deBoer_model/preprocess_data/core_promoter_data_with_rc_df_e7_e30_separate.csv`

For a clean "combine e7 and e30" pretraining dataset, this should be the
starting point unless EDA shows the `unknown` bucket has a deliberate role.

| Bucket | Original rows | Rows with RC copies | Proposed 80/10/10 original-row split |
|---|---:|---:|---|
| e7 | 56,360 | 112,720 | 45,088 train / 5,636 val / 5,636 test |
| e30 | 49,593 | 99,186 | 39,674 train / 4,959 val / 4,960 test |
| total | 105,953 | 211,906 | 84,762 train / 10,595 val / 10,596 test |

The target used by the current derived tables is standardized `expression`.
Upstream files indicate this is a z-scored log10-transformed expression value,
with raw positive values also present in
`formatted_data/1e7_1e30_combined/bib170bib200varCountsbib189normModel.csv`.
The current derived table does not carry per-variant barcode-count columns.
`Core_Promoter_Model/code/UMI_counts_all.npy` exists, but it is not joined to
the training-ready table as a barcode-quality field, so barcode-aware splitting
is not available yet for legacy e7/e30 unless we reconstruct provenance.

### New Lib1 promoter data

Current Lib1 promoter table:

`/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/promoters/L1_final_fastqs1-5_sublibrary_Promoter_subset.csv`

Observed shape is 7,894 rows. One-shot scoring found 7,893 usable sequence
rows after sequence validation. The table has `number_of_barcodes`,
`DNA_bc_counts_sum`, `RNA_bc_counts_sum`, and `RNA/DNA`.

Target for Lib1 promoter modeling should be `log2(RNA/DNA)` unless an analysis
notebook argues for a different transform. Current summary:

| Field | Value |
|---|---:|
| RNA/DNA rows > 0 | 7,894 |
| log2(RNA/DNA) mean | 1.877 |
| log2(RNA/DNA) std | 0.641 |
| log2(RNA/DNA) min / max | -3.807 / 5.358 |
| rows with >= 2 barcodes | 6,812 |
| rows with >= 4 barcodes | 4,593 |
| rows with >= 8 barcodes | 1,932 |
| rows with >= 16 barcodes | 301 |

Most promoter inserts are 50 nt: 7,774 rows are exactly 50 nt, with a small
tail from 41 to 51 nt and one invalid length-1 row. Current one-shot scoring
pads promoter sequences to 84 nt, matching the legacy e7/e30 model input.

## Split Policy

Other Phase 1 CRE work uses two main split conventions:

| Dataset or runner | Total rows | Train % | Val % | Test % | Notes |
|---|---:|---:|---:|---:|---|
| UTR5 Hani Lib1+Lib2 Phase 3 | 31,782 | 80.00 | 10.00 | 10.00 | fold column |
| Seelig intron A5SS | 265,044 | 80.00 | 10.00 | 10.00 | fold column |
| Lib1 enhancer scratch config | 4,788 | 80.00 | 10.00 | 10.00 | `test_min_barcodes=1` |
| Lib1 enhancer HQ8 fine-tune | 4,787 | 89.72 | 5.14 | 5.14 | HQ8-only val/test |
| In-house UTR5 June HPO | 8,331 | 95.69 | 0.86 | 3.44 | small HQ8 heldout pool |

For legacy e7/e30, use 80/10/10 because the dataset is large and has no usable
barcode-quality column. Split original variants first, stratified by
`complexity` and expression quantile bins, then apply train-only RC
augmentation. Do not split after adding stored RC rows.

For new Lib1 promoter, use a barcode-aware split. The primary recommendation is
80/10/10 over all usable rows, with validation and test drawn from rows with
`number_of_barcodes >= 4`. This gives enough heldout rows without consuming
nearly all HQ8 examples. Report secondary evaluation slices for `>= 8` and
`>= 16` barcodes.

If we decide that Lib1 heldout must be HQ8-only, use a smaller total holdout and
record that the split is no longer comparable to the 80/10/10 pretraining
datasets.

## Phase 0: EDA And Data Product

Create a small promoter EDA notebook or script before launching HPO.

Required checks:

- confirm whether `core_promoter_data_with_rc_df_e7_e30_separate.csv` is the
  canonical legacy pretraining source,
- explain or exclude the `unknown` bucket in `core_promoter_data_with_rc_df_comb_apr1.csv`,
- reconcile 105,953 to 105,955 legacy original-row counts across raw and
  derived files,
- verify the legacy target transform from raw expression to z-scored log10
  `expression`,
- check duplicate sequences and any e7/e30 overlap before splitting,
- generate a stable split manifest with row IDs, split labels, source library,
  expression bins, and split hashes,
- verify no original sequence or reverse complement leaks across train, val,
  and test,
- write `split_membership_rows.csv`, `split_membership_summary.csv`, and
  `run_manifest.json` for every training batch.

Output should be a new learn-ready legacy promoter table with explicit
`train`/`val`/`test` split labels and no stored RC rows by default.

## Phase 1: Legacy e7/e30 Pretraining

Train smaller, current architectures on the split-safe legacy e7/e30 table:

- `UTR_BassetVL`,
- `ResNet1DRegressor`,
- optionally `BassetVL` only as a continuity baseline.

Avoid re-centering the plan around the older heavy deBoer transformer-style
architecture unless it is used only as historical context.

Run a reverse-complement ablation:

- no RC augmentation,
- train-only RC augmentation through the dataloader,
- optional stored-RC-copy comparison only if needed for historical parity.

Validation chooses checkpoints and hyperparameters. Test is reported once per
selected run family. Required metrics: Pearson, Spearman, Pearson R2,
coefficient-of-determination R2, MSE, and loss for train/val/test, using the
updated logging contract.

## Phase 2: New Lib1 Promoter From Scratch

Build a Lib1 promoter runner rather than reusing the legacy promoter loader
unchanged.

Recommended defaults:

- input sequence: `Promoter`,
- input length: 84 nt via right-padding with `N`,
- target: `log2(RNA/DNA)`,
- train target standardization: fit on train only,
- primary heldout quality: `number_of_barcodes >= 4`,
- primary split: 80/10/10 total, val/test drawn from the barcode-qualified pool,
- secondary reporting: all rows, `>= 4`, `>= 8`, and `>= 16` barcode slices.

Run scratch baselines with the same architectures that survived legacy
pretraining. Keep reverse-complement augmentation as an explicit ablation rather
than a default assumption.

## Phase 3: Legacy-To-Lib1 Fine-Tuning

Fine-tune the best legacy e7/e30 checkpoint on the new Lib1 promoter data.

Minimum fine-tuning arms:

- head-only or final-layer-only,
- last convolutional stage plus head,
- full unfreeze.

Use the same Lib1 split membership as the scratch runs. Compare against Lib1
scratch on the same val/test rows. If the legacy-pretrained model wins only on
validation but not on the untouched test set, keep it as exploratory rather than
promoted.

## Promotion Criteria

A promoter checkpoint can become a Phase 2 candidate encoder only if:

- the data product has explicit train/val/test rows,
- no RC leakage exists across splits,
- checkpoint selection uses validation only,
- final claims use held-out test metrics,
- Lib1 metrics are reported by barcode-quality slice,
- scratch and legacy-to-Lib1 fine-tune are compared on identical Lib1 splits,
- the model card records whether the seed was legacy e7/e30 pretrained or Lib1
  scratch.

If none of the promoter routes beat a simple baseline on Lib1 test correlation,
the Phase 2 combinatorial plan should mark promoter as pending/random-init
rather than pretending the legacy pretraining solved the Lib1 promoter task.

## Immediate Task List

- [ ] Write the legacy promoter EDA and split-generation script.
- [ ] Create the e7/e30-only 80/10/10 split-safe learn-ready CSV.
- [ ] Add promoter configs/launchers for split-safe legacy pretraining.
- [ ] Run legacy pretraining smoke tests with and without RC augmentation.
- [ ] Run focused HPO for the best legacy architecture family.
- [ ] Build the Lib1 promoter scratch/fine-tune runner with barcode-aware
      split manifests.
- [ ] Run Lib1 scratch baselines.
- [ ] Fine-tune the best legacy e7/e30 checkpoint on Lib1.
- [ ] Write the promoter decision notebook and update the Phase 1 matrix.
