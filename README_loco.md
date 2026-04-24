# README_loco

This is the local working README for `boda2_EU`, a personalized extension of `[sjgosai/boda2](https://github.com/sjgosai/boda2)` for training deep learning models on different CRE parts rather than only the original enhancer-centric MPRA workflow.

The main local goal of this repo is now:

- enhancer modeling on one-hot encoded sequence data
- promoter modeling
- 5'UTR and 3'UTR expression / RNA activity prediction
- sweep-driven architecture search with W&B
- notebook-based HPO result analysis
- future transfer learning onto new in-house data

This file is meant to help reboot the project quickly after context drift.

## What This Repo Became

Upstream `boda2` / CODA is a modular framework for model training, inference, and sequence design on regulatory DNA data. In this local copy, the framework was extended into a broader CRE modeling sandbox:

- `boda/` now contains custom data modules and model classes beyond the original enhancer use case.
- `src/` was reorganized into workflow-oriented subdirectories rather than a flat script collection.
- `tutorials/` became the practical place for HPO result extraction, model comparison, and evaluation.
- `analysis/` remains a larger exploratory notebook archive and is less curated.

The practical mental model is:

- `boda/` = reusable library code
- `src/` = executable workflows
- `tutorials/` = curated notebooks for running and analyzing experiments
- `analysis/` = older exploratory notebooks and manuscript/dev work
- `docker/` = environment / deployment support

## Repo Map

### Core library

- `boda/data/`
  - original MPRA loaders plus local additions for promoter and UTR tasks
- `boda/model/`
  - original Basset-family models plus `UTR_BassetVL`
- `boda/graph/`
  - training wrappers such as `CNNBasicTraining` and `CNNTransferLearning`
- `boda/generator/`
  - sequence design / optimization modules

### Executable workflows

- `src/learn/`
  - training entrypoints, W&B sweep configs, local launch scripts, and cached W&B run outputs
- `src/design/`
  - sequence generation scripts
- `src/analysis/`
  - reusable inference / downstream scripts such as contribution scores and VCF prediction
- `src/util_scripts/`
  - docker / deployment helpers

### Notebook layers

- `tutorials/`
  - curated notebooks for training, HPO extraction, evaluation, and design
- `analysis/`
  - broader exploratory notebook archive; useful, but not always in sync

## Local Extensions That Matter

These are the key modifications that make this repo useful for multi-CRE modeling:

- `PromoterDataModule` in `boda/data/mpra_datamodule.py`
  - reads promoter CSVs, handles RC filtering, pads sequences, standardizes expression, and uses dataset-provided train/val splits
- `HaniGoozardi_RNA_Activity_DataModule` in `boda/data/mpra_datamodule.py`
  - supports processed 5'UTR and 3'UTR RNA activity libraries with fold-based splitting and cell-type filtering
- `UTR_BassetVL` in `boda/model/basset.py`
  - shorter-sequence Basset variant using same-padded convs and optional adaptive pooling, appropriate for promoter / UTR tasks
- `train_wandb_log.py` in `src/learn/`
  - the real modern training entrypoint for sweep-driven work

## Current Supported Problem Families


| CRE part | Main data module                                                         | Main model       | Main graph            | Typical goal                                     |
| -------- | ------------------------------------------------------------------------ | ---------------- | --------------------- | ------------------------------------------------ |
| Enhancer | `MPRA_DataModule`                                                        | `BassetBranched` | `CNNTransferLearning` | multi-output MPRA regression / transfer learning |
| Promoter | `PromoterDataModule`                                                     | `UTR_BassetVL`   | `CNNBasicTraining`    | single-output promoter expression regression     |
| 5'UTR    | `HaniGoozardi_RNA_Activity_DataModule` or `UTR_Polysome_MPRA_DataModule` | `UTR_BassetVL`   | `CNNBasicTraining`    | translation / RNA activity prediction            |
| 3'UTR    | `HaniGoozardi_RNA_Activity_DataModule`                                   | `UTR_BassetVL`   | `CNNBasicTraining`    | RNA activity prediction                          |


## Canonical Training Entry Points

### Primary modern path

- `src/learn/train_wandb_log.py`
  - use this for W&B sweeps and most current experiments
  - saves a model artifact tarball and writes `model_saved_path` into W&B run summary

### Older / alternative paths

- `src/learn/train.py`
  - more generic older CLI training path
- `src/learn/previous_train.py`
  - older W&B-oriented trainer; likely superseded
- `src/learn/fixed_utr_train.sh`
  - fixed hand-written launch commands for older 5'UTR runs
- `src/learn/deploy_wandb_agent_train.sh`
  - W&B agent fan-out launcher, but currently promoter-specific in practice
- `src/learn/launch/`
  - curated task-oriented launch layer for the reorganized config tree
- `src/learn/run_registry/`
  - machine-readable best-run and sweep-launch bookkeeping

## Sweep Config Inventory

The hand-authored configs that matter are now organized under `src/learn/configs/` by
CRE family, target family, and model family:


| Config                                                                                          | Role                              | Notes                                                                                    |
| ----------------------------------------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------- |
| `enhancer/malinois_mpra/basset_branched/enhancer__malinois_mpra__basset_branched__transfer_baseline.yml` | enhancer baseline / replay config | really a fixed grid-style enhancer transfer config rather than a broad exploratory sweep |
| `promoter/deboer_core/utr_bassetvl/promoter__deboer_core__utr_bassetvl__bayes.yml`            | promoter Bayesian sweep           | explores RC augmentation, conv widths/kernels, FC head, optimizer, LR, scheduler        |
| `utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__egfp_1.yml`                  | 5'UTR polysome fixed run          | recreated from the historical fixed polysome path for eGFP library 1                    |
| `utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__egfp_2.yml`                  | 5'UTR polysome fixed run          | recreated from the historical fixed polysome path for eGFP library 2                    |
| `utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__mcherry_1.yml`               | 5'UTR polysome fixed run          | recreated from the historical fixed polysome path for mCherry library 1                 |
| `utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__mcherry_2.yml`               | 5'UTR polysome fixed run          | recreated from the historical fixed polysome path for mCherry library 2                 |
| `utr5/hani_rna_activity/utr_bassetvl/utr5__hani_rna_activity__utr_bassetvl__bayes.yml`        | 5'UTR Hani lib1 sweep             | uses processed Hani CSV, fold-based split, `target_cell_type=c2`                        |
| `utr3/hani_rna_activity/utr_bassetvl/utr3__hani_rna_activity__utr_bassetvl__bayes.yml`        | broad 3'UTR sweep                 | larger architecture search, includes `adaptive_pool_output_size`                        |
| `utr3/hani_rna_activity/utr_bassetvl/utr3__hani_rna_activity__utr_bassetvl__focused_bayes__2025-06-16.yml` | focused 3'UTR refinement sweep    | tighter second-pass search after a broader run                                          |


The common pattern across these YAMLs is:

- choose `data_module`, `model_module`, `graph_module`
- define dataset path and split behavior
- define architecture search space
- define optimizer / scheduler space
- define trainer behavior and checkpoint metric
- define W&B project and run naming
- define local artifact destination

## Sweep Naming and Storage Reality

Right now, sweep organization is functional but inconsistent:

- config filenames are flat and mix task, phase, and date conventions
- artifact roots are inconsistent between `src/local_artifacts` and `src/learn/local_artifacts`
- W&B run cache is stored inside `src/learn/wandb/`, which makes `src/` look larger and noisier than it really is
- `deploy_wandb_agent_train.sh` is named generically but is effectively a promoter launcher

Current authored-config structure:

- `src/learn/configs/enhancer/malinois_mpra/basset_branched/`
- `src/learn/configs/promoter/deboer_core/utr_bassetvl/`
- `src/learn/configs/utr5/polysome/utr_bassetvl/`
- `src/learn/configs/utr5/hani_rna_activity/utr_bassetvl/`
- `src/learn/configs/utr3/hani_rna_activity/utr_bassetvl/`
- `src/learn/launch/`
- `src/learn/run_registry/`
- `src/learn/wandb/` kept explicitly as generated-output-only

Recommended naming convention for future authored configs:

- `<cre_family>__<target_family>__<model_family>__<stage>__<yyyy-mm-dd>.yml`

Example:

- `utr3__hani_rna_activity__utr_bassetvl__focused_bayes__2025-06-16.yml`

Comparison rule of thumb:

- if two model families should be compared directly, keep them under the same `cre_family/target_family`
- keep the filename stem the same apart from the `model_family` token

## Tutorial / Notebook Map

Important note: the curated directory is `tutorials/`, not `Tutorials/`.

These notebooks appear to be the most useful reboot entrypoints:

- `tutorials/hani_utr_2025_data_inspection.ipynb`
  - raw-to-processed UTR library inspection and preprocessing
- `tutorials/extract_wandb_sweep.ipynb`
  - generic sweep result extraction and dataframe-based HPO analysis
- `tutorials/extract_sweep_result/may26_utr3_sweep.ipynb`
  - 3'UTR sweep extraction and best-run analysis
- `tutorials/extract_sweep_result/jul17_utr5_cross_lib_sweep_comparison.ipynb`
  - 5'UTR sweep comparison and cross-library generalization
- `tutorials/get_HPO_5utr_polysome.ipynb`
  - older 5'UTR HPO comparisons across libraries
- `tutorials/unified_cre_model_evaluation.ipynb`
  - unified model comparison across CRE families
- `tutorials/evaluate_local_utr_models.ipynb`
  - local artifact / local W&B metadata recovery without relying fully on API access
- `tutorials/check_promoter_data.ipynb`
  - promoter dataset sanity-check notebook
- `tutorials/run_training_and_design.ipynb`
  - direct training-to-design workflow example

## Reboot Workflow

For new work, think of the project as a 6-stage pipeline:

1. Prepare a stable processed dataset.
2. Pick the correct data module and model family.
3. Define a sweep or fixed run config.
4. Launch training and save artifacts consistently.
5. Analyze HPO results in `tutorials/`.
6. Promote the best model into reuse or transfer learning.

### Path A: Train from scratch on a new dataset

Use this when the new in-house dataset is large enough or sufficiently different that full retraining is reasonable.

Checklist:

- preprocess into a clean table with stable column names
- decide whether the problem is enhancer, promoter, 5'UTR, or 3'UTR-like
- choose the closest existing YAML as a template
- keep artifact outputs in one canonical subtree
- analyze by validation metric, then verify generalization in notebook analysis

### Path B: Transfer learning onto new in-house data

Use this when you already have a strong parent model and want fast adaptation.

Checklist:

- reuse an existing architecture and compare it against scratch training
- use `CNNTransferLearning` when freezing / initializing from a parent is helpful
- evaluate zero-shot parent performance before fine-tuning
- compare fine-tuned vs scratch vs parent-only baselines
- record the promoted artifact path and training context in one place

## Current In-House Data You Can Use Now

Immediate enhancer dataset:

- `/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/enhancers/20251218_np_fastq1_500000NPreads_enh_variants_bc_sum_avg_expression.txt`

Observed columns:

- `Enhancers`
- `DNA_Counts_Sum`
- `RNA_Counts_Sum`
- `n_barcodes`
- `RNA_DNA_Ratio_raw`
- `RNA_DNA_Ratio_log10_scaled`
- `RNA_DNA_Ratio_log10_scaled_zscore`

Why this is useful:

- it already looks like a sequence-to-expression table
- it can support a first reboot experiment on enhancer prediction right away
- it is a good candidate for testing the current enhancer path and for later transfer-learning comparisons

Suggested first-pass mapping for this dataset:

- sequence column: `Enhancers`
- target candidate: `RNA_DNA_Ratio_log10_scaled_zscore`
- fallback target candidate: `RNA_DNA_Ratio_log10_scaled`

Before launching a serious sweep, verify:

- sequence lengths are consistent or intentionally padded
- whether low-count rows should be filtered
- whether reverse complements are biologically appropriate for the task
- whether train/val/test split should be by random split, held-out design batch, or another provenance-aware grouping

## Known Inconsistencies and Gotchas

- `tutorials/` is lowercase, but it is easy to mentally refer to it as `Tutorials/`
- `utr3__hani_rna_activity__utr_bassetvl__bayes.yml` currently names `UTR3_RNA_Activity_DataModule`, but the exported local UTR/Hani class present in `boda/data/__init__.py` is `HaniGoozardi_RNA_Activity_DataModule`
- artifact roots are inconsistent between `src/local_artifacts/...` and `src/learn/local_artifacts/...`
- `src/learn/wandb/` contains generated run outputs, not curated source material
- `deploy_wandb_agent_train.sh` starts with `!/bin/bash` instead of `#!/bin/bash`
- the root `README.md` still reflects the upstream CODA mental model more than the current local multi-CRE workflow

## Suggested Cleanup Order

If the goal is to make this repo easy to restart and maintain, do the following in order:

1. Declare one canonical artifact root.
2. Split authored sweep configs by CRE family, target family, and model family.
3. Move or clearly label generated W&B output as non-source material.
4. Normalize sweep filenames so cross-model comparisons are obvious.
5. Add one small manifest or registry per experiment family documenting:
  - dataset path
  - data module
  - model class
  - graph class
  - sweep config
  - W&B project / sweep ID
  - best artifact(s)
6. Add curated launch scripts under `src/learn/launch/` so training starts from the reorganized config tree rather than ad hoc commands.
7. Keep `src/learn/configs/README.md` and `src/learn/README.md` aligned with the on-disk layout.

## Best Current Mental Model

When restarting this project, do not think "this is the old enhancer repo."

Think:

- this is a modular local CRE modeling framework
- `boda/` is the reusable engine
- `src/learn/` is the experiment launcher layer
- `tutorials/` is where you remember how sweeps were interpreted
- the next valuable milestone is to connect the new in-house enhancer dataset to a clean, documented training + evaluation path

## Immediate Next Experiments

High-value next steps for the reboot:

1. Run a small fixed enhancer baseline on the MattLee enhancer dataset using the current local stack.
2. Decide whether enhancer reboot should stay with `BassetBranched` or also test `UTR_BassetVL`-style simpler backbones on shorter constructs.
3. Normalize one new sweep config for the in-house enhancer data under a cleaner naming convention.
4. Add a small manifest notebook or CSV that records best runs and artifact paths across enhancer, promoter, 5'UTR, and 3'UTR efforts.
