# HPO Repo Reboot Plan

This document turns the current repo state into a concrete HPO-focused roadmap.

Primary goals:

1. recover the best current model choices and experiment history
2. reorganize the repo so HPO across multiple model classes is easier
3. make experiment recovery robust even when local artifacts are missing
4. make config layout comparison-friendly across model families on the same target
5. add a notebook support layer so analysis logic moves out of one-off notebooks

## Current Findings

### 1. Best historical enhancer direction

From `tutorials/malinois_BO_Sup_table_3.txt`:

- the best overall enhancer family is still `BassetBranched` with `CNNTransferLearning`
- the best historical row by average test performance is:
  - timestamp: `20240104_071417`
  - model: `BassetBranched`
  - graph: `CNNTransferLearning`
  - loss: `L1KLmixed`
  - average test score across K562 / HepG2 / SKNSH: about `0.8867`
- `BassetVL` is competitive, but slightly behind
- among the top 50 runs in the BO table:
  - `BassetBranched`: 33
  - `BassetVL`: 17

Conclusion:

- for enhancer reboot, `BassetBranched` should remain the default baseline to beat
- `BassetVL` is still worth keeping as a lighter comparator

### 2. Best cached W&B runs by project / target family

Recovered from `src/learn/wandb`:

- `promoter_optimization`
  - best cached run: `run-20250910_164516-404zkdns`
  - model: `UTR_BassetVL`
  - `epoch_end_val_r2`: `0.4119`
- `utr3_rna_activity_optimization`
  - best cached run: `run-20250617_105607-j94k79zh`
  - model: `UTR_BassetVL`
  - `epoch_end_val_r2`: `0.4511`
- `utr5_hani_rna_activity`
  - best cached run: `run-20250714_100009-2z7reh8i`
  - model: `UTR_BassetVL`
  - `epoch_end_val_r2`: `0.5664`

Important distinction:

- the above `utr5_hani_rna_activity` result is for the newer Hani 5'UTR RNA activity target
- it is not the same task as the older 5'UTR polysome / ribosome-load optimization workflow
- the older polysome workflow lives mainly in:
  - `tutorials/get_HPO_5utr_polysome.ipynb`
  - `src/learn/fixed_utr_train.sh`
  - W&B project: `boda2_EU-src`
  - data module: `UTR_Polysome_MPRA_DataModule`

Recovered polysome sweep identity from the notebook:

- `egfp_1` sweep: `rp7qguqc`
- `egfp_2` sweep: `awnbbtop`
- `mcherry_1` sweep: `4mxeeug3`
- `mcherry_2` sweep: `50qg6ejn`

Recovered top within-library notebook results:

- `egfp_1`: about `0.9459`
- `egfp_2`: about `0.9006`
- `mcherry_1`: about `0.8091`
- `mcherry_2`: about `0.8775`

Conclusion:

- your current default YAML model choices are consistent with the best cached promoter / UTR RNA-activity runs
- `UTR_BassetVL` is the right default for promoter / UTR work until a stronger architecture is added
- 5'UTR should be treated as two separate benchmarks:
  - `5'UTR polysome`
  - `5'UTR Hani RNA activity`

### 3. Artifact reality

Important practical finding:

- the checked-in repo currently contains W&B metadata under `src/learn/wandb/`
- but the referenced `src/learn/local_artifacts/...` trees are not present in the repo snapshot I can see
- the W&B summaries still preserve the intended artifact paths, for example:
  - promoter: `/home/minhang/synBio_AL/boda2_EU/src/learn/local_artifacts/promoter/sweep/sept10_sweep/...`
  - 3'UTR: `/home/minhang/synBio_AL/boda2_EU/src/learn/local_artifacts/utr3/sweep/june26_utr3_lib1/...`
  - 5'UTR: `/home/minhang/synBio_AL/boda2_EU/src/learn/local_artifacts/utr5/sweep/hani_sweep_jul14/...`

Conclusion:

- current experiment recovery depends more on W&B metadata than local artifact presence
- the repo needs a manifest layer for experiment recovery

### 4. Current graph layer is already generic enough

`boda/graph/cnn_prediction.py` is model-agnostic enough for new model classes.

As long as a model exposes:

- `forward`
- `criterion`
- `n_outputs`

then `CNNBasicTraining` and `CNNTransferLearning` can already train it.

Conclusion:

- adding a ResNet-style model does not require a graph rewrite first
- the bigger need is model organization and HPO config organization

## High-Priority Repo Organization Updates

These are the most valuable structural improvements for the HPO goal.

### A. Split configs by CRE family, target family, and model family

Current problem:

- the previous authored config layout was effectively flat
- configs did not make target-family-vs-model-family comparisons explicit
- model-to-model comparisons on the same task would have been easy to scatter across unrelated paths

Recommended structure:

- use `src/learn/configs/<cre_family>/<target_family>/<model_family>/<config>.yml`
- current concrete layout:
  - `src/learn/configs/enhancer/malinois_mpra/basset_branched/`
  - `src/learn/configs/promoter/deboer_core/utr_bassetvl/`
  - `src/learn/configs/utr5/hani_rna_activity/utr_bassetvl/`
  - `src/learn/configs/utr3/hani_rna_activity/utr_bassetvl/`
- future model families should be added as sibling directories under the same `cre_family/target_family`
- when two configs are intended for direct comparison, keep the filename stem the same except for the `model_family` token

Benefits:

- makes model-vs-task HPO explicit
- separates biologically different 5'UTR targets that currently share too much mental namespace
- avoids mixing biological task naming with architecture naming
- makes it easy to add one new model family at a time
- makes cross-model comparisons much easier to audit later

### B. Add a best-run manifest layer

Current problem:

- best runs are remembered through notebook memory, W&B project names, and timestamps
- artifacts are not centrally indexed

Recommended addition:

- `plan/phase1_lib1/learn/best_runs_snapshot.md` later, for human-readable summaries
- `src/learn/run_registry/` or `metadata/run_registry/`
- one CSV or YAML manifest with columns like:
  - `cre_part`
  - `dataset`
  - `project`
  - `sweep_id`
  - `run_id`
  - `timestamp`
  - `config_path`
  - `model_module`
  - `graph_module`
  - `comparison_group`
  - `val_metric`
  - `artifact_path`
  - `notes`

Benefits:

- lets you recover the HPO story even if local artifacts disappear
- helps separate "latest run" from "best run"

### C. Separate authored configs from generated W&B state

Current problem:

- `src/learn/wandb/` sits beside code and configs
- it is useful for provenance, but noisy

Recommended policy:

- keep `src/learn/wandb/` as archived run cache if you want it
- document clearly that it is generated metadata, not source code
- consider eventually moving it to `metadata/wandb_cache/`

Benefits:

- cleaner code navigation
- less confusion about what should be edited manually

### D. Add one HPO entrypoint README for `src/learn/`

Recommended file:

- `src/learn/README.md`

It should answer:

- which trainer is canonical
- where configs live
- how to launch a sweep
- how to identify best runs
- where artifacts should go
- where notebooks pick results up from

### E. Create a notebook support layer

Current problem:

- notebooks encode a lot of one-off logic for W&B recovery and artifact lookup

Recommended addition:

- `src/analysis/hpo_results_eval_utils.py`

Functions to include:

- list runs from local W&B cache
- resolve best run by metric
- load run config and summary payloads
- parse `model_saved_path`
- load artifact metadata into a dataframe
- normalize historical artifact roots
- support notebook-side filtering for apples-to-apples model comparisons

Benefits:

- fewer notebook-specific hacks
- easier transition between exploration and production-ish evaluation
- easier re-use across multiple result notebooks

## Subsequent Workstreams

The following are intentionally not part of this reorg pass, but they are still
important next-phase plans and should stay in this roadmap.

### Model Expansion Plan

#### Priority 1: add a ResNet-style sequence backbone

This is the best next architecture addition.

Why:

- stronger inductive bias than a plain 3-layer conv stack
- still compatible with sequence-local motif learning
- much more realistic next step than going straight to a transformer
- should plug into existing graph logic cleanly

Recommended implementation approach:

1. add a new file such as `boda/model/resnet.py`
2. keep the CLI contract consistent with current model modules
3. expose the class through `boda/model/__init__.py`
4. create one minimal smoke-test config before adding large sweep spaces

Recommended class design:

- `ResNet1D` or `CRE_ResNet1D`
- inputs: one-hot sequence tensor `[B, 4, L]`
- stack:
  - stem conv
  - residual blocks
  - optional downsampling / pooling
  - global or adaptive pooling
  - linear head
- outputs:
  - single-output regression for promoter / UTR
  - multi-output regression for enhancer

CLI hyperparameters to support:

- `stem_channels`
- `stem_kernel_size`
- `block_channels`
- `n_blocks`
- `block_kernel_size`
- `downsample_every`
- `pool_type`
- `pool_output_size`
- `fc_channels`
- `n_linear_layers`
- `linear_dropout_p`
- `activation`
- `use_batch_norm`
- `use_weight_norm`
- `loss_criterion`

#### Priority 2: do not change graph classes yet

Recommendation:

- keep `CNNBasicTraining` and `CNNTransferLearning`
- only generalize them if a new model actually breaks assumptions

This avoids premature refactoring.

#### Priority 3: keep transformer as a later branch

Reason:

- more hyperparameters
- weaker immediate interpretability / debugging path
- more likely to complicate HPO before the repo organization is ready

### Enhancer Reboot Plan

Enhancer should be the top priority once the reorg / recovery layer is stable.

#### Default baseline to use first

Use:

- `BassetBranched`
- `CNNTransferLearning`
- `L1KLmixed`

This is the strongest historical direction from the Malinois BO table.

#### Comparator model to keep

Also include:

- `BassetVL`
- `CNNTransferLearning`

This gives you a lighter baseline and protects against overcommitting to branching complexity.

#### Recommended enhancer HPO phases

##### Phase 1: reproduce baseline families on current enhancer data

Goal:

- verify the pipeline still works end-to-end
- confirm that the best old model family still dominates on current data

Models:

- `BassetBranched`
- `BassetVL`

Keep transfer learning on for both if using Malinois initialization.

##### Phase 2: narrow enhancer search around winning family

If `BassetBranched` still wins:

- focus HPO on:
  - dropout
  - number of branched layers
  - branched width
  - LR
  - weight decay
  - frozen epochs
  - reverse complements

If `BassetVL` is surprisingly competitive:

- treat it as a stronger simpler baseline before adding ResNet

##### Phase 3: introduce ResNet as challenger

Only after:

- baseline enhancer HPO is reproducible
- artifact/run recovery is documented

Then compare:

- `BassetBranched`
- `BassetVL`
- `ResNet1D`

### MattLee Enhancer Dataset Plan

Dataset:

- `/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/single_part_variant_level/enhancers/20251218_np_fastq1_500000NPreads_enh_variants_bc_sum_avg_expression.txt`

Observed columns:

- `Enhancers`
- `DNA_Counts_Sum`
- `RNA_Counts_Sum`
- `n_barcodes`
- `RNA_DNA_Ratio_raw`
- `RNA_DNA_Ratio_log10_scaled`
- `RNA_DNA_Ratio_log10_scaled_zscore`

#### Immediate evaluation strategy

First decide whether to do:

- zero-shot evaluation of Malinois / pretrained enhancer model
- or direct fine-tuning / retraining

Realistically, because the output schema differs from the original three-cell-line enhancer task, the first useful test is probably:

1. inspect sequence lengths and filtering needs
2. build a single-target enhancer datamodule variant or adapter
3. evaluate a pretrained enhancer backbone as initialization, not necessarily as direct prediction
4. compare transfer learning against training from scratch

#### No-flank scratch comparison

New comparison path:

- config: `src/learn/configs/enhancer/bashor_in_house/lib1_enhancer_fastqs1_5__scratch_no_flank_basic__bayes.yml`
- launcher: `src/learn/launch/lib1_enhancer_fastqs1_5_scratch_no_flank_sweep.sh`
- source CSV: `/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/single_part_variant_level/enhancers/L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.csv`
- learn-ready table: `src/learn/derived_data/enhancer/bashor_in_house/lib1_fastqs1_5_0filtered_out__learn_ready.tsv`

Rationale:

- the previous in-house enhancer scratch configs use `Lib1EnhancerDataModule` defaults, which pad short enhancer inserts to `600` bp with BODA/Malinois MPRA flanking sequence
- the no-flank comparison uses the same learn-ready target table but sets `padding_mode: neutral`, `padded_seq_len: 216`, and `input_len: 216`
- this is not literally variable-length training; the raw enhancer lengths vary, so fixed-length models still need a neutral tensor pad
- neutral `N` bases encode as zeros, avoiding injected biological flank sequence while keeping BassetVL and ResNet shape-compatible

Initial smoke check:

- split sizes: train `3830`, val `479`, test `479`
- batch shape: `[batch, 4, 216]`
- `BassetVL(input_len=216)` and `ResNet1DRegressor(input_len=216)` both forward successfully

## 5'UTR Task Split

This split should become explicit in the repo and docs.

### 5'UTR polysome

Characteristics:

- legacy but very strong signal
- target is polysome / ribosome-load style activity
- main data module: `UTR_Polysome_MPRA_DataModule`
- key notebook: `tutorials/get_HPO_5utr_polysome.ipynb`
- fixed-run launcher: `src/learn/fixed_utr_train.sh`
- W&B project recovered from notebook: `boda2_EU-src`

Why it matters:

- this is currently the strongest 5'UTR result family in the repo
- it demonstrates that the current codebase can achieve high within-library and useful cross-library generalization

### 5'UTR Hani RNA activity

Characteristics:

- newer processed-library workflow
- target is RNA activity
- main data module: `HaniGoozardi_RNA_Activity_DataModule`
- main config: `src/learn/configs/utr5/hani_rna_activity/utr_bassetvl/utr5__hani_rna_activity__utr_bassetvl__bayes.yml`
- W&B project: `utr5_hani_rna_activity`

Why it matters:

- this is the cleaner modern workflow
- but it should not overwrite or replace the older polysome benchmark in planning documents

### Hani observed-head branched Lib1 update

Status as of 2026-05-04:

- implemented observed-head wide Lib1 preprocessing for branched Hani UTR models
- implemented `UTR3_Branched_RNA_Activity_DataModule` and `UTR5_Branched_RNA_Activity_DataModule`
- model baseline: `BassetBranched`
- graph: `CNNBasicTraining`
- monitored metric: `epoch_end_val_pearson_r2`

Dataset sizes:

- 3'UTR Lib1 observed heads: `c1`, `c2`, `c4`, `c6`, `c13`, `c17`
  - train `22741`, val `2843`, test `2842`
- 5'UTR Lib1 observed heads: `c1`, `c2`, `c4`, `c6`, `c17`
  - train `17288`, val `2161`, test `2160`

Stage 1 sweep results:

- 3'UTR sweep `54r4667a`
  - runs completed: `32`
  - best run: `it06cy6q`
  - best `epoch_end_val_pearson_r2`: `0.4278`
  - test `test_pearson_r2`: `0.4163`
  - test mean Pearson: `0.6427`
- 5'UTR sweep `5wraz7oh`
  - runs completed: `32`
  - best monitored run: `j4z89e01`
  - best `epoch_end_val_pearson_r2`: `0.5067`
  - test `test_pearson_r2`: `0.4636`
  - test mean Pearson: `0.6802`
  - best test run in the same sweep: `o4ipczqg`, test `test_pearson_r2` `0.4717`

Metric note:

- `epoch_end_val_pearson_r2` is Pearson correlation squared after flattening all outputs
- this is the historical repo HPO metric, not standard coefficient-of-determination R2
- standard regression R2 is logged separately as `*_cod_r2`

Modeling interpretation:

- reverse-complement augmentation was harmful for both UTR branched sweeps, consistent with UTRs being directional regulatory elements for RNA processing / translation
- enhancer RC augmentation should stay a separate empirical choice because enhancer activity is less tied to transcript direction, but flanking/vector context still matters
- the 5'UTR branched baseline is strong enough that a branched `UTR_BassetVL` rewrite is not the immediate next blocker

Recommended next phase:

1. run focused Stage 2 HPO for both UTRs with `use_reverse_complements: false`
2. keep `BassetBranched` as the first baseline while narrowing LR, weight decay, branch depth, branch width, and dropout
3. after Stage 2, rerun the top two or three configs with multiple seeds
4. only then compare against branched `UTR_BassetVL` or ResNet-style alternatives

### Important blocker

The local file you referenced:

- `/home/minhang/synBio_AL/boda2_EU/tutorials/malinois_artifacts__20211113_021200__287348.tar.gz`

does not appear to exist in the local repo snapshot I can see.

Implication:

- either use the upstream public Malinois artifact URL
- or locate the tarball elsewhere on disk before wiring notebook evaluation to a local path

## Notebook Support Plan

Immediate notebook objective:

- migrate repeated W&B cache / artifact recovery logic into `src/analysis/hpo_results_eval_utils.py`

Recommended first notebook updates:

1. switch existing result-inspection notebooks to import the shared utility layer
2. build dataframe views from local W&B cache instead of repeating ad hoc loaders
3. add comparison filters keyed by task family and model family
4. write best-run selections back into the manifest layer

## Concrete Next Actions

Recommended execution order:

1. reorganize authored configs on disk by `cre_family/target_family/model_family`
2. keep `src/learn/README.md` and `src/learn/configs/README.md` aligned with the new layout
3. add a best-run manifest with explicit comparison-group metadata
4. move notebook-side run recovery into `src/analysis/hpo_results_eval_utils.py`
5. update analysis notebooks to consume the shared utilities and manifest

## Decisions To Lock In Soon

These decisions will remove a lot of future confusion:

- one canonical artifact root
- one canonical location for generated W&B cache
- one run registry file for "best known" runs
- one naming convention for configs that includes CRE family, target family, and model family
- one comparison-group convention so cross-model benchmarks are easy to recover later

## Summary

The repo is already strong enough to restart HPO without a major refactor.

What it needs most is:

- clearer experiment bookkeeping
- model-family-aware config organization
- one manifest for best runs
- one reusable notebook support layer for result evaluation

That sequence gives you the highest chance of comparing model families cleanly without losing track of what already worked.
