# Best Runs Snapshot

This file is a lightweight human-readable registry of the best-known runs recovered so far.

It separates tasks that use different targets, even when they share the same CRE part.

## Enhancer

Source:

- `tutorials/malinois_BO_Sup_table_3.txt`

Best known historical family:

- model: `BassetBranched`
- graph: `CNNTransferLearning`
- loss: `L1KLmixed`
- best row timestamp: `20240104_071417`
- mean test score across K562 / HepG2 / SKNSH: about `0.8867`

Strong comparator:

- model: `BassetVL`
- graph: `CNNTransferLearning`
- best row timestamp: `20240106_024527`
- mean test score across K562 / HepG2 / SKNSH: about `0.8861`

## Promoter

Source:

- local W&B cache in `src/learn/wandb`

Best cached run:

- project: `promoter_optimization`
- run: `run-20250910_164516-404zkdns`
- model: `UTR_BassetVL`
- graph: `CNNBasicTraining`
- `epoch_end_val_r2`: `0.4119`

## 3'UTR RNA Activity

Source:

- local W&B cache in `src/learn/wandb`

Best cached run:

- project: `utr3_rna_activity_optimization`
- run: `run-20250617_105607-j94k79zh`
- model: `UTR_BassetVL`
- graph: `CNNBasicTraining`
- `epoch_end_val_r2`: `0.4511`

## 3'UTR Hani Observed-Head Branched RNA Activity

Source:

- W&B sweep launched from `src/learn/launch/utr3_hani_basset_branched_sweep.sh`
- local run registry: `src/learn/run_registry/runs.csv`
- sweep id: `54r4667a`

Dataset:

- data module: `UTR3_Branched_RNA_Activity_DataModule`
- processed table: `3UTR_lib1_branched_observed_heads.csv`
- observed heads: `c1`, `c2`, `c4`, `c6`, `c13`, `c17`
- split sizes: train `22741`, val `2843`, test `2842`

Best Stage 1 run:

- project: `utr3__hani_rna_activity__scratch__basset_branched`
- run: `it06cy6q`
- model: `BassetBranched`
- graph: `CNNBasicTraining`
- monitor: `epoch_end_val_pearson_r2`
- best monitored value: `0.4278`
- test `test_pearson_r2`: `0.4163`
- test mean Pearson: `0.6427`

Stage 1 interpretation:

- `epoch_end_val_pearson_r2` is Pearson correlation squared after flattening outputs, not standard coefficient-of-determination R2
- standard coefficient-of-determination is logged separately as `*_cod_r2`
- reverse-complement augmentation underperformed for this directional UTR task
- Stage 2 should fix `use_reverse_complements: false` and run a narrower HPO around the best Stage 1 region

## 5'UTR Hani RNA Activity

Source:

- local W&B cache in `src/learn/wandb`

Best cached run:

- project: `utr5_hani_rna_activity`
- run: `run-20250714_100009-2z7reh8i`
- model: `UTR_BassetVL`
- graph: `CNNBasicTraining`
- `epoch_end_val_r2`: `0.5664`

Important note:

- this is not the same target as the older polysome-based 5'UTR work

## 5'UTR Hani Observed-Head Branched RNA Activity

Source:

- W&B sweep launched from `src/learn/launch/utr5_hani_basset_branched_sweep.sh`
- local run registry: `src/learn/run_registry/runs.csv`
- sweep id: `5wraz7oh`

Dataset:

- data module: `UTR5_Branched_RNA_Activity_DataModule`
- processed table: `5UTR_lib1_branched_observed_heads.csv`
- observed heads: `c1`, `c2`, `c4`, `c6`, `c17`
- split sizes: train `17288`, val `2161`, test `2160`

Best Stage 1 runs:

- best monitored run: `j4z89e01`
  - project: `utr5__hani_rna_activity__scratch__basset_branched`
  - model: `BassetBranched`
  - monitor: `epoch_end_val_pearson_r2`
  - best monitored value: `0.5067`
  - test `test_pearson_r2`: `0.4636`
  - test mean Pearson: `0.6802`
- best held-out test run in the same sweep: `o4ipczqg`
  - best monitored value: `0.5011`
  - test `test_pearson_r2`: `0.4717`
  - test mean Pearson: `0.6863`

Per-head test Pearson for `j4z89e01`:

- `c1`: `0.706`
- `c2`: `0.715`
- `c4`: `0.588`
- `c6`: `0.721`
- `c17`: `0.671`

Stage 1 interpretation:

- the observed-head branched model is competitive enough that `UTR_BassetVL` branching is not an immediate blocker
- `c4` is the weakest 5'UTR head, but still carries useful signal
- reverse-complement augmentation underperformed here too; Stage 2 should fix `use_reverse_complements: false`
- recommended next HPO: focused `BassetBranched` Stage 2 with about `64` runs per UTR, then multi-seed confirmation of the best few configs

## 5'UTR Polysome

Source:

- `tutorials/get_HPO_5utr_polysome.ipynb`
- `src/learn/fixed_utr_train.sh`

Recovered task identity:

- W&B project: `boda2_EU-src`
- data module: `UTR_Polysome_MPRA_DataModule`

Recovered sweep IDs:

- `egfp_1`: `rp7qguqc`
- `egfp_2`: `awnbbtop`
- `mcherry_1`: `4mxeeug3`
- `mcherry_2`: `50qg6ejn`

Recovered top within-library results from the notebook:

- `egfp_1`
  - top artifact timestamp: `20250415_101214`
  - top R²: about `0.9459`
- `egfp_2`
  - top artifact timestamp: `20250416_123247`
  - top R²: about `0.9006`
- `mcherry_1`
  - top artifact timestamp: `20250417_180941`
  - top R²: about `0.8091`
- `mcherry_2`
  - top artifact timestamp: `20250420_042012`
  - top R²: about `0.8775`

Interpretation:

- the polysome benchmark is a first-class 5'UTR success case and should stay visible in planning
- it should not be collapsed into the `utr5_hani_rna_activity` line item

## Immediate Use

When deciding what to work on next:

- use the enhancer row to choose the first reboot baseline
- use the promoter / UTR rows to preserve best-known defaults
- use the 5'UTR polysome rows as evidence that 5'UTR work has two separate target families in this repo
