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
