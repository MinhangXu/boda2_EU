# `tutorials/lib1_tasks` Guide

This directory is the notebook workspace for Lib1-centered interpretation,
pretraining decisions, fine-tuning diagnostics, and in-house transfer analysis.
It should explain what happened, what was learned, and what should happen next;
it should not be the primary home for reusable training code.

## Workflow Contract

Use this split:

- `plan/`
  - durable reasoning, status, and decision criteria
- `src/learn/`
  - pretraining/HPO code, launchers, configs, run registry, and generated HPO
    roots
- `src/finetune/`
  - transfer-learning and learning-curve scripts plus generated finetune output
- `tutorials/lib1_tasks/`
  - notebooks and plots that interpret those runs and turn them into decisions

When adding a notebook, make the first markdown cell name the question it
answers, the run/output paths it reads, and the decision it is meant to support.

## Directory Map

- `in_house_EDA/`
  - In-house Lib1 data checks before modeling claims.
  - Current UTR EDA notebook:
    `in_house_utr_eda_may2026.ipynb`.
  - Plot outputs live under `in_house_EDA/plots/`.
- `pretraining_CRE_public_data/`
  - Public-data pretraining and HPO interpretation.
  - Covers public CRE HPO summaries, Hani UTR architecture choices, PARADE
    checkpoint evaluation, Seelig intron decisions, and Hani 5'UTR Lib1+Lib2
    Phase 3 scratch-HPO analysis.
  - Presentation-ready CSV/PNG/SVG outputs live under `presentation_plots/`.
  - See `pretraining_CRE_public_data/README.md` for canonical notebook status.
- `pretrain_CRE_inhouse_data/`
  - In-house Lib1 one-shot transfer diagnostics and from-scratch HPO
    interpretation for promoter, intron, and 3 Prime UTR.
  - See `pretrain_CRE_inhouse_data/README.md` for canonical notebook status.
- `fine_tuning/enhancer_finetune_w_boda_pretrain/`
  - Enhancer Lib1 fine-tuning and learning-curve notebooks connected to
    `src/finetune/finetune_sweep_scripts/lib1_enhancer/`.
- `fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/`
  - Hani 5'UTR Lib2 Phase 2 fine-tuning and in-house FivePrime proxy analysis
    connected to `src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/`.
  - See `fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/README.md`
    for Phase 2, Phase 2 v2, and June in-house notebook status.
- `fine_tuning/markdown/`
  - Short written interpretations and collaborator-facing summaries from older
    enhancer fine-tuning passes.
- `reusable_plots/`
  - Stable plot assets that notebooks may reuse for presentation context.

Older root-level notebooks in this folder are historical and can be moved into
one of the thematic subdirectories when touched again.

## Current Status Index

| Area | Status | Primary notebooks |
|---|---|---|
| In-house UTR EDA | active/reference | `in_house_EDA/in_house_utr_eda_may2026.ipynb` |
| Public CRE/Hani UTR pretraining | active/reference | `pretraining_CRE_public_data/parade_released_checkpoint_eval_may2026.ipynb`, `pretraining_CRE_public_data/utr_hani_architecture_choices_may2026.md` |
| Seelig intron pretraining | active/reference | `pretraining_CRE_public_data/intron_seelig_a5ss_sd1_pretraining_hpo_decision_may2026.ipynb` |
| PARADE/BODA checkpoint comparison | reference | `pretraining_CRE_public_data/parade_released_checkpoint_eval_may2026.ipynb` |
| Hani 5'UTR Phase 2 fine-tuning | active/canonical v2 | `fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/hani_utr5_lib2_phase2_v2_finetune_analysis_may2026.ipynb` |
| Hani 5'UTR Phase 3 scratch HPO | active | `pretraining_CRE_public_data/hani_utr5_lib1_lib2_phase3_scratch_hpo_analysis_may2026.ipynb` |
| In-house 5'UTR PARADE/BODA HPO | active | `fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/inhouse_utr5_parade_resnet_small_hpo_analysis_jun2026.ipynb` |
| In-house 5'UTR scratch baseline | ready-to-run scaffold | `pretrain_CRE_inhouse_data/README.md`; launchers `../../src/learn/launch/lib1_fiveprime_scratch_resnet1d_sweep.sh`, `../../src/learn/launch/lib1_fiveprime_scratch_utr_bassetvl_sweep.sh` |
| Promoter/Intron in-house one-shot | active/reference | `pretrain_CRE_inhouse_data/promoter_intron_inhouse_pretrained_eval_may2026.ipynb` |
| In-house all-part scratch HPO | active/canonical | `pretrain_CRE_inhouse_data/lib1_inhouse_scratch_hpo_best_models_june2026.ipynb` |
| Enhancer Lib1 fine-tuning | active/reference | notebooks under `fine_tuning/enhancer_finetune_w_boda_pretrain/` |

## Cleanup Companion

Use
`../../plan/repo_hygiene/lib1_tasks_run_analysis_backtracking_checklist_june2026.md`
as the manual review ledger for deciding which notebooks and generated run
roots should be canonical, local archive, or deletion candidates.

Use
`../../plan/phase1_lib1/phase1_library1_thread_matrix_june2026.md`
as the Phase 1 thread matrix for assigning CRE part, Training Regime, Thread
Function, run-script provenance, analysis notebook provenance, and Phase 2 gap
status.

## Output Hygiene

- Keep generated plots near the notebook family that produced them.
- Prefer `presentation_plots/<analysis_name>/` for multi-file analysis outputs.
- Commit small CSV/PNG/SVG decision artifacts when they are useful provenance.
- Keep large raw predictions, model checkpoints, W&B logs, and per-seed run
  outputs in ignored generated-output folders under `src/learn` or
  `src/finetune`.

## Naming Pattern

Prefer names that include:

- biological task: `enhancer`, `utr5`, `intron`, `fivePrime`
- experiment phase or purpose: `pretraining`, `finetune`, `eda`, `decision`
- date when the notebook is a decision snapshot: `may2026`, `apr2026`

Example:

```text
hani_utr5_lib1_lib2_phase3_scratch_hpo_analysis_may2026.ipynb
```
