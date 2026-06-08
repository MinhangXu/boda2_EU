# FivePrime Fine-Tuning Analysis Index

This directory holds human-in-the-loop analysis for 5 Prime UTR transfer and
in-house fine-tuning experiments connected to
`src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/`.

## Notebook Status

| Notebook | Reads | Role | Status |
|---|---|---|---|
| `hani_utr5_lib2_phase2_finetune_analysis_may2026.ipynb` | `src/finetune/learning_curve/hani_utr5_lib2_resnet1d_1mmy39ku_phase2_may2026` | First Phase 2 Lib2 fine-tune pass. | historical, superseded by v2 |
| `hani_utr5_lib2_phase2_v2_finetune_analysis_may2026.ipynb` | `src/finetune/learning_curve/hani_utr5_lib2_resnet1d_1mmy39ku_phase2_v2_may2026` | Validation-first Phase 2 v2 decision notebook. | canonical |
| `inhouse_utr5_parade_resnet_small_hpo_analysis_jun2026.ipynb` | `src/finetune/learning_curve/inhouse_utr5_parade_resnet_small_hpo_jun2026` and `src/finetune/learning_curve/inhouse_utr5_parade_resnet_downsample_top_configs_jun2026` | In-house FivePrime PARADE vs BODA ResNet HPO and downsample follow-up. | canonical |

## Current Takeaways

- Phase 2 v2 improves Lib2 validation performance relative to the pretrained
  Lib1 model while paying a modest Lib1-retention cost.
- Phase 3 scratch comparison lives in
  `tutorials/lib1_tasks/pretraining_CRE_public_data/` because it is a
  `src/learn` HPO analysis, not a fine-tune run.
- The June in-house notebook is the current place to compare PARADE and BODA
  ResNet behavior on in-house FivePrime measurements.

## Output Policy

Keep `plots/<analysis_name>/` for curated notebook outputs. Keep generated run
roots under ignored `src/finetune/learning_curve/`. Raw predictions and
per-epoch diagnostics should stay generated/local unless a small curated sample
is explicitly needed for a test or figure.
