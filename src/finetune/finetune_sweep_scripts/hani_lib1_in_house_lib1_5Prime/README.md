# Hani 5'UTR Lib2 Phase 2 Fine-Tuning

This folder implements Phase 2 from
`../../../../plan/finetune/parade_released_checkpoint_eval_and_finetune_plan_may2026.md`.

The main runner fine-tunes the current canonical BODA 5'UTR ResNet1D artifact
from run `1mmy39ku` on Hani/Goodarzi 5'UTR Lib2.

## Scripts

- `hani_utr5_lib2_finetune.py`
  - Aggregates Lib2 replicate rows to one row per uppercased 50 nt sequence and
    cell head.
  - Creates a deterministic sequence-level train/val/test split.
  - Fine-tunes `1mmy39ku` with conservative transfer settings.
  - Evaluates pretrained and fine-tuned models on Lib2 val/test, Lib1 test
    retention, and in-house exact-length `FivePrime` candidates.
- `run_hani_utr5_lib2_finetune_parallel.sh`
  - Launches one seed per GPU slot and combines outputs.
- `combine_hani_utr5_lib2_outputs.py`
  - Combines per-seed CSV summaries into an output-root `combined/` directory.

## Quick Preview

```bash
PREVIEW_ONLY=1 bash src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/run_hani_utr5_lib2_finetune_parallel.sh
```

## Default Run

```bash
bash src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/run_hani_utr5_lib2_finetune_parallel.sh
```

Useful overrides:

```bash
GPU_IDS="0 1 2 3" SEED_LIST="7 11 13 17" bash src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/run_hani_utr5_lib2_finetune_parallel.sh
UNFREEZE_SCOPES="head_only last_stage_plus_head full" bash src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/run_hani_utr5_lib2_finetune_parallel.sh
TARGET_SCALER_SOURCE="lib2_train" bash src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/run_hani_utr5_lib2_finetune_parallel.sh
```

The default target scaler is `pretrained_lib1_train`, which keeps the model in
the same output coordinate as the source Lib1 pretrained artifact. Use
`TARGET_SCALER_SOURCE=lib2_train` to instead normalize Lib2 targets with the
Lib2 train split.

## Key Outputs

- `lib2_sequence_split_manifest.csv`
- `lib2_sequence_split_audit.csv`
- `model_comparison_summary.csv`
- `per_head_metrics.csv`
- `inhouse_fiveprime_metrics.csv`
- `per_epoch_diagnostics.csv`
- `lib2_test_model_ranking.csv`
- `runs/*/finetuned_model.pt`
