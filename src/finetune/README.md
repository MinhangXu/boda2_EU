# `src/finetune` Guide

This directory holds downstream adaptation and learning-curve experiments. It
is the bridge between pretrained models promoted from `src/learn` and the
analysis notebooks under `tutorials/lib1_tasks`.

## Mental Model

Use `src/finetune` for executable fine-tuning workflows:

- load a pretrained artifact or checkpoint
- define a public or in-house adaptation split
- run transfer-learning or learning-curve sweeps
- write per-run and combined summaries under `learning_curve/`
- hand results to notebooks for interpretation and decision-making

Use `tutorials/lib1_tasks` for human-in-the-loop analysis. Do not turn notebooks
into the source of truth for reusable training logic.

## Directory Map

- `finetune_sweep_scripts/`
  - Authored Python runners and shell launch drivers.
  - Task folders keep related `.py`, `.sh`, and local README files together.
- `finetune_sweep_scripts/lib1_enhancer/`
  - Enhancer Lib1 fine-tuning and learning-curve runners.
  - Covers targeted random/HQ splits, barcode-bin sweeps, comparable-bin
    analyses, exact-low-barcode runs, and multiseed aggregation.
- `finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/`
  - Hani/Goodarzi 5'UTR Lib2 Phase 2 fine-tuning from the canonical BODA Lib1
    pretrained `ResNet1DRegressor` run `1mmy39ku`.
  - See `../../plan/finetune/parade_released_checkpoint_eval_and_finetune_plan_may2026.md`
    and `../../plan/finetune/notion_connect_hani_lib1_2_inhouse_utr_update_may2026.md`
    for the plan and decision context.
- `learning_curve/`
  - Generated run outputs: manifests, per-seed folders, combined summaries,
    predictions, logs, and cache folders.
  - This is ignored as generated state. Promote only small, deliberate summary
    files with `git add -f` when they become curated references.
- `cache/`
  - Generated task caches; ignored.
- `analyses_cache_april1/`
  - Older generated analysis cache; ignored and kept only as local provenance.

## Script Roles

- Python runner scripts define data loading, split construction, model loading,
  fine-tuning, evaluation, and output writing.
- Shell driver scripts define concrete sweep surfaces, GPU assignment, seeds,
  output roots, and preview/smoke behavior.
- Combiner scripts aggregate per-seed or per-run summaries into a `combined/`
  output directory.
- Notebook analyses should consume the generated CSV/JSON summaries and avoid
  re-implementing training logic.

## Current Status Index

| Area | Status | Notes |
|---|---|---|
| Lib1 enhancer fine-tuning | active/reference | Scripts are now grouped under `finetune_sweep_scripts/lib1_enhancer/`; next cleanup target is extracting shared utilities for checkpoint loading, splits, metrics, and aggregation. |
| Hani 5'UTR Lib2 Phase 2 transfer | active | Implemented under `hani_lib1_in_house_lib1_5Prime/`; analysis lives in `tutorials/lib1_tasks/fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/`. |
| `learning_curve/` result trees | generated | Keep local for inspection, but treat as output state rather than source code. |
| reusable fine-tune library | pending | Preferred next refactor is a small `src/finetune/lib/` package for shared sequence, split, checkpoint, metric, training, and summary helpers. |

## Recommended Future Structure

If the script families keep growing, move shared logic into:

```text
src/finetune/lib/pathing.py
src/finetune/lib/sequences.py
src/finetune/lib/checkpoints.py
src/finetune/lib/splits.py
src/finetune/lib/metrics.py
src/finetune/lib/training.py
src/finetune/lib/summaries.py
```

Keep task-specific runners thin. A good runner should mostly define the
experiment surface and delegate reusable mechanics to `src/finetune/lib/`.
