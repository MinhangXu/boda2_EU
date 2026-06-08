# Plan Directory

This directory is the repo's planning and decision center. Keep executable code
in `src/`, exploratory notebooks in `tutorials/`, and the durable reasoning
that coordinates them here.

`README_loco.md` stays at the repo root as broad local memory. `plan/` should
hold time-bounded implementation plans, experiment roadmaps, decision records,
and follow-up task maps.

## Organization

- `learn/`
  - Mirrors `src/learn/`.
  - Use for pretraining, HPO, run-registry, dataset-preparation, model-promotion,
    and foundation-encoded sequence-to-expression plans.
- `finetune/`
  - Mirrors `src/finetune/`.
  - Use for transfer-learning, fine-tuning, public-to-in-house adaptation, and
    downstream evaluation plans that connect scripts with analysis notebooks.
- `active_learning/`
  - Use for acquisition strategy, sequence-landscape diagnostics, exploration
    versus exploitation, and adapters that may span `src/analysis`,
    `src/design`, and future active-learning loops.
- `combinatorial/`
  - Use for multi-part GRE training plans that span part-specific pretrained
    encoders, fusion models, and long-sequence modeling.
- `repo_hygiene/`
  - Use for backtracking checklists, GitHub inclusion decisions, generated-state
    cleanup notes, and context-compression audits.

This split is intentionally a little coarser than the code tree. A plan should
live where the primary decision happens, even if the implementation touches
several folders. Cross-link freely when a plan spans multiple systems.

## Status Vocabulary

- `active`: current driver for near-term implementation or analysis.
- `implemented`: core scaffold or requested phase exists; keep for provenance
  and future iteration.
- `partially implemented`: some phases are done, but acceptance criteria or
  follow-up phases remain open.
- `blocked`: technically planned, but waiting on data, hardware, environment,
  or a decision gate.
- `deferred`: useful later, not the next repo-management priority.
- `reference`: stable context, snapshot, or explanatory brief.

## Current Index

| Plan | Status | Home | Notes |
|---|---|---|---|
| [`learn/hpo_repo_reboot_plan.md`](learn/hpo_repo_reboot_plan.md) | partially implemented | `src/learn` | Original HPO repo reorg map; configs, launchers, registry, and generated-state policy are now real enough to treat this as provenance plus remaining cleanup. |
| [`learn/best_runs_snapshot.md`](learn/best_runs_snapshot.md) | reference | `src/learn/run_registry` | Human-readable best-run snapshot; refresh after major HPO batches or model-promotion decisions. |
| [`learn/intron_seelig_hal_pretraining_plan.md`](learn/intron_seelig_hal_pretraining_plan.md) | partially implemented | `src/learn`, `boda/data` | Seelig A5SS scalar path exists; richer HAL/paper-comparable phases remain future work. |
| [`learn/utr_hani_basset_branched_hpo_illustration_brief.md`](learn/utr_hani_basset_branched_hpo_illustration_brief.md) | reference | `src/learn` | Architecture/HPO explanation for Hani UTR BassetBranched runs and presentation material. |
| [`learn/evo2_foundation_encoded_seq2expr_plan.md`](learn/evo2_foundation_encoded_seq2expr_plan.md) | blocked | `src/foundation/evo2`, `src/learn/evo2`, `boda/*` | Hash-smoke scaffold and cached-embedding training path exist; real local Evo2 extraction is hardware/toolchain gated. |
| [`finetune/parade_released_checkpoint_eval_and_finetune_plan_may2026.md`](finetune/parade_released_checkpoint_eval_and_finetune_plan_may2026.md) | partially implemented | `src/finetune`, `src/learn`, `tutorials/lib1_tasks` | Phase 1/1.5 evaluation and EDA exist; Phase 2 fine-tuning and Phase 3 scratch-HPO work now have follow-up analysis. |
| [`finetune/notion_connect_hani_lib1_2_inhouse_utr_update_may2026.md`](finetune/notion_connect_hani_lib1_2_inhouse_utr_update_may2026.md) | active | `src/finetune`, `tutorials/lib1_tasks` | Current May 2026 transfer summary connecting Hani Lib1/Lib2 and in-house FivePrime diagnostics. |
| [`finetune/lib1_enhancer_scratch_and_finetune_rehydration_june2026.md`](finetune/lib1_enhancer_scratch_and_finetune_rehydration_june2026.md) | active | `src/learn`, `src/finetune`, `tutorials/lib1_tasks` | Enhancer scratch, transfer, barcode-count, and notebook rehydration map. |
| [`active_learning/sequence_landscape_fast_implementation_plan.md`](active_learning/sequence_landscape_fast_implementation_plan.md) | active | `src/analysis/sequence_landscape_adapters` | Implementation plan for sequence-space diagnostics around transfer improvement, retention loss, and in-house placement. |
| [`active_learning/sequence_landscape_active_learning_exploration_exploitation.md`](active_learning/sequence_landscape_active_learning_exploration_exploitation.md) | reference | future active-learning loop | Higher-level acquisition strategy notes for exploration, exploitation, calibration probes, and landscape-aware selection. |
| [`combinatorial/multi_part_training_strategy_june2026.md`](combinatorial/multi_part_training_strategy_june2026.md) | active | future `src/learn`, `src/finetune`, `tutorials/lib1_tasks` | Strategy for segmented, pretrained-encoder, and full-long-sequence combinatorial GRE training. |
| [`repo_hygiene/lib1_tasks_run_analysis_backtracking_checklist_june2026.md`](repo_hygiene/lib1_tasks_run_analysis_backtracking_checklist_june2026.md) | active | `tutorials/lib1_tasks`, generated outputs | Manual checklist for deciding which notebooks/output roots become canonical, local archive, or deletion candidates. |

## June 2026 Modeling Status Matrix

| Part class | Pretraining | Fine-tune | From scratch | Current decision surface | Next action |
|---|---|---|---|---|---|
| Enhancer | BODA2/Malinois checkpoint available | Lib1 transfer and barcode-count studies mostly complete | Current in-house scratch evidence is weak | `finetune/lib1_enhancer_scratch_and_finetune_rehydration_june2026.md` | Decide which enhancer notebooks to keep canonical; run only the missing scratch comparison if it changes decisions. |
| Promoter | Public pretraining complete | Pending | Public scratch/HPO roots exist | `pretraining_CRE_public_data/promoter_intron_inhouse_pretrained_eval_may2026.ipynb` | Compare scratch and fine-tune on in-house promoter data. |
| 5 Prime UTR | Hani Lib1/Lib2 pretraining complete | Phase 2 v2 and June in-house HPO complete | Phase 3 Lib1+Lib2 scratch complete | Phase 2 v2, Phase 3 scratch, and June in-house notebooks | Pick the seed checkpoint for combinatorial scenario 2. |
| Intron | Seelig pretraining complete | Pipeline in progress | Public scratch complete | Seelig pretraining notebook plus promoter/intron one-shot notebook | Resolve in-house one-shot mismatch before fine-tune claims. |
| 3 Prime UTR | Hani pretraining complete | Pending | Public scratch/HPO roots exist | Public CRE/PARADE summary notebooks | Run from-scratch versus fine-tune comparison. |

## Adding A New Plan

Add a plan when a future reader needs the "why", not only the code. Good plan
files answer:

- What question or decision is this work trying to resolve?
- Which code paths, launch scripts, notebooks, data products, and run outputs
  are involved?
- What is already implemented, what is pending, and what would count as a
  promotion or stop decision?
- Where should generated outputs live, and what should not be committed?

If a plan becomes mostly historical, keep it in place and mark it `reference`
or `implemented` here instead of deleting it. The goal is to compress context,
not erase the path that got us here.
