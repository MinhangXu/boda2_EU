# Plan Directory

This directory is the repo's planning and decision center. Keep executable code
in `src/`, exploratory notebooks in `tutorials/`, and the durable reasoning
that coordinates them here.

`README_loco.md` stays at the repo root as broad local memory. `plan/` should
hold time-bounded implementation plans, experiment roadmaps, decision records,
and follow-up task maps.

## Organization

- `phase1_lib1/`
  - Use for single-part Library 1 modeling plans, including the `learn/` and
    `finetune/` subfolders that mirror the earlier plan layout.
  - Holds the Phase 1 thread matrix, part-specific gap plans, pretraining/HPO
    notes, transfer-learning plans, and model-promotion records.
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
| [`phase1_lib1/dedup_phase1_rerun_july2026/README.md`](phase1_lib1/dedup_phase1_rerun_july2026/README.md) | reference | `src/learn`, `src/analysis`, `tutorials/lib1_tasks/pretrain_CRE_inhouse_data` | Canonical status and navigation for the July 2026 deduplicated Lib1 campaign; Stages 1--4, the fixed-budget final refits, and the one-time locked final-test evaluation are complete. |
| [`phase1_lib1/tac_campaign_july2026/README.md`](phase1_lib1/tac_campaign_july2026/README.md) | active | `src/analysis`, `plan/phase1_lib1/tac_campaign_july2026` | Post-TAC decision queue, report source, and follow-up priorities built from the closed deduplicated mean-expression campaign. |
| [`phase1_lib1/barcode_count_modeling_july2026/README.md`](phase1_lib1/barcode_count_modeling_july2026/README.md) | active | `raw_data_bashor/mpra_eda_tool`, later `src/learn` | Cross-repo roadmap for Lib1 barcode-count EDA, precise 3A/3B estimands, repaired spread targets, Poisson/NB likelihoods, hierarchical modeling gates, and the link to the fixed dedup mean-expression baseline. |
| [`repo_hygiene/git_data_and_artifact_policy.md`](repo_hygiene/git_data_and_artifact_policy.md) | reference | repository-wide | Rules for sensitive data, exact split manifests, generated results, oversized artifacts, local-only July commits, and the deferred move from the public fork to private development. |
| [`phase1_lib1/phase1_library1_thread_matrix_june2026.md`](phase1_lib1/phase1_library1_thread_matrix_june2026.md) | active | `tutorials/lib1_tasks`, `src/learn`, `src/finetune` | Local mirror of the Notion Phase 1 Library 1 board; maps CRE-part threads to training regimes, thread functions, run scripts, notebooks, and Phase 2 gaps. |
| [`phase1_lib1/promoter_phase1.md`](phase1_lib1/promoter_phase1.md) | active | `src/learn`, `src/finetune`, `Core_Promoter_Model` | Split-safe promoter plan for legacy in-house e7/e30 pretraining, Lib1 scratch training, and legacy-to-Lib1 fine-tuning. |
| [`phase1_lib1/learn/hpo_repo_reboot_plan.md`](phase1_lib1/learn/hpo_repo_reboot_plan.md) | partially implemented | `src/learn` | Original HPO repo reorg map; configs, launchers, registry, and generated-state policy are now real enough to treat this as provenance plus remaining cleanup. |
| [`phase1_lib1/learn/best_runs_snapshot.md`](phase1_lib1/learn/best_runs_snapshot.md) | reference | `src/learn/run_registry` | Human-readable best-run snapshot; refresh after major HPO batches or model-promotion decisions. |
| [`phase1_lib1/learn/intron_seelig_hal_pretraining_plan.md`](phase1_lib1/learn/intron_seelig_hal_pretraining_plan.md) | partially implemented | `src/learn`, `boda/data` | Seelig A5SS scalar path exists; richer HAL/paper-comparable phases remain future work. |
| [`phase1_lib1/learn/utr_hani_basset_branched_hpo_illustration_brief.md`](phase1_lib1/learn/utr_hani_basset_branched_hpo_illustration_brief.md) | reference | `src/learn` | Architecture/HPO explanation for Hani UTR BassetBranched runs and presentation material. |
| [`phase1_lib1/learn/evo2_foundation_encoded_seq2expr_plan.md`](phase1_lib1/learn/evo2_foundation_encoded_seq2expr_plan.md) | blocked | `src/foundation/evo2`, `src/learn/evo2`, `boda/*` | Hash-smoke scaffold and cached-embedding training path exist; real local Evo2 extraction is hardware/toolchain gated. |
| [`phase1_lib1/finetune/parade_released_checkpoint_eval_and_finetune_plan_may2026.md`](phase1_lib1/finetune/parade_released_checkpoint_eval_and_finetune_plan_may2026.md) | partially implemented | `src/finetune`, `src/learn`, `tutorials/lib1_tasks` | Phase 1/1.5 evaluation and EDA exist; Phase 2 fine-tuning and Phase 3 scratch-HPO work now have follow-up analysis. |
| [`phase1_lib1/finetune/notion_connect_hani_lib1_2_inhouse_utr_update_may2026.md`](phase1_lib1/finetune/notion_connect_hani_lib1_2_inhouse_utr_update_may2026.md) | active | `src/finetune`, `tutorials/lib1_tasks` | Current May 2026 transfer summary connecting Hani Lib1/Lib2 and in-house FivePrime diagnostics. |
| [`phase1_lib1/finetune/lib1_enhancer_scratch_and_finetune_rehydration_june2026.md`](phase1_lib1/finetune/lib1_enhancer_scratch_and_finetune_rehydration_june2026.md) | active | `src/learn`, `src/finetune`, `tutorials/lib1_tasks` | Enhancer scratch, transfer, barcode-count, and notebook rehydration map. |
| [`active_learning/sequence_landscape_fast_implementation_plan.md`](active_learning/sequence_landscape_fast_implementation_plan.md) | active | `src/analysis/sequence_landscape_adapters` | Implementation plan for sequence-space diagnostics around transfer improvement, retention loss, and in-house placement. |
| [`active_learning/sequence_landscape_active_learning_exploration_exploitation.md`](active_learning/sequence_landscape_active_learning_exploration_exploitation.md) | reference | future active-learning loop | Higher-level acquisition strategy notes for exploration, exploitation, calibration probes, and landscape-aware selection. |
| [`combinatorial/multi_part_training_strategy_june2026.md`](combinatorial/multi_part_training_strategy_june2026.md) | active | future `src/learn`, `src/finetune`, `tutorials/lib1_tasks` | Strategy for segmented, pretrained-encoder, and full-long-sequence combinatorial GRE training. |
| [`repo_hygiene/lib1_tasks_run_analysis_backtracking_checklist_june2026.md`](repo_hygiene/lib1_tasks_run_analysis_backtracking_checklist_june2026.md) | active | `tutorials/lib1_tasks`, generated outputs | Manual checklist for deciding which notebooks/output roots become canonical, local archive, or deletion candidates. |

## June 2026 Modeling Status Matrix

See [`phase1_lib1/phase1_library1_thread_matrix_june2026.md`](phase1_lib1/phase1_library1_thread_matrix_june2026.md)
for the fuller Phase 1 thread taxonomy, run/notebook ledger, and gap register.

| Part class | Pretraining | Fine-tune | From scratch | Current decision surface | Next action |
|---|---|---|---|---|---|
| Enhancer | BODA2/Malinois checkpoint available | Lib1 transfer and barcode-count studies mostly complete | Current in-house scratch evidence is weak | `phase1_lib1/finetune/lib1_enhancer_scratch_and_finetune_rehydration_june2026.md` | Decide which enhancer notebooks to keep canonical; run only the missing scratch comparison if it changes decisions. |
| Promoter | Legacy in-house e7/e30 pretraining exists but needs split-safe rerun | Pending | June Lib1 scratch HPO complete for ResNet1D and BassetVL | `phase1_lib1/phase1_library1_thread_matrix_june2026.md`, `phase1_lib1/promoter_phase1.md` | Analyze scratch HPO, decide modal50 versus allvalid/51-padded policy, then compare against legacy-to-Lib1 fine-tune. |
| 5 Prime UTR | Hani Lib1/Lib2 pretraining complete | Phase 2 v2 and June in-house HPO complete | Phase 3 Lib1+Lib2 scratch complete; in-house exact/modal50 ResNet1D/BassetVL scratch scaffolds added | Phase 2 v2, Phase 3 scratch, June in-house notebooks, and `src/learn/launch/lib1_fiveprime_scratch_resnet1d_sweep.sh` / `src/learn/launch/lib1_fiveprime_scratch_utr_bassetvl_sweep.sh` | Run in-house scratch baselines, then pick the seed checkpoint for combinatorial scenario 2. |
| Intron | Seelig pretraining complete | Pipeline in progress | June Lib1 modal80 ResNet scratch HPO complete | Seelig pretraining notebook plus in-house scratch HPO notebook | Confirm architecture/RC and decide whether Seelig fine-tune is still needed. |
| 3 Prime UTR | Hani pretraining complete | Pending | June Lib1 modal100 ResNet/Basset scratch HPO complete | In-house scratch HPO notebook plus public CRE/PARADE summaries | Synthesize BassetVL focused RC factorial and compare against any length-context/fine-tune branch. |

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
