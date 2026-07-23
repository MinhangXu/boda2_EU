# Lib1 Dedup Stage 2 To Targeted 3'UTR Handoff

**Status date:** 2026-07-13
**Purpose:** give the next Codex task one focused starting point without
requiring it to reconstruct the Stage 1/2 conversation.

## Current State

Stage 2 is complete and analyzed:

- 660/660 analysis cells are complete;
- 66 base configurations form 132 five-fold OOF RC arms and 66 RC pairs;
- pooled five-fold OOF Pearson on raw `log2_RNA_DNA` and `prediction_raw`
  remains primary;
- the audit loader has not been instantiated or scored; and
- no Stage 3 weighted-loss cells have launched.

The user approved a bounded 20-30-configuration 3'UTR UTRBassetVL targeted
HPO. This approval is a decision to design and run the bounded search, not an
authorization to guess its search space, silently choose its cell accounting,
or touch the frozen audit set.

## Read These First

1. [Stage 2 analysis notebook](../../../tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/03_stage2_paired_rc_analysis.ipynb)
2. [Executed Stage 2 notebook](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/executed_notebooks/03_stage2_paired_rc_analysis__executed.ipynb)
3. [Stage 2 decision report](lib1_dedup_stage2_analysis_report_july2026.md)
4. [Pre-Stage-2 protocol amendment](lib1_dedup_pre_stage2_protocol_amendment_july2026.md)
5. [Full campaign plan](lib1_dedup_phase1_hpo_rerun_plan_july2026.md)
6. [Current campaign status](README.md)

The machine-readable evidence is under
`src/learn/outputs/analysis/lib1_dedup_stage2_july2026/`; the prior 10-config
portfolio is frozen in the Stage 2 analysis manifest and its
`__utr3_utrbassetvl_selected_configs` side products.

## Why A Targeted Search Was Approved

The leading UTRBassetVL arm (`86969bcf`, RC off) reached pooled OOF Pearson
0.501, versus 0.331 for the leading RC-off ResNet1D arm. UTRBassetVL also wins
the best-arm comparison under RC on (0.399 versus 0.332). This is evidence of
better best-config headroom, not broad architecture dominance: the two route
distributions have nearly identical medians and the UTRBasset result is
variable across configs and folds.

Within the 10 UTRBassetVL policies:

- the tested learning rates span 0.00005 to 0.001863;
- the winner is near the old 0.002 upper boundary;
- learning rate has a positive exploratory association with best validation
  Pearson in all five folds under both RC policies;
- higher learning rate is associated with lower training MSE; and
- no policy systematically reaches the 220-epoch cap.

Therefore simply increasing patience or total epochs is not the indicated
next move. The signal supports a bounded optimizer/regularization search. The
reported Spearman p-values treat repeated config-fold cells as independent and
are descriptive only; do not use them as publication inference.

## Binding Constraints For The Next Task

Before generating a launch manifest, freeze in a dated amendment or design
record:

1. the exact number of proposed configurations (within 20-30);
2. every search dimension and bound, with special attention to expanding the
   old learning-rate ceiling;
3. whether the bounded search trains every config across all five folds and
   both RC policies, or uses a cheaper screening phase followed by a declared
   promotion rule;
4. the resulting number of training cells—"20-30 configs" alone is not a cell
   count;
5. selection metrics and tie/stability rules, retaining pooled OOF Pearson as
   primary when a complete five-fold arm exists;
6. how new candidates will be compared with the existing 10 UTRBassetVL and
   10 ResNet1D Stage 2 candidates without erasing provenance; and
7. W&B project/group/run-name conventions and local output paths.

Use only the existing deduplicated 3'UTR data product and development splits.
Keep model seed, target definition, high-barcode validation estimand, raw OOF
export, and audit isolation consistent with Stage 2 unless a change is
explicitly justified and labeled. Do not score or instantiate the audit
loader, and do not generate the Stage 3 weighted manifest yet.

The next task should first produce a dry-run manifest, validation report, and
one-row pilot command. Full execution remains a user-run operation unless the
user explicitly asks otherwise.

## Decisions That Remain Outside This Targeted HPO

- The present 3'UTR Stage 3 five are provisional until the targeted-HPO result
  is analyzed.
- The 5'UTR ResNet1D diversity slot remains a separate judgment.
- The numerical tolerance for "no material" RMSE/COD degradation is not yet
  frozen.
- Enhancer transfer has historically supported barcode-weighted MSE, but the
  standardized `CNNBassetBranchedScopedTransfer` path still needs a tested
  weights-required integration before Stage 3.
- The audit remains inaccessible until the final Stage 3 and refit protocol
  are preregistered.

## Suggested Next-Task Prompt

> Using the Stage 2-to-targeted-3'UTR handoff as the entry point, design and
> implement the bounded 3'UTR UTRBassetVL targeted HPO protocol. First freeze
> the exact search space, screening/promotion design, cell accounting, OOF
> selection rule, W&B organization, and audit-isolation checks in a dated
> amendment. Then generate and validate a dry-run manifest and give me a
> one-row pilot command. Do not launch the full campaign or instantiate the
> audit loader.
