# Lib1 Dedup Phase 1 Rerun (July 2026)

This directory is the canonical navigation and current-status page for the
five-part deduplicated Library 1 rerun. The dated documents beside this file
are durable protocol, explanation, and implementation records; update current
campaign status here first rather than duplicating it across those records.

## Current Status

Stage 1, Stage 2, the bounded targeted 3'UTR HPO, Stage 3, the final refits,
and the one-time locked final-test evaluation are complete. User-facing prose
uses **final test**; historical internal keys, paths, and hashes containing
`audit` are retained for reproducibility. The
first-pass Stage 2 analysis notebook and decision report were completed on
2026-07-13. The 3'UTR protocol was frozen on 2026-07-14 as 24 new
configurations across all five folds and both RC modes (240 cells); all 240
cells and the development-only analysis completed on 2026-07-14. Stage 3 then
completed all 450 weighted cells and its development-only analysis on
2026-07-15. The standardized weighted Enhancer-transfer integration is strict
and tested. The analyzer resolves 450 immutable unweighted plus 450 weighted
cells, all 180 five-fold OOF arms, and five selected part policies. The dated
Stage 3 analysis handoff records the exact selections. On 2026-07-16, 15
fixed-budget all-non-audit refits completed, their checkpoint allowlist was
frozen, and the separately authorized audit scorer evaluated the five natural
audit partitions once. On 2026-07-17 the Stage 4 downsampling contract and
660-row dry-run manifest were frozen and independently validated. By
2026-07-18 all 660 development-only cells, 132 pooled OOF tracks, the frozen
2,000-replicate bootstrap analysis, and the presentation report completed.
No current final-test product was read by Stage 4.

- Stage 1 completed 885 exact-dedup replay rows and 25 paired pre-dedup
  diagnostic rows.
- Stage 2 has 660 complete analysis cells: 50 verified Stage 1 reuse cells and
  610 new launches.
- The completed product contains 132 five-fold out-of-fold arms and 66 paired
  RC configurations.
- The primary metric remains pooled five-fold OOF Pearson on raw
  `log2_RNA_DNA` and `prediction_raw`.
- Inferred Intron masks remain sensitivity categories, not true synthesis
  subset labels.
- The post-Stage-2 Intron estimand audit is complete. The secondary final-audit
  reporting rule was accepted into the dated Stage 3 amendment before any
  audit access; it does not change the frozen natural-mixture audit or Stage 2
  selection.
- All Stage 3 runs and development analyses have `n_test=0`. The audit was
  opened only after the five policies, fixed epochs, 15 refits, and checkpoint
  allowlist were frozen. The primary audit predictor is the three-seed raw
  prediction mean; audit results cannot return to selection or calibration.
- Stage 3 selects Enhancer K562/full transfer with RC on and unweighted loss;
  Promoter, Intron, 3'UTR, and 5'UTR select RC off with barcode-weighted loss.
- Targeted 3'UTR accounting is 240/240 completed cells, 48/48 complete OOF
  arms, and 120/120 fold-level RC pairs. Every provenance record has
  `n_test=0`; no test metric or test prediction was produced.
- The targeted analysis compares 88 total 3'UTR arms with source provenance
  intact. It identifies 27 constant-prediction fold cells, supports RC off as
  the current unweighted default, and supplies seven UTRBassetVL plus three
  ResNet1D members to the exact Stage 3 ten-config 3'UTR portfolio.
- Primary audit Pearson is 0.365249 Enhancer, 0.443849 Promoter, 0.681348
  Intron, 0.452441 3'UTR, and 0.512086 5'UTR. Enhancer is the main
  generalization/calibration limitation (audit COD R2 0.003804).
- Intron audit natural pooled Pearson is 0.681348 versus 0.473334 after
  within-inferred-stratum centering; the residual exact-80 stratum is weakest
  at 0.206120. These remain inferred masks, not verified sublibraries.

## Read Order And Document Roles

1. [Reader guide](lib1_dedup_stage1_to_stage2_reader_guide_july2026.md) —
   plain-language concepts, terminology, and rationale.
2. [Full campaign plan](lib1_dedup_phase1_hpo_rerun_plan_july2026.md) — formal
   scientific contract across campaign stages.
3. [Pre-Stage-2 amendment](lib1_dedup_pre_stage2_protocol_amendment_july2026.md)
   — binding Intron-estimand and challenger-lane additions frozen before
   Stage 2.
4. [Stage 1 implementation checks](lib1_dedup_phase1_stage1_implementation_checks_july2026.md)
   — technical audit appendix for data, manifests, W&B, and launch safety.
5. [Intron estimand and challenge-set protocol](lib1_dedup_intron_estimand_and_challenge_set_protocol_july2026.md)
   — post-Stage-2 diagnosis of composition-assisted pooled correlation,
   validation of the proposed balanced/high-barcode set, and the frozen
   pre-audit secondary reporting rule.
6. [Stage 3 weighted-loss amendment](lib1_dedup_stage3_protocol_amendment_july14_2026.md)
   — binding ten-config portfolios, paired weighted-loss design, selection
   gates, W&B organization, Intron sensitivity rule, and audit isolation.
7. [Post-presentation interpretation addendum](lib1_dedup_post_presentation_interpretation_addendum_july17_2026.md)
   — user-facing terminology, experiment hierarchy, exact final-refit flow,
   target units, the prospective Intron challenge-set boundary, and the
   Enhancer-transfer RC confound.
8. [Stage 4 downsampling amendment](lib1_dedup_stage4_downsampling_protocol_amendment_july17_2026.md)
   — binding tiered portfolio, six-point primary grid, inner-checkpoint/outer-OOF
   design, nested subset seeds, bounded-curve analysis, and final-test isolation.

The plan and amendment jointly define the protocol. The guide explains it;
the implementation-check document records how Stage 1 satisfied it.

## Executable Analysis And Products

- [Stage 2 analysis and decision report](lib1_dedup_stage2_analysis_report_july2026.md)
- [Targeted 3'UTR HPO analysis report](lib1_dedup_utr3_targeted_hpo_analysis_report_july14_2026.md)
- [Stage 1 exact-replay selection notebook](../../../tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/01_exact_replay_selection.ipynb)
- [Intron inferred-mask analysis notebook](../../../tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/02_intron_inferred_mask_strata_analysis.ipynb)
- [Executed Intron inferred-mask notebook with plots](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/executed_notebooks/02_intron_inferred_mask_strata_analysis__executed.ipynb)
- [Stage 2 paired-RC analysis notebook](../../../tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/03_stage2_paired_rc_analysis.ipynb)
- [Executed Stage 2 notebook](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/executed_notebooks/03_stage2_paired_rc_analysis__executed.ipynb)
- [Targeted 3'UTR HPO analysis notebook](../../../tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/04_utr3_targeted_hpo_analysis.ipynb)
- [Executed targeted 3'UTR notebook](../../../src/learn/outputs/analysis/lib1_dedup_utr3_targeted_hpo_july2026/executed_notebooks/04_utr3_targeted_hpo_analysis__executed.ipynb)
- [Stage 2 analysis program](../../../src/analysis/lib1_dedup_stage2_analysis.py)
- [Targeted 3'UTR HPO analysis program](../../../src/analysis/lib1_dedup_utr3_targeted_hpo_analysis.py)
- [Stage 3 paired loss/RC analysis program](../../../src/analysis/lib1_dedup_stage3_analysis.py)
- [Stage 3 paired loss/RC interpretation notebook](../../../tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/05_stage3_paired_rc_loss_analysis.ipynb)
- [Rendered Stage 3 plus locked-audit notebook report](../../../src/learn/outputs/audit/lib1_dedup_final_audit_july2026/reporting/05_stage3_paired_rc_loss_analysis__executed.html)
- [Frozen audit products](../../../src/learn/outputs/audit/lib1_dedup_final_audit_july2026/frozen_products/audit_summary.json)
- [Final-refit and audit amendment](lib1_dedup_final_refit_and_audit_protocol_amendment_july16_2026.md)
- [Pre-audit implementation reconciliation](lib1_dedup_final_refit_implementation_reconciliation_july16_2026.md)
- [Diagram-ready full workflow handoff](lib1_dedup_phase1_workflow_relationships_july16_2026.md)
- [TAC presentation and illustration brief](lib1_mean_expression_tac_presentation_and_illustration_brief_july18_2026.md)
- [Executed Stage 3 notebook](../../../src/learn/outputs/analysis/lib1_dedup_stage3_weighted_loss_july2026/executed_notebooks/05_stage3_paired_rc_loss_analysis__executed.ipynb)
- [Rendered Stage 3 notebook report](../../../src/learn/outputs/analysis/lib1_dedup_stage3_weighted_loss_july2026/executed_notebooks/05_stage3_paired_rc_loss_analysis__executed.html)
- [Stage 3 analysis and next-stage handoff](lib1_dedup_stage3_analysis_and_next_stage_handoff_july15_2026.md)
- [Stage 3 machine-readable summary](../../../src/learn/outputs/analysis/lib1_dedup_stage3_weighted_loss_july2026/stage3_analysis_summary.json)
- [Stage 3 selected policies](../../../src/learn/outputs/analysis/lib1_dedup_stage3_weighted_loss_july2026/stage3_selected_part_policies.json)
- [Stage 2 reporting program](../../../src/analysis/lib1_dedup_stage2_reporting.py)
- [Intron estimand sensitivity reporting program](../../../src/analysis/lib1_dedup_intron_sensitivity_reporting.py)
- [Lib1 analysis and figure conventions](../../../tutorials/lib1_tasks/ANALYSIS_CONVENTIONS.md)
- [Stage 2 machine-readable summary](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_analysis_summary.json)
- [Stage 2 OOF metrics](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_oof_metrics.csv)
- [Stage 2 fold metrics](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_oof_fold_metrics.csv)
- [Stage 2 paired-RC metrics](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_rc_pair_metrics.csv)
- [Stage 2 Intron sensitivity metrics](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_intron_sensitivity_stratum_metrics.csv)
- [Stage 2 Intron fold-trained baseline](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_intron_stratum_mean_baselines.csv)
- [Stage 2 Intron estimand sensitivity summary](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_intron_sensitivity_reporting_summary.json)
- [Stage 2 reporting products](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_reporting_summary.json)
- [Targeted 3'UTR machine-readable summary](../../../src/learn/outputs/analysis/lib1_dedup_utr3_targeted_hpo_july2026/utr3_targeted_hpo_analysis_summary.json)
- [Stage 2 to targeted-3'UTR handoff](lib1_dedup_stage2_to_targeted_utr3_handoff_july2026.md)
- [Frozen targeted-3'UTR HPO amendment](lib1_dedup_targeted_utr3_hpo_protocol_amendment_july14_2026.md)
- [Frozen Stage 3 weighted-loss amendment](lib1_dedup_stage3_protocol_amendment_july14_2026.md)
- [Frozen Stage 4 downsampling amendment](lib1_dedup_stage4_downsampling_protocol_amendment_july17_2026.md)
- [Stage 4 manifest generator](../../../src/learn/generate_lib1_dedup_stage4_downsampling_manifest.py)
- [Stage 4 independent verifier](../../../src/learn/verify_lib1_dedup_stage4_downsampling_manifest.py)
- [Stage 4 resume-safe campaign runner](../../../src/learn/run_lib1_dedup_stage4_downsampling_campaign.py)
- [Stage 4 development-OOF analysis program](../../../src/analysis/lib1_dedup_stage4_downsampling_analysis.py)
- [Stage 4 presentation-reporting program](../../../src/analysis/lib1_dedup_stage4_reporting.py)
- [Stage 4 analysis notebook](../../../tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/06_stage4_downsampling_analysis.ipynb)
- [Stage 4 analysis and decision handoff](lib1_dedup_stage4_downsampling_analysis_and_handoff_july18_2026.md)

## Next Decision Gate

Stage 3 selection, the one-time final test, and Stage 4 sample-efficiency
analysis are closed. The next gate is a data-acquisition/design decision, not
another automatic training launch. The Stage 4 handoff recommends:

1. joint highest generic-volume priority for 5'UTR and 3'UTR, followed by
   Promoter;
2. targeted, position-balanced and within-stratum Intron acquisition rather
   than optimizing pooled Pearson alone;
3. lowest generic-volume priority for Enhancer under the tested transfer
   route; and
4. no numerical 10x- or 100x-beyond-full claim from the unstable curve-family
   fits.

The five Stage 3 policies remain frozen. Sparse Stage 4 alternatives are
sensitivity anchors and do not authorize post hoc model reselection. Any
reopened configuration comparison or new held-out evaluation needs a
separately dated protocol and must not reuse the existing final test as model
selection or calibration data.

Separately, obtain verified construct-to-sublibrary provenance if the team
wants a true Intron synthesis-pool analysis. Do not relabel the current
sequence-inferred mask report as verified sublibrary performance.

## Upstream And Later-Stage Context

These records remain in their broader homes because they either precede this
campaign or govern work beyond the completed unweighted Stage 2 screen:

- [Deduplicated data-product update](../../repo_hygiene/barcode_level_dedup_update_july6_2026.md)
- [June sweep relationships](../lib1_sweep_workflow_relationships_june2026.md)
- [June outer-seed HPO plan](../lib1_outer_seed_prior_hpo_plan_june2026.md)
- [Barcode-weighted loss plan](../learn/lib1_barcode_weighted_loss_plan_june2026.md)
- [Barcode-threshold downsampling plan](../lib1_barcode_threshold_downsampling_plan_june2026.md)
- [Barcode-level uncertainty discussion brief](../learn/barcode_level_uncertainty_discussion_context_july7_2026.md)
