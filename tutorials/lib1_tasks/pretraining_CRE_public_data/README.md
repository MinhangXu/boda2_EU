# Public CRE Pretraining Analysis Index

This directory is the notebook layer for public-data pretraining and checkpoint
promotion. The executable HPO configs and launchers live under `src/learn`;
generated run roots live under ignored `src/learn/outputs/` or
`src/learn/local_artifacts/`.

## Current Canonical Notebooks

| Notebook | Role | Current status |
|---|---|---|
| `parade_released_checkpoint_eval_may2026.ipynb` | PARADE released-checkpoint comparison, BODA current checkpoint context, and in-house UTR scoreability diagnostics. | canonical reference |
| `intron_seelig_a5ss_sd1_pretraining_hpo_decision_may2026.ipynb` | Seelig A5SS SD1 intron pretraining decision. | canonical intron pretraining decision |
| `hani_utr5_lib1_lib2_phase3_scratch_hpo_analysis_may2026.ipynb` | 5 Prime UTR Lib1+Lib2 from-scratch Phase 3 HPO and Phase 2 comparison. | canonical 5 Prime scratch comparison |

## See Also

Promoter/intron one-shot diagnostics that score in-house Lib1 data with
promoted public checkpoints live in
`../pretrain_CRE_inhouse_data/promoter_intron_inhouse_pretrained_eval_may2026.ipynb`.

In-house Lib1 scratch HPO analysis for promoter, intron, UTRs, and enhancer
lives in
`../pretrain_CRE_inhouse_data/lib1_inhouse_scratch_hpo_best_models_june2026.ipynb`.

Retired public-HPO notebooks and their generated presentation plots were
removed in June 2026 after the standardized Lib1 in-house scratch orchestrator
finished across all parts. The retained public-data launch/config surface is
the Seelig intron BassetBranched sweep plus the compact Hani UTR delta-aux
BassetBranched and cell-conditioned ResNet1D sweep launchers/configs.

## Output Policy

Commit small CSV/PNG/SVG scorecards under `presentation_plots/` only when they
are part of a decision record. Keep raw predictions, checkpoints, W&B run
folders, and local model artifacts out of Git.

Large prediction CSVs should move to an ignored `raw_predictions/` folder or be
regenerated from the notebook when needed.

## Open Gaps

- Intron needs architecture confirmation and a decision on whether Seelig
  fine-tuning is still worth running after strong modal80 scratch results.
