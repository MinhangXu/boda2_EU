# Lib1 HPO Cleanup Decisions, June 2026

Status: executed locally on 2026-06-11. Tracked deletions are staged in git;
ignored/untracked local artifact directories were removed from disk.

## Protected Files

Current standardized Lib1 in-house scratch HPO launch/config files remain the
primary surface:

- `src/learn/launch/lib1_inhouse_scratch_orchestrator.sh`
- `src/learn/launch/_wandb_helpers.sh`
- `src/learn/launch/lib1_promoter_scratch_resnet1d_sweep.sh`
- `src/learn/launch/lib1_promoter_scratch_promoter_bassetvl_sweep.sh`
- `src/learn/launch/lib1_intron_scratch_resnet1d_sweep.sh`
- `src/learn/launch/lib1_threeprime_scratch_resnet1d_sweep.sh`
- `src/learn/launch/lib1_threeprime_scratch_utr_bassetvl_sweep.sh`
- `src/learn/launch/lib1_fiveprime_scratch_resnet1d_sweep.sh`
- `src/learn/launch/lib1_fiveprime_scratch_utr_bassetvl_sweep.sh`
- `src/learn/launch/lib1_enhancer_no_flank_hq8_scratch_resnet1d_sweep.sh`
- `src/learn/launch/lib1_enhancer_no_flank_hq8_scratch_bassetvl_sweep.sh`
- `src/learn/configs/promoter/bashor_in_house/resnet1d/lib1_promoter__scratch_resnet1d__bayes.yml`
- `src/learn/configs/promoter/bashor_in_house/promoter_bassetvl/lib1_promoter__scratch_promoter_bassetvl__bayes.yml`
- `src/learn/configs/introns/bashor_in_house/resnet1d/lib1_intron_modal80__scratch_resnet1d__bayes.yml`
- `src/learn/configs/utr3/bashor_in_house/resnet1d/lib1_threeprime__scratch_resnet1d__bayes.yml`
- `src/learn/configs/utr3/bashor_in_house/utr_bassetvl/lib1_threeprime__scratch_utr_bassetvl__bayes.yml`
- `src/learn/configs/utr5/bashor_in_house/resnet1d/lib1_fiveprime_modal50__scratch_resnet1d__bayes.yml`
- `src/learn/configs/utr5/bashor_in_house/utr_bassetvl/lib1_fiveprime_modal50__scratch_utr_bassetvl__bayes.yml`
- `src/learn/configs/enhancer/bashor_in_house/resnet1d/lib1_enhancer_no_flank_hq8__scratch_resnet1d__bayes.yml`
- `src/learn/configs/enhancer/bashor_in_house/bassetvl/lib1_enhancer_no_flank_hq8__scratch_bassetvl__bayes.yml`

The compact public-data HPO surface retained for future reference:

- `src/learn/launch/introns_seelig_a5ss_sd1_basset_branched_sweep.sh`
- `src/learn/launch/utr3_hani_basset_branched_delta_aux_sweep.sh`
- `src/learn/launch/utr5_hani_basset_branched_delta_aux_sweep.sh`
- `src/learn/launch/utr3_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh`
- `src/learn/launch/utr5_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh`
- `src/learn/launch/utr_hani_resnet1d_cell_conditioned_delta_aux_sweeps.sh`
- `src/learn/configs/introns/seelig_2015/basset_branched/introns__seelig_2015_a5ss_sd1__scratch__basset_branched.yml`
- `src/learn/configs/utr3/hani_rna_activity/basset_branched/utr3__hani_rna_activity__basset_branched__delta_aux_bayes.yml`
- `src/learn/configs/utr5/hani_rna_activity/basset_branched/utr5__hani_rna_activity__basset_branched__delta_aux_bayes.yml`
- `src/learn/configs/utr3/hani_rna_activity/resnet1d/utr3__hani_rna_activity__resnet1d__cell_conditioned_delta_aux_bayes.yml`
- `src/learn/configs/utr5/hani_rna_activity/resnet1d/utr5__hani_rna_activity__resnet1d__cell_conditioned_delta_aux_bayes.yml`

## Deleted Source Families

Launchers/configs removed:

- Legacy Malinois enhancer BassetNonBranched sweeps.
- Legacy Bashor enhancer scratch basic/weighted/FASTQ1-5 sweeps.
- Public DeBoer promoter BassetVL/ResNet1D/UTR-BassetVL sweeps.
- Legacy in-house promoter E7/E30 train90k sweeps.
- Old all-region pilot and public-dataset batch launchers.
- Superseded 3 Prime UTR RC-factorial grid launch/configs.
- Superseded Hani UTR broad/focused BassetBranched sweeps.
- Superseded Hani UTR broad/focused ResNet1D challenger sweeps.
- Superseded Hani UTR UTR-BassetVL sweeps.
- Superseded 5 Prime UTR Lib1+Lib2 phase3 scratch launcher/config.
- Superseded 5 Prime polysome UTR-BassetVL launch/configs.
- Old enhancer scratch setup note
  `src/learn/configs/enhancer/bashor_in_house/bashor_lab_collab_thread2_scratch_sweeps.md`.

Notebooks removed:

- `tutorials/lib1_tasks/pretraining_CRE_public_data/cre_public_pretraining_hpo_analysis.ipynb`
- `tutorials/lib1_tasks/pretraining_CRE_public_data/hani_utr_basset_branched_hpo_presentation_summary.ipynb`
- `tutorials/lib1_tasks/pretraining_CRE_public_data/public_cre_hpo_presentation_summary.ipynb`

Generated public-summary presentation plots removed from
`tutorials/lib1_tasks/pretraining_CRE_public_data/presentation_plots/`:

- `hani_utr_branched/`
- `best_r2_by_region_any_split.*`
- `best_test_performance_by_region.*`
- `best_validation_by_region_phase.*`
- `top5_checkpoint_metric_by_region.*`
- `utr_parade_context_comparison.*`
- `utr_test_set_leaders.csv`
- `utr_validation_vs_test_pearson.*`
- `validation_selected_region_winners.csv`
- `validation_selected_winner_summary.*`

## Deleted Local Artifacts

Large ignored/untracked artifact directories removed from disk:

- Legacy enhancer scratch basic/weighted outputs and local artifacts.
- Legacy enhancer FASTQ1-5 scratch local artifacts.
- Legacy Malinois enhancer local artifacts.
- Public DeBoer promoter local artifacts.
- Legacy promoter E7/E30 local artifacts.
- Old promoter/3UTR/5UTR sweep local artifacts.
- Superseded per-project `src/learn/outputs/hpo_runs/by_project/` directories
  for deleted enhancer, promoter, Hani UTR broad/focused/BassetVL/challenger,
  5 Prime Lib1+Lib2 phase3, and polysome sweeps.
- Retired root-level project directories
  `src/learn/promoter__legacy_e7_e30__scratch__resnet1d_train90k` and
  `src/learn/utr3__bashor_in_house__threeprime_modal100__scratch__utr_bassetvl_focused_rc_factorial_fp32`.

Kept artifact directories:

- Current standardized Lib1 in-house scratch artifacts for promoter, intron,
  3 Prime UTR, 5 Prime UTR, and enhancer.
- Current standardized per-project HPO run roots were moved from
  `src/learn/<wandb_project>/` into
  `src/learn/outputs/hpo_runs/by_project/<wandb_project>/`.
- `src/learn/outputs/hpo_runs/by_project/introns__seelig_2015_a5ss_sd1__scratch__basset_branched`
- `src/learn/outputs/hpo_runs/by_project/utr3__hani_rna_activity__delta_aux__basset_branched`
- `src/learn/outputs/hpo_runs/by_project/utr5__hani_rna_activity__delta_aux__basset_branched`
- `src/learn/outputs/hpo_runs/by_project/utr3__hani_rna_activity__cell_conditioned_delta_aux__resnet1d`
- `src/learn/outputs/hpo_runs/by_project/utr5__hani_rna_activity__cell_conditioned_delta_aux__resnet1d`

## Deferred

- `src/learn/wandb` was left untouched. It is still useful as a local W&B cache
  until compact config snapshots for comparison cohorts are fully exported.
