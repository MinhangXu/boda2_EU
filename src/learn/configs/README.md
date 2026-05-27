# `src/learn/configs` Guide

This directory now organizes authored configs by:

1. CRE family
2. target family
3. model family

The layout is:

- `configs/<cre_family>/<target_family>/<model_family>/<config>.yml`

Current authored configs:

- `configs/enhancer/malinois_mpra/basset_branched/enhancer__malinois_mpra__basset_branched__transfer_baseline.yml`
- `configs/enhancer/bashor_in_house/lib1_enhancer_fastqs1_5__scratch_basic__bayes.yml`
- `configs/enhancer/bashor_in_house/lib1_enhancer_fastqs1_5__scratch_weighted__bayes.yml`
- `configs/legacy/enhancer/malinois_mpra/basset_nonbranched/enhancer__malinois_mpra__basset_nonbranched__single_head_k562__bayes.yml`
- `configs/legacy/enhancer/malinois_mpra/basset_nonbranched/enhancer__malinois_mpra__basset_nonbranched__single_head_combined__bayes.yml`
- `configs/promoter/deboer_core/utr_bassetvl/promoter__deboer_core__utr_bassetvl__bayes.yml`
- `configs/promoter/deboer_core/utr_bassetvl/promoter__deboer_core__utr_bassetvl__focused_bayes.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__bayes__egfp_1.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__bayes__egfp_2.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__bayes__mcherry_1.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__bayes__mcherry_2.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__egfp_1.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__egfp_2.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__mcherry_1.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__mcherry_2.yml`
- `configs/utr5/hani_rna_activity/utr_bassetvl/utr5__hani_rna_activity__utr_bassetvl__bayes.yml`
- `configs/utr5/hani_rna_activity/utr_bassetvl/utr5__hani_rna_activity__utr_bassetvl__focused_bayes.yml`
- `configs/utr5/hani_rna_activity/resnet1d/utr5__hani_rna_activity_lib1_lib2__resnet1d__phase3_scratch_bayes.yml`
- `configs/utr3/hani_rna_activity/utr_bassetvl/utr3__hani_rna_activity__utr_bassetvl__bayes.yml`
- `configs/utr3/hani_rna_activity/utr_bassetvl/utr3__hani_rna_activity__utr_bassetvl__focused_bayes__2025-06-16.yml`
- `configs/utr3/hani_rna_activity/utr_bassetvl/utr3__hani_rna_activity__utr_bassetvl__focused_bayes_2026_04.yml`

## Naming Convention

Use:

- `<cre_family>__<target_family>__<model_family>__<stage>.yml`
- add `__<yyyy-mm-dd>` only when the stage name alone is not enough

Examples:

- `utr3__hani_rna_activity__utr_bassetvl__bayes.yml`
- `utr3__hani_rna_activity__resnet__bayes.yml`
- `enhancer__malinois_mpra__basset_branched__transfer_baseline.yml`

## Comparing Models Fairly

To compare models on the same task:

- keep them in sibling model-family directories under the same `cre_family/target_family`
- keep the filename stem the same after the model-family token whenever the search stage is meant to be comparable
- record the shared comparison in the run registry or manifest layer rather than relying on path memory

Example comparison-ready structure:

- `configs/utr3/hani_rna_activity/utr_bassetvl/utr3__hani_rna_activity__utr_bassetvl__bayes.yml`
- `configs/utr3/hani_rna_activity/resnet/utr3__hani_rna_activity__resnet__bayes.yml`

That keeps model comparisons local to one target family instead of scattering them across the repo.

For enhancer single-head transfer experiments, the directory name may describe the
comparison intent more directly than the exact model class. For example,
the archived `configs/legacy/.../basset_nonbranched/` path maps to the
`BassetVL` model class.

The preferred enhancer single-head path is currently the combined pan-cell target:

- `combined_activity_zmean`
- derived from `K562_mean`, `HepG2_mean`, and `SKNSH_mean`
- built by `src/learn/prepare_enhancer_single_head_dataset.py`

Phase 3 Hani 5'UTR Lib1+Lib2 scratch HPO intentionally keeps the combined
target family in the filename/project name (`hani_rna_activity_lib1_lib2`) so
Lib1-only and Lib1+Lib2 production-pretraining branches do not share a W&B
bucket. Its derived input table is built by
`src/learn/prepare_hani_utr5_lib1_lib2_phase3_dataset.py`.
