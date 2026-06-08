# Lib1 Tasks Run/Analysis Backtracking Checklist

Generated: 2026-06-08

Purpose: give a paper-notebook style checklist for deciding which notebooks,
generated result roots, and run scripts remain useful context. This file is for
manual review. It does not mean the listed files should be deleted
automatically.

## How To Use This Checklist

For each row:

- Open the analysis notebook and confirm the result root still exists locally.
- Check whether the unique takeaway is already captured in a newer notebook,
  `plan/`, or `tutorials/lib1_tasks/README.md`.
- Mark the final decision in the rightmost column:
  - `keep canonical`: commit or keep as the source-of-truth analysis surface.
  - `archive/local`: keep locally, but do not add to GitHub.
  - `delete local`: safe to remove local generated state after the takeaway is
    captured elsewhere.
  - `needs recap`: write a small summary before removing anything.

Prefer deleting generated outputs before deleting notebooks. Notebooks compress
context; raw run forests usually expand it.

## Enhancer Fine-Tuning And Scratch Backtracking

Canonical enhancer question map:

| Question | Canonical record | Notebook placement | Generated data action |
|---|---|---|---|
| Transfer baseline | `plan/finetune/lib1_enhancer_scratch_and_finetune_rehydration_june2026.md` | early transfer notebooks moved to local-only `archive/` | delete local caches only after the rehydration plan is accepted |
| Random split / AL readiness | `plan/finetune/lib1_enhancer_scratch_and_finetune_rehydration_june2026.md` | random-split notebooks moved to local-only `archive/` | keep ignored until no rerun is needed; then delete local roots |
| Barcode training quality | `may13_2026_bc_training_eval.ipynb` plus `barcode_range_comparable_bins_decision_analysis_may2026.ipynb` | keep top-level canonical | keep ignored run roots until final barcode policy is written |
| Barcode test quality | `may13_2026_bc_test_eval.ipynb` | keep top-level canonical | keep ignored run roots until final barcode policy is written |
| HQ8 multiseed HPO | `may15_2026_hq8_multiseed_hpo_analysis.ipynb` | keep top-level canonical | keep ignored run roots until enhancer decision is final |

The local-only archive lives at
`tutorials/lib1_tasks/fine_tuning/enhancer_finetune_w_boda_pretrain/archive/`
and is ignored by Git. The top-level enhancer notebook folder should now
contain only the four canonical notebooks named above.

| Done | Approx date | Run/output root | Runner or config family | Analysis notebook(s) | Unique takeaway | Suggested action | Decision | Generated data action |
|---|---|---|---|---|---|---|---|---|
| [x] | 2026-03-25 | `src/finetune/learning_curve/lib1_enhancer_mar25_b2` | `lib1_enhancer_learning_curve_finetune_updated.py` | `archive/finetune_learning_curve_sweep.ipynb` | First B2-only transfer learning curve baseline. | Archive/local; do not make canonical. | moved to local archive | delete local root after rehydration plan is accepted |
| [x] | 2026-03-28 | `src/finetune/learning_curve/lib1_enhancer_mar28_broad_hpo` | `lib1_enhancer_learning_curve_finetune_updated.py` | `archive/complete_finetune_learning_curve_sweep_apr1_2026.ipynb` | Cache-first broad-HPO reconstruction; current visible table mostly behaves like B2 reference. | Archive/local after confirming key numbers are in the enhancer rehydration plan. | moved to local archive | delete local root after rehydration plan is accepted |
| [x] | 2026-04-01 | `src/finetune/learning_curve/lib1_enhancer_targeted_apr2026` | `lib1_enhancer_learning_curve_finetune_updated.py` | `archive/targeted_finetune_learning_curve_HQ_split_apr1_2026.ipynb`; duplicate root-level `fine_tuning/targeted_finetune_learning_curve_HQ_split_apr1_2026.ipynb` remains outside this folder | HQ-first learning curve; more data helps but train size and quality composition are confounded. | Archive thematic copy; review duplicate root-level notebook separately. | thematic copy moved to local archive | delete local root after final enhancer recap if no rerun is needed |
| [x] | 2026-04-03 | `src/finetune/learning_curve/lib1_enhancer_targeted_random_all_per_seed_apr2026` | `lib1_enhancer_learning_curve_finetune_split_options.py` | `archive/targeted_finetune_learning_curve_random_split_apr3_2026.ipynb`; duplicate root-level `fine_tuning/targeted_finetune_learning_curve_random_split_apr3_2026.ipynb` remains outside this folder | Harder random-all split gives lower but more AL-realistic signal. | Archive thematic copy; covered by rehydration plan. | thematic copy moved to local archive | delete local root after final enhancer recap if no rerun is needed |
| [x] | 2026-04-05 | `src/finetune/learning_curve/lib1_enhancer_long_epoch_k562_b3_bcap8_random_split_apr2026` | `lib1_enhancer_learning_curve_long_epoch_single_head_b3_random_split.py` | `archive/long_epoch_learning_curve_analysis_apr5.ipynb`; duplicate root-level `fine_tuning/long_epoch_learning_curve_analysis_apr5.ipynb` remains outside this folder | Longer training did not obviously beat shorter targeted/follow-up recipes. | Archive/local. | thematic copy moved to local archive | delete local root after final enhancer recap if no rerun is needed |
| [x] | 2026-04-14 | `src/finetune/learning_curve/lib1_enhancer_followup_diagnostic_random_split_apr2026_v1` | `lib1_enhancer_learning_curve_followup_diagnostic_random_split.py` | `archive/AL_ready_random_split_followup_diagnostic_apr2026.ipynb` | Stronger random-split diagnostic; useful AL-readiness sanity check. | Archive/local; summarized in rehydration plan. | moved to local archive | keep ignored until rehydration plan is accepted, then delete local if desired |
| [x] | 2026-04-19 to 2026-05-04 | `src/learn/outputs/hpo_runs/by_project/enhancer__bashor_in_house__*` | `src/learn/configs/enhancer/bashor_in_house/*.yml`, `src/learn/launch/lib1_enhancer*scratch*.sh` | historical `lib1_scratch_basic_sweep_diagnostics_and_test_eval.ipynb` deleted on 2026-06-08 | From-scratch enhancer training does not generalize well enough to be primary path. | Rehydration plan now carries the retained conclusion. | deleted obsolete notebook copies | generated HPO roots stay ignored; delete only after registry/plan recap is trusted |
| [x] | 2026-04-29 | `src/finetune/learning_curve/lib1_enhancer_filtered_raw_ratio_random_hq_holdout_hq_first_b2_b3_bcaps10_bestlr_apr29_2026` | `lib1_enhancer_learning_curve_filtered_raw_ratio_split_options.py` | `archive/hq_first_b2_b3_learning_curve_analysis_may01_2026.ipynb` | Updated fastqs1-5 HQ-first transfer baseline. | Archive/local; later barcode notebooks separate barcode count more cleanly. | moved to local archive | delete local root after final enhancer recap if no rerun is needed |
| [x] | 2026-04-30 | `src/finetune/learning_curve/lib1_enhancer_barcode_range_stage1_hq4_hq8_apr2026` | `lib1_enhancer_barcode_range_finetuning.py`, `run_lib1_barcode_range_stage1.sh` | `archive/barcode_range_learning_curve_analysis_apr30_2026.ipynb` | First direct barcode-bin learning curve. | Archive/local; May comparable-bin and exact-low-barcode analyses supersede it. | moved to local archive | delete local root after final enhancer recap if no rerun is needed |
| [x] | 2026-05-04 | `src/finetune/learning_curve/lib1_enhancer_barcode_range_comparable_bins_hq4_hq8_b2_b3_bcap10_30_seed5_may2026` | `lib1_enhancer_barcode_range_comparable_bins_finetuning.py`, `run_lib1_barcode_range_comparable_bins_parallel.sh` | `fine_tuning/enhancer_finetune_w_boda_pretrain/barcode_range_comparable_bins_decision_analysis_may2026.ipynb`; also used in May13 training eval | Equal-N comparison of low/mid/high barcode bins. | Keep canonical. | top-level canonical | keep ignored run root until barcode policy is final |
| [x] | 2026-05-06 to 2026-05-12 | `lib1_enhancer_exact_low_barcode_*`, `lib1_enhancer_barcode_range_comparable_bins_loggrid_*`, `lib1_enhancer_random_train_test_quality_*` | exact-low, comparable-loggrid, and random train/test quality runners | `may13_2026_bc_training_eval.ipynb`, `may13_2026_bc_test_eval.ipynb`; `archive/lib1_may2026_followup_low_barcode_quality_analysis.ipynb` | Separates training barcode count from held-out measurement quality; exact 3-barcode and 4-6 bins can be useful. | Keep May13 training/test notebooks canonical; archive the overlapping follow-up notebook. | May13 notebooks top-level canonical; older synthesis moved to local archive | keep ignored run roots until barcode policy is final |
| [x] | 2026-05-15 | `src/finetune/learning_curve/lib1_enhancer_threshold_hq8_random_mixed_b2_allheads_8seed_absgrid_may2026` and `lib1_enhancer_branched_only_k562_hq8_4seed_noearly_250epoch_may2026` | `run_lib1_threshold_hq8_multiseed_hpo_parallel.sh`, `run_lib1_branched_only_k562_noearly_hq8_parallel.sh` | `fine_tuning/enhancer_finetune_w_boda_pretrain/may15_2026_hq8_multiseed_hpo_analysis.ipynb` | Most mature enhancer barcode/HQ8 multiseed HPO snapshot. | Keep canonical. | top-level canonical | keep ignored run roots until enhancer decision is final |
| [x] | early transfer | `src/finetune/cache/lib1_enhancer/aggregates` | `lib1_enhancer_transfer_multiseed.py` | `archive/tewhey_model_on_lib1_enhancer.ipynb`; `tewhey_model_on_lib1_enhancer_multiseed_summary.ipynb` remains root-level outside this folder | Establishes pretrained Malinois/BODA2 transfer is useful. | Archive detailed early notebook; rely on rehydration plan. | detailed notebook moved to local archive | delete cache only after rehydration plan is accepted |

## UTR, Promoter, And Intron Backtracking

| Done | Approx date | Run/output root | Runner or config family | Analysis notebook(s) | Unique takeaway | Superseded or repeated by | Suggested action | Decision |
|---|---|---|---|---|---|---|---|---|
| [x] | 2026-05-04 to 2026-05-13 | Hani UTR public HPO projects in `src/learn/outputs/hpo_runs/by_project/utr*__hani_rna_activity__*` | `src/learn/configs/utr{3,5}/hani_rna_activity/*`, `src/learn/launch/utr*_hani_*` | `pretraining_CRE_public_data/public_cre_hpo_presentation_summary.ipynb`, `hani_utr_basset_branched_hpo_presentation_summary.ipynb`, `parade_released_checkpoint_eval_may2026.ipynb` | Selects and contextualizes UTR pretrained models and PARADE comparison. | Phase 2/Phase 3 5 Prime decision notebooks for 5 Prime; 3 Prime still mostly pretraining/reference. | Keep the summary notebooks; generated HPO roots remain ignored. | keep canonical/reference summaries; do not commit generated HPO roots |
| [x] | 2026-05-15 to 2026-05-17 | `src/learn/outputs/hpo_runs/by_project/introns__seelig_2015_a5ss_sd1__scratch__basset_branched` | Seelig intron config/launcher | `pretraining_CRE_public_data/intron_seelig_a5ss_sd1_pretraining_hpo_decision_may2026.ipynb` | Intron public pretraining works, but one-shot in-house transfer later fails. | Promoter/intron one-shot eval. | Keep pretraining decision notebook; next notebook should be fine-tune pipeline. | keep canonical intron pretraining notebook |
| [x] | 2026-05-19 | `src/finetune/learning_curve/hani_utr5_lib2_resnet1d_1mmy39ku_phase2_may2026` | first `hani_utr5_lib2_finetune.py` run | `fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/hani_utr5_lib2_phase2_finetune_analysis_may2026.ipynb` | First Phase 2 fine-tune pass. | Phase 2 v2 split-safe HPO notebook. | Archive/local unless historical provenance is needed. | not canonical; leave local unless a historical appendix is desired |
| [x] | 2026-05-26 | `src/finetune/learning_curve/hani_utr5_lib2_resnet1d_1mmy39ku_phase2_v2_may2026` | `hani_utr5_lib2_finetune.py`, `run_hani_utr5_lib2_finetune_parallel.sh` | `fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/hani_utr5_lib2_phase2_v2_finetune_analysis_may2026.ipynb` | Clean validation-first Phase 2 result: Lib2 improves with modest Lib1 retention cost. | Phase 3 scratch comparison is complementary, not a replacement. | Keep canonical. | keep canonical |
| [x] | 2026-05-20 to 2026-05-27 | `src/learn/outputs/hpo_runs/by_project/utr5__hani_rna_activity_lib1_lib2__phase3_scratch__resnet1d` | `utr5_hani_lib1_lib2_resnet1d_phase3_scratch_sweep.sh`, phase3 scratch config | `pretraining_CRE_public_data/hani_utr5_lib1_lib2_phase3_scratch_hpo_analysis_may2026.ipynb` | Scratch Lib1+Lib2 ResNet run `4eq96xxd` is competitive with Phase 2 fine-tune and has better Lib1 retention. | Current 5 Prime model-comparison source. | Keep canonical. | keep canonical |
| [x] | 2026-05-27 | `src/finetune/learning_curve/promoter_intron_pretrained_inhouse_eval_may2026` | `promoter_intron_inhouse_one_shot_eval.py` | `pretrain_CRE_inhouse_data/promoter_intron_inhouse_pretrained_eval_may2026.ipynb` | Promoter one-shot transfer weak positive; intron one-shot transfer negative. | Future promoter/intron fine-tune pipeline. | Keep canonical diagnostic. | keep canonical diagnostic |
| [x] | 2026-06-03 to 2026-06-07 | `src/finetune/learning_curve/inhouse_utr5_parade_resnet_small_hpo_jun2026`, `inhouse_utr5_parade_resnet_downsample_top_configs_jun2026` | `inhouse_utr5_parade_resnet_finetune.py`, `run_inhouse_utr5_parade_resnet_small_hpo_parallel.sh` | `fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/inhouse_utr5_parade_resnet_small_hpo_analysis_jun2026.ipynb` | In-house FivePrime PARADE vs BODA ResNet HPO and downsample follow-up. | Current in-house 5 Prime decision surface. | Keep canonical. | keep canonical |

## EDA And Presentation Notebooks

| Done | Notebook or artifact | Unique takeaway | Suggested action | Decision |
|---|---|---|---|---|
| [ ] | `in_house_EDA/apr22_udpated_enhancer_dataset_exploratory.ipynb` | In-house enhancer data shape and target sanity before modeling. | Keep if it remains the only data-readiness record; otherwise archive. | keep/reference unless covered by enhancer rehydration plan |
| [x] | `in_house_EDA/in_house_utr_eda_may2026.ipynb` | In-house UTR sequence/length/barcode readiness. | Keep canonical UTR EDA. | keep canonical |
| [ ] | `pretraining_CRE_public_data/cre_public_pretraining_hpo_analysis.ipynb` | Early public CRE HPO analysis. | Archive if `public_cre_hpo_presentation_summary.ipynb` covers the same conclusions. | archive/local candidate |
| [x] | `pretraining_CRE_public_data/public_cre_hpo_presentation_summary.ipynb` | Current public CRE presentation summary across regions. | Keep canonical summary. | keep canonical/reference |
| [x] | `pretraining_CRE_public_data/parade_released_checkpoint_eval_may2026.ipynb` | PARADE released-checkpoint validation and in-house UTR scoreability context. | Keep canonical reference. | keep canonical/reference |

## Generated Data Cleanup Policy

- Do not commit `src/learn/local_artifacts/`, `src/learn/outputs/`, or
  `src/finetune/learning_curve/` result forests.
- For enhancer, the local generated roots tied only to archived notebooks can
  be deleted after the rehydration plan and this checklist capture their
  takeaways.
- For UTR/promoter/intron, keep generated roots ignored until the canonical
  notebooks above are committed and no rerun/debugging is needed.
- If a notebook still points to an old pre-reorg path, fix the path only when
  the notebook is being promoted to canonical.

## Generated Run Roots With No `lib1_tasks` Follow-Up Notebook Detected

These are the best local deletion candidates after a quick sanity check.

| Done | Artifact root | Why it looks orphaned | Suggested action | Decision |
|---|---|---|---|---|
| [ ] | `src/finetune/learning_curve/_smoke_branched_only_k562_noearly` | Smoke/debug output; no analysis notebook. | Delete local. | approved delete-local candidate |
| [ ] | `src/finetune/learning_curve/_launcher_smoke_branched_only_k562` | Launcher smoke output; no analysis notebook. | Delete local. | approved delete-local candidate |
| [ ] | `src/finetune/learning_curve/_launcher_smoke_no_capture_branched_only_k562` | Launcher smoke output; no analysis notebook. | Delete local. | approved delete-local candidate |
| [ ] | `src/finetune/learning_curve/_launcher_preview_debug_branched_only_k562` | Preview/debug output; no analysis notebook. | Delete local. | approved delete-local candidate |
| [ ] | `src/finetune/learning_curve/lib1_enhancer_filtered_raw_ratio_random_all_lc_apr25_2026` | No notebook text references this root. | Needs recap or delete local if superseded by Apr29 HQ-holdout and May barcode analyses. | delete-local candidate after one-line recap |
| [ ] | `src/finetune/learning_curve/lib1_enhancer_targeted_random_hq_val_test_per_seed_apr21_2026` | No notebook text references this root. | Needs recap or delete local if not used in current decisions. | delete-local candidate after one-line recap |

## Learn HPO Roots Not Explicitly Referenced By A `lib1_tasks` Notebook

Many `src/learn/outputs/hpo_runs/by_project/*` roots are summarized through
`src/learn/run_registry/runs.csv` rather than opened directly by notebooks. Do
not delete these solely because they are absent from notebook text. Treat this
as a prompt to ask: "Is the registry summary enough?"

The May 2026 repo reorg moved generated HPO roots under
`src/learn/outputs/hpo_runs/by_project/`, so some notebooks may still rely on
the run registry or old W&B cache paths rather than the moved directory names.
For those cases, prefer fixing notebook path references only when the notebook
is promoted to canonical. Older tasks such as
`utr5__polysome__scratch__utr_bassetvl` can remain ignored/local unless they
become active again.

High-priority review buckets:

- `enhancer__bashor_in_house__*`: covered by scratch notebook plus enhancer
  rehydration plan, even though output roots are not named in notebook text.
- `promoter__deboer_core__*`: covered by public CRE summaries, but lacks a
  dedicated in-house fine-tune analysis notebook.
- `utr3__hani_rna_activity__*`: covered by public CRE and PARADE-context
  summaries; 3 Prime fine-tune is still pending.
- `utr5__hani_rna_activity__*`: covered by public CRE summaries and later
  Phase 2/Phase 3 5 Prime notebooks.
- `utr5__polysome__scratch__utr_bassetvl`: no `lib1_tasks` follow-up notebook
  was detected; root-level tutorial notebooks may still cover it.

## Run Scripts Without A Dedicated Follow-Up Notebook

This list means "no one notebook names this script as its main driver." It does
not mean the script is unused. Many scripts are covered indirectly through an
output root.

Likely support-only, no separate notebook needed:

- `src/finetune/finetune_sweep_scripts/**/combine_*outputs.py`
- `src/finetune/finetune_sweep_scripts/lib1_enhancer/combine_learning_curve_seed_outputs.py`

Covered by output-root notebooks, but often not named directly:

| Script | Result root(s) | Follow-up status |
|---|---|---|
| `run_lib1_barcode_range_stage1.sh` | `lib1_enhancer_barcode_range_stage1_hq4_hq8_apr2026` | analyzed by archived Apr30 barcode notebook; superseded |
| `run_lib1_barcode_range_comparable_bins_parallel.sh` | `lib1_enhancer_barcode_range_comparable_bins_hq4_hq8_b2_b3_bcap10_30_seed5_may2026` | analyzed by canonical comparable-bin and May13 training notebooks |
| `run_lib1_barcode_range_comparable_bins_loggrid_parallel.sh` | `lib1_enhancer_barcode_range_comparable_bins_loggrid_hq4_hq8_cap1000_b1_b2*` | analyzed by canonical May13 training notebook |
| `run_lib1_exact_low_barcode_parallel.sh` | `lib1_enhancer_exact_low_barcode_hq4_hq8_cap500_b1_b2*` | analyzed by canonical May13 training notebook |
| `run_lib1_random_train_test_quality_parallel.sh` | `lib1_enhancer_random_train_test_quality_finebins_cap3000_b2*` | analyzed by canonical May13 test notebook |
| `run_lib1_threshold_hq8_multiseed_hpo_parallel.sh` | `lib1_enhancer_threshold_hq8_random_mixed_b2_allheads_8seed_absgrid_may2026` | analyzed by canonical May15 HQ8 notebook |
| `run_lib1_branched_only_k562_noearly_hq8_parallel.sh` | `lib1_enhancer_branched_only_k562_hq8_4seed_noearly_250epoch_may2026` | analyzed by canonical May15 HQ8 notebook |
| `run_hani_utr5_lib2_finetune_parallel.sh` | `hani_utr5_lib2_resnet1d_1mmy39ku_phase2_v2_may2026` | analyzed by canonical Phase 2 v2 notebook |
| `run_inhouse_utr5_parade_resnet_small_hpo_parallel.sh` | `inhouse_utr5_parade_resnet_small_hpo_jun2026`, `inhouse_utr5_parade_resnet_downsample_top_configs_jun2026` | analyzed by canonical June in-house 5 Prime notebook |

Needs a notebook or status note if the results matter:

- `src/learn/launch/utr5_polysome_fixed_all.sh`
- `src/learn/launch/utr5_polysome_utr_bassetvl_sweep.sh`
- `src/learn/launch/promoter_deboer_compare_architectures.sh`
- `src/learn/launch/enhancer_malinois_basset_branched_baseline.sh`

Already covered by planning or registry summaries, but not a one-to-one
notebook:

- most `src/learn/launch/utr3_hani_*` and `src/learn/launch/utr5_hani_*`
  public HPO launchers
- most `src/learn/configs/utr{3,5}/hani_rna_activity/**`
- `src/learn/configs/promoter/deboer_core/**`

## Personal Review Order

1. Start with enhancer fine-tuning rows from Mar25 to May15.
2. Mark which single enhancer notebook is canonical for each question:
   transfer baseline, random split, barcode training, barcode test quality,
   HQ8 multiseed.
3. Remove or archive duplicate root-level notebooks only after the thematic
   subdir copy is chosen.
4. Review UTR/Promoter/Intron rows, keeping Phase 2 v2, Phase 3 scratch, June
   in-house 5 Prime, and promoter/intron one-shot as the current canonical set.
5. Delete only smoke/debug generated roots first. They have the lowest risk.
6. For any large generated root, write a one-paragraph recap in this file or a
   nearby README before deleting local files.
