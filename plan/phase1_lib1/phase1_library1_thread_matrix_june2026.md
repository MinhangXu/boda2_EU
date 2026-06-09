# Phase 1 Library 1 Thread Matrix

Generated: 2026-06-08

This file is the local repo mirror of the Notion "Phase 1 - Library 1" board.
It summarizes in-house Library 1 single-part modeling threads by CRE part,
training regime, thread function, run scripts, and analysis notebooks.

Use it for Phase 1 synthesis: decide what is ready to combine, scale, or
redesign in Phase 2. Use
`repo_hygiene/lib1_tasks_run_analysis_backtracking_checklist_june2026.md` for
cleanup decisions about duplicate notebooks and generated result roots.

## Source Of Truth Pointers

| Source | Role |
|---|---|
| `tutorials/lib1_tasks/README.md` | Notebook workspace contract and canonical notebook index. |
| `plan/README.md` | Repo planning index and compact modeling status matrix. |
| `plan/repo_hygiene/lib1_tasks_run_analysis_backtracking_checklist_june2026.md` | Manual ledger for canonical/archive/delete decisions. |
| `src/learn/run_registry/best_runs.csv` | Curated public-data and pretraining run winners. |
| `src/learn/run_registry/runs.csv` | Per-run learn/HPO registry. |
| `src/learn/run_registry/sweep_launches.csv` | Sweep launcher/config provenance. |
| Notion page `Phase 1 - Library 1` | Live Research OS board and property taxonomy. |

## Tag Vocabulary

### Training Regime

Use one primary value per thread row.

| Regime | Use when |
|---|---|
| Public-data pretraining | The run trains or evaluates on public data before any in-house adaptation. |
| Legacy-data pretraining | The run trains on older in-house or non-Lib1 data that plays the same seeding role as public pretraining for Phase 1 comparisons. |
| Public -> in-house fine-tuning | The run starts from public/pretrained weights and adapts or evaluates on in-house Library 1. |
| In-house scratch training | The run trains directly from in-house Library 1 without public pretrained weights. |
| Cross-regime comparison | The analysis explicitly compares multiple training routes. |
| No model training / analysis only | The work is EDA, one-shot scoring, diagnostics, planning, repo structure, or synthesis. |

### Thread Function

Use multiple values when appropriate.

`Model benchmark`; `Data / assay diagnostic`; `Experiment-design insight`;
`Model-selection / promotion`; `Transfer / retention diagnostic`;
`Landscape / representation diagnostic`; `Infrastructure / tooling`;
`Meeting / synthesis`; `Phase-transition evidence`.

## Phase 1 Decision Surface

| CRE part | Promotable artifact now? | Canonical route candidate | Main evidence | Phase 1 gaps before Phase 2 |
|---|---|---|---|---|
| Enhancer | Yes, as a transfer route rather than scratch. | BODA2/Malinois -> Lib1 fine-tune, with `B2_with_RC` as baseline and barcode-aware variants as secondary arms. | Scratch generalization is weak; transfer, barcode-bin, HQ8, and random-split analyses show usable signal and clarify barcode policy. | Pick canonical split policy for go/no-go claims; decide barcode policy for train versus val/test; decide whether to ensemble seeds/init heads. |
| Core promoter | Not yet for new in-house Lib1. | Legacy in-house e7/e30 pretraining exists, but needs a split-safe rerun before being used as the promoter seed. | Legacy e7/e30 HPO and one-shot Lib1 diagnostic exist; one-shot transfer is weak positive. | Rebuild e7/e30 train/val/test split, test RC augmentation, then run Lib1 scratch and legacy -> Lib1 fine-tune comparisons. |
| 5 Prime UTR | Close, but route choice is still open. | Either Hani Lib1/Lib2 ResNet scratch `4eq96xxd`, Hani Lib1 pretrained `1mmy39ku` after Phase 2 fine-tune, or June in-house fine-tuned model depending on promotion metric. | Hani public pretraining promoted `1mmy39ku`; Phase 2 v2 Lib2 fine-tune improves Lib2 with modest Lib1 retention cost; Phase 3 scratch is competitive with better Lib1 retention; June in-house HPO compares BODA/ResNet and PARADE with downsample follow-up. | Choose canonical production seed for combinatorial scenario 2; decide whether in-house RNA/DNA proxy can be a selection target or only a diagnostic. |
| Intron | Public pretraining yes; in-house no. | Seelig A5SS public pretraining as a starting point only. | Seelig public HPO works; promoter/intron one-shot shows negative in-house intron transfer. | Resolve target/length/assay mismatch; build real fine-tune pipeline before promotion. |
| 3 Prime UTR | Public Hani checkpoint yes; in-house no. | Hani 3 Prime ResNet `zlipechs` for public context; in-house route pending. | Public Hani pretraining and PARADE/BODA comparison exist; in-house ThreePrime EDA shows current public 240 nt checkpoints are not exact-length compatible with mostly 100 nt in-house inserts. | Decide 100 nt in-house model versus length-context redesign; run scratch/fine-tune comparison after length policy. |
| Cross-part / infrastructure | Yes, as context plumbing. | Use this matrix plus run registry, tutorial indices, sequence-landscape adapters, and combinatorial plan. | Repo reorg has separated plan, src, notebooks, generated roots, and cleanup ledger. | Keep Phase 1 board focused on active research threads; promote one checkpoint per included part before starting combinatorial scenario 2. |

## Question Coverage

| Modeling question | Best-covered parts | Evidence threads | Remaining gaps |
|---|---|---|---|
| Which pretrained checkpoint or architecture should seed a part? | Enhancer, Core promoter, 5 Prime UTR, 3 Prime UTR, Intron | Malinois/BODA2 enhancer baseline; legacy e7/e30 promoter HPO; Hani UTR ResNet/Basset/branched HPO; Seelig intron HPO; PARADE comparison. | Core promoter needs a split-safe legacy rerun and current promotion notebook; 3 Prime public checkpoint is not in-house compatible without a length policy. |
| Can in-house Library 1 train a model from scratch? | Enhancer, partly 5 Prime | Enhancer scratch sweeps show weak generalization; 5 Prime Phase 3 public Lib1+Lib2 scratch is competitive but is not in-house scratch. | Promoter, intron, 3 Prime, and in-house 5 Prime scratch routes are not fully resolved. |
| Does pretraining transfer to in-house Library 1? | Enhancer, 5 Prime, promoter/intron diagnostic | Enhancer fine-tuning works; in-house 5 Prime BODA/PARADE HPO is active; legacy e7/e30 promoter one-shot is weak positive and intron one-shot negative. | Promoter/intron need true fine-tuning; 3 Prime cannot be scored directly without length redesign. |
| Which output head or cell context should be used? | Enhancer, 5 Prime UTR, 3 Prime UTR | Enhancer init-head sweeps over K562/HepG2/SKNSH; Hani UTR observed-head and cell-conditioned ablations; in-house 5 Prime cell-head HPO. | Convert head comparisons into per-part defaults for Phase 2 candidate scoring. |
| What unfreeze scope should be used? | Enhancer, 5 Prime UTR | Enhancer sweeps over `head_only`, `branched_only`, `linear_all_head`, `conv3_plus`, `full`; 5 Prime Phase 2 and June in-house sweeps compare `head_only`, `last_stage_plus_head`, `full`. | Promoter/intron/3 Prime fine-tune scopes remain untested. |
| How do barcode counts affect training? | Enhancer, 5 Prime UTR | Enhancer comparable-bin, exact-low-barcode, random train/test quality, and HQ8 studies; in-house 5 Prime HPO uses thresholded training pools. | Promoter/intron/3 Prime need barcode-aware split definitions before training claims. |
| How do barcode counts affect evaluation metrics? | Enhancer, UTR EDA | Enhancer HQ4/HQ8 and quality-resolved test bins separate test measurement quality from train quality; UTR EDA defines HQ8 candidate pools. | Need the same held-out barcode policy for promoter, intron, and 3 Prime. |
| How much would more in-house data help? | Enhancer, 5 Prime UTR | Enhancer learning curves and barcode log-grid; in-house 5 Prime downsample follow-up. | Translate learning-curve slopes into experimental sample-size recommendations per CRE part. |
| Does fine-tuning retain public-task performance? | 5 Prime UTR | Phase 2 v2 reports Lib2 improvement and Lib1 retention cost; Phase 3 scratch comparison gives a retention-friendly alternative. | Enhancer retention is less relevant because the downstream target is in-house enhancer; promoter/intron/3 Prime lack fine-tune runs. |
| Which diagnostics change Phase 2 design? | Enhancer, 5 Prime, 3 Prime, Intron | Enhancer barcode policy; 5 Prime route selection and in-house proxy caution; 3 Prime length mismatch; intron one-shot mismatch. | Convert these into Phase 2 workstream gates with explicit run manifests. |

## Run And Notebook Evidence Ledger

The ledger uses run families rather than every individual W&B run. A single row
should become one Notion thread or one tightly related group of Notion threads.

| Thread ID | CRE part | Training Regime | Thread Function | Runner / launch files | Analysis notebooks and plans | Phase 1 readout |
|---|---|---|---|---|---|---|
| P1-ENH-00 public enhancer baseline | Enhancer | Public-data pretraining | Model benchmark; Model-selection / promotion | `src/learn/launch/enhancer_malinois_basset_branched_baseline.sh`; `src/learn/launch/enhancer_malinois_basset_nonbranched_single_head_combined_sweep.sh`; `src/learn/configs/enhancer/malinois_mpra/basset_branched/enhancer__malinois_mpra__basset_branched__transfer_baseline.yml` | `plan/phase1_lib1/learn/best_runs_snapshot.md`; `tutorials/lib1_tasks/extract_single_enhancer_output_HPO.ipynb` | BODA2/Malinois checkpoint is the enhancer pretrained baseline used for Lib1 adaptation. |
| P1-ENH-01 in-house enhancer scratch | Enhancer | In-house scratch training | Model benchmark; Data / assay diagnostic; Phase-transition evidence | `src/learn/launch/lib1_enhancer_scratch_compare_loss_modes.sh`; `src/learn/launch/lib1_enhancer_scratch_weighted_sweep.sh`; `src/learn/launch/lib1_enhancer_fastqs1_5_scratch_compare_loss_modes.sh`; `src/learn/launch/lib1_enhancer_fastqs1_5_scratch_no_flank_sweep.sh`; configs in `src/learn/configs/enhancer/bashor_in_house/` | `plan/phase1_lib1/finetune/lib1_enhancer_scratch_and_finetune_rehydration_june2026.md`; `src/learn/configs/enhancer/bashor_in_house/bashor_lab_collab_thread2_scratch_sweeps.md`; obsolete scratch notebook was removed after recap | Scratch does not generalize well enough to be the primary enhancer route; updated data and no-flank checks did not fix the gap. |
| P1-ENH-02 early BODA2 transfer | Enhancer | Public -> in-house fine-tuning | Model benchmark; Transfer / retention diagnostic; Model-selection / promotion | `src/finetune/finetune_sweep_scripts/lib1_enhancer/lib1_enhancer_transfer_multiseed.py` | `tutorials/lib1_tasks/tewhey_model_on_lib1_enhancer_multiseed_summary.ipynb`; `plan/phase1_lib1/finetune/lib1_enhancer_scratch_and_finetune_rehydration_june2026.md` | Pretrained enhancer features transfer; reverse-complement B2 is a strong baseline and beats scratch. |
| P1-ENH-03 enhancer learning curves and split realism | Enhancer | Public -> in-house fine-tuning | Experiment-design insight; Model benchmark; Phase-transition evidence | `src/finetune/finetune_sweep_scripts/lib1_enhancer/lib1_enhancer_learning_curve_finetune_updated.py`; `lib1_enhancer_learning_curve_finetune_split_options.py`; `lib1_enhancer_learning_curve_followup_diagnostic_random_split.py`; `lib1_enhancer_learning_curve_long_epoch_single_head_b3_random_split.py` | Historical notebooks now local/archive or root-level duplicates: `targeted_finetune_learning_curve_HQ_split_apr1_2026.ipynb`, `targeted_finetune_learning_curve_random_split_apr3_2026.ipynb`, `long_epoch_learning_curve_analysis_apr5.ipynb`; summaries in `tutorials/lib1_tasks/fine_tuning/markdown/` and enhancer rehydration plan | More data helps most under deeper unfreezing; random-all splits are stricter and more AL-realistic than HQ-first splits. |
| P1-ENH-04 enhancer barcode training policy | Enhancer | Public -> in-house fine-tuning | Data / assay diagnostic; Experiment-design insight; Phase-transition evidence | `lib1_enhancer_barcode_range_finetuning.py`; `lib1_enhancer_barcode_range_comparable_bins_finetuning.py`; `lib1_enhancer_exact_low_barcode_finetuning.py`; `run_lib1_barcode_range_stage1.sh`; `run_lib1_barcode_range_comparable_bins_parallel.sh`; `run_lib1_barcode_range_comparable_bins_loggrid_parallel.sh`; `run_lib1_exact_low_barcode_parallel.sh` | `tutorials/lib1_tasks/fine_tuning/enhancer_finetune_w_boda_pretrain/barcode_range_comparable_bins_decision_analysis_may2026.ipynb`; `may13_2026_bc_training_eval.ipynb` | Barcode count matters, but low-barcode rows are not simply useless; exact 3-barcode and 4-6 barcode bins can train useful models in equal-N settings. |
| P1-ENH-05 enhancer held-out quality and HQ8 HPO | Enhancer | Public -> in-house fine-tuning | Data / assay diagnostic; Model-selection / promotion; Phase-transition evidence | `lib1_enhancer_random_train_test_quality_finetuning.py`; `run_lib1_random_train_test_quality_parallel.sh`; `run_lib1_threshold_hq8_multiseed_hpo_parallel.sh`; `run_lib1_branched_only_k562_noearly_hq8_parallel.sh` | `tutorials/lib1_tasks/fine_tuning/enhancer_finetune_w_boda_pretrain/may13_2026_bc_test_eval.ipynb`; `may15_2026_hq8_multiseed_hpo_analysis.ipynb` | HQ8 heldout gives cleaner model-selection metrics; quality-resolved test bins show evaluation barcode count must be reported separately from training barcode count. |
| P1-PROM-01 legacy e7/e30 promoter pretraining | Core promoter | Legacy-data pretraining | Model benchmark; Model-selection / promotion | `src/learn/launch/promoter_deboer_utr_bassetvl_sweep.sh`; `src/learn/launch/promoter_deboer_compare_architectures.sh`; configs in `src/learn/configs/promoter/deboer_core/` | `plan/phase1_lib1/promoter_phase1.md`; `plan/phase1_lib1/learn/best_runs_snapshot.md`; `tutorials/lib1_tasks/pretraining_CRE_public_data/public_cre_hpo_presentation_summary.ipynb` | Legacy in-house e7/e30 promoter model evidence exists, but it needs a proper val/test split before becoming a Library 1 seed decision. |
| P1-PROM-02 promoter one-shot in-house transfer | Core promoter | No model training / analysis only | Data / assay diagnostic; Transfer / retention diagnostic; Phase-transition evidence | `src/finetune/finetune_sweep_scripts/promoter_intron_inhouse_one_shot_eval.py` | `tutorials/lib1_tasks/pretrain_CRE_inhouse_data/promoter_intron_inhouse_pretrained_eval_may2026.ipynb` | Legacy e7/e30 promoter checkpoint has weak positive one-shot signal on new Lib1 promoter data; needs real scratch versus fine-tune follow-up. |
| P1-UTR5-01 Hani public 5 Prime pretraining | 5 Prime UTR | Public-data pretraining | Model benchmark; Model-selection / promotion | `src/learn/launch/utr5_hani_utr_bassetvl_sweep.sh`; `utr5_hani_basset_branched_sweep.sh`; `utr5_hani_basset_branched_focused_sweep.sh`; `utr5_hani_resnet1d_sweep.sh`; `utr5_hani_basset_branched_delta_aux_sweep.sh`; `utr5_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh` | `tutorials/lib1_tasks/pretraining_CRE_public_data/public_cre_hpo_presentation_summary.ipynb`; `hani_utr_basset_branched_hpo_presentation_summary.ipynb`; `utr_hani_architecture_choices_may2026.md`; `src/learn/run_registry/best_runs.csv` | Validation-selected public 5 Prime model is ResNet1D run `1mmy39ku`; delta-aux and PARADE-like variants are ablations, not promoted. |
| P1-UTR5-02 PARADE/BODA checkpoint context | 5 Prime UTR | No model training / analysis only | Model benchmark; Transfer / retention diagnostic; Data / assay diagnostic | Released PARADE checkpoints under `external_models/parade/`; BODA best runs from `src/learn/run_registry/best_runs.csv` | `tutorials/lib1_tasks/pretraining_CRE_public_data/parade_released_checkpoint_eval_may2026.ipynb`; `plan/phase1_lib1/finetune/parade_released_checkpoint_eval_and_finetune_plan_may2026.md` | PARADE is useful as a benchmark/context model, but should not displace BODA unless it wins on validation or in-house ranking behavior. |
| P1-UTR5-03 Hani Lib2 Phase 2 fine-tune | 5 Prime UTR | Cross-regime comparison | Model benchmark; Transfer / retention diagnostic; Model-selection / promotion | `src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/hani_utr5_lib2_finetune.py`; `run_hani_utr5_lib2_finetune_parallel.sh`; `combine_hani_utr5_lib2_outputs.py` | `tutorials/lib1_tasks/fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/hani_utr5_lib2_phase2_v2_finetune_analysis_may2026.ipynb`; `plan/phase1_lib1/finetune/notion_connect_hani_lib1_2_inhouse_utr_update_may2026.md` | Fine-tuning from `1mmy39ku` improves Lib2 metrics with modest Lib1 retention cost; in-house proxy does not clearly favor the selected full-unfreeze checkpoint. |
| P1-UTR5-04 Hani Lib1+Lib2 Phase 3 scratch | 5 Prime UTR | Public-data pretraining | Cross-regime comparison; Model-selection / promotion; Phase-transition evidence | `src/learn/launch/utr5_hani_lib1_lib2_resnet1d_phase3_scratch_sweep.sh`; `src/learn/configs/utr5/hani_rna_activity/resnet1d/utr5__hani_rna_activity_lib1_lib2__resnet1d__phase3_scratch_bayes.yml` | `tutorials/lib1_tasks/pretraining_CRE_public_data/hani_utr5_lib1_lib2_phase3_scratch_hpo_analysis_may2026.ipynb` | Scratch Lib1+Lib2 ResNet run `4eq96xxd` is competitive with Phase 2 and has better Lib1 retention, making it a serious Phase 2 seed candidate. |
| P1-UTR5-05 in-house FivePrime HPO and downsample | 5 Prime UTR | Public -> in-house fine-tuning | Data / assay diagnostic; Model benchmark; Experiment-design insight; Phase-transition evidence | `src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/inhouse_utr5_parade_resnet_finetune.py`; `run_inhouse_utr5_parade_resnet_small_hpo_parallel.sh`; `combine_inhouse_utr5_finetune_outputs.py` | `tutorials/lib1_tasks/fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/inhouse_utr5_parade_resnet_small_hpo_analysis_jun2026.ipynb` | Compares BODA ResNet and PARADE on in-house FivePrime RNA/DNA with barcode-aware splits and downsample follow-up; current in-house 5 Prime decision surface. |
| P1-UTR-EDA-01 in-house UTR data readiness | 5 Prime UTR; 3 Prime UTR | No model training / analysis only | Data / assay diagnostic; Experiment-design insight; Phase-transition evidence | Notebook-only analysis | `tutorials/lib1_tasks/in_house_EDA/in_house_utr_eda_may2026.ipynb`; plots under `tutorials/lib1_tasks/in_house_EDA/plots/in_house_utr_eda_may2026/` | FivePrime is mostly exact 50 nt and scoreable; ThreePrime is mostly 100 nt and not compatible with current 240 nt public checkpoints without a new length policy. |
| P1-UTR3-01 Hani public 3 Prime pretraining | 3 Prime UTR | Public-data pretraining | Model benchmark; Model-selection / promotion | `src/learn/launch/utr3_hani_utr_bassetvl_sweep.sh`; `utr3_hani_basset_branched_sweep.sh`; `utr3_hani_basset_branched_focused_sweep.sh`; `utr3_hani_resnet1d_sweep.sh`; `utr3_hani_basset_branched_delta_aux_sweep.sh`; `utr3_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh` | `tutorials/lib1_tasks/pretraining_CRE_public_data/public_cre_hpo_presentation_summary.ipynb`; `hani_utr_basset_branched_hpo_presentation_summary.ipynb`; `utr_hani_architecture_choices_may2026.md`; `src/learn/run_registry/best_runs.csv` | Validation-selected public 3 Prime model is ResNet1D run `zlipechs`; it is a public benchmark, not yet an in-house Library 1 route. |
| P1-INTRON-01 Seelig intron public pretraining | Intron | Public-data pretraining | Model benchmark; Model-selection / promotion | `src/learn/launch/introns_seelig_a5ss_sd1_basset_branched_sweep.sh`; `src/learn/configs/introns/seelig_2015/basset_branched/introns__seelig_2015_a5ss_sd1__scratch__basset_branched.yml` | `tutorials/lib1_tasks/pretraining_CRE_public_data/intron_seelig_a5ss_sd1_pretraining_hpo_decision_may2026.ipynb`; `plan/phase1_lib1/learn/intron_seelig_hal_pretraining_plan.md` | Public intron pretraining works on Seelig A5SS SD1, but does not yet solve in-house intron activity. |
| P1-INTRON-02 intron one-shot in-house transfer | Intron | No model training / analysis only | Data / assay diagnostic; Transfer / retention diagnostic; Phase-transition evidence | `src/finetune/finetune_sweep_scripts/promoter_intron_inhouse_one_shot_eval.py` | `tutorials/lib1_tasks/pretrain_CRE_inhouse_data/promoter_intron_inhouse_pretrained_eval_may2026.ipynb` | One-shot in-house intron transfer is negative, likely reflecting target and sequence-context mismatch; fine-tune claims should wait. |
| P1-XPART-01 sequence landscape diagnostics | Cross-part / infrastructure | No model training / analysis only | Landscape / representation diagnostic; Infrastructure / tooling; Experiment-design insight | `src/analysis/sequence_landscape_adapters/` | `plan/active_learning/sequence_landscape_fast_implementation_plan.md`; `plan/active_learning/sequence_landscape_active_learning_exploration_exploitation.md` | Provides the analysis bridge for asking where in-house sequences sit relative to public train/test and which errors or retention failures are neighborhood-specific. |
| P1-XPART-02 combinatorial transition plan | Cross-part / infrastructure | No model training / analysis only | Infrastructure / tooling; Meeting / synthesis; Phase-transition evidence | Future `src/learn/configs/combinatorial/` and `src/finetune/finetune_sweep_scripts/combinatorial/` | `plan/combinatorial/multi_part_training_strategy_june2026.md` | Phase 2 scenario 2 should wait until included part encoders have promoted checkpoints or documented random/scratch initialization choices. |

## Phase 1 Gap Register

| Gap | CRE part | Needed run or analysis | Why it matters for Phase 2 |
|---|---|---|---|
| Canonical enhancer policy | Enhancer | Write final split/barcode decision note from May13/May15 notebooks; optionally run only the missing scratch comparison if it changes the decision. | Determines whether enhancer enters Phase 2 as a promoted transfer encoder and how heldout claims are reported. |
| In-house promoter comparison | Core promoter | Rebuild legacy e7/e30 train/val/test split, rerun lightweight promoter pretraining with RC ablation, then run legacy -> Lib1 fine-tune and Lib1 scratch HPO; create a promoter-specific decision notebook. | Needed before promoter can become a combinatorial encoder instead of random/scratch initialization. |
| 5 Prime canonical seed | 5 Prime UTR | Compare Phase 2 v2, Phase 3 scratch `4eq96xxd`, and June in-house HPO under the same promotion criteria. | Chooses the first 5 Prime encoder for combinatorial scenario 2 and candidate scoring. |
| In-house intron mismatch | Intron | Debug target, length, and transform policy; then build a real fine-tune pipeline. | Prevents importing a public splice model that is not predictive for the in-house intron assay. |
| 3 Prime length policy | 3 Prime UTR | Decide whether to train a 100 nt in-house ThreePrime model or redesign context to match 240 nt public models. | Current public 3 Prime checkpoint cannot be fairly evaluated on the in-house table. |
| Shared barcode policy | All in-house parts | Define train barcode thresholds separately from validation/test barcode thresholds in every run manifest. | Avoids mixing model improvement with measurement-quality artifacts. |
| Phase 2 workstream order | Cross-part | Promote included single-part artifacts or document explicit pending/random-init choices. | Prevents combinatorial modeling from absorbing unresolved single-part uncertainty. |

## Recommended Notion Thread Seeds

These are good first rows for the Notion Phase 1 board.

| Thread title | CRE part | Priority | Training Regime | Thread Function |
|---|---|---|---|---|
| Enhancer Lib1 transfer and barcode policy | Enhancer | High | Public -> in-house fine-tuning | Model-selection / promotion; Data / assay diagnostic; Phase-transition evidence |
| Enhancer Lib1 scratch feasibility | Enhancer | Medium | In-house scratch training | Model benchmark; Phase-transition evidence |
| 5 Prime canonical seed selection | 5 Prime UTR | High | Cross-regime comparison | Model-selection / promotion; Transfer / retention diagnostic; Phase-transition evidence |
| In-house FivePrime BODA/PARADE HPO | 5 Prime UTR | High | Public -> in-house fine-tuning | Model benchmark; Data / assay diagnostic; Experiment-design insight |
| Promoter in-house scratch/fine-tune gap | Core promoter | High | Cross-regime comparison | Model benchmark; Phase-transition evidence |
| Intron one-shot mismatch and fine-tune gate | Intron | High | No model training / analysis only | Data / assay diagnostic; Transfer / retention diagnostic; Phase-transition evidence |
| 3 Prime length-policy gate | 3 Prime UTR | High | No model training / analysis only | Data / assay diagnostic; Experiment-design insight; Phase-transition evidence |
| Phase 2 combinatorial entry gate | Cross-part / infrastructure | Medium | No model training / analysis only | Infrastructure / tooling; Meeting / synthesis; Phase-transition evidence |
