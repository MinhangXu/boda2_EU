# BODA-First UTR Fine-Tuning and PARADE Checkpoint Evaluation Plan

## Goal

Use our validation-selected BODA UTR pretrained models first for downstream adaptation, then decide later whether PARADE should enter the stack as an inference baseline, a teacher model, or a fine-tunable pretrained backbone.

The immediate modeling priority is BODA-first:

1. Fine-tune the current 5'UTR BODA ResNet1D winner `1mmy39ku` on Hani/Goodarzi 5'UTR Lib2.
2. Evaluate the Lib2 fine-tuned model on held-out Lib2 validation/test splits and on Lib1 retention metrics.
3. Test whether Lib2 fine-tuning improves ranking/correlation on in-house exact-length `FivePrime` candidates.
4. Only after that baseline is understood, decide whether to run a larger HPO pretraining pass on Lib1+Lib2 or bring PARADE into the fine-tuning stack.

Primary notebook:

`tutorials/lib1_tasks/pretraining_CRE_public_data/parade_released_checkpoint_eval_may2026.ipynb`

Primary outputs:

`tutorials/lib1_tasks/pretraining_CRE_public_data/presentation_plots/parade_released_checkpoint_eval_may2026/`

In-house UTR EDA notebook:

`tutorials/lib1_tasks/in_house_EDA/in_house_utr_eda_may2026.ipynb`

In-house UTR EDA outputs:

`tutorials/lib1_tasks/in_house_EDA/plots/in_house_utr_eda_may2026/`

Released checkpoints:

- 3'UTR: `/home/minhang/synBio_AL/external_models/parade/parade/predictor/regression_multiple/saved_models/model-utr3-deltas-epoch=9-step=1330.ckpt`
- 5'UTR: `/home/minhang/synBio_AL/external_models/parade/parade/predictor/regression_multiple/saved_models/model-utr5-deltas-epoch=9-step=840.ckpt`

## Phase 1: Notebook Evaluation

The notebook should answer four immediate questions.

1. Do the released PARADE checkpoints reproduce the author-reported Lib1 and Lib2 activity metrics closely enough in our local environment?
2. How do released PARADE checkpoint predictions compare with our current validation-selected BODA UTR pretrained models?
3. Does PARADE's delta output explain anything different from absolute activity on Lib1/Lib2?
4. Can the released 5'UTR checkpoint and our current 5'UTR BODA model score in-house `FivePrime` candidates in a useful way?

Data inputs:

- Hani/Goodarzi processed Lib1/Lib2 tables:
  - `/home/minhang/synBio_AL/opt_EU_learn_n_design/utr_hani_2025/processed_utr_data/3UTR_lib1_branched_observed_heads.csv`
  - `/home/minhang/synBio_AL/opt_EU_learn_n_design/utr_hani_2025/processed_utr_data/5UTR_lib1_branched_observed_heads.csv`
  - `/home/minhang/synBio_AL/opt_EU_learn_n_design/utr_hani_2025/processed_utr_data/3UTR_lib2_processed.csv`
  - `/home/minhang/synBio_AL/opt_EU_learn_n_design/utr_hani_2025/processed_utr_data/5UTR_lib2_processed.csv`
- Current BODA UTR models from:
  - `src/learn/run_registry/best_runs.csv`
- In-house candidate tables:
  - `/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/FivePrimes/L1_final_fastqs1-5_sublibrary_FivePrime_subset.csv`
  - `/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/ThreePrimes/L1_final_fastqs1-5_sublibrary_ThreePrime_subset.csv`

Metrics:

- Per-cell Pearson for activity.
- Paper-style average activity Pearson: correlate the per-sequence mean predicted activity with the per-sequence mean observed activity across overlapping cell types.
- Flattened activity Pearson across sequence-cell pairs.
- PARADE-only delta Pearson, where the observed delta is `activity_c - mean(activity over evaluated heads)`.
- For in-house candidates, Pearson/Spearman between predicted average activity and `log2(RNA/DNA)`, clearly labeled as a construct-level proxy rather than a direct Hani bin-mass target.

Case handling:

- Treat DNA sequence case as biological no-op.
- Uppercase sequences before model inference.
- Preserve the original sequence string as the join key for scoring, because Goodarzi Lib2 contains mixed-case sequence annotations and silently uppercasing the key can break truth/prediction alignment.

Important cell-type mapping:

- `c1`: MDA-MB-231
- `c2`: HepG2
- `c4`: Jurkat
- `c5`: BxPC-3
- `c6`: SW480
- `c13`: PA-1
- `c15`: A549
- `c17`: NALM6

## Current Expected Comparison Models

Use only current canonical BODA UTR rows from `best_runs.csv` for the headline comparison:

- 3'UTR BODA current: `zlipechs`, `ResNet1DRegressor`
- 5'UTR BODA current: `1mmy39ku`, `ResNet1DRegressor`

Keep the previous cell-conditioned delta-aux ResNet as an ablation, not as the headline canonical model, unless we explicitly add it back into this notebook as a separate diagnostic.

## Sequence-Length Guardrail

This is a real modeling constraint, not bookkeeping noise.

- PARADE 5'UTR checkpoint expects 50-nt inserts.
- BODA current 5'UTR ResNet1D was trained on 50-nt inserts.
- PARADE 3'UTR checkpoint expects 240-nt inserts.
- BODA current 3'UTR ResNet1D was trained on 240-nt inserts.
- Our in-house `FivePrime` candidates are mostly 50 nt, so exact-length scoring is appropriate.
- Our in-house `ThreePrime` candidates are mostly 100 nt, so exact-length scoring is not appropriate for the current 3'UTR checkpoints.

Default policy:

- Score exact-length valid DNA sequences only.
- Report excluded candidate counts by reason.
- Do not pad/truncate `ThreePrime` sequences in the headline analysis.

Optional later sensitivity analysis:

- Add a deliberately labeled `N`-padding or flanking-context experiment for `ThreePrime`.
- Treat that as exploratory distribution scoring, not a valid apples-to-apples checkpoint evaluation.

## Phase 1.5: In-House UTR EDA Gate

Before committing to Lib2 fine-tuning or combined Lib1+Lib2 pretraining, use the in-house EDA notebook to decide the valid data surface.

Current EDA findings:

- `FivePrime` has 8,461 rows, 8,461 unique candidate sequences, and 8,331 exact 50-nt candidates. This is mostly compatible with the current 5'UTR pretrained models.
- `ThreePrime` has 7,258 rows, 7,258 unique candidate sequences, and a modal length of 100 nt. It has zero exact 240-nt candidates, so it is not directly compatible with the current 3'UTR pretrained models.
- A barcode threshold of 8+ leaves 1,818 valid/finite `FivePrime` rows and 810 valid/finite `ThreePrime` rows. This should be treated as a high-quality candidate pool, not the only possible training set.
- The in-house 5'UTR GC distribution is lower than Hani 5'UTR Lib1/Lib2, so even exact-length 5'UTR scoring is still a domain-shift test.

Decision gate:

- For 5'UTR, proceed with exact-length scoring/fine-tuning experiments after writing explicit barcode-aware train/val/test split files.
- For 3'UTR, decide the length policy before any scoring claim. The preferred first serious branch is a dedicated 100-nt `ThreePrime` model or full-construct model, rather than padding/truncating into a 240-nt public-checkpoint input.
- Preserve an untouched in-house holdout before any active-learning calibration, teacher distillation, or fine-tuning on in-house rows.

## Phase 2: BODA-First 5'UTR Lib2 Fine-Tuning

Start from the current canonical 5'UTR BODA model:

- Run ID: `1mmy39ku`
- Model: `ResNet1DRegressor`
- Source training data: Hani/Goodarzi 5'UTR Lib1
- Current role: validation-selected 5'UTR pretrained model in `best_runs.csv`

This phase has two parts: the completed/current Phase 2 v1 run, then a recommended Phase 2 v2 broader HPO run with a stricter split policy.

### Phase 2 v1: Current Implementation

Primary code and artifacts:

- Training script: `src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/hani_utr5_lib2_finetune.py`
- Launcher: `src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/run_hani_utr5_lib2_finetune_parallel.sh`
- Combiner: `src/finetune/finetune_sweep_scripts/hani_lib1_in_house_lib1_5Prime/combine_hani_utr5_lib2_outputs.py`
- Analysis notebook: `tutorials/lib1_tasks/fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/hani_utr5_lib2_phase2_finetune_analysis_may2026.ipynb`
- Output root: `src/finetune/learning_curve/hani_utr5_lib2_resnet1d_1mmy39ku_phase2_may2026/`

Lib2 preprocessing in v1:

- Source file: `/home/minhang/synBio_AL/opt_EU_learn_n_design/utr_hani_2025/processed_utr_data/5UTR_lib2_processed.csv`
- Raw Lib2 rows: 97,133
- Unique exact 50-nt Lib2 sequences before requiring all five heads: 11,367
- Retained Lib2 sequences with all five heads: 10,174
- Heads retained: `c1`, `c2`, `c4`, `c6`, `c17`
- The large raw-to-modeled-row drop is expected because raw rows include replicate and cell-type observations; modeling is done after sequence/head aggregation.

Split policy in v1:

- One deterministic sequence-level split seed: `split_seed=42`
- Split fractions: 80% train, 10% validation, 10% test
- Resulting sequence counts: 8,140 train, 1,017 validation, 1,017 test
- Split implementation: hash uppercased sequence, sort by hash, assign test first, validation second, train last
- Important limitation: all training seeds used the same sequence split, so v1 estimates optimizer/training variability but not split-to-split variability.

Sweep shape in v1:

- Training seeds: `7`, `11`, `13`
- Unfreeze scopes: `head_only`, `last_stage_plus_head`, `full`
- Head learning rates: `1e-4`, `3e-4`
- Backbone learning rate: `1e-5`
- Target scaler: `pretrained_lib1_train`
- Epoch cap: 100
- Early stopping patience: 20
- Optimizer: AdamW
- Scheduler: constant learning rate

Current interpretation:

- The best validation-selected config is `full` unfreezing with `head_lr=3e-4`.
- The selected single checkpoint is seed 7, `full`, `head_lr=3e-4`.
- Lib2 Pearson gains are real but modest in absolute `r`; COD R2 and RMSE improve more strongly, which suggests much of the gain is calibration/error-scale improvement rather than a pure ranking reshuffle.
- Lib1 retention should remain a promotion gate because full unfreezing can trade off some Lib1 performance.
- Do not promote from v1 alone as the final answer. Treat it as evidence that Lib2 transfer works and as a guide for the v2 HPO space.

### Phase 2 v2: Broad HPO With Fixed Splitting

The v2 run should separate the final evaluation question from the hyperparameter search question.

Outer split:

- Create one untouched final Lib2 test manifest using a fixed `final_test_seed`.
- Assign splits at the unique-sequence level only.
- Stratify the final test draw by average activity quantile and GC-content quantile.
- Hold out 10% of retained Lib2 sequences as final test.
- Exclude the final test set from HPO ranking, early stopping, checkpoint selection, and plot-driven decisions.

Inner HPO splits:

- Use the remaining 90% of Lib2 sequences as the HPO pool.
- Create three inner train/validation manifests with distinct `inner_split_seed`s.
- Each inner validation set should be 10% of the HPO pool.
- Use the same stratification variables as the outer split: average activity quantile and GC-content quantile.
- Save `outer_final_test_manifest.csv`, `inner_split_manifest_{id}.csv`, `split_policy.json`, and `split_audit.csv`.

Why training seeds are still needed:

- A split seed controls which sequences are assigned to train/validation/test.
- A training seed controls stochastic training behavior on a fixed split: batch order, optimizer noise, any randomly initialized or reset layers, dropout if present, and GPU nondeterminism.
- Three split seeds do not replace training seeds; they answer a different variance question.
- For cost control, v2 can screen many configs with one training seed per inner split, then rerun the best configs with multiple training seeds.

Recommended v2 stages:

1. Screening: run all candidate configs across the three inner split seeds with one training seed per split. Rank by mean inner-validation average-activity Pearson, with Lib1 retention as a secondary gate.
2. Confirmation: take the top three hyperparameter configs from screening and rerun each across the same three inner split seeds with three training seeds. Pick the final config using mean performance, variance, and retention.
3. Final evaluation: freeze the config and selection rule, train on the HPO pool, and evaluate the untouched final Lib2 test once. Multiple final training seeds can be reported as uncertainty, but they should not be used to choose a different config.

The "three finalists" are the top three hyperparameter configs after screening, not three final test sets and not three split seeds.

Recommended v2 search space:

- Unfreeze scopes: `last_stage_plus_head`, `full`; keep `head_only` as a lightweight control if compute allows.
- Head learning rates: `1e-4`, `3e-4`; optionally `1e-3` only for head-only or carefully monitored runs.
- Backbone learning rates: `3e-6`, `1e-5`, `3e-5`.
- Target scalers: `pretrained_lib1_train`, `lib2_train`.
- Epoch cap: 200 or 250.
- Patience: 30 to 40.
- Optional freeze warmup: 0, 3, or 5 epochs.
- Weight decay: keep `1e-4`; optionally add `3e-5`.
- Scheduler variants if implemented: constant, cosine decay with warmup, and ReduceLROnPlateau.

Implementation changes needed for v2:

- Decouple split manifest creation from model training.
- Teach the training script to consume explicit outer and inner split manifests.
- Add `stage`, `outer_split_seed`, `inner_split_seed`, `training_seed`, and `split_id` to run tags, metric rows, and output directories.
- Disable final-test evaluation during screening and confirmation.
- Keep final-test metrics absent or masked until the final evaluation stage.
- Add scheduler options only if scheduler HPO is included in the run.
- Combined outputs should include config, split IDs, training seed, selected epoch, validation metrics, Lib1 retention metrics, and final-test metrics only for final-evaluation runs.

Promotion criteria after v2:

- Clear mean improvement on held-out Lib2 final test versus pretrained `1mmy39ku`.
- No unacceptable Lib1 retention penalty.
- Stable performance across inner split seeds and confirmation training seeds.
- Calibration/error metrics improve alongside Pearson/Spearman, not only one metric in isolation.

## Phase 3: Optional BODA Lib1+Lib2 Pretraining HPO

A combined Lib1+Lib2 pretraining run may produce a stronger production pretrained model than Lib1-only HPO, but it changes the evaluation question.

Why the authors likely did not train the headline predictor this way:

- Lib2 is useful as an external generalization benchmark.
- Training on Lib2 removes the clean Lib1-to-Lib2 transfer test unless a new held-out Lib2 split is reserved.
- The paper needed evidence that the predictor generalized to a separate library, while our active-learning objective may care more about the strongest production model.

If we do Lib1+Lib2 pretraining:

- Use a library-aware split strategy that prevents sequence leakage.
- Reserve a Lib2 test set that is never used for HPO or early stopping.
- Track Lib1-only, Lib2-only, and combined validation/test metrics separately.
- Compare against Lib2 fine-tuning from `1mmy39ku`, not only against scratch Lib1+Lib2 training.
- Do not use in-house holdout performance for HPO selection; keep it as an external deployment check.

Recommended order:

1. Fine-tune `1mmy39ku` on Lib2 and evaluate carefully.
2. If helpful, sweep transfer-learning hyperparameters.
3. Then consider Lib1+Lib2 pretraining HPO as a production-pretraining branch.

## Phase 4: Deferred PARADE Integration Route

After BODA-first Lib2 fine-tuning and optional Lib1+Lib2 HPO are understood, pick one of three PARADE routes.

Route A: inference baseline only.

- Keep PARADE checkpoints outside BODA.
- Use the notebook/script to add PARADE scores as features, benchmarks, or candidate annotations.
- Lowest engineering cost and lowest risk.

Route B: teacher model.

- Use PARADE predictions, especially per-cell activity and delta, as auxiliary labels or distillation targets.
- Fine-tune BODA/ResNet models on our Lib1/in-house data with an extra teacher-alignment term.
- Useful if PARADE generalizes better to Goodarzi Lib2 but direct checkpoint fine-tuning is cumbersome.

Route C: port or wrap the PARADE LegNet backbone.

- Add a BODA-compatible wrapper around PARADE `RNARegressor` / `LegNetClassifier`.
- Load encoder/backbone weights from the released checkpoints.
- Replace or augment the head for downstream Lib1 tasks.
- Highest engineering cost, but closest to true pretrained model fine-tuning.

Recommended PARADE move after the BODA-first baseline:

Keep Route A as a reporting baseline. Only do Route B or Route C if PARADE adds value beyond BODA Lib2 fine-tuning or a BODA Lib1+Lib2 production pretraining run.

## Phase 5: PARADE Fine-Tuning Stack Design

If PARADE is promoted beyond inference baseline, implement a new stack rather than forcing the checkpoint into `CNNTransferLearning`.

Proposed modules:

- `boda/model/parade_legnet.py`
  - BODA wrapper for the PARADE LegNet-style backbone.
  - Supports loading released checkpoint weights.
  - Exposes encoder-only and full two-head modes.
- `boda/data/utr_cell_conditioned_datamodule.py`
  - Generalized sequence-cell data module with channels matching PARADE: sequence, positional, and cell-condition channels.
  - Reuses the BODA cell-conditioned dataset where possible.
- `boda/graph/parade_transfer.py`
  - Fine-tuning graph for activity-only, delta+activity, and optional teacher-distillation losses.
- `src/learn/configs/utr5/bashor_in_house/parade_legnet/`
  - 5'UTR in-house fine-tuning configs.
- `src/learn/configs/utr3/bashor_in_house/parade_legnet/`
  - Only after we decide a valid 3'UTR length/context policy.

## Differences From The Authors' Original Setup

Even if we load their checkpoint exactly, our downstream use will differ unless we intentionally reproduce all of these choices:

- Architecture: PARADE uses a LegNet-style convolutional classifier/regressor; our canonical BODA winners use ResNet1D.
- Input channels: PARADE uses sequence + positional + broadcast cell-condition channels.
- Outputs: PARADE predicts two values per sequence-cell pair, delta and absolute activity; our canonical BODA models predict absolute activity heads in a branched multi-output format.
- Cell conditioning: PARADE is long-form cell-conditioned; canonical BODA ResNet is wide multi-head.
- Reverse complement: PARADE prediction uses no reverse-complement test-time augmentation in the public prediction helper; training notebooks may use augmentation settings that should be checked before claiming exact replication.
- Sequence context: PARADE checkpoints are tied to the Hani/Goodarzi insert lengths and construct conventions.
- Target scale: PARADE activity is trained on the paper's processed bin-mass style target; our in-house `RNA/DNA` is a construct-level expression proxy.

## Deferred PARADE Decision Criteria

Promote PARADE into downstream fine-tuning only if at least one of the following is true:

- Released checkpoint metrics reproduce author Lib1/Lib2 values locally.
- PARADE materially outperforms BODA current models on Lib2 generalization.
- PARADE predictions correlate with in-house 5'UTR `RNA/DNA` better than BODA current predictions.
- PARADE delta outputs provide a useful cell-specific signal that absolute activity alone misses.

Do not promote PARADE simply because it is closer to the paper. It should earn its way in through validation or clearly useful candidate-ranking behavior.
