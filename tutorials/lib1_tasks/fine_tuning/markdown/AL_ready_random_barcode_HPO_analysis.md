# AL-Ready Random-Split Follow-Up Analysis

This note is a handoff for building a new analysis notebook around the finished follow-up diagnostic run:

- output dir: `/home/minhang/synBio_AL/boda2_EU/src/finetune/learning_curve/lib1_enhancer_followup_diagnostic_random_split_apr2026_v1`
- runner script: `/home/minhang/synBio_AL/boda2_EU/src/finetune/finetune_sweep_scripts/lib1_enhancer_learning_curve_followup_diagnostic_random_split.py`
- base training loop: `/home/minhang/synBio_AL/boda2_EU/src/finetune/finetune_sweep_scripts/lib1_enhancer_learning_curve_finetune_updated.py`

The point of this run is **not** to re-test whether random split is harder than the earlier `hq_first` setup. That had already been addressed by the earlier random-split run. This follow-up is a **mechanism / training-dynamics / AL-readiness diagnostic** under the random-split regime.

## Exact run command

```bash
cd "/home/minhang/synBio_AL"
CUDA_VISIBLE_DEVICES=0 python "boda2_EU/src/finetune/finetune_sweep_scripts/lib1_enhancer_learning_curve_followup_diagnostic_random_split.py" \
  --outdir "/home/minhang/synBio_AL/boda2_EU/src/finetune/learning_curve/lib1_enhancer_followup_diagnostic_random_split_apr2026_v1" \
  --init_heads K562 HepG2 SKNSH \
  --seeds 17 23 19 31 \
  --train_size_fracs 0.75 1.0 \
  --frozen_epochs_grid 2 0 \
  --b3_bcap 8 \
  --val_frac 0.1 \
  --test_frac 0.1 \
  --head_lr 5e-4 \
  --backbone_lr 1e-4 \
  --patience 50 \
  --max_epochs 200
```

## What this follow-up changes relative to the earlier random-split run

Earlier random-split baseline:

- `/home/minhang/synBio_AL/boda2_EU/src/finetune/learning_curve/lib1_enhancer_targeted_random_all_per_seed_apr2026`

This follow-up changes the setup in these ways:

1. Keeps `split_strategy = random_all_per_seed`, but reduces holdouts from `0.15 / 0.15` to `0.10 / 0.10`.
2. Narrows to the top-end train-size regime only:
   - actual train sizes in this run are `2297` and `3063`
3. Uses a fixed strong LR pair instead of another HPO sweep:
   - `head_lr = 5e-4`
   - `backbone_lr = 1e-4`
4. Tests warmup directly:
   - `frozen_epochs = 2`
   - `frozen_epochs = 0`
5. Adds comprehensive per-epoch logging for:
   - train metrics
   - validation metrics
   - test metrics
6. Logs per-epoch `R2`, `Pearson`, and `Spearman`, plus test metrics at each epoch.

So this run is meant to answer:

- does `frozen_epochs = 2` versus `0` matter?
- are deeper scopes really better once we look at train/val/test dynamics?
- is the model more useful for **ranking** or for **calibrated regression**?
- are we closer to having an AL-ready scoring model?

## Output files

Main outputs in the run directory:

- `learning_curve_runs.csv`
- `learning_curve_histories.csv`
- `learning_curve_summary_mean_std.csv`
- `unfreeze_scope_summary_mean_std.csv`
- `zero_shot_by_seed.csv`
- `run_manifest.json`

Useful manifest facts:

- `split_strategy = random_all_per_seed`
- `val_frac = 0.1`
- `test_frac = 0.1`
- `per_epoch_train_metrics_logged = true`
- `per_epoch_test_metrics_logged = true`

## Cache pickles

Per-run cache payloads live under:

- `/home/minhang/synBio_AL/boda2_EU/src/finetune/learning_curve/lib1_enhancer_followup_diagnostic_random_split_apr2026_v1/cache/runs`

Each payload contains:

- `fit_info`
- `history_df`
- `pred_df`
- `train_metrics`
- `val_metrics`
- `test_metrics`

Important note:

- `pred_df` is the **test-split prediction table** for that run, not a train/val prediction table.
- observed columns in `pred_df`:
  - `Enhancers`
  - `n_barcodes`
  - `RNA_DNA_Ratio_log10_scaled`
  - `row_id`
  - `pred`

This means the cache pickles are enough for held-out ranking / top-K / disagreement analyses.

## Working definition of an AL-ready model

For this notebook, I want to use a pragmatic definition:

- stable held-out ranking across seeds and init heads
- decent top-K enrichment on unseen sequences
- a validation criterion that is at least reasonably aligned with the downstream ranking objective
- a useful uncertainty / disagreement signal
- no obvious dependence on a brittle training recipe

## Already-known result that should be included early

At the largest train size (`3063`), averaged across seeds x init heads (`n = 12` runs per cell):

### B2_with_RC | full

- `frozen = 0`:
  - train `R2 = 0.258`
  - val `R2 = 0.087`
  - test `R2 = 0.102`
  - train `Spearman = 0.423`
  - test `Spearman = 0.258`
  - mean `best_epoch = 0.33`

- `frozen = 2`:
  - train `R2 = 0.227`
  - val `R2 = 0.095`
  - test `R2 = 0.121`
  - train `Spearman = 0.382`
  - test `Spearman = 0.263`
  - mean `best_epoch = 1.58`

Interpretation:

- removing warmup increases train fit but hurts held-out metrics for `full`
- the deep model still early-stops extremely early on average
- this is operationally important for the AL loop because it changes which training recipe should be trusted to score candidates

### B3_with_RC_weighted_bcap_8 | full

- `frozen = 0`:
  - train `R2 = 0.233`
  - val `R2 = 0.085`
  - test `R2 = 0.100`
  - train `Spearman = 0.405`
  - test `Spearman = 0.257`
  - mean `best_epoch = 0.25`

- `frozen = 2`:
  - train `R2 = 0.222`
  - val `R2 = 0.094`
  - test `R2 = 0.118`
  - train `Spearman = 0.377`
  - test `Spearman = 0.264`
  - mean `best_epoch = 1.58`

### Sanity-check result

For `branched_only`, `frozen_epochs` should make no difference. The notebook should show that as a sanity check.

## High-priority notebook analyses

These are the main analyses I want.

### 1. Warmup delta analysis

Goal:

- directly compare `frozen_epochs = 0` versus `2`

Suggested plots / tables:

- delta plots for:
  - `train_r2`
  - `val_r2`
  - `test_r2`
  - `train_spearman`
  - `test_spearman`
  - `best_epoch`
- organize by:
  - `setting`
  - `unfreeze_scope`
  - `train_size`

Question:

- does dropping warmup improve generalization or mostly just increase train fit?

### 2. Checkpoint-selection alignment

Goal:

- determine whether selecting the checkpoint by validation loss is aligned with what we might care about for AL

For each run, compare:

- epoch of minimum `val_loss_standardized`
- epoch of maximum `test_r2`
- epoch of maximum `test_spearman`

Then summarize:

- how often the test-optimal epoch is later than the val-loss-selected epoch
- how much held-out `R2` and `Spearman` are left on the table by selecting on `val_loss`

Important caveat:

- this is an **offline diagnostic only**
- do **not** use test results to choose the deployed model
- the goal is to understand alignment of the selection rule

### 3. Rank-vs-calibration analysis

Goal:

- figure out whether the model is more useful for ranking than for calibrated regression

Suggested analyses:

- compare `test_r2` versus `test_spearman` across:
  - epochs
  - settings
  - scopes
  - warmup values
- inspect whether some recipes improve ranking more than effect-size fit

Interpretation:

- if `Spearman` improves while `R2` stays flat, the model may already be useful for rank-based acquisition

### 4. Stability across seeds and init heads

Goal:

- estimate whether a given recipe is stable enough to use in an AL loop

Suggested plots:

- boxplots / stripplots / mean +/- SD for:
  - `test_r2`
  - `test_spearman`
  - `best_epoch`
- stratify by:
  - `setting`
  - `unfreeze_scope`
  - `frozen_epochs`
  - `train_size`

Question:

- are some recipes more brittle than others?

### 5. Simplicity-versus-performance frontier

Goal:

- decide whether `full` is really worth the extra adaptation / sensitivity versus `branched_only` or `conv3_plus`

Suggested view:

- compare the best warmup for each scope
- examine:
  - mean test metrics
  - variance across seeds / heads
  - mean best epoch

Question:

- if a simpler scope is nearly as good and more stable, that may be the better AL deployment choice

### 6. Optional comparison to the earlier random-split run

This is optional and should be presented carefully.

Earlier random-split run:

- `/home/minhang/synBio_AL/boda2_EU/src/finetune/learning_curve/lib1_enhancer_targeted_random_all_per_seed_apr2026`

If comparing:

- do it qualitatively, not as an apples-to-apples leaderboard
- remind the reader that this follow-up changed:
  - holdout fractions
  - train sizes
  - logging
  - LR search versus fixed LR pair
  - warmup grid

## Most AL-relevant analyses from cache pickles

These should use the cached `pred_df` test predictions.

### 1. Top-K hit-rate / lift curves

Goal:

- evaluate how useful the model is for exploitation on unseen sequences

Suggested metrics:

- mean observed activity in top-K predictions
- enrichment over random
- hit rate in top `5%`, `10%`, and `20%`

This is probably the closest offline proxy to exploitation quality in an AL loop.

### 2. Tail-ranking analysis

Goal:

- check whether the model ranks the highest-activity test sequences well

Suggested analyses:

- precision / recall style summaries for the top observed tail
- overlap between top predicted and top observed sequences

This is important because global `Spearman` can hide whether the high-value tail is actually being ordered correctly.

### 3. Prediction disagreement as an uncertainty proxy

Goal:

- see whether disagreement across init heads acts like a usable uncertainty signal

Suggested approach:

- for the same test split, align predictions by `row_id`
- compare predictions across init heads
- compute:
  - per-sequence prediction variance / range
  - error versus disagreement

Question:

- do sequences with larger disagreement also have larger prediction error?

If yes, that is useful for exploration-heavy acquisition.

### 4. Top-K stability

Goal:

- determine whether the top predicted candidates are stable across heads or settings

Suggested analyses:

- overlap of top-K predicted sequences across:
  - init heads
  - scopes
  - warmup values

Question:

- if top-K sets are unstable, pure exploitation is riskier

### 5. Calibration in the top tail

Goal:

- test whether the model is systematically over- or under-estimating the highest predicted sequences

Suggested analyses:

- predicted versus observed in top prediction bins
- top-decile calibration views

Question:

- does the model overestimate the highest-scoring candidates?

## Important implementation notes for the notebook

1. Do not blindly pool predictions across different test splits.
   - the split composition changes across seeds
   - comparisons that rely on per-sequence alignment should only compare runs with the same test split

2. Use the split hashes in `learning_curve_runs.csv` when needed:
   - `train_row_id_hash`
   - `val_row_id_hash`
   - `test_row_id_hash`

3. For disagreement analyses across init heads, only compare runs that share:
   - `seed`
   - `setting`
   - `unfreeze_scope`
   - `frozen_epochs`
   - `train_size`
   - same `test_row_id_hash`

4. Remember that `best_epoch` is the validation-loss-selected epoch and is averaged when summarized.

## Suggested notebook structure

1. Load run metadata and summarize what changed versus earlier runs
2. Warmup delta analysis
3. Per-epoch train / val / test trajectory analysis
4. Checkpoint-selection alignment
5. Rank-vs-calibration analysis
6. Stability across seeds and init heads
7. Cache-pickle analyses:
   - top-K enrichment
   - disagreement
   - top-K stability
   - top-tail calibration
8. Final recommendation:
   - which recipe looks most AL-ready?
   - what signal should acquisition use most heavily: score, rank, uncertainty, or diversity?

## Desired final takeaway from the notebook

The notebook should answer:

1. Which model recipe is most trustworthy under the random-split regime?
2. Is the model more useful for ranking than for calibrated regression?
3. Does `frozen_epochs = 2` outperform `0` for the deep scopes?
4. Is there a usable disagreement / uncertainty signal for exploration?
5. Are we comfortable using one of these recipes as the scoring model in the first AL loop iteration?
