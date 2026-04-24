# Targeted Run Interpretation: HQ-First Split and Next Checks

This note summarizes the current interpretation of the targeted April 2026 `hq_first` run and connects it to the in-progress `random_all_per_seed` rerun.

Main analysis surface:

- `boda2_EU/tutorials/lib1_tasks/fine_tuning/targeted_finetune_learning_curve_apr2026_analysis.ipynb`

Targeted run artifacts:

- `boda2_EU/src/finetune/learning_curve/lib1_enhancer_targeted_apr2026`

In-progress rerun:

- `boda2_EU/src/finetune/learning_curve/lib1_enhancer_targeted_random_all_per_seed_apr2026`

## Current interpretation from the `hq_first` targeted run

The cleanest summary so far is:

- the selected checkpoint that generalizes best often does not fit the training set very strongly
- held-out performance still improves with increasing train size, especially for `conv3_plus` and `full`
- for deeper unfreezing, the selected `best_epoch` becomes quite small at larger train sizes

That combination suggests limited effective adaptation of the pretrained backbone under the current setup, but not necessarily zero learning.

More precisely:

- `train_r2`, `train_pearson`, and `train_loss_standardized` in `learning_curve_runs.csv` are evaluated after reloading the best-validation checkpoint
- `best_epoch` is selected by validation loss, not by train loss
- for `conv3_plus` and `full`, `frozen_epochs = 2`, so the deeper layers only begin unfreezing at epoch `2`
- when `best_epoch` is around `2-4`, the selected model only had a short window for deeper adaptation

So a weaker-than-expected training `R^2` does not mean "training did not happen." It means the validation-selected checkpoint stayed relatively close to the pretrained solution.

## Important caveat: train size and quality composition are entangled

This run uses `train_sampling_mode = hq_first`.

That means:

- training sizes `161` and `322` are HQ-only
- the first mixed-quality train size is `643`
- after `643`, increasing train size also changes the quality mix, not just the number of examples

So in this run alone we cannot cleanly separate:

- effect of more data
- effect of noisier / lower-quality data entering
- effect of validation-based early stopping

This is the main reason to avoid over-claiming from the targeted `hq_first` curves.

## What seems safe to say now

The current `hq_first` run supports the following claims:

- more in-house data still helps validation and test performance
- deeper unfreezing is where most of the gains appear
- the training procedure often selects an early checkpoint for `conv3_plus` and `full`
- the selected checkpoint frequently shows a smaller train-vs-held-out gap than one might expect from a more fully adapted model

What is not yet safe to conclude:

- that the in-house data is fundamentally too weak to support deeper fine-tuning
- that exploitation is exhausted and future rounds should focus mostly on exploration
- that the train-curve behavior is caused only by early stopping rather than by the shifting HQ/LQ mix

## Why the `random_all_per_seed` rerun matters

The in-progress rerun changes the question in a useful way:

- it uses `split_strategy = random_all_per_seed`
- it uses `train_sampling_mode = random`
- it removes the fixed HQ-first curriculum that confounds train size with train-set composition
- it keeps attention on the better transfer regimes (`B2`, `B3 weighted`) and deeper scopes

This rerun is the right next test because it will tell us whether the current pattern is robust when:

- train size grows without the same deterministic HQ-to-LQ transition
- validation/test difficulty also varies with seed

Interpretation guide once that run finishes:

- if train `R^2` still drops while `best_epoch` still collapses early for deeper scopes, then the case gets stronger that the current model/optimizer/validation setup is conservative on this in-house task
- if the pattern weakens substantially, then part of the current story was driven by the `hq_first` curriculum rather than by a hard limit on learnability

Here "conservative" means the training pipeline prefers checkpoints that stay close to the pretrained initialization instead of moving far enough to fit the in-house task strongly.

## Should we increase patience and max epochs?

Probably yes as a targeted follow-up experiment, but not because we should expect "grokking."

### Short answer

- increasing `patience` and `max_epochs` is a reasonable diagnostic
- waiting for "grokking behavior" is probably not the right framing here

### Why "grokking" is probably not the main hypothesis

Classic grokking is usually discussed in much more stylized settings:

- algorithmic or low-noise tasks
- very long training after train accuracy is already high
- delayed generalization emerging far after memorization

That is not the pattern here. Here we see:

- modest train fit, not near-perfect train fit
- early validation-selected checkpoints
- noisy biological regression data
- transfer learning from a large pretrained model

So the more plausible possibilities are:

- patience is too short for deeper unfreezing to show benefit after epoch `2`
- the validation criterion is too noisy or too strict
- the current learning-rate / early-stopping combination favors very conservative solutions
- later epochs mostly increase train fit without improving held-out performance

### What a longer-training check would actually tell us

If you rerun a small subset with larger `patience` and `max_epochs`, there are two informative outcomes:

1. Later best-validation checkpoints improve held-out metrics.

That would suggest the current setup is stopping too early and is underestimating how much the in-house data can help deeper fine-tuning.

2. Train metrics keep improving but validation/test stay flat or worsen.

That would suggest the current limitation is not simply "insufficient epochs"; it would mean longer training mostly buys overfitting, not better adaptation.

## Practical recommendation

I would not launch a giant new sweep for this immediately. A more efficient next step is:

1. Wait for `lib1_enhancer_targeted_random_all_per_seed_apr2026` to finish.
2. Compare whether low `best_epoch` for `conv3_plus` and `full` persists under random sampling.
3. If it does, run a small follow-up on only the top 1-2 settings with:
   - larger `patience` such as `20-30`
   - larger `max_epochs` such as `120-200`
   - the same train sizes, or just the largest few train sizes
4. Check whether the best-validation checkpoint moves later and whether held-out metrics actually improve.

That experiment would answer the practical question much more directly than invoking grokking.

## Working conclusion for now

The current targeted `hq_first` run suggests:

- more in-house data is still useful
- most benefit comes from deeper unfreezing
- the selected deeper models often stop very early, indicating limited effective adaptation under the current training setup

But because train size and quality composition are entangled in this run, the strongest claims should wait for the `random_all_per_seed` rerun.

If the same early-stop-plus-low-train-fit pattern survives in that rerun, then it becomes much more defensible to say that the present transfer-learning setup is conservative on this task and that simply exploiting the current in-house pool harder may have limited returns without either more diverse data or a changed training protocol.
