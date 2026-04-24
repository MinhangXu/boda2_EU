# Bashor Lab Collab Thread 1: HPO and Estimate for +1000 Samples

This note summarizes the transfer-learning learning-curve work for the Bashor Lab collaboration, focusing on what we learned from the archived B2-only baseline, the targeted April 2026 follow-up, and the current in-progress random-split rerun.

## What we analyzed

We used two notebooks as the main analysis surfaces:

- `tutorials/lib1_tasks/fine_tuning/finetune_learning_curve_sweep.ipynb`
- `tutorials/lib1_tasks/fine_tuning/targeted_finetune_learning_curve_apr2026_analysis.ipynb`

The older `finetune_learning_curve_sweep.ipynb` notebook stays focused on the archived Mar 25 B2-only baseline run in `src/finetune/learning_curve/lib1_enhancer_mar25_b2`. It is the stable reference point for how transfer learning behaved before the broader B1/B2/B3 targeted follow-up.

The targeted April notebook analyzes `src/finetune/learning_curve/lib1_enhancer_targeted_apr2026`. That run narrowed the search space and was intended to answer a more practical question: for threshold-1 training, which transfer setup and unfreezing strategy actually looks best once we do a smaller HPO sweep and inspect the learning curves carefully.

## What the targeted April run did

The targeted April run:

- compared `B1_no_RC`, `B2_with_RC`, and `B3_with_RC_weighted_bcap_10`
- fixed attention on `train_threshold = 1`
- compared `branched_only`, `conv3_plus`, and `full`
- swept `head_lr` over `2e-4` and `5e-4`
- swept `backbone_lr` over `1e-5` and `1e-4`
- evaluated train sizes from `161` to `3215`
- aggregated across seeds and initialization heads

The notebook reports:

- `runs: (3024, 71)`
- `histories: (71222, 28)`
- `summary: (756, 84)`
- `scope_summary: (36, 33)`

It also notes an important data-composition breakpoint:

- the HQ-only pool size before lower-quality examples enter is `614`
- the first mixed-quality train size in that sweep is `643`

So the learning curves are not only about sample count; after `643` samples they also reflect the transition from HQ-only training examples to a mixture that includes lower-quality examples.

## How HPO was selected

In `targeted_finetune_learning_curve_apr2026_analysis.ipynb`, the best HPO was selected separately for each `setting x unfreeze_scope` combination by ranking:

1. mean validation `R^2`
2. mean test `R^2`
3. validation loss

That gives a compact way to compare settings without overfitting to one seed or one initialization head.

Across the stronger settings, the preferred hyperparameters were fairly consistent:

- `head_lr = 5e-4` was favored for the top-performing combinations
- `backbone_lr = 1e-4` tended to help the deeper-unfreezing settings
- `full` generally beat `conv3_plus`, which generally beat `branched_only`

Selected best-HPO summary from the targeted run:

- `B2_with_RC + full`: mean test `R^2 ~= 0.179` across the whole curve summary
- `B3_with_RC_weighted_bcap_10 + full`: mean test `R^2 ~= 0.184` across the whole curve summary
- at the largest train size (`3215`), `B3_with_RC_weighted_bcap_10 + full` reached test `R^2 ~= 0.231`
- at the same largest train size, `B2_with_RC + full` reached test `R^2 ~= 0.226`

The take-home message is that reverse-complement augmentation helps, barcode-weighted B3 is modestly better than plain B2 at the top end, and the main step-change comes from unfreezing more of the backbone rather than staying in the most frozen regime.

## Estimated impact of another 1000 samples

Using the targeted April learning curves, we can make a rough estimate of how much another `~1000` training examples might help generalization.

This estimate is intentionally approximate. It uses the tail of the learning curves from the previous targeted run, so it should be treated as directional rather than as a final claim.

### Best-case transfer setups

For the best `full` fine-tuning settings, the last segment of the learning curve still shows some upward slope:

- `B2_with_RC + full` goes from test `R^2 ~= 0.216` at `2411` samples to `~0.226` at `3215`
- `B3_with_RC_weighted_bcap_10 + full` goes from test `R^2 ~= 0.222` at `2411` samples to `~0.231` at `3215`

A simple extrapolation from that tail suggests another `1000` samples would likely buy about:

- `+0.01` to `+0.012` absolute test `R^2` for the strongest `full` settings

That would put the expected test `R^2` roughly in the range:

- `B2_with_RC + full`: around `0.238` to `0.240`
- `B3_with_RC_weighted_bcap_10 + full`: around `0.241` to `0.243`

### More frozen settings

The expected gain is much smaller if we do not unfreeze deeply:

- `conv3_plus`: likely only about `+0.002` to `+0.003` test `R^2`
- `branched_only`: near zero to `+0.003` test `R^2`

So the main conclusion is:

- more data is still worthwhile if we are willing to use `full` fine-tuning
- more data alone is unlikely to materially improve generalization if we keep most of the backbone frozen

## Current rerun and why it matters

The current run is:

```bash
CUDA_VISIBLE_DEVICES=0 python boda2_EU/src/finetune/finetune_sweep_scripts/lib1_enhancer_learning_curve_finetune_split_options.py \
  --outdir /home/minhang/synBio_AL/boda2_EU/src/finetune/learning_curve/lib1_enhancer_targeted_random_all_per_seed_apr2026 \
  --split_strategy random_all_per_seed \
  --split_seed 7 \
  --val_frac 0.15 \
  --test_frac 0.15 \
  --include_b2 --include_b3 \
  --b3_bcaps 8 10 \
  --train_thresholds 1 \
  --unfreeze_scopes branched_only conv3_plus full \
  --head_lrs 2e-4 5e-4 \
  --backbone_lrs 1e-5 1e-4 \
  --train_size_fracs 0.05 0.1 0.2 0.35 0.5 0.75 1.0 \
  --train_sampling_mode random \
  --seeds 17 23 19 31
```

This rerun differs from the earlier targeted analysis in a few important ways:

- it uses `random_all_per_seed`
- it uses moving validation and test splits rather than relying on the earlier fixed-split framing
- it keeps `B2` and `B3 weighted` but drops `B1`
- it uses `train_sampling_mode = random`
- it tests `b_cap = 8` and `10` for B3 weighted

The purpose of this rerun is not to replace the earlier targeted result, but to check whether the earlier gains are robust when the splits themselves vary by seed.

At the time of writing, this run is still in progress. The run manifest exists, but the summary CSV outputs are not yet present, so it is too early to report final generalization numbers from it.

## Bottom line

Current best evidence suggests:

- `full` fine-tuning remains the most promising transfer regime
- `B3` weighted is slightly stronger than plain `B2` at the largest train sizes
- another `~1000` samples would probably help, but mostly for the deeper-unfreezing setups
- a realistic expectation is about `+0.01` absolute `R^2` for the best `full` settings, not a dramatic jump

That is large enough to matter, but not large enough to expect a qualitative change in behavior unless the extra examples also improve the quality or diversity of the training pool.
