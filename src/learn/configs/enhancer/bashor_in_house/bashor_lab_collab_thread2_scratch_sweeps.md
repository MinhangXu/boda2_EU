# Bashor Lab Collab Thread 2: Scratch Training Sweeps

This note summarizes what we set up with:

- `src/learn/launch/lib1_enhancer_scratch_weighted_sweep.sh`
- `src/learn/launch/lib1_enhancer_scratch_compare_loss_modes.sh`

The goal of these runs was to study lib1 enhancer prediction from scratch, rather than as transfer learning from the pretrained Malinois checkpoint.

## Main question

These scratch runs were meant to answer two related questions:

1. How well can the architectures learn the lib1 enhancer task from scratch?
2. Does barcode-weighted training help generalization relative to an unweighted loss?

The architecture comparison is:

- `BassetVL`
- `ResNet1DRegressor`

The loss/training comparison is:

- a basic unweighted setup
- a barcode-weighted setup

## Top-level launcher

`src/learn/launch/lib1_enhancer_scratch_compare_loss_modes.sh` is the high-level wrapper for the comparison.

It switches on `MODE`:

- `MODE=basic`
- `MODE=weighted`

For `basic`, it launches:

- config: `src/learn/configs/enhancer/bashor_in_house/lib1_enhancer__scratch_basic__bayes.yml`
- comparison group: `enhancer__bashor_lib1__scratch_basic__bassetvl_vs_resnet1d`

For `weighted`, it launches:

- config: `src/learn/configs/enhancer/bashor_in_house/lib1_enhancer__scratch_weighted__bayes.yml`
- comparison group: `enhancer__bashor_lib1__scratch_weighted__bassetvl_vs_resnet1d`

In both modes, the script then calls the shared W&B helper to create or reuse a sweep and fan out multiple `wandb agent` workers across GPUs.

Defaults in the launcher are:

- `NUM_AGENTS=4`
- `NUM_RUNS=8`
- `GPU_LIST=0 1 2 3 4 5 6 7`

So this script is the umbrella entry point for the "basic vs weighted" scratch-training comparison.

## Dedicated weighted launcher

`src/learn/launch/lib1_enhancer_scratch_weighted_sweep.sh` is the more focused launcher for the barcode-weighted study.

It hard-codes:

- config: `configs/enhancer/bashor_lib1/scratch/lib1_enhancer__scratch_weighted__bayes.yml` in the script
- comparison group: `enhancer__bashor_lib1__scratch_weighted__bassetvl_vs_resnet1d`
- launch script tag: `launch/lib1_enhancer_scratch_weighted_sweep.sh`

It also exports a default local dataset path:

- `/home/minhang/synBio_AL/opt_EU_learn_n_design/MattLee_lib1/enhancers/20251218_np_fastq1_500000NPreads_enh_variants_bc_sum_avg_expression.txt`

And it uses these default launcher settings:

- `NUM_AGENTS=4`
- `NUM_RUNS=8`
- `GPU_LIST=0 1 2 3`

This script is the focused entry point when the main interest is the weighted-loss scratch-training setup.

## What the basic scratch config does

`lib1_enhancer__scratch_basic__bayes.yml` defines a Bayesian sweep for scratch training with:

- `program: train_wandb_log.py`
- `method: bayes`
- optimization target: `epoch_end_val_pearson_r2`
- `graph_module: CNNBasicTraining`
- `loss_criterion: MSELoss`
- `barcode_weighting: false`

Other important pieces:

- `model_module: [BassetVL, ResNet1DRegressor]`
- `use_reverse_complements: [true, false]`
- `lr`: log-uniform from `1e-5` to `5e-3`
- `weight_decay`: log-uniform from `1e-6` to `1e-2`
- `max_epochs: 50`
- `stopping_patience: 15`

Data setup:

- dataset: lib1 enhancer expression table
- target: `RNA_DNA_Ratio_log10_scaled`
- barcode column: `n_barcodes`
- `test_min_barcodes: 4`
- `val_frac_within_hq: 0.25`
- `test_frac_within_hq: 0.25`
- `train_sampling_mode: hq_first`

So the basic config is the unweighted scratch baseline.

## What the weighted scratch config does

`lib1_enhancer__scratch_weighted__bayes.yml` keeps the same overall sweep structure, but changes the training graph and enables barcode weighting:

- optimization target: `epoch_end_val_pearson_r2`
- `graph_module: CNNWeightedRegressionTraining`
- `barcode_weighting: true`
- `barcode_weight_cap: [8.0, 10.0, 15.0]`
- `barcode_weight_min: 0.1`

It still compares:

- `BassetVL`
- `ResNet1DRegressor`

and still sweeps:

- reverse-complement augmentation on/off
- learning rate
- weight decay

The core idea is to give higher-confidence barcode-supported examples more influence, while capping that influence so a few heavily measured examples do not dominate training.

## What the shared W&B helper contributes

Both launchers use `src/learn/launch/_wandb_helpers.sh`.

That helper is responsible for:

- creating the W&B sweep if needed
- resolving the W&B entity and project
- recording the sweep launch in `src/learn/run_registry/sweep_launches.csv`
- launching multiple `wandb agent` processes across the requested GPUs

So the sweep launch flow is:

1. choose `basic` or `weighted`
2. select the appropriate config and comparison-group label
3. create or reuse the W&B sweep
4. spread agents across GPUs
5. log the launch metadata in the local registry

## Bottom line

The scratch-training work was set up to compare:

- architecture choice: `BassetVL` vs `ResNet1DRegressor`
- augmentation choice: reverse complements on/off
- objective choice: unweighted basic loss vs barcode-weighted loss

In short:

- `lib1_enhancer_scratch_compare_loss_modes.sh` is the umbrella script for basic vs weighted comparisons
- `lib1_enhancer_scratch_weighted_sweep.sh` is the dedicated weighted launcher
- the corresponding Bayesian sweep YAMLs define the actual model, loss, and search space for those experiments
