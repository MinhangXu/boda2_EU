# Lib1 Barcode-Weighted Loss Math and Implementation Plan

Generated: 2026-06-15

Status: planning note for a weighted-loss follow-up to the Lib1 outer-seed
scratch HPO work.

2026-06-16 implementation update:

- `CNNWeightedRegressionTraining` now inherits `CNNBasicTraining`'s canonical
  validation/test and multi-loader logging path, and overrides only the
  training loss.
- The weighted graph accepts the same logging args as `CNNBasicTraining`, so
  an outer-seed config can switch graph modules without dropping
  `log_per_output_metric_details` or `log_legacy_metric_aliases`.
- The selected robust configs are written into
  `src/learn/generate_lib1_weighted_loss_followup_manifest.py`.
- The generated 60-run weighted manifest tag is
  `lib1_outer_seed_selected_barcode_weighted_june2026`.
- The existing 60 selected unweighted outer-seed rows are preserved as the
  paired baseline rather than rerun by default.

## Trigger

The residual-vs-barcode diagnostics in
`tutorials/lib1_tasks/pretrain_CRE_inhouse_data/outer_seed_prior_lib1_orchestrator_jun15.ipynb`
show the pattern we would expect if low-barcode rows are noisier training
labels:

- signed residual medians are useful for bias, and they are not the whole
  story;
- absolute residual / IQR tends to be larger at lower barcode count;
- validation and test rows are mostly high-barcode, so an unweighted model can
  spend too much capacity fitting low-support labels that do not match the
  held-out measurement-quality regime.

This makes barcode-weighted regression worth testing beyond the enhancer
experiments, especially for promoter, intron, 3'UTR, and 5'UTR scratch runs.

## Code Location Map

Weighted sequence training is not primarily implemented as a new class in
`boda/model/loss_functions.py`. That file exposes ordinary torch loss classes
and mixed losses such as `MSEKLmixed`; the standard model criterion is still
configured by `loss_criterion=MSELoss` there.

The active barcode-weighted sequence path is:

- `boda/data/bashor_datamodule.py`
  - CLI/config knobs: `barcode_weighting`, `barcode_weight_cap`, and
    `barcode_weight_min` are defined in the data module arguments.
  - weight formula: `_barcode_weight(n)` computes
    `clip(log1p(n) / log1p(barcode_weight_cap), barcode_weight_min, 1.0)`.
  - batch plumbing: when `barcode_weighting=true`, the data frame gets a
    `sample_weight` column and datasets return `(x, y, w)`.
  - the Lib1 part-specific modules `Lib1EnhancerDataModule`,
    `Lib1ThreePrimeDataModule`, `Lib1PromoterDataModule`,
    `Lib1IntronDataModule`, and `Lib1FivePrimeDataModule` all inherit this.
- `boda/graph/cnn_weighted_regression.py`
  - `CNNWeightedRegressionTraining` accepts `(x, y)` or `(x, y, w)`.
  - weighted MSE is:

```text
per_sample_i = mean_j (y_hat_ij - y_ij)^2
loss = sum_i w_i * per_sample_i / sum_i w_i
```

  - weights are used for training loss only; validation/test metrics are
    unweighted so weighted and unweighted runs remain comparable.
- `boda/graph/__init__.py` registers `CNNWeightedRegressionTraining`.
- The analogous embedding path is
  `boda/graph/embedding_prediction.py::WeightedEmbeddingRegressionTraining`,
  with weights supplied by `boda/data/embedding_datamodule.py`.
- The older enhancer fine-tuning scripts under
  `src/finetune/finetune_sweep_scripts/lib1_enhancer/` use the same clipped
  log barcode-weight formula and the same normalized weighted MSE.

Current in-house sequence configs already contain the inactive switch:

```yaml
barcode_weighting:
  value: false
graph_module:
  value: CNNBasicTraining
```

To run the current implementation, the paired weighted config needs:

```yaml
barcode_weighting:
  value: true
graph_module:
  value: CNNWeightedRegressionTraining
```

## Math

For construct `i`, let:

```text
x_i = sequence
y_i = observed aggregate activity
b_i = number of barcodes
f_theta(x_i) = model prediction
```

A useful observation model is:

```text
y_i = f_true(x_i) + eps_i
E[eps_i] = 0
Var(eps_i | b_i) = sigma_i^2
```

If `y_i` is an average over barcode-level observations and each barcode has
roughly independent measurement noise, then a simple approximation is:

```text
sigma_i^2 ~= sigma_construct^2 + sigma_barcode^2 / b_i
```

When `sigma_construct^2` is small, inverse-variance weighting would make
`w_i` roughly proportional to `b_i`. When there is irreducible construct-level
or preprocessing noise, raw `w_i=b_i` is too aggressive because the gain from
additional barcodes saturates.

The Gaussian negative log-likelihood with known variance is:

```text
NLL_i = 0.5 * ((y_i - f_theta(x_i))^2 / sigma_i^2 + log sigma_i^2)
```

If we do not model `sigma_i^2` explicitly, weighted least squares uses:

```text
L(theta) = sum_i w_i * (y_i - f_theta(x_i))^2 / sum_i w_i
```

where `w_i` is a reliability proxy. The current BODA proxy is:

```text
w_i = min(1, max(w_min, log1p(b_i) / log1p(b_cap)))
```

This is a conservative compromise:

- low-barcode rows are not discarded;
- weights increase monotonically with barcode support;
- high-barcode variants cannot dominate a mini-batch;
- the normalized denominator keeps the loss scale stable across batches and
  makes learning rates more comparable to unweighted MSE.

Raw barcode-count weighting is still a possible ablation, but it should not be
the first default unless we have evidence that residual variance falls nearly
as `1 / b_i` with little saturation.

## Current Implementation Assessment

The weighted MSE math itself is reasonable and matches the enhancer
fine-tuning implementation. It does not need a conceptual rewrite before a
first experiment.

What looks good:

- The data module already carries barcode weights for all in-house Lib1 part
  classes.
- The loss is normalized by `sum(w)`, so changing barcode composition does not
  wildly change the batch loss scale.
- Validation and test metrics are deliberately unweighted.
- The default log/cap transform is safer than raw barcode-count weights for
  small datasets with skewed barcode distributions.

What should be improved before making this part of the standardized
outer-seed HPO workflow:

1. Make `CNNWeightedRegressionTraining` match `CNNBasicTraining`'s canonical
   validation/test logging contract.

   `CNNBasicTraining` handles multiple validation loaders from
   `epoch_eval_splits=[train, val, test]` and writes canonical W&B history
   keys. The weighted graph currently has its own validation/test methods.
   For standardized paired comparisons, the cleanest fix is to let the
   weighted graph inherit the basic validation/test behavior and override only
   training loss.

2. Make weighted configs swappable with basic configs.

   `CNNBasicTraining` exposes config knobs such as
   `log_per_output_metric_details` and `log_legacy_metric_aliases`. The
   weighted graph currently has a narrower graph argument surface. If we simply
   flip `graph_module` in an existing YAML, unknown args can fail parsing
   unless they are removed. The weighted graph should accept and pass these
   through to the parent.

3. Add a guard or clear provenance when weights are absent.

   `CNNWeightedRegressionTraining` falls back to the unweighted criterion when
   a batch lacks `w`. That is useful for compatibility, but a standardized
   weighted run should fail fast or warn loudly if
   `barcode_weighting=false`.

4. Use `train_eval_dataloader()` for final train-set evaluation when available.

   `train_wandb_log.py` currently builds final train metrics from
   `train_dataloader()`. For RC-augmented or weighted datasets, the cleaner
   diagnostic loader is `train_eval_dataloader()`, which avoids training-time
   augmentation while still ignoring weights for comparable metrics.

5. Add sequence-level smoke tests.

   The embedding weighted path has a toy test. Add the same coverage for
   `BashorDataModule` plus `CNNWeightedRegressionTraining`: batch shape,
   hand-calculated weighted MSE, and multi-loader validation compatibility.

## Experiment Design

Use weighted loss as a paired follow-up to the no-RC outer-seed prior HPO, not
as an unlabelled change inside that experiment.

Initial scope:

- parts: promoter, intron, 3'UTR, 5'UTR;
- optional enhancer sanity check only if we want to compare against the older
  enhancer weighted story;
- same split seeds as outer-seed prior:

```text
split_seed = [101, 202, 303, 404, 505]
model_seed = 1701
use_reverse_complements = false
```

Paired comparison:

- For each selected `(part, architecture, config_id, split_seed)`, run:
  - unweighted baseline, already planned or already completed;
  - weighted loss with the same hyperparameters, data split, model seed, and
    checkpoint monitor.
- Change only:
  - `barcode_weighting=true`;
  - `graph_module=CNNWeightedRegressionTraining`;
  - `logger_project`, `comparison_group`, `manifest_tag`, and artifact roots.

Weight settings:

- Start with the current package default:

```text
barcode_weight_cap = 8
barcode_weight_min = 0.1
```

- Add `b_cap=16` only if residual-scale plots suggest meaningful
  heteroscedasticity above 8 barcodes.
- Do not start with raw `w=b_i`; reserve it as a later ablation if clipped-log
  weighting helps but still appears underpowered.

Metrics:

- selection metric: validation Pearson only;
- report: validation/test Pearson, Spearman, COD R2, MSE/RMSE, and train
  metrics;
- diagnostics: signed residual vs `n_barcodes`, absolute residual/IQR vs
  `n_barcodes`, and metrics split by `train_low_barcode`, `train_hq`, `val`,
  and `test`;
- inference: paired bootstrap deltas across shared held-out row IDs.

Success criteria:

- weighted runs improve mean validation Pearson across split seeds without a
  test-set tradeoff;
- COD R2 or RMSE does not deteriorate while Pearson improves;
- residual-scale curves flatten for low-barcode train rows or at least stop
  forcing high-barcode heldout performance down;
- the improvement is not explained by one lucky split seed.

## Implementation Steps

1. Patch `CNNWeightedRegressionTraining`. Done 2026-06-16.

   Keep `_weighted_mse` and `training_step`, but inherit the basic graph's
   validation/test epoch handling. Add the basic graph logging args to its
   argument parser and constructor.

2. Add sequence weighted-loss tests. Partly done 2026-06-16.

   Create a tiny synthetic `DNARegressionDataset` or `BashorDataModule` fixture
   and verify:

```text
observed_loss == sum(w * squared_error) / sum(w)
```

   Also smoke-test `epoch_eval_splits=[train, val, test]`.

   A direct unit test was added for weighted train loss and unweighted
   validation loss. Full test execution requires the BODA training environment
   because this local shell does not have `torch` or `pytest` installed.

3. Add a weighted manifest mode. Done 2026-06-16.

   Extend or sibling
   `src/learn/generate_lib1_outer_seed_prior_hpo_manifest.py` so it can emit a
   weighted manifest with:

```text
manifest_tag = lib1_outer_seed_selected_barcode_weighted_june2026
barcode_weighting = true
barcode_weight_cap = 8
barcode_weight_min = 0.1
graph_module = CNNWeightedRegressionTraining
```

   Keep the unweighted manifest immutable for clean comparison.

4. Run a pilot.

   One top config per part, one split seed, `b_cap=8`. Confirm:

- no parser leftovers;
- W&B history contains train/val/test canonical metrics;
- final artifacts and `runs.csv` rows are written;
- residual plots can be regenerated from checkpoint predictions.

5. Run the paired follow-up.

   Use the same config IDs and split seeds as the selected unweighted
   outer-seed set. Analyze by paired deltas, not by independent ranking.

6. Decide whether to broaden.

   If weighted loss helps promoter/intron/UTR consistently, promote it to the
   next standardized scratch HPO recipe. If it only helps under narrow
   barcode distributions, keep it as a part-specific option.

## Open Questions

- Should `b_cap` be part-specific? The residual plots for each part should
  answer this better than a universal guess.
- Should we learn a heteroscedastic variance head instead of using barcode
  count as a fixed proxy? This is attractive later, but the paired weighted
  experiment is cheaper and easier to interpret first.
- Should the heldout metric itself ever be barcode-weighted? For model
  selection, no: keep validation/test unweighted so the question remains
  "how well do predictions match heldout constructs?" Weighted heldout metrics
  can be reported only as diagnostics.
