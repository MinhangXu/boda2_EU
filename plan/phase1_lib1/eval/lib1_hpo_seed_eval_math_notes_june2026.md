# Lib1 HPO Seed-Evaluation Math Notes

Generated: 2026-06-14

This note keeps the statistical framing from the HPO trust-diagnostics work in
a durable plan file. It supports:

- `plan/phase1_lib1/lib1_outer_seed_prior_hpo_plan_june2026.md`
- `tutorials/lib1_tasks/pretrain_CRE_inhouse_data/lib1_inhouse_scratch_hpo_seed_split_diagnostics_june2026.ipynb`

## Observation Model

For one HPO run, the observed heldout Pearson is not just "the config quality."
It is better thought of as:

```text
observed Pearson
  = true config quality
  + split difficulty
  + training randomness
  + finite heldout sampling noise
  + measurement/model noise
```

The previous broad HPO swept hyperparameters and seeds jointly, so these terms
were entangled. The new outer-seed design tries to separate config quality from
split difficulty by evaluating the same config on the same set of split seeds.

## Fisher Pearson CI

For a Pearson correlation `r` measured on `n` heldout examples, the usual
Fisher approximation is:

```text
z = atanh(r)
SE(z) = 1 / sqrt(n - 3)
CI_z = z +/- 1.96 * SE(z)
CI_r = tanh(CI_z)
```

Interpretation:

- This CI captures finite-heldout sampling uncertainty for one fixed run and
  one fixed split.
- It does not include split difficulty, model-init randomness, hyperparameter
  selection bias, or label noise uncertainty.
- It is useful for avoiding overinterpretation of tiny differences among runs
  evaluated on small validation/test sets.

## Bootstrap CI

A nonparametric bootstrap resamples heldout rows with replacement and recomputes
the metric on each resample.

For one model on one split:

```text
for b in 1..B:
    sample n heldout rows with replacement
    compute metric_b
CI = percentile(metric_b, [2.5, 97.5])
```

This can be used for Pearson, Spearman, RMSE, COD R2, or median absolute
residual.

## Paired Bootstrap Delta

When two models are evaluated on the same row IDs, compare them with a paired
bootstrap:

```text
for b in 1..B:
    sample row IDs with replacement
    delta_b = metric(model_A on sampled rows) - metric(model_B on sampled rows)
CI_delta = percentile(delta_b, [2.5, 97.5])
```

Use this only when the row IDs are shared. In the Lib1 data module, the same
part/data policy and `split_seed` should produce the same validation/test row
IDs. The notebook reconstructs split hashes to audit this.

Interpretation:

- For Pearson, Spearman, and COD R2, positive delta favors model A.
- For RMSE, negative delta favors model A.
- If the delta CI excludes 0, the row-level evidence supports a difference on
  that shared split.

## Outer-Seed Aggregation

For the new 600-run design, the primary unit is:

```text
(part, config_id)
```

Each config is evaluated on five split seeds:

```text
split_seed = [101, 202, 303, 404, 505]
```

Recommended summary per config:

- mean validation Pearson across seeds,
- standard deviation across seeds,
- min/max validation Pearson,
- mean validation Spearman,
- mean validation COD R2,
- mean validation RMSE,
- rank consistency across seeds.

With five seeds, the mean/std should be read as practical robustness evidence,
not as a high-precision estimate of an underlying distribution.

## Test Set Role

The validation set selects configs. The test set is final or diagnostic
evidence.

Avoid selecting the final config by test Pearson. If a test-selected run is
shown for diagnostics, label it explicitly as diagnostic rather than
validation-selected.

## Barcode-Aware Diagnostics

The train/val/test curve weirdness can be explained if low-barcode training
labels are noisier than high-barcode validation/test labels.

Useful checks:

- evaluate the same checkpoint on `train_all`, `train_hq`,
  `train_low_barcode`, `val`, and `test`;
- plot signed residual vs `n_barcodes` to check bias;
- plot absolute residual or residual IQR vs `n_barcodes` to check noise/scale;
- report COD R2 alongside Pearson so scale failures are not hidden.

The signed residual median line answers "is the model biased at this barcode
count?" The vertical spread or absolute residual plot answers "are labels or
predictions noisier at this barcode count?"
