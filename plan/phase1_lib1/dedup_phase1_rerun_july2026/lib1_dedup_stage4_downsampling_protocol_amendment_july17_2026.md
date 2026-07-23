# Lib1 Dedup Stage 4 Downsampling Protocol Amendment

**Frozen:** July 17, 2026
**Campaign:** `lib1_dedup_phase1_rerun_july2026`
**Stage:** `stage4_downsampling`
**Status:** frozen for dry-run manifest generation; no training launched by this amendment

## 1. Question and estimand

Stage 4 asks:

> How does held-out development performance change as the number of unique,
> exact-deduplicated training constructs increases, while the selected model
> configuration, part-specific RC/loss policy, split system, optimizer, and
> evaluation target remain fixed?

`N` always means unique construct IDs before reverse-complement augmentation.
The primary estimand is the change in pooled five-fold development OOF Pearson
for the selected Stage 3 configuration of each CRE part. Raw-scale RMSE and COD
R2 are calibration outcomes. This stage estimates sample efficiency; it does
not reopen architecture, RC, weighted-loss, or configuration selection.

The x-axis is logarithmic because the planned comparisons are multiplicative.
This does **not** assume that Pearson grows exponentially with `N`.

## 2. Final-test isolation

The frozen final test set is outside Stage 4.

- No final-test loader may be created.
- No final-test metric, prediction, checkpoint, or result may be read by the
  generator, verifier, runner, or analysis.
- Final-test rows are skipped before their sequence, target, or barcode fields
  are parsed by the Stage 4 data mode.
- `evaluate_test_after_fit=false`, `prediction_splits=oof`, and
  `epoch_eval_splits=train val` are mandatory.
- The 14-config development portfolio was proposed before the final-test result
  was opened and remains independent of that result.
- The Enhancer scratch lane is a predeclared route diagnostic, not a response
  to final-test performance.

## 3. Outer OOF and inner checkpoint folds

For outer OOF fold `k`:

```text
outer OOF fold       = k
inner checkpoint fold = (k + 1) mod 5
training pool         = train_only rows + the other three development folds
```

The inner fold alone drives `val_pearson`, early stopping, and best-checkpoint
selection. The outer fold is evaluated once with the loaded best checkpoint
and exported as `oof` predictions. Registry `val_*` metrics are therefore
inner-checkpoint metrics and must never be analyzed as Stage 4 OOF results.

The eligible full training-pool minima after excluding both folds are:

| Part | Minimum eligible `full` N |
|---|---:|
| Enhancer | 4,145 |
| Promoter | 6,889 |
| Intron | 7,158 |
| 3'UTR | 6,385 |
| 5'UTR | 7,396 |

Thus `N=4,000` is feasible in every part and fold. In Stage 4, `full` means all
eligible rows remaining after the paired outer and inner development folds are
excluded; it is not the larger Stage 3 outer-only training pool.

## 4. Frozen portfolios and policies

The machine-readable Stage 4 portfolio contains 15 configurations:

- five primary configurations: the selected Stage 3 configuration for each
  part;
- nine alternatives: the remaining predeclared Stage 4 one-SE portfolio
  configurations (two each for Enhancer, Promoter, Intron, and 3'UTR; one for
  5'UTR);
- one diagnostic Enhancer scratch configuration.

Part policies are fixed:

| Part/lane | RC | Loss | Role |
|---|---|---|---|
| Enhancer primary + alternatives | on | unweighted MSE | selected transfer-policy sensitivity |
| Enhancer scratch diagnostic | off | unweighted MSE | route-shape diagnostic only |
| Promoter | off | barcode-weighted MSE | selected Stage 3 policy |
| Intron | off | barcode-weighted MSE | selected Stage 3 policy |
| 3'UTR | off | barcode-weighted MSE | selected Stage 3 policy |
| 5'UTR | off | barcode-weighted MSE | selected Stage 3 policy |

The Enhancer scratch configuration is the best development-admissible scratch
route (`basecfg_7bb5763f52f3678922d64e5026e75fa14b79bde606319b207a5f8b30885f87b8`).
It differs from transfer in architecture, input policy, and RC, so it does not
isolate initialization. Its narrow diagnostic question is whether a reasonable
scratch route closes any of the transfer gap as `N` grows.

## 5. Sizes, nested subset replicates, and accounting

### Primary configurations

```text
N = 40, 250, 400, 2,500, 4,000, full
finite-N subset seeds = 104729, 130363, 155921
full subset replicates = 1 (a subset seed cannot change the full pool)
```

Each smaller finite subset is an exact prefix of the same stable-ID-sorted,
seeded permutation used by every larger subset in its replicate track. The
same part/fold/seed prefix is shared across configurations. Model/training seed
is fixed at `1701` to isolate sampling variation from optimization variation.

Accounting: `5 configs x 5 folds x (5 finite sizes x 3 seeds + 1 full) = 400`.

### Alternative configurations

```text
N = 40, 400, 4,000, full
subset seed = 104729
```

These sparse anchors are configuration-sensitivity checks, not statistical
replicates. They test whether the selected configuration's learning-curve shape
is unusually optimistic or pessimistic and whether rankings reverse between
low and high `N`. `40 -> 400 -> 4,000` gives two exact decades and `full`
connects the sparse curve to the maximum Stage 4 pool. Dense three-seed curves
for all nine alternatives would spend most of the budget re-testing model
choice rather than estimating sample-number uncertainty.

Accounting: `9 configs x 5 folds x 4 sizes = 180`.

### Enhancer scratch diagnostic

The scratch diagnostic uses the complete primary grid and the same three finite
subset seeds.

Accounting: `1 config x 5 folds x (5 finite sizes x 3 seeds + 1 full) = 80`.

### Total

| Part | Rows |
|---|---:|
| Enhancer | 200 |
| Promoter | 120 |
| Intron | 120 |
| 3'UTR | 120 |
| 5'UTR | 100 |
| **Total** | **660** |

Direct observed contrasts include `40 -> 400`, `250 -> 2,500`, and
`400 -> 4,000` (10x), plus `40 -> 4,000` (100x). Performance at 10x or 100x
the current full dataset remains an extrapolation rather than an observed
contrast.

## 6. Training and data contract

- Exact deduplicated Lib1 learn-ready datasets and frozen split manifests are
  SHA-bound.
- `train_min_barcodes=1`; no barcode threshold is crossed.
- Sampling is random within the fixed eligible pool, after stable-ID sorting.
- Target normalization is fitted only on the selected training subset and then
  applied to the inner and outer folds.
- Raw predictions are produced by inverse-transforming with that training-only
  mean and standard deviation.
- Stage 3 optimizer, architecture, batch size, maximum epochs, minimum epochs,
  early-stopping patience, transfer head/scope, RC, and loss settings remain
  fixed per configuration.
- Barcode-weighted rows require the strict weighted graph and a three-item
  training batch; silent unweighted fallback is forbidden.
- RC may expand training examples but never changes the reported unique `N`.
- Artifacts use `artifact_retention=none`; OOF predictions and compact
  provenance are mandatory.

Every row records and verifies the training-pool, selected-training, inner-fold,
outer-OOF, normalization, dataset, split-manifest, command, and row hashes.

## 7. W&B organization

Entity:

```text
minhangxu1998-baylor-college-of-medicine
```

Per-part project:

```text
<part>__bashor_in_house__dedup_exact_v1__stage4_downsampling_development
```

Group:

```text
lib1_dedup_phase1_rerun_july2026__stage4__<part>__<lane>__<config-short-id>
```

Job type is `stage4_downsampling_cell`. Run names encode part, lane, config,
outer fold, inner fold, `N`, subset replicate/seed, and model seed. W&B identity
must resolve before fitting, and the exact manifest command is immutable.

## 8. Primary analysis

For each primary part/configuration and each `N`:

1. concatenate the five outer-fold prediction tables within each subset track;
2. calculate pooled OOF Pearson, raw RMSE, and raw COD R2;
3. summarize across the three finite-N subset tracks;
4. calculate paired observed differences for all predeclared 10x and 100x
   comparisons;
5. report best epoch, optimizer steps, and train-versus-inner-validation gap.

For Intron, every `N` also requires:

- within-inferred-stratum-centered Pearson;
- metrics for each frozen inferred-mask stratum;
- stratum counts and target/prediction means.

The sparse alternatives and Enhancer scratch lane are shown separately. They
cannot replace the primary configuration or drive a new model selection.

## 9. Bootstrap uncertainty

The bootstrap quantifies uncertainty in observed OOF metrics and curve-derived
contrasts; it does not create new biological samples or remove extrapolation
risk.

- Use 2,000 deterministic bootstrap replicates.
- Resample outer-OOF construct IDs with replacement within each outer fold.
- Use the same resampled IDs for all `N`, subset tracks, and compared
  configurations so paired differences remain paired.
- For finite-N primary curves, also resample the three subset tracks with
  replacement. `full` is reused because it has one unique training subset.
- Recalculate pooled metrics and refit both curve families in every successful
  bootstrap replicate.
- Report percentile 95% intervals and the number of failed/degenerate fits.

## 10. Bounded curve families and disagreement

Curve fits are secondary summaries of the observed points, not truth and not a
selection gate. Pearson is transformed to Fisher-z space so predictions remain
bounded after the inverse transform.

### Saturating power law

```text
z(N) = z_inf - A * N^(-alpha),  A > 0, alpha > 0
r(N) = tanh(z(N))
```

This family permits a long, slowly improving tail.

### Exponential saturation

```text
z(N) = z_inf - A * exp(-N / tau),  A > 0, tau > 0
r(N) = tanh(z(N))
```

This family approaches its asymptote more quickly. RMSE is fit with analogous
positive, decreasing forms in raw-loss space.

Both are useful because the six observed points may not identify the far-tail
shape. Report each family's fitted curve, leave-one-size-out error, prediction
at `full`, and projected `full -> 10x full` gain. Family disagreement is the
absolute difference between their projected gains and is displayed directly;
it is not averaged away. Close projections with acceptable held-out-size error
support a more stable scenario estimate. Large disagreement means the available
data do not determine the tail, and only observed contrasts should be used for
decisions. A 100x-full projection may be tabulated as exploratory but is never
presented as an inferred result.

No arbitrary pass/fail threshold is attached to curve-family disagreement.

## 11. Failure and retry rules

- A valid completed cell is immutable.
- Only exact technical retries are allowed: same manifest SHA, command, cell
  ID, data/split hashes, seeds, subset, and hyperparameters.
- Failed and partial attempts are retained.
- A disappointing learning-curve point is not a retry reason.
- No row, size, subset seed, or portfolio membership may be added after viewing
  Stage 4 performance without a separately dated sensitivity amendment.

## 12. Launch gate

Before any training launch:

1. generate the 660-row dry-run manifest and machine-readable portfolio;
2. independently verify all counts, hashes, nested prefixes, fold isolation,
   command contracts, W&B identities, and absence of final-test access;
3. run one explicitly confirmed pilot row;
4. reconcile its OOF prediction IDs and provenance against the manifest; and
5. require a separate explicit confirmation for the full campaign.
