# Lib1 Dedup Stage 2 Analysis And Decision Report

**Analysis date:** 2026-07-13
**Campaign:** `lib1_dedup_phase1_rerun_july2026`
**Stage:** `stage2_paired_rc`

## Decision Summary

Stage 2 is complete and scientifically usable: all 660 planned cells resolved,
all 132 five-fold OOF arms are complete, and all 66 base configurations have
paired RC-off/on predictions on identical constructs. The frozen audit loader
has not been instantiated.

The main conclusions are:

1. The pretrained Enhancer route clearly outperforms scratch training. The two
   full-unfreeze source-head policies are effectively tied; the numerical
   K562/full leader should not be presented as evidence that K562 is the
   biologically superior source head.
2. Intron remains the strongest scratch task. Its leading model beats a
   fold-training-fitted inferred-mask-mean baseline, but performance is still
   heterogeneous across the inferred sensitivity categories.
3. The 3'UTR UTRBassetVL challenger produces the best individual result, but
   the lane median is almost unchanged from the ResNet1D core and fold/config
   instability is substantial. Its learning-rate and training-fit evidence
   meets the protocol gate for considering a bounded targeted HPO.
4. Promoter and 5'UTR are broadly consistent with the earlier results. Stage 2
   contains no 5'UTR transfer challenger: all ten 5'UTR configurations are
   scratch/core candidates, comprising eight UTRBassetVL and two ResNet1D
   configurations.
5. RC augmentation is route-specific. It is supported for the Enhancer
   transfer route and for one Intron configuration, but not as a general
   default for Promoter, either 3'UTR lane, or 5'UTR.
6. Stage 3 must not launch automatically. The bounded 3'UTR targeted HPO is
   approved, but its search protocol and resulting candidate identities, the
   numerical interpretation of "no material" RMSE/COD degradation, and the
   standardized weighted scoped-Enhancer integration must be frozen first.

## Reproducible Evidence

The canonical executable and machine-readable outputs are:

- [Stage 2 analyzer](../../../src/analysis/lib1_dedup_stage2_analysis.py)
- [completion summary](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_analysis_summary.json)
- [cell completion table](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_cell_completion.csv)
- [OOF metrics](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_oof_metrics.csv)
- [fold-level OOF metrics](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_oof_fold_metrics.csv)
- [OOF predictions](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_oof_predictions.tsv.gz)
- [paired-RC metrics](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_rc_pair_metrics.csv)
- [fold-level paired-RC metrics](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_rc_fold_pair_metrics.csv)
- [Intron sensitivity metrics](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_intron_sensitivity_stratum_metrics.csv)
- [Intron fold-trained and explanatory baselines](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/stage2_intron_stratum_mean_baselines.csv)
- [Intron mask explanation and balance-diagnostic notebook](../../../tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/02_intron_inferred_mask_strata_analysis.ipynb)
- [executed Intron notebook with plots](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/executed_notebooks/02_intron_inferred_mask_strata_analysis__executed.ipynb)
- [Stage 2 reporting program](../../../src/analysis/lib1_dedup_stage2_reporting.py)
- [Stage 2 reporting summary](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_reporting_summary.json)
- [Intron estimand and challenge-set protocol](lib1_dedup_intron_estimand_and_challenge_set_protocol_july2026.md)
- [Intron estimand sensitivity reporting program](../../../src/analysis/lib1_dedup_intron_sensitivity_reporting.py)
- [Intron estimand sensitivity summary](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_intron_sensitivity_reporting_summary.json)
- [Stage 2 paired-RC analysis notebook](../../../tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/03_stage2_paired_rc_analysis.ipynb)
- [executed Stage 2 notebook](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/executed_notebooks/03_stage2_paired_rc_analysis__executed.ipynb)
- [reusable positional base-composition utility](../../../src/analysis/sequence_composition.py)
- [Stage 2 to targeted-3'UTR handoff](lib1_dedup_stage2_to_targeted_utr3_handoff_july2026.md)

All results below use raw `log2_RNA_DNA` and `prediction_raw` unless explicitly
described as normalized training MSE.

## Completeness And Audit Isolation

| Contract item | Result |
|---|---:|
| Analysis cells | 660 / 660 |
| Stage 1 reuse cells | 50 |
| Newly run Stage 2 cells | 610 |
| Complete OOF arms | 132 / 132 |
| Complete paired-RC configurations | 66 / 66 |
| Audit loader instantiated | no |

Each arm contains exactly one held-out prediction per development construct:
979 Enhancer, 1,545 Promoter, 1,061 Intron, 525 3'UTR, or 1,438 5'UTR
constructs. RC mates have identical construct IDs and raw targets.

The accounting is `50 core + 6 Enhancer transfer + 10 3'UTR challenger = 66`
base configurations. Each configuration has RC off and on, yielding 132 OOF
arms; each arm contains five fold-trained cells, yielding 660 cells. Pairing
the two arms for each base configuration gives 66 OOF RC pairs (and 330
fold-level pairs). An OOF arm is therefore a five-model prediction product,
not one training run.

Here the audit loader means the test `DataLoader` that would materialize the
frozen high-barcode audit constructs and pass them through a selected model.
Stage 2 validates that audit IDs remain disjoint, but creates no audit-test
dataset, predictions, or metrics.

## Route-Level OOF Results

The distribution column combines both RC arms within each provenance lane. It
is descriptive: core and challenger portfolios were selected by different
historical procedures and are not random samples from an architecture.

| Part / lane | Arms | Pearson median [range] | RMSE median [range] | COD R2 median [range] |
|---|---:|---:|---:|---:|
| Enhancer core scratch | 20 | 0.116 [-0.049, 0.202] | 1.252 [0.932, 8.764] | -0.946 [-94.295, -0.079] |
| Enhancer transfer challenger | 12 | 0.527 [0.490, 0.565] | 0.773 [0.752, 0.793] | 0.259 [0.219, 0.297] |
| Intron core scratch | 20 | 0.654 [0.626, 0.682] | 0.538 [0.502, 0.616] | 0.381 [0.189, 0.461] |
| Promoter core scratch | 20 | 0.455 [0.403, 0.477] | 0.437 [0.430, 0.462] | 0.192 [0.097, 0.220] |
| 3'UTR ResNet1D core scratch | 20 | 0.251 [-0.034, 0.332] | 0.705 [0.661, 0.934] | -0.035 [-0.817, 0.088] |
| 3'UTR UTRBassetVL challenger | 20 | 0.252 [0.019, 0.501] | 0.690 [0.618, 0.927] | 0.007 [-0.792, 0.204] |
| 5'UTR core scratch | 20 | 0.460 [0.418, 0.519] | 0.385 [0.371, 0.424] | 0.188 [0.012, 0.246] |

The best arm in each lane is:

| Part / lane | Config, RC | Pearson | Spearman | RMSE | COD R2 | Calibration slope | Raw bias |
|---|---|---:|---:|---:|---:|---:|---:|
| Enhancer scratch | `5d9f63c2`, on | 0.202 | 0.157 | 1.095 | -0.487 | 0.286 | +0.469 |
| Enhancer transfer | K562/full `6e6b2b97`, on | 0.565 | 0.426 | 0.759 | 0.285 | 0.754 | +0.003 |
| Intron | `6079cd38`, on | 0.682 | 0.629 | 0.502 | 0.461 | 0.974 | +0.043 |
| Promoter | `00175f1c`, off | 0.477 | 0.528 | 0.430 | 0.220 | 1.137 | +0.031 |
| 3'UTR ResNet1D | `585fba9a`, on | 0.332 | 0.202 | 0.684 | 0.024 | 0.698 | +0.178 |
| 3'UTR UTRBassetVL | `86969bcf`, off | 0.501 | 0.230 | 0.618 | 0.204 | 1.290 | +0.128 |
| 5'UTR | `9dd728c0`, off | 0.519 | 0.537 | 0.371 | 0.244 | 0.843 | +0.055 |

Calibration is defined as
`observed = intercept + slope * prediction_raw`; bias is
`mean(prediction_raw - observed)`. The 3'UTR challenger gain is much larger in
Pearson than in Spearman and its slope is 1.29, so the current leader should
not be summarized by Pearson alone.

### Enhancer route comparison

All six transfer configurations occupy a narrow and clearly superior Pearson
range of 0.490-0.565. Under RC on, K562/full scores 0.5647 and HepG2/full
scores 0.5623. HepG2/full has slightly better RMSE/COD R2 (0.7525/0.2974
versus 0.7591/0.2851), while K562/full has slightly higher Pearson. This is a
near tie between two equally complex policies, not evidence for source-head
specific biology. The route comparison also changes initialization and input
policy (600-nt fixed MPRA flank context versus 216-nt neutral padding), so it
establishes the better prediction route but does not identify which component
caused the improvement.

### Fold stability

Median standard deviation of the five fold-specific Pearson values was:

| Part / lane | RC off | RC on |
|---|---:|---:|
| Enhancer scratch | 0.050 | 0.043 |
| Enhancer transfer | 0.060 | 0.069 |
| Intron | 0.059 | 0.050 |
| Promoter | 0.089 | 0.075 |
| 3'UTR ResNet1D | 0.098 | 0.093 |
| 3'UTR UTRBassetVL | 0.110 | 0.143 |
| 5'UTR | 0.039 | 0.028 |

The leading fold vectors are K562/full Enhancer transfer RC-on
`[0.604, 0.574, 0.558, 0.613, 0.509]`, Intron `6079cd38` RC-on
`[0.769, 0.646, 0.647, 0.677, 0.679]`, 3'UTR UTRBassetVL `86969bcf` RC-off
`[0.555, 0.359, 0.564, 0.716, 0.476]`, and 5'UTR `9dd728c0` RC-off
`[0.535, 0.557, 0.504, 0.534, 0.500]`. The 3'UTR result has the greatest
selection uncertainty; five folds are stability diagnostics, not five
independent biological replicates.

## Strict RC Decision

The formal campaign rule is stricter than the legacy convenience Boolean
retained by the analyzer:

```text
mean fold delta >= 0.005
positive delta in at least 4/5 folds
no material RMSE or COD-R2 degradation
```

The analyzer and reporting table now also emit the binding Pearson fold gate.
The older `mean > 0` / no-more-than-two-negative Boolean remains only for
backward compatibility and is not used for the decision. Using a conservative
zero-tolerance interpretation of “no material degradation” gives:

| Part / lane | Mean pooled RC-on-minus-off Pearson | Positive configs | Strict passes |
|---|---:|---:|---:|
| Enhancer scratch | +0.022 | 7/10 | 0/10 |
| Enhancer transfer | +0.030 | 6/6 | 5/6 |
| Intron | +0.004 | 6/10 | 1/10 |
| Promoter | -0.009 | 1/10 | 0/10 |
| 3'UTR ResNet1D | -0.044 | 4/10 | 0/10 |
| 3'UTR UTRBassetVL | -0.043 | 3/10 | 0/10 |
| 5'UTR | -0.038 | 0/10 | 0/10 |

Across all 66 configurations, the mean and median pooled Pearson changes are
-0.0137 and -0.0091; 27/66 are positive. This pooled cross-route summary is
descriptive only because configuration portfolios and parts differ.

The five strict Enhancer-transfer passes are HepG2 conv3-plus, HepG2 full,
HepG2 branched-only, K562 full, and K562 branched-only. K562 conv3-plus
improves pooled Pearson but improves in only three folds. For Intron, only
`6079cd38` passes both the original pooled rule and the amendment's additional
within-stratum guard: mean fold changes are +0.0219 pooled and +0.0567
within-stratum, with one negative fold for each.

Therefore RC on is supported for the Enhancer transfer route. It is supported
for the specific Intron `6079cd38` policy, not as an unqualified Intron-wide
claim. RC off remains the conservative default elsewhere while Stage 3 still
crosses both RC modes to estimate the RC-by-loss interaction.

## Intron Sensitivity-Stratum Result

The leading `6079cd38` RC-on arm has pooled Pearson 0.6821,
within-stratum-centered Pearson 0.4510, macro-stratum Pearson 0.3503, and
minimum-stratum Pearson 0.1758. Its per-category results are:

| Inferred sensitivity category | n | Pearson | RMSE | COD R2 | Calibration slope | Bias |
|---|---:|---:|---:|---:|---:|---:|
| `mask1_specific` | 374 | 0.599 | 0.534 | 0.350 | 1.027 | +0.059 |
| `mask2_not_mask1` | 365 | 0.277 | 0.498 | 0.067 | 0.769 | +0.026 |
| `mask3_residual` | 322 | 0.176 | 0.467 | 0.020 | 0.819 | +0.045 |

The development target means are 2.726, 1.968, and 1.847 respectively. A
mask-only baseline fitted using each fold's training rows and then applied to
that fold's held-out rows gives Pearson 0.5728, RMSE 0.5620, and COD R2
0.3246. The leading model therefore exceeds the mask-only baseline by 0.1093
Pearson and 0.1363 COD R2, showing real within-category signal, but the weak
residual-category result remains a robustness limitation.

The full-OOF decomposition makes the pooled-versus-conditional distinction
more precise: 32.9% of target variance, 70.6% of prediction variance, and
70.6% of target-prediction covariance lie between inferred category means.
Equal-stratum weighting changes pooled Pearson only from 0.6821 to 0.6785
because the category counts are already close to equal; it does not remove
between-category covariance. The problem is therefore an estimand mismatch
from mixture-induced between-category covariance, not leakage or an invalid
natural-library score. See the
[dedicated Intron protocol and figure guide](lib1_dedup_intron_estimand_and_challenge_set_protocol_july2026.md).

Here `mask3_residual` means an otherwise valid canonical 80-nt sequence that
matches the unconstrained all-`N` Mask 3 but not the broader `GT...AG` Mask 2;
it does not mean model error, low barcode support, or failed synthesis. The
1,061 development constructs already satisfy the held-out minimum of eight
barcodes. A position/base-balanced analysis is appropriate as a secondary
development-only sensitivity analysis, but literal 25% A/C/G/T at every
position conflicts with the designed boundary constraints and must not replace
the frozen primary validation/audit estimand after results have been seen.

The Stage 2 notebook now shows A/C/G/T frequency at each of the 80 aligned
positions for all 1,061 development constructs, separated into the inferred
categories (374/365/322). This makes the mask-derived sequence signatures
visible and helps explain how a CNN can recognize a category. It does not
recover synthesis membership, establish causal bases, capture higher-order
motifs, or measure prediction performance; the per-category score table must
remain beside it.

The earlier approximately 0.703 mask-mean Pearson was a fold-0 diagnostic, not
the complete Stage 2 OOF estimand. The fold-training-fitted OOF baseline above
is the appropriate Stage 2 comparison. These categories remain inferred
sequence-mask sensitivity labels, not recovered synthesis-subset membership.

## 3'UTR Learning Histories And Targeted-HPO Decision

`train_mse` is calculated after standardizing the target using each fold's
training rows. A normalized training MSE of 1 is approximately the error of
predicting the training mean. A run that reaches only 0.94 has reduced that
baseline error by about 6%; it is not a raw-log2 RMSE and does indicate weak
training fit for that run.

Across all 100 Stage 2 UTRBassetVL histories:

| Diagnostic | RC off | RC on |
|---|---:|---:|
| Median best epoch | 35.5 | 36.0 |
| Median best-checkpoint train MSE | 0.840 | 0.913 |
| Runs with best-checkpoint train MSE >= 0.90 | 20/50 | 28/50 |
| Median stopping epoch | 80.5 | 81.0 |

All 660 local W&B histories were materialized. One Enhancer scratch run
(`1szhlfg8`) has an invalid trailing local W&B block; its complete epoch rows
were recovered and the warning is recorded in
`stage2_learning_history_summary.csv`. This does not affect the 3'UTR history
diagnostic or any OOF prediction metric.

The leading `86969bcf` RC-off configuration is not itself uniformly stuck at
0.94: its median train MSE at the best checkpoint is 0.692 across folds
(range 0.468-0.941), with median best epoch 32. No challenger has systematic
220-epoch cap saturation. More epochs alone are therefore not the indicated
fix.

There is, however, a repeated optimizer-boundary signal. Among the ten frozen
challenger configurations, log learning rate versus fold validation Pearson
has Spearman rho +0.355 (`p=0.011`) for RC off and +0.379 (`p=0.0066`) for RC
on; the association is positive in every fold. Higher learning rate is also
associated with lower train MSE (rho -0.411 off and -0.371 on), and lower
train MSE is associated with higher validation Pearson (rho -0.418 off and
-0.301 on). The winning learning rate, 0.001863, is close to the historical
sweep ceiling of 0.002. The quoted p-values are the ordinary two-sided
Spearman approximations applied to 50 config-fold rows per RC panel. They are
unadjusted diagnostics: the calculation does not account for repeated
learning rates within configs, overlapping fold-training sets, or the several
diagnostics examined, so they are not publication-grade independent-replicate
tests and are omitted from the presentation plot.

This satisfies the amendment's gate for a bounded 20-30 configuration
targeted 3'UTR UTRBassetVL search, which is now approved: the challenger wins in four
of five folds against the leading stable ResNet arm, while existing policies
show repeatable training-fit and upper-learning-rate-boundary behavior. The
evidence is still observational and confounded by the purposefully selected
K=10 portfolio, so it does not justify a broad replay or an automatic launch.
The targeted space must be frozen first, must use only development data, and
must keep the audit loader unavailable.

## Provisional Stage 3 Top Five

Selecting each base configuration by its better Stage 2 RC arm and then
ranking pooled OOF Pearson gives the following provisional prefixes. Prefixes
are unique within the Stage 2 manifest; this table is not yet the immutable
Stage 3 selection manifest.

| Part | Provisional five, in pooled-rank order |
|---|---|
| Enhancer | `6e6b2b97` K562/full on; `e53d6596` HepG2/full on; `3f7d963d` HepG2/conv3-plus on; `f199d009` K562/conv3-plus on; `404c9e99` K562/branched-only on |
| Intron | `6079cd38` on; `58481a47` on; `0ee9e54c` off; `873605b1` off; `767a6d28` off |
| Promoter | `00175f1c` off; `bff24362` off; `e10d0e2b` off; `0c0cefe7` off; `9b929319` off |
| 3'UTR | `86969bcf` UTRBassetVL off; `6b80f0ea` UTRBassetVL off; `acad3448` UTRBassetVL on; `4c3c7c47` UTRBassetVL on; `585fba9a` ResNet1D on |
| 5'UTR | `9dd728c0`, `25d3b0fb`, `e3b85c86`, `99b40ac8`, and `bee0f2b5`, all UTRBassetVL off |

Before freezing this set:

1. freeze and run the approved targeted 3'UTR search, then use its
   development-only evidence to resolve the five 3'UTR identities;
2. apply the one-standard-error/simple-model rule rather than treating tiny
   numerical differences as real ranks;
3. decide whether 5'UTR should replace fifth-ranked UTRBassetVL `bee0f2b5`
   (Pearson 0.4778) with the best ResNet1D `ffd49926` (Pearson 0.4657) as the
   near-tied architecture-diverse representative;
4. decide whether the Enhancer weighted-loss set should be the pure top five
   transfer policies or retain one intentionally labeled scratch anchor; and
5. generate an immutable full-ID selection manifest with input policy,
   architecture, source head/scope, and Stage 2 evidence.

The pure ranking correctly retains one ResNet1D candidate for 3'UTR. It does
not imply that the 3'UTR challenger lane broadly dominates: the challenger
lane median Pearson is 0.252, essentially the core median of 0.251, and the
gain is concentrated in a few UTRBassetVL configurations.

## Weighted-Loss Handoff

After the top-five decision is frozen, Stage 3 adds 250 weighted mates:

```text
5 parts x 5 base configs x 5 folds x 2 RC modes = 250 new runs
```

The 250 unweighted Stage 2 cells are reused, yielding the planned paired
`RC x loss` factorial. Evaluation remains unweighted on the same high-barcode
development constructs; barcode weighting changes the contribution of
training examples, not the primary validation estimand. Weighted-minus-
unweighted decisions must use the same strict rule as RC and report the
RC-by-loss interaction.

The notebook phrase "new weighted rows if frozen" means `5 selected configs ×
5 folds × 2 RC modes = 50` not-yet-run weighted training cells for one part,
but only after those five full config identities are locked. It does not mean
that 50 weighted rows already exist. Across all parts, the eventual analysis
contains 250 reused unweighted cells plus 250 new weighted cells.

Enhancer transfer can be barcode-weighted and has been before. Historical
`src/finetune` custom loops consumed `(x, y, w)` and minimized
`sum(w * squared_error) / sum(w)` across the branched-only, conv3-plus, and
full scopes. Those completed results establish feasibility but used older
pre-dedup tables, different splits, and in one family a log10 target; they
cannot substitute for the current dedup five-fold Stage 3 pairs.

The implementation prerequisite is narrower. The standardized Stage 2 class
`CNNBassetBranchedScopedTransfer` currently inherits a training step that
unpacks only `(x, y)` and applies ordinary MSE, so reusing it unchanged would
discard a third weight tensor. Stage 3 therefore needs the already-established
weighted arithmetic integrated into a tested scoped-transfer graph while
preserving source-head loading, warm-up/unfreezing, optimizer groups, and exact
RC/fold pairing. A strict weights-required guard must prevent a run labeled
weighted from receiving a two-tensor batch. The standard scratch and
UTRBassetVL routes likewise retain explicit arithmetic and missing-weight
checks when switched to the weighted graph.

## Resolved Analysis Implementation And Remaining Freeze Decisions

The analyzer now materializes the formal Pearson fold gate, 660 fold-metric
rows, raw calibration for every arm, and the exact fold-training-fitted Intron
mask-mean baseline. The reporting program materializes all 660 local learning
histories, best-epoch/convergence summaries, optimizer-boundary diagnostics,
strict RC review, and the provisional top-five/diversity table. The executed
notebook renders these products without constructing a DataModule.

Remaining work is decision or Stage 3 implementation work, not missing Stage
2 evidence:

- define or explicitly retain the conservative zero-tolerance interpretation
  of "no material" RMSE/COD degradation;
- freeze, run, and analyze the approved bounded 3'UTR targeted HPO;
- freeze the one-standard-error and 5'UTR diversity judgments;
- decide whether Enhancer's weighted set is the pure transfer top five or
  retains a separately labeled scratch anchor;
- integrate and test genuinely weighted standardized scoped Enhancer transfer;
- record model-size evidence consistently across scratch and transfer routes;
  and
- generate an immutable full-ID Stage 3 selection/launch manifest only after
  those decisions are signed off.

## Publication Guardrails

- These are validation-selected OOF results across 132 arms. A winning score
  is not an unbiased final generalization estimate after config, architecture,
  and RC selection.
- Fold training sets overlap heavily; fold scores are stability diagnostics,
  not independent replicates for a five-sample significance test.
- Core and challenger portfolios have different historical selection priors.
  Report their full distributions and provenance rather than only winners.
- Enhancer transfer changes initialization and input context together, so the
  route win cannot be attributed to pretraining alone.
- The 3'UTR Pearson gain is accompanied by low Spearman, calibration error,
  and high fold variability; all should appear beside the headline Pearson.
- Intron inferred masks are sensitivity categories only.
- For Intron, never report pooled Pearson as evidence of uniform within-design
  ranking without the mask-mean baseline, within-centered, macro, minimum,
  and per-category results beside it.
- Keep the frozen audit inaccessible until the Stage 3 policy and final refit
  protocol have been preregistered. Do not return from audit results to HPO.
