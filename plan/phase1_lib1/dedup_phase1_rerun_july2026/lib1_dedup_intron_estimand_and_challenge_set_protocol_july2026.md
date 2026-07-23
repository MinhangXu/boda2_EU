# Lib1 Dedup Intron Estimand And Challenge-Set Protocol

**Draft date:** 2026-07-14
**Status:** post-Stage-2 development analysis complete; Stage 3 inferred-mask
sensitivity rule frozen by the 2026-07-14 Stage 3 amendment; final-audit
reporting rule frozen but not executed
**Audit status:** frozen audit loader not instantiated or scored

## Purpose And Scope

This document answers the concern that the pooled Intron correlation is
"inflated" and routes the collaborator's proposed position-balanced,
high-barcode evaluation set into the existing five-fold-development plus
frozen-audit framework.

The concern is substantively correct only after naming the intended claim.
The pooled Pearson correlation is a valid estimate of ranking performance for
the observed mixed Intron library. It is not data leakage and it is not a
mathematically invalid score. It is **composition-assisted relative to a claim
about ranking sequences within a fixed Intron design regime**. The problem is
therefore:

> **mixture-induced between-category covariance and heterogeneous conditional
> performance, creating an estimand mismatch when pooled correlation is used
> to imply uniform within-design-regime ranking.**

The three sequence-mask labels remain inferred sensitivity categories. They
are not recovered synthesis-pool membership and must not be renamed as true
subsets.

This document does not change the completed Stage 2 primary metric, rerun any
model, alter a fold, or move an audit row. The later dated Stage 3 amendment
authorizes development-only Stage 3 implementation under this sensitivity
rule but does not authorize a full launch or audit access. Because the
development outcomes were already available when the additional balancing
checks were written, those checks are explicitly post hoc and descriptive.
The final-audit reporting rule was accepted into the dated Stage 3 amendment
before any audit predictions were generated or inspected and is now binding
for the eventual one-time audit.

## The Statistical Problem

Let (G\) be the inferred sequence-mask category, (Y\) the raw
`log2_RNA_DNA` target, and \(\widehat Y\) the raw held-out prediction. The law
of total covariance gives

\[
\operatorname{Cov}(Y,\widehat Y)
=
\mathbb E\!\left[\operatorname{Cov}(Y,\widehat Y\mid G)\right]
+
\operatorname{Cov}\!\left(\mathbb E[Y\mid G],
                            \mathbb E[\widehat Y\mid G]\right).
\]

The first term is within-category ranking signal. The second term rewards a
model for recognizing a mask-compatible sequence pattern and assigning the
category an appropriate mean activity. Both terms are part of natural-library
performance, but only the first directly addresses ranking within a fixed
design regime.

Pearson correlation is not additive, so `0.682 - 0.451` must not be presented
as an exact amount of statistical "inflation." The two numbers use different
covariance and variance denominators. The defensible statement is that the
pooled result is substantially composition-assisted, followed by the complete
metric and covariance decomposition below.

## What Completed Stage 2 Already Answers

The leading Intron arm is `basecfg_6079cd38...`, RC on, with exactly one OOF
prediction for each of 1,061 development constructs.

| Estimand | Pearson | What it answers |
|---|---:|---|
| Natural pooled OOF | 0.6821 | Ranking in the observed mixed library |
| Equal-stratum weighted pooled OOF | 0.6785 | Pooled ranking after giving each inferred category one-third mass |
| Fold-training-fitted mask-mean baseline | 0.5728 | How far category recognition plus category means can go without within-category sequence ranking |
| Within-stratum centered | 0.4510 | Ranking after subtracting observed and predicted category means |
| Equal-stratum weighted within-centered | 0.4435 | Within-category ranking with equal category mass |
| Macro-stratum | 0.3503 | Unweighted mean of three category-specific correlations |
| Minimum stratum | 0.1758 | Worst category-specific correlation |
| Mask 1 compatible | 0.5986 | Ranking among 374 Mask-1-compatible constructs |
| Mask 2, not Mask 1 | 0.2765 | Ranking among 365 Mask-2-not-1 constructs |
| Residual exact-80 | 0.1758 | Ranking among 322 valid exact-80 constructs outside Mask 2 |

The leader exceeds the leakage-safe mask-mean baseline by 0.1093 Pearson and
0.1363 COD \(R^2\). The model therefore learns real sequence signal beyond
category means. The low Mask-2-not-1 and residual correlations nevertheless
show that the pooled score does not establish uniform performance across
design regimes.

The full OOF covariance decomposition is more direct:

| Quantity | Between inferred categories | Within inferred categories |
|---|---:|---:|
| Target variance | 32.9% | 67.1% |
| Prediction variance | 70.6% | 29.4% |
| Target-prediction covariance | 70.6% | 29.4% |

Thus almost one third of target variance, but more than two thirds of the
leader's prediction variance and target-prediction covariance, is associated
with category mean differences. This is the most precise version of the
collaborator's concern.

The pre-Stage-2 amendment already required pooled, within-centered,
macro-stratum, minimum-stratum, and per-stratum metrics. Those results
**substantially address category-driven composition assistance**. They do not
test every possible sequence-composition shift, and they do not make the
current audit a controlled external challenge library.

## Recommended Figures

The smallest persuasive figure package is:

1. [Pooled, mask-mean baseline, and within-centered OOF triptych](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/figures/stage2_intron_pooled_baseline_centered_triptych.png).
   This is the clearest intuitive explanation of why pooled Pearson is not the
   mean of the three category-specific correlations.
2. [OOF estimand audit](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/figures/stage2_intron_estimand_audit.png).
   Its panels show the variance/covariance decomposition, the metric ladder,
   and the pooled-to-within gap in all five held-out folds.
3. [Per-category OOF calibration facets](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/figures/stage2_intron_per_stratum_oof_calibration.png).
   Every panel uses the same raw x/y limits and a common identity line.

Useful supplements are the existing target/prediction violin plot, the
position/base-frequency heatmap, and the
[barcode-threshold sensitivity plot](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/figures/stage2_intron_barcode_threshold_sensitivity.png).
The composition heatmap explains why category recognition is possible; it is
not itself performance evidence.

For a publication-ready forest plot, add construct-resampling intervals while
preserving development-fold proportions. Such intervals are conditional on
the five already-trained models. Do not treat the five overlapping-training
folds as five independent biological replicates.

## Validity Of A Literal Position-Balanced Set

The collaborator's proposal is useful as the intuition for a separate
composition stress test, but literal 25% A/C/G/T at every position is not a
neutral repair of the current evaluation distribution.

Let \(S\) be the union of Mask 1 compatible and Mask 2-not-1. Every sequence
in \(S\) has

\[
X_1=G,\qquad X_2=T,\qquad X_{79}=A,\qquad X_{80}=G.
\]

For any target distribution \(q\) with 25% G at position 1,

\[
q(S) \le q(X_1=G) = 0.25.
\]

The same restriction follows at the other fixed boundary positions. Equal
allocation to the three inferred categories requires \(q(S)=2/3\), while the
current development mixture has

\[
p(S)=\frac{374+365}{1061}=69.7\%.
\]

Therefore literal equal bases and equal categories cannot both hold. Literal
within-category 25% base frequencies are also impossible at the fixed and
IUPAC-restricted positions of Masks 1 and 2.

A target-free linear-program check over the actual 1,061 development
sequences strengthens this conclusion. It allowed arbitrary nonnegative
construct weights summing to one and required every one of the 320
position/base marginals to equal 0.25. Exact balance was infeasible. The
closest minimax solution still missed at least one marginal by 3.704
percentage points. At that optimum, the residual-category mass is constrained
to approximately 91.54%--91.57%. One returned optimum used 156 positive
weights, had Kish effective sample size about 95, and placed about 27.3 times
uniform weight on its largest row. The optimum weights are not unique, so the
ESS and support count are diagnostics; the infeasibility, minimum deviation,
and optimized residual-mass range are the durable conclusions.

Even a feasible first-order balance would not balance dinucleotides, k-mers,
GC tracts, motif combinations, or higher-order sequence grammar. It would
also not balance the target distribution. A result from such a set would
answer performance under a deliberately different sequence distribution, not
reveal a universally "unbiased" number.

## What High Barcode Count Does And Does Not Do

All 1,061 development constructs already have at least eight barcodes. Barcode
count is a measurement-support proxy; it does not remove category mean shifts
or guarantee a representative sequence distribution. If barcode recovery is
associated with sequence or activity, filtering can also change the estimand.

The current development-only threshold check is:

| Minimum barcodes | n | Pooled Pearson | Within-centered | Macro-stratum | Minimum-stratum |
|---:|---:|---:|---:|---:|---:|
| 8 | 1,061 | 0.682 | 0.451 | 0.350 | 0.176 |
| 10 | 560 | 0.721 | 0.421 | 0.284 | 0.088 |
| 12 | 307 | 0.736 | 0.489 | 0.299 | -0.029 |

The pooled number increases with a higher cutoff, while worst-category
evidence deteriorates and sample size falls. This is exactly why a single
favorable high-barcode cutoff must not replace the complete evaluation suite.
These thresholds were examined after Stage 2 and are descriptive, not a new
independent test set.

## Routing Into Five-Fold Development And Frozen Audit

### Completed five-fold development product

No retraining or resplitting is needed. For every fixed config/RC arm:

1. retain exactly one held-out prediction for every development construct;
2. keep natural-mixture pooled OOF Pearson as the campaign primary metric;
3. always co-report within-centered, macro, minimum, per-category calibration,
   and the fold-training-fitted mask-mean baseline for Intron;
4. report the equal-stratum sensitivity with target-free weights

   \[
   w_i \propto \frac{1/3}{\widehat p(G_i)},
   \]

   including both weighted pooled and weighted within-centered correlation,
   the weight range, and effective sample size; and
5. label the `>=8`, `>=10`, and `>=12` barcode results as post-Stage-2
   development sensitivities, with category counts beside every score.

Equal-stratum weighting changes the leader's pooled Pearson only from 0.6821
to 0.6785 because the observed counts, 374/365/322, are already close to equal.
It does not remove between-category mean covariance; that is why weighted
within-centered and per-category results remain mandatory.

### Frozen rule for the final audit

Do not inspect audit category counts, targets, or predictions now. The dated
Stage 3 amendment accepts the following deterministic reporting rule before
the audit loader is instantiated:

1. freeze model/config, RC policy, loss, checkpoint/refit policy, preprocessing,
   and every evaluation formula;
2. evaluate the full natural audit once and retain its pooled Pearson, RMSE,
   COD \(R^2\), Spearman, and calibration as the primary audit result;
3. apply the already-defined inferred-mask function and report
   within-centered, macro, minimum, and all per-category metrics on the same
   predictions;
4. optionally apply the fixed equal-stratum formula above, using labels only,
   and report pooled plus within-centered weighted results and effective
   sample size;
5. if the barcode sensitivity is retained, use the fixed cutoffs 8, 10, and
   12, display every resulting `n`, and suppress category-specific Pearson
   when a category has fewer than 30 rows or zero target/prediction variance;
6. do not revise a cutoff, weight rule, category rule, model, or claim after
   audit outcomes are visible; and
7. do not return from audit results to HPO.

The full 265-row natural audit remains primary. No post hoc balanced subset
replaces it, and no row is moved between development and audit.

### A genuinely new external challenge library

If the publication requires a composition-controlled external-validity test,
design and synthesize it separately after the final model is locked:

- allocate constructs across declared design families or, if true membership
  remains unavailable, clearly labeled inferred mask categories;
- preserve every fixed boundary and balance only mutable positions over each
  mask's allowed alphabet;
- cover joint GC, motif, and k-mer structure rather than only marginal base
  frequencies;
- separate sequence-composition design from the assay-QC/barcode-depth rule;
- preregister sample size, exclusions, barcode-QC threshold, and analysis;
  and
- treat it as an external challenge result, not a replacement for the frozen
  audit or a new source of HPO feedback.

## Reproducible Products

Run the development-only reporting layer with:

```bash
cd "$(git rev-parse --show-toplevel)"
conda run -n boda_env python src/analysis/lib1_dedup_intron_sensitivity_reporting.py
```

The program reads only existing development OOF outputs and the split
manifest. It does not construct a DataModule or audit loader. Its products are:

- [reporting program](../../../src/analysis/lib1_dedup_intron_sensitivity_reporting.py)
- [machine-readable summary](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_intron_sensitivity_reporting_summary.json)
- [estimand table](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_intron_leader_estimand_summary.csv)
- [variance/covariance decomposition](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_intron_leader_signal_decomposition.csv)
- [five-fold estimands](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_intron_leader_fold_estimands.csv)
- [barcode-threshold sensitivity](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_intron_leader_barcode_threshold_sensitivity.csv)
- [fixed-boundary constraints](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_intron_literal_base_balance_constraints.csv)
- [position-balance linear-program result](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_intron_literal_position_balance_lp.csv)

This protocol extends the interpretation in the
[Stage 2 analysis report](lib1_dedup_stage2_analysis_report_july2026.md) and the
[pre-Stage-2 amendment](lib1_dedup_pre_stage2_protocol_amendment_july2026.md).
