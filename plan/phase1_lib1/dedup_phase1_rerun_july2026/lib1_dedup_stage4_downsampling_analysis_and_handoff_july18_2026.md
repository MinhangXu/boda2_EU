# Lib1 Dedup Stage 4 Downsampling Analysis And Handoff

**Analysis date:** 2026-07-18
**Status:** 660/660 development-only cells complete; presentation analysis
complete; Stage 4 does not authorize model reselection
**Final-test isolation:** no current final-test loader instantiated, product
read, or metric computed

## Completion And Estimand

The frozen manifest SHA-256 is
`dd6abda4726846f482536a235093b2ed9aa5a36b12591613c400601dcb27a84a`.
All 660 unique cells completed and reconcile to 132 pooled outer-fold OOF
tracks and 72 learning-curve points. The campaign contains:

- five frozen primary configurations at
  `N=[40,250,400,2500,4000,full]`;
- three nested training-subset tracks at each finite primary N and one full-N
  realization per outer fold;
- nine predeclared portfolio alternatives at sparse
  `N=[40,400,4000,full]` sensitivity anchors;
- one dense Enhancer scratch diagnostic; and
- five outer OOF folds, fixed model seed 1701, and a distinct inner
  checkpoint fold.

The primary estimand is pooled five-fold development-OOF Pearson. The frozen
bootstrap uses 2,000 replicates with seed `20260717`, resampling constructs
within outer folds and finite-N subset tracks while preserving comparisons
across training sizes.

The modeled target is construct-mean log2 RNA/DNA. Training-subset mean and
standard deviation are used for optimization-time normalization, and
predictions are inverse-transformed. All reported predictions, RMSE, COD R2,
slopes, and biases are therefore evaluated on the raw log2 RNA/DNA scale;
Pearson is invariant to this affine transform.

## Primary Sample-Size Result

The common `400 -> 4,000` decade is the cleanest cross-part marginal-data
comparison. Every interval is entirely above zero.

| Part | Mean full N | Full Pearson | Full COD R2 | Observed 400 -> 4,000 delta r (95% CI) | Observed 40 -> 4,000 delta r (95% CI) |
|---|---:|---:|---:|---:|---:|
| Enhancer | 4,145 | 0.516 | 0.239 | +0.065 [0.034, 0.096] | +0.128 [0.076, 0.183] |
| Promoter | 6,889 | 0.455 | 0.196 | +0.194 [0.148, 0.242] | +0.312 [0.236, 0.384] |
| Intron, pooled | 7,159 | 0.656 | 0.423 | +0.078 [0.043, 0.117] | +0.263 [0.159, 0.384] |
| 3'UTR | 6,385 | 0.309 | 0.062 | +0.242 [0.070, 0.389] | +0.273 [0.095, 0.440] |
| 5'UTR | 7,397 | 0.513 | 0.242 | +0.254 [0.196, 0.308] | +0.397 [0.340, 0.454] |

The `40 -> 4,000` column is a directly observed 100x training-size contrast,
not a forecast to 100x the current full dataset. The only tested overall 10x
interval that crosses zero is 3'UTR `40 -> 400`; its strong
`400 -> 4,000` improvement shows delayed takeoff rather than evidence that
data do not help.

Finite-N points average three nested subset tracks. Full N has one
training-subset realization per outer fold. In particular, Enhancer adds only
about 145 constructs from 4,000 to full, and its point estimate changes from
0.530 to 0.516. This small reversal is sampling/training variation, not
evidence that additional data are harmful.

## Intron: Pooled Performance Is Not The Biological Answer

Full-N pooled Intron Pearson is 0.656, but within-inferred-stratum-centered
Pearson is 0.394. The centered `400 -> 4,000` gain is +0.231 with 95% CI
[0.153, 0.313], much larger than the pooled gain of +0.078. Full-N
within-stratum Pearson is:

| Inferred design/mask stratum | N | Target mean | Full-N Pearson |
|---|---:|---:|---:|
| Mask 1 compatible | 374 | 2.726 | 0.500 |
| Mask 2, not 1 | 365 | 1.968 | 0.333 |
| Residual | 322 | 1.847 | 0.103 |

The separated target means confirm that between-design-group expression can
inflate pooled Pearson. Centering removes the three mean offsets and asks
whether the model predicts variation inside the groups. These sequence-based
groups are not measured splicing classes. They neither prove splice
efficiency nor create the team's requested position-balanced 80-bp
evaluation set.

The practical Intron conclusion is therefore not simply "collect more random
Introns." Prioritize verified and underrepresented within-group variation,
and predeclare a new evaluation set whose base distribution is balanced at
each of the 80 positions. If synthesis-sublibrary or measured-splicing labels
become available, analyze them as verified metadata rather than relabeling
the current inferred masks.

## Enhancer: Operational Transfer Advantage, Not A Causal Pretraining Claim

At full N, the selected K562/full-transfer + RC route has Pearson 0.516, RMSE
0.783, and COD R2 0.239. The scratch ResNet1D diagnostic has Pearson 0.082,
RMSE 1.241, and COD R2 -0.910. The selected route is therefore operationally
decisive for the present campaign.

This is not a controlled pretraining-only contrast: architecture, input
framing, RC, and initialization differ. The correct claim is that this tested
scratch route does not close the gap. Also, the selected transfer route gains
only about +0.004 mean Pearson from 2,500 to 4,000, making generic Enhancer
volume the lowest current marginal-data priority.

## Portfolio And Calibration Sensitivity

Stage 4 does not reopen selection. Its sparse alternatives show that the
direction of the learning-curve result is reasonably portfolio-robust, but
the primary config is not numerically best at every anchor:

- Enhancer rank 2 is +0.026 at 4,000 and +0.034 at full versus primary;
- 3'UTR rank 5 is -0.088 at 4,000 but +0.076 at full, a single-track rank
  reversal;
- Promoter and Intron primary configs lead their alternatives at high N; and
- 5'UTR is effectively tied at full (+0.001 for its alternative).

These are matched-track point deltas without a frozen paired-bootstrap
interval. They cannot trigger post hoc model replacement. If configuration
selection is deliberately reopened, it needs a separately dated confirmation
with repeated subset/model seeds and an evaluation plan that does not reuse
the already-opened final test.

Full-N observed-on-prediction slopes and mean prediction-minus-target biases
are:

| Part | Slope (ideal 1) | Bias (ideal 0) |
|---|---:|---:|
| Enhancer | 0.758 | +0.006 |
| Promoter | 1.285 | +0.003 |
| Intron | 0.895 | +0.030 |
| 3'UTR | 0.937 | +0.126 |
| 5'UTR | 0.798 | +0.026 |

Promoter predictions are compressed in range, while 3'UTR retains the clearest
positive mean shift. Any future affine calibration must be fit using
development predictions only and evaluated on genuinely new held-out data;
the current final test must not become calibration training data.

## What The Previous Downsampling Studies Add

The previous studies are context, not an absolute-performance leaderboard:

- the June pre-dedup scratch campaign averaged five configurations across
  five split seeds, used barcode-threshold pools, and evaluated against a
  different high-barcode held-out design;
- the pre-dedup exact-one-barcode diagnostic used three configurations and
  five split seeds; and
- the historical Enhancer curve used eight per-seed random splits and the
  K562/full-transfer route at `N=[50,400,1000,2000,full]`.

The replicated qualitative findings are useful: Promoter and both UTRs
continue to benefit from scale, 3'UTR again has delayed takeoff, and Enhancer
transfer again shows diminishing returns. The old pooled Intron result looked
flatter; the current centered analysis reveals continued within-stratum
learning. The old barcode-threshold endpoints also showed no consistent
benefit from discarding one-barcode rows, supporting the current
`train_min_barcodes=1` contract. Differences in absolute Pearson cannot be
attributed to deduplication, architecture, or any single protocol change.

## Curve-Family Limitation

The bounded power-law and exponential fits are tail-sensitivity scenarios.
They compare fitted(full) with fitted(10x full); they do not compare observed
full performance with an observed future experiment. Power-law Pearson
asymptotes hit the allowed `r=0.999909` boundary for Enhancer, Promoter,
3'UTR, and 5'UTR, and several RMSE fits hit zero. Family disagreement is large
for multiple parts. Do not quote a numerical 10x- or 100x-beyond-full gain,
average the two families, or interpret the fitted boundary as a biological
ceiling.

Stage 4 full-N values also must not be compared directly with Stage 3 values:
Stage 4 excludes both the outer OOF fold and a distinct inner checkpoint fold,
so its training pool and checkpoint protocol are intentionally more
conservative.

## Decisions For The Next Work

1. Keep the five Stage 3 policies frozen; Stage 4 supports data-acquisition
   decisions, not model reselection.
2. Treat 5'UTR and 3'UTR as the joint highest generic data-volume priorities,
   followed by Promoter. Their gain intervals overlap, so do not claim a
   precise ordering between the two UTRs.
3. Treat Intron as a targeted-design priority: collect within-stratum and
   position-balanced variation rather than optimizing the inflated pooled
   Pearson alone.
4. Treat generic Enhancer volume as lowest priority under the selected
   transfer route. Run a matched architecture/RC/input experiment only if the
   causal value of pretraining itself matters.
5. Use the observed contrasts for budgeting. Do not base acquisition counts
   on the unstable beyond-full curve projections.
6. Before any new deployment claim, predeclare how raw-scale calibration will
   be assessed on genuinely new data, especially for Promoter and 3'UTR.

## Presentation And Reproducibility Artifacts

- analysis notebook:
  `tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/06_stage4_downsampling_analysis.ipynb`;
- self-contained HTML presentation:
  `src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/stage4_downsampling_analysis_presentation.html`;
- executive report and scorecard:
  `src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/stage4_executive_summary.md` and
  `stage4_decision_scorecard.csv`;
- slide PNG/PDF files:
  `src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/`;
- machine-readable core contract:
  `src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/stage4_analysis_contract.json`.

The executed notebook contains no error outputs. Thirty Stage 4 analysis,
data-contract, manifest-contract, and presentation-reporting tests pass.

Frozen presentation hashes after execution:

- analysis contract: `1adff7a855c00745499741a849edf7009e1f727bebe80904c96f10295e350e7c`;
- executive-summary JSON: `78df446ff2865448223688d15206df572da6907a2e3fd1654c54b7911dcda982`;
- self-contained HTML: `49e4141a74a3cdda734cf0ae01bc18308ce2b4d7ec5dbb025aad08ab610f7949`;
- executed notebook: `1ab7dd124a48332b191a953c0e6fc61e6c50e941cc3ee72e206972654dcbdf69`.
