# Lib1 Dedup Post-Presentation Interpretation Addendum

**Addendum date:** 2026-07-17
**Status:** explanatory record after the locked final-test evaluation; no model
selection, retraining, or final-test rescoring is authorized by this document

## Terminology Policy

User-facing reports use **locked final test set** and **one-time final-test
evaluation**. The word `audit` is reserved for an investigative/root-cause
process. Historical code keys, partition values, filenames, hashes, W&B names,
and output directories that contain `audit` remain unchanged so provenance and
reproducibility are not broken.

The experiment hierarchy is:

1. A **construct row** has one sequence and one construct-level target.
2. A **base config** fixes architecture, optimizer/training hyperparameters,
   initialization or transfer route, and input policy. Fold, RC, and loss are
   not part of the base-config identity.
3. A **cell** is one trained `(part, base config, RC, loss, fold)` run.
4. An **arm** is one `(part, base config, RC, loss)` condition summarized over
   its five fold cells and five-fold OOF predictions.
5. A **pair** contains arms that differ in exactly one tested intervention,
   such as weighted versus unweighted loss.
6. A **gate** is the predeclared paired evidence rule for that intervention.
7. An **admissible arm** is an arm allowed into part-specific selection.
8. The **selected policy** is the one selected arm per part, after which its
   fixed epoch budget and three final-refit seeds are attached.

Stage 3 contained 50 base configs, 180 arms, and 900 fold cells. The
part-specific admissible pools contained 17 Enhancer, 14 Promoter, 18 Intron,
13 3'UTR, and 19 5'UTR arms.

## Gate Pass, Admissibility, And The One-SE Pool

Admissibility is evaluated at the arm level. A complete RC-off/unweighted arm
with finite required metrics is the baseline. A weighted arm must pass its
weighted-versus-unweighted loss gate; an RC-on arm must pass its same-loss RC
gate; an RC-on/weighted arm must pass both. Non-finite required selection
metrics fail closed.

Therefore, `gate_pass` in the weighted-minus-unweighted plot means that the
**loss intervention pair** passed its full paired loss gate. It is an input to
admissibility, not a synonym for admissibility. For an RC-off weighted arm,
loss-gate passage plus finite metrics is sufficient. An RC-on weighted arm
also needs its RC gate.

The paired gate requires:

- mean of five fold-Pearson deltas at least `+0.005`;
- at least four positive fold deltas;
- pooled RMSE increase and COD R2 decrease within the frozen part-specific
  allowances; and
- for Intron, nonnegative mean within-inferred-stratum-centered delta and no
  more than two negative centered folds.

Only admissible arms within the same CRE part enter its selection pool. The
numerical best is the arm with the largest pooled OOF Pearson. Its SE is the
standard deviation of 10,000 Pearson values obtained by resampling held-out
construct rows within each fold, not the SD or SE of five fold scores. The
one-SE set is every admissible arm satisfying

```text
pooled OOF Pearson >= best admissible Pearson - bootstrap SE(best)
```

The frozen rule then prefers the largest minimum-fold Pearson. Intron next
uses minimum-stratum Pearson and within-stratum-centered Pearson. RMSE and COD
R2 follow, and complexity preferences are only exact-tie resolvers. This is
why 3'UTR chose admissible rank 5: it sacrificed a small amount of pooled
Pearson inside the one-SE band for a better worst fold.

## Exact Development-To-Final-Test Flow

The five fold checkpoints are used only to create OOF evidence and determine
the fixed epoch budget. The campaign did **not** refit all configs or directly
test those five checkpoints.

1. Every candidate arm trains five fold cells; each development construct is
   held out once and receives one OOF prediction.
2. Development-only OOF evidence selects exactly one policy per part.
3. The fixed epoch count is the median of the selected arm's five zero-based
   best-epoch indices, plus one completed epoch.
4. That one selected policy is trained anew at seeds 1701, 1702, and 1703 on
   every eligible non-final-test row. There is no validation loader, no early
   stopping, and the final epoch is retained.
5. Those 15 frozen checkpoints are evaluated once on the locked final test.
   The primary predictor is the arithmetic mean of the three seeds' raw
   per-construct predictions. Neural weights are not averaged.

The seeds were fixed prospectively as the established campaign seed 1701 plus
two consecutive stochastic replicates, 1702 and 1703. They were not selected
using final-test performance.

## Target And Plot Units

The raw construct-level target is

```text
log2(RNA_bc_counts_sum / DNA_bc_counts_sum)
```

with aggregate exact-deduplicated barcode counts and no pseudocount. A
one-unit increase is a twofold increase in the RNA/DNA ratio.

Training does standardize this target using the mean and sample SD fitted only
on the current training rows. Fold-specific training therefore has
fold-specific normalization, and each final refit fits normalization on all
of its non-final-test rows. Every exported prediction is inverse-transformed:

```text
prediction_raw = prediction_standardized * training_SD + training_mean
```

Consequently, the calibration figure is not z-scored. Both axes are raw
`log2(RNA/DNA)` in the same mathematical unit for every part, although each
part has a different raw distribution and range. The top row shows individual
constructs and the bottom row bins the same points into hexagons; color is the
number of constructs in a hexagon. The rows use identical x/y limits within a
part. Density changes the visibility of overlapping points, not Pearson or the
calibration fit.

## Intron: What The Current Result Establishes

The Mask-1-compatible final-test sequences form a high-expression cluster:
their observed mean is 2.765, versus 1.912 for Mask-2-not-1 and 1.894 for the
residual group. This is consistent with the team's library-design account
that Mask 1 sequences splice strongly, but the current artifacts contain
sequence-inferred masks rather than verified sublibrary membership or direct
splicing measurements. Until provenance or junction evidence is supplied,
the claim-safe name is **Mask-1-compatible high-expression sequences expected
to splice strongly**.

The natural pooled Pearson of 0.681 contains both within-group ranking and
between-group mean separation. Centering observed and predicted values within
the three inferred masks gives 0.473. The latter demonstrates real residual
ranking signal while confirming that the pooled number is composition
assisted.

The current 265-row final test was stable-ID hash sampled from the natural
high-barcode pool; it was not position balanced. Its dominant splice-boundary
frequencies are G at position 1 (78.9%), T at position 2 (78.1%), A at position
79 (75.5%), and G at position 80 (76.6%). These closely mirror development,
so this is a natural-mixture test rather than an unusually skewed draw.

“All N at every position” should mean approximately 25% A/C/G/T across the
selected sequences, not literal `N` characters. Literal global balance is
infeasible on the current sequence support. Mask-1- and Mask-2-compatible
sequences all fix `GT` at positions 1-2 and `AG` at 79-80; forcing 25% at those
positions caps their combined mass at 25%, compared with about 70% naturally.
Even arbitrary weighting of the 265 rows cannot attain exact balance; the
closest minimax weighting misses at least one positional marginal by 7.703
percentage points and has Kish effective sample size about 52.

Because the final-test outcomes were already viewed before this proposal, no
subset or reweighting of these rows can become a new untouched final test. It
may be reported only as a post-hoc sensitivity and must not return to HPO.

For a genuinely prospective external Intron challenge set:

- recover verified synthesis/design-family labels;
- retain required splice boundaries and balance only positions mutable under
  each family's allowed alphabet;
- predeclare family quotas and minimum per-family sample sizes;
- isolate exact sequences, design parents, and motif siblings across data
  roles;
- select using sequence and frozen QC fields only, never targets or model
  predictions;
- optimize position discrepancy jointly with GC, motif, and selected k-mer
  coverage, with deterministic tie-breaking and an infeasibility report; and
- freeze the model, estimands, metrics, intervals, and suppression rules before
  evaluating once.

This challenge answers a distribution-shift question. It complements rather
than replaces the natural-mixture final test.

## Enhancer RC Interpretation

The selected Enhancer policy transfers the exact Malinois BassetBranched
checkpoint. The SHA-bound checkpoint serializes
`use_reverse_complements=True`, and the documented Malinois recipe also used
RC augmentation. All six Stage 3 pretrained BassetBranched transfer configs
showed positive mean RC effects, while the four scratch anchors were a
different ResNet1D architecture and none passed an RC gate.

The safe conclusion is therefore route-specific:

> RC was helpful within the pretrained BassetBranched Enhancer transfer
> family, but this campaign cannot separate a genuine Lib1 RC effect from
> architecture or compatibility with an RC-trained source representation.

A causal follow-up requires matched BassetBranched controls that differ only
in source-checkpoint RC history or initialization. The completed final-test
result does not authorize changing the already frozen Enhancer policy.

## Finality

This addendum changes terminology and interpretation only. It does not alter
the selected policies, final checkpoints, final-test rows, scores, or claims
of statistical independence. Internal legacy identifiers containing `audit`
remain intact and should be translated to “final test” only at the
presentation/reporting boundary.
