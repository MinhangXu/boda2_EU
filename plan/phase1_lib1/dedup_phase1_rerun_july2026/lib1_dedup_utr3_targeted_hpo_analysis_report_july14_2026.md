# Lib1 Dedup Targeted 3'UTR HPO Analysis Report

**Date:** 2026-07-14
**Status:** development-only analysis complete; Stage 3 five not yet frozen
**Binding design:** `lib1_dedup_targeted_utr3_hpo_protocol_amendment_july14_2026.md`

## Scope And Safety

The bounded targeted campaign completed all 240 planned cells: 24 exact
UTRBassetVL configurations, five development folds, and paired RC off/on.
This analysis reconciles the frozen manifest, registry, compact provenance,
local W&B summaries, and validation-only prediction exports. It does not
construct a DataModule, instantiate an audit loader, load audit targets, call
`trainer.test`, or score an audit prediction. No Stage 3 manifest is generated
by this analysis.

The registry is the completion source of truth. Rows 2--240 have launcher
completion markers; row 1 was the separately completed pilot and is supported
by the same registry, prediction, provenance, and W&B evidence. The final
monitor snapshot is stale and is not used for completion accounting.

Verified accounting:

- 240/240 completed targeted cells, all with 105 validation predictions;
- 48/48 complete targeted OOF arms with 525 unique development constructs;
- 24/24 complete targeted RC pairs;
- 0 populated test metrics and `n_test=0` in every compact provenance record;
- 40 existing Stage 2 3'UTR comparator arms recomputed from their frozen OOF
  prediction product; and
- 88 combined arms from 44 base configurations and 46,200 OOF predictions.

Every targeted model has 291,845 parameters. Targeted training consumed 19.45
GPU-hours in total; median per-cell fit time was 251 seconds.

## Primary Result

The numerical targeted-HPO winner is:

`basecfg_6cb459958ae1a16e112bdacc6e03c9e02fc12cdc85ed951cfcd25ada7856a517`

It is configuration 10, RC off, with learning rate 0.002, weight decay 0.0007,
and linear dropout 0.50.

| Metric | Value |
|---|---:|
| Pooled raw OOF Pearson | 0.506134 |
| Pooled Spearman | 0.185349 |
| Pooled RMSE | 0.634657 |
| Pooled COD R2 | 0.160517 |
| Minimum fold Pearson | 0.353599 |

The leading Stage 2 UTRBassetVL incumbent remains very competitive:

`basecfg_86969bcf79247695d2c27ce1466d4eab2373e5e1f3645da99f24ebf4c59c0fbe`

Its RC-off arm has Pearson 0.501355, Spearman 0.230066, RMSE 0.617865, COD R2
0.204353, and minimum fold Pearson 0.359391. The new winner gains only 0.004779
Pearson while worsening every listed secondary metric. A construct-paired,
fold-stratified descriptive bootstrap gives a 95% interval of approximately
[-0.0470, 0.0655] for new winner minus incumbent and a bootstrap probability
of improvement of about 0.586. This is not a confirmatory comparison because
the targeted winner was selected on these same OOF rows.

The best Stage 2 ResNet arm has Pearson 0.332251. The combined evidence no
longer supplies a performance-based reason to reserve a 3'UTR ResNet diversity
slot.

## Frozen One-Standard-Error Rule

The numerical winner's fold-stratified, 10,000-resample bootstrap standard
error is 0.069741, making the one-SE cutoff 0.436393. Fifteen targeted arms
from 12 configurations are eligible. Applying the preregistered deterministic
tie order selects:

`basecfg_1e3a0c9f053271a63a4da596c588484b52c56cf65fe6fb791bd909e15c3b9def`

Its RC-off arm has Pearson 0.442002, RMSE 0.637708, COD R2 0.152428, and the
best eligible minimum-fold Pearson, 0.390561. This is the preferred targeted
arm under the frozen stability-first rule; it is not the numerical winner and
does not by itself define five Stage 3 configurations.

The original amendment froze the resample count and seed but did not specify
within-fold row order, NumPy generator, or sample-SD convention. This analysis
records the reproducible implementation as NumPy `default_rng`/PCG64, each
fold sorted by `construct_id`, five within-fold samples per replicate, and
sample standard deviation (`ddof=1`). Using native prediction-file row order
changes the Monte Carlo SE slightly but does not change the 15-arm set or the
preferred arm.

## Search-Surface And RC Findings

For RC off, mean arm Pearson by learning rate was 0.439 at 0.001, 0.471 at
0.002, 0.407 at 0.004, and 0.331 at 0.006. Weight decay and dropout had much
smaller marginal effects. The targeted evidence therefore supports a local
optimizer optimum around learning rate 0.002; it does not support extending
the learning-rate boundary upward.

Twenty-seven of 240 fold cells exported exactly constant predictions. Twenty-
six were RC-on cells: five at learning rate 0.004 and 21 at 0.006. The only
RC-off collapse occurred at 0.006. Different constant levels across folds can
produce a nonzero pooled correlation even when a fold correlation is
undefined, so finite-fold counts and fold-level results are mandatory beside
pooled metrics.

RC on improved pooled Pearson for only one of 24 configurations. Across the
grid, mean pooled RC-on-minus-off Pearson was -0.1864. No configuration passed
the planned gate of mean fold delta at least 0.005 and positive deltas in at
least four folds; none passed that gate plus the current zero-tolerance
RMSE/COD guard. RC off is therefore the development-selected unweighted
default for 3'UTR. Stage 3 must still train both RC states to preserve the
frozen RC-by-loss factorial and to test interaction with weighted training.

## 3'UTR Stage 3 Portfolio Decision

The pure pooled-Pearson top five base configurations are:

1. `basecfg_6cb459958ae1a16e112bdacc6e03c9e02fc12cdc85ed951cfcd25ada7856a517`
2. `basecfg_86969bcf79247695d2c27ce1466d4eab2373e5e1f3645da99f24ebf4c59c0fbe`
3. `basecfg_7b1f881265b0fc0aee9e2b601352b93e064e37bee808c65b6b038e6a260e2062`
4. `basecfg_8b14e9e7f2f26e52985dda2dec8f128c9da9a31662a64015dca76a993b4cd5b4`
5. `basecfg_1becdea28bb6a22dbb61a48222baf1cbce413ac6e405691c9bda4b1da6253f90`

A less redundant, evidence-balanced review set is recommended:

1. `basecfg_6cb459958ae1a16e112bdacc6e03c9e02fc12cdc85ed951cfcd25ada7856a517`
   -- numerical targeted winner;
2. `basecfg_86969bcf79247695d2c27ce1466d4eab2373e5e1f3645da99f24ebf4c59c0fbe`
   -- Stage 2 incumbent and stronger RMSE/COD anchor;
3. `basecfg_1e3a0c9f053271a63a4da596c588484b52c56cf65fe6fb791bd909e15c3b9def`
   -- preregistered one-SE stability choice;
4. `basecfg_0417b66646a3d1e1f7b7f00178f106a004221338769a86ef415d6b583d4a3b05`
   -- near-best secondary metrics and low fold SD; and
5. `basecfg_7b1f881265b0fc0aee9e2b601352b93e064e37bee808c65b6b038e6a260e2062`
   -- next strongest distinct targeted optimizer policy.

This recommendation deliberately trades two small pooled-rank advantages for
the frozen stability choice and stronger secondary/fold behavior. It is a
review proposal, not an immutable selection. A dated full-ID record must state
whether the pure top five or balanced principle governs before Stage 3
manifest generation.

If the balanced set is frozen, four targeted configurations contribute 40
reusable unweighted cells and the Stage 2 incumbent contributes 10, for 50
reusable 3'UTR cells. Stage 3 then adds exactly 50 weighted mates. The Stage 3
generator must allow exact unweighted reuse provenance from either the Stage 2
or targeted-HPO manifest.

## What The HPO Does And Does Not Decide

The HPO closes the last part-specific candidate search, supplies exact 3'UTR
config identities and reusable unweighted cells, diagnoses optimizer/RC
stability, and removes the earlier need for a weak ResNet performance slot. It
does not demonstrate weighted-loss benefit, determine RC under weighted
training, choose candidates for the other CRE parts, or estimate frozen-audit
generalization.

## Remaining Gate Before Stage 3 Implementation

The following items remain open across the five parts:

1. Freeze a global one-SE/simple-model and five-slot fill procedure, including
   the bootstrap implementation, the simplicity measure, stability and
   secondary-metric tie order, and consistent parameter-count evidence.
2. Freeze the exact five full 3'UTR IDs using either the pure or balanced
   portfolio principle above.
3. Decide 5'UTR's fifth slot: UTRBassetVL `bee0f2b5` versus architecture-
   diverse ResNet `ffd49926`.
4. Decide whether Enhancer uses the pure transfer top five or replaces one
   transfer policy with a separately labeled scratch anchor.
5. Resolve Intron's rank-five/rank-six robustness tie under its
   within-centered/minimum-stratum amendment.
6. Freeze numerical allowed RMSE increase and COD-R2 decrease for both RC and
   weighted-loss gates, including floating tolerance and whether the rule is
   config-specific or part-level. Explicit zero tolerance plus a small numeric
   epsilon is the most conservative current option.
7. Freeze the exact clipped-log barcode weight and normalized weighted-MSE
   contract, then implement strict weights-required behavior. The scoped
   Enhancer transfer path currently discards a third tensor, and the generic
   weighted path currently permits a two-tensor unweighted fallback.
8. Test weighted arithmetic, missing-weight failure, unweighted validation,
   RC duplication of training examples and weights only, and preservation of
   scoped-transfer source-head/scope/optimizer/reset behavior.
9. Create and independently verify a 250-new-cell Stage 3 manifest paired with
   250 immutable unweighted reuse cells, add an exact W&B stage contract, and
   run representative weighted pilots before any full launch.

Audit access remains blocked throughout this gate. Before the later audit
stage, the plan must also reconcile its 60-run four-arm sensitivity design
with the requirement to freeze a validation-selected RC/loss default, obtain
Intron secondary-reporting sign-off, and pre-budget any optional extra 5'UTR
finalist.
