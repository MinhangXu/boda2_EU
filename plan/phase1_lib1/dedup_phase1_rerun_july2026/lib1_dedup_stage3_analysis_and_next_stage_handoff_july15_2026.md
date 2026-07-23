# Lib1 Dedup Stage 3 Analysis And Next-Stage Handoff

**Analysis date:** 2026-07-15
**Status:** Stage 3 development analysis complete; five development policies
selected; final-refit/audit implementation not yet authorized
**Audit status:** loader not instantiated; targets and IDs not loaded;
predictions and metrics not computed; inferred audit-stratum counts not
inspected

## Completion And Reproducibility

The frozen Stage 3 campaign completed all 450 new barcode-weighted cells. The
analysis resolves 450 immutable unweighted cells plus the 450 weighted cells,
for 900/900 cells and 180/180 five-fold OOF arms. It contains 90 loss-arm
pairs, 450 loss fold pairs, 80 RC-arm pairs, 400 RC fold pairs, and 40 complete
non-3'UTR RC-by-loss factorials. Registry, local provenance, validation
prediction IDs, and read-only W&B records reconcile with no duplicate
completion, retry, or failed row. Every run has `n_test=0` and blank test
metrics.

Frozen analysis evidence:

- analysis summary SHA-256:
  `1d339f3278446e959213f415cc9353e480672702fb539ed4eb7ade39e300693e`;
- output artifact index SHA-256:
  `1cc7011ae01f0f2d64fa6d6e433d69388e3592d7c7be268b45c9aa7007acd78a`;
- five-policy JSON SHA-256:
  `f0f818fb6b4b722726e5a98edb1e525f2c66f6ff1155772d07a0b0a71769464c`;
- five-policy CSV SHA-256:
  `cfa20785a36e01540af41885dafacead4622a27d4174decbab34a25efeea9782`;
- one-SE review SHA-256:
  `cc3416bd10b4bec5791e484f7e172b087928c012bab5e68eb5078d9cf355437f`;
- executed interpretation notebook SHA-256:
  `e0ccdad735876aa716808aef18642726ec2763dd724e09cc4e9b4bbb40cb09eb`.

The analyzer admitted 81/180 arms. It reset
`numpy.random.default_rng(20260714)` independently for each part and used
10,000 within-fold row-bootstrap replicates for each best-arm one-SE band.

## Frozen Development Selections

These are deterministic products of the predeclared admissibility and
part-specific one-SE rule. They must not be manually replaced after audit
access.

| Part | Selected full config | Architecture / route | RC | Loss | Pooled Pearson | Minimum fold | RMSE | COD R2 |
|---|---|---|---|---|---:|---:|---:|---:|
| Enhancer | `basecfg_6e6b2b979116f3e9cd83a8747792d89a97918ce57e72949f810c309afa068036` | BassetBranched, K562 full transfer | on | unweighted | 0.564722 | 0.508781 | 0.759080 | 0.285096 |
| Promoter | `basecfg_bff24362f7f5a2013947c22336ec779dc986c42124230dae5ff4fcc9904a5d0d` | PromoterBassetVL scratch | off | barcode weighted | 0.478157 | 0.394573 | 0.430790 | 0.215190 |
| Intron | `basecfg_58481a479285bf26af4a9813d37abecc1e6a548795eb3f606fe4d5758ecc4a86` | ResNet1D scratch | off | barcode weighted | 0.690313 | 0.648779 | 0.494948 | 0.476208 |
| 3'UTR | `basecfg_7b1f881265b0fc0aee9e2b601352b93e064e37bee808c65b6b038e6a260e2062` | UTRBassetVL scratch | off | barcode weighted | 0.492697 | 0.391328 | 0.623931 | 0.188653 |
| 5'UTR | `basecfg_9dd728c0df617152551b366c304a265d52be567ad04fb35dbdcecd406235d315` | UTRBassetVL scratch | off | barcode weighted | 0.542062 | 0.525582 | 0.360010 | 0.289171 |

## What Stage 3 Learned

### Paired intervention evidence for the selected policies

| Part | Accepted intervention | Mean five-fold Pearson delta | Positive folds | Pooled Pearson delta | RMSE change | COD R2 decrease | Result |
|---|---|---:|---:|---:|---:|---:|---|
| Enhancer | RC on minus off, unweighted | +0.030561 | 4/5 | +0.029375 | -0.034354 | -0.066174 | RC on passes; weighting is not selected |
| Promoter | weighted minus unweighted, RC off | +0.012066 | 4/5 | +0.009229 | -0.001526 | -0.005569 | weighting passes; RC stays off |
| Intron | weighted minus unweighted, RC off | +0.014876 | 5/5 | +0.016010 | -0.012949 | -0.027766 | weighting passes; RC stays off |
| 3'UTR | weighted minus unweighted, RC off | +0.023818 | 4/5 | +0.010479 | +0.002889 | +0.007497 | weighting passes within its part margins; RC was never enabled |
| 5'UTR | weighted minus unweighted, RC off | +0.017774 | 5/5 | +0.022674 | -0.011268 | -0.045192 | weighting passes; RC stays off |

Across the portfolios, loss gates passed 4/20 Enhancer arms, 8/20 Promoter,
11/20 Intron, 3/10 3'UTR, and 17/20 5'UTR. RC gates passed 10/20 Enhancer
arms, 2/20 Intron, and 0/20 for both Promoter and 5'UTR. The descriptive
factorial interaction is heterogeneous and does not enter selection. In the
selected configs, weighting is helpful at RC off for Promoter, Intron, and
5'UTR; it is not helpful for the selected Enhancer transfer policy.

### Why the 3'UTR numerical winner was not selected

The admissible numerical winner was
`basecfg_0417b66646a3d1e1f7b7f00178f106a004221338769a86ef415d6b583d4a3b05`
with pooled Pearson 0.547563. Its best-arm bootstrap SE was 0.079458, giving a
one-SE threshold of 0.468106 and nine arms in the band. The frozen ordering
then preferred `basecfg_7b1f...` because its minimum fold Pearson was higher:
0.391328 versus 0.371979. This is the intended stability preference, not a
claim that the selected arm has the highest pooled point estimate. The wide
band should be carried forward as uncertainty, not used to reopen HPO.

### Intron inferred-mask sensitivity

For the selected Intron config, barcode weighting improved natural pooled
Pearson from 0.674303 to 0.690313, within-stratum-centered Pearson from
0.435373 to 0.470215, equal-stratum pooled Pearson from 0.670186 to 0.686040,
and minimum-stratum Pearson from 0.093314 to 0.106677. Its five-fold centered
delta was +0.033066 and passed in the frozen gate.

| Inferred development stratum | N | Unweighted Pearson | Weighted Pearson | Weighted calibration slope |
|---|---:|---:|---:|---:|
| `mask1_specific` | 374 | 0.590332 | 0.607160 | 0.983532 |
| `mask2_not_mask1` | 365 | 0.291214 | 0.374129 | 0.960680 |
| `mask3_residual` | 322 | 0.093314 | 0.106677 | 0.423921 |

The conclusion is therefore “consistent improvement with remaining
stratum-limited performance,” not uniform resolution of Intron biology. These
are inferred sequence-mask strata, not verified synthesis sublibraries.

### Raw-scale calibration diagnostics

Primary selection and any primary audit comparison remain on raw predictions.
Selected pooled observed-on-prediction slopes are 0.754 Enhancer, 1.317
Promoter, 0.977 Intron, 1.181 3'UTR, and 0.899 5'UTR. Mean prediction bias is
small for four parts but is +0.152 for 3'UTR. These support reporting
calibration explicitly. They do not authorize fitting a correction on audit
outcomes.

## Candidate Fixed-Epoch Budgets

The older final-refit plan defines the fixed budget as the integer median of
the five selected arm's zero-based `best_epoch` values, plus one training
epoch. The exact development evidence is:

| Part | Five zero-based best epochs | Median | Candidate fixed epochs |
|---|---|---:|---:|
| Enhancer | 5, 5, 4, 3, 187 | 5 | 6 |
| Promoter | 32, 75, 42, 47, 43 | 43 | 44 |
| Intron | 24, 16, 14, 23, 20 | 20 | 21 |
| 3'UTR | 27, 36, 64, 35, 8 | 35 | 36 |
| 5'UTR | 86, 58, 82, 91, 59 | 82 | 83 |

These are reproducible candidates, not yet an audit authorization. The
Enhancer fold-4 value of 187 is a conspicuous learning-dynamics outlier; the
predeclared median rule is robust to it, but the six-epoch final budget and
the exact transfer warm-up/optimizer transition must be stated explicitly in
the final-refit amendment.

## Decisions Required Before The Next Launch

1. **Resolve the stale audit-count conflict.** The older plan specifies four
   RC-by-loss arms per part and three seeds (60 refits). That is inconsistent
   with the now-frozen one-arm-per-part decisions and is literally impossible
   for RC-off-only 3'UTR. The recommended replacement is five selected policies
   times seeds `[1701, 1702, 1703]` = **15 refits**. Auditing all alternatives
   would instead be a 54-run nonselective analysis and would require a new
   explicit amendment.
2. **Freeze the refit/checkpoint contract.** Use every non-audit row, no
   validation loader, no early stopping, the fixed epoch budgets above, the
   selected RC/loss settings, and final-epoch checkpoint retention. Enhancer
   must preserve its K562 head, full-transfer scope, two-epoch warm-up,
   differential learning rates, and optimizer reset.
3. **Separate training from audit scoring.** Train and reconcile all refits
   with `n_test=0`, freeze a checkpoint allowlist with hashes, and only then
   permit a separately confirmed one-time audit evaluator.
4. **Choose the seed claim.** Recommended: report all three seeds and use the
   arithmetic mean of raw seed predictions as the primary audit predictor,
   while naming seed 1701 as the canonical neural checkpoint for downstream
   CRE integration. Never choose a seed from audit performance and never
   average neural weights.
5. **Freeze calibration before audit.** Recommended: raw predictions remain
   primary for Pearson, Spearman, RMSE, and COD R2; calibration slope,
   intercept, and bias are diagnostics. Any optional affine correction must
   be fit and hashed from development OOF predictions only, never audit rows.
6. **Freeze audit claims and retry rules.** Audit failure may limit a part's
   downstream claim but cannot select a different HPO arm. Only exact technical
   retries are allowed after audit visibility.
7. **Retain the exact Intron audit reporting rule.** Use the natural 265-row
   audit once, report natural and inferred-mask views on the same predictions,
   include barcode cutoffs 8/10/12 with `n`, suppress category Pearson for
   `n < 30` or zero variance, and do not create a balanced replacement audit.

## Proposed Stage 4 Learning-Curve Shortlist

If the next stage means the planned downsampling study rather than final
audit, keep the selected RC/loss policy fixed and use only distinct configs
inside that part's one-SE band. A defensible development-only shortlist is:

- Enhancer: `6e6b2b979116`, `e53d6596a16e`, `3f7d963d6d64`;
- Promoter: `bff24362f7f5`, `9b9293193ecd`, `9821907e1ab3`;
- Intron: `58481a479285`, `5b5d2d82cef9`, `6079cd38f32d`;
- 3'UTR: `7b1f881265b0`, `0417b66646a3`, `1becdea28bb6`;
- 5'UTR: `9dd728c0df61`, `e3b85c86fe40` only.

The two 5'UTR ResNet controls under the selected weighted/RC-off policy have
pooled Pearson 0.483266 and 0.478342, both below the 0.521874 one-SE threshold.
There is therefore no development evidence to force an architecture-diverse
third 5'UTR config. With three configs for the other four parts and two for
5'UTR, the existing five-fold, seven-size design would contain 490 rather than
525 runs.

The existing size grid `[100, 250, 500, 1000, 2000, 3500, full]` is feasible
under its explicit `train_min_barcodes=1` contract. Minimum eligible fold
training-pool sizes are 4,341 Enhancer, 7,198 Promoter, 7,370 Intron, 6,490
3'UTR, and 7,684 5'UTR. The much smaller HQ-only counts are the validation
stratum, not the Stage 4 sampling pool. Running Stage 4 before the final audit
would strengthen audit isolation but changes the older launch order and needs
a dated amendment.

## Readiness Conclusion

Stage 3 development selection is complete. The project is ready to freeze a
post-Stage-3 refit/evaluation amendment and implement a dry-run-only 15-refit
manifest. It is **not** yet ready to instantiate the audit loader. For later
combinatorial CRE integration, one canonical refit checkpoint per part and a
five-part checkpoint-bundle manifest must be frozen separately from any
three-seed scoring ensemble.
