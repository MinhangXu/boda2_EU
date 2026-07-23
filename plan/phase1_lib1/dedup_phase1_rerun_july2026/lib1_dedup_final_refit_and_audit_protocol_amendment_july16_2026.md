# Lib1 Dedup Final-Refit And Audit Protocol Amendment

**Amendment date:** 2026-07-16
**Status:** frozen post-Stage-3 protocol; implementation and the gated final
audit are authorized
**Authorization record:** on 2026-07-16, after reviewing the completed Stage 3
development analysis, the collaborator explicitly requested final audit-test
evaluation of the one selected single-part model policy for each CRE part.

## Decision And Scope

This amendment replaces the stale 60-refit audit design in
`lib1_dedup_phase1_hpo_rerun_plan_july2026.md`. Stage 3 selected exactly one
development policy per CRE part, and 3'UTR has no RC-on arm. The final campaign
therefore contains exactly:

```text
5 frozen part policies x 3 predeclared model seeds = 15 final refits
```

It has two strictly separated phases:

1. train and reconcile all 15 final refits without constructing an audit
   loader, then freeze a SHA-256-bound checkpoint allowlist;
2. in a separately confirmed invocation, open each frozen audit partition once
   and score only the 15 allowlisted checkpoints.

No other config, RC arm, loss arm, model seed, epoch budget, checkpoint, or
calibration rule may enter the scorer. Audit results cannot return a part to
HPO or select an alternative model. Stage 4 downsampling remains a later
development-only study and is not a prerequisite for this final audit.

## Frozen Part Policies

The complete `base_config_id`, rather than its short display form, is the
config identity.

| Part | Full selected config | Architecture / route | RC | Loss | Fixed completed epochs |
|---|---|---|---|---|---:|
| Enhancer | `basecfg_6e6b2b979116f3e9cd83a8747792d89a97918ce57e72949f810c309afa068036` | `BassetBranched`; Malinois pretrained; K562 source head; `full` transfer | on | unweighted MSE | 6 |
| Promoter | `basecfg_bff24362f7f5a2013947c22336ec779dc986c42124230dae5ff4fcc9904a5d0d` | `PromoterBassetVL`; scratch | off | clipped-log barcode-weighted MSE | 44 |
| Intron | `basecfg_58481a479285bf26af4a9813d37abecc1e6a548795eb3f606fe4d5758ecc4a86` | `ResNet1DRegressor`; scratch | off | clipped-log barcode-weighted MSE | 21 |
| 3'UTR | `basecfg_7b1f881265b0fc0aee9e2b601352b93e064e37bee808c65b6b038e6a260e2062` | `UTR_BassetVL`; scratch | off | clipped-log barcode-weighted MSE | 36 |
| 5'UTR | `basecfg_9dd728c0df617152551b366c304a265d52be567ad04fb35dbdcecd406235d315` | `UTR_BassetVL`; scratch | off | clipped-log barcode-weighted MSE | 83 |

For every weighted policy, retain the Stage 3 training weights exactly:

```text
w_i = clip(log1p(n_barcodes_i) / log1p(8), 0.1, 1.0)
loss = sum_i(w_i * per-example MSE_i) / sum_i(w_i)
```

Enhancer must retain the selected K562 source-head slice, Malinois MPRA-flank
input contract, two completed warm-up epochs, transition to full unfreezing,
differential head/backbone learning rates, and optimizer-state reset at the
scope transition. RC is training augmentation only; every validation, audit,
and exported prediction uses the original sequence orientation.

## Fixed-Epoch Basis

Stage 3 stored `best_epoch` as a zero-based epoch index. The frozen final budget
is the integer median of the five selected-arm indices plus one completed
training epoch:

| Part | Five zero-based Stage 3 best epochs | Integer median | Fixed completed epochs |
|---|---|---:|---:|
| Enhancer | 5, 5, 4, 3, 187 | 5 | 6 |
| Promoter | 32, 75, 42, 47, 43 | 43 | 44 |
| Intron | 24, 16, 14, 23, 20 | 20 | 21 |
| 3'UTR | 27, 36, 64, 35, 8 | 35 | 36 |
| 5'UTR | 86, 58, 82, 91, 59 | 82 | 83 |

The Enhancer fold-4 value of 187 is retained as provenance; the predeclared
median makes the final budget robust to that outlier. The final refit must not
reinterpret a zero-based index as an epoch count.

## Frozen Final-Refit Contract

Model seeds are exactly `[1701, 1702, 1703]`. For a given part, all three seeds
must use the identical resolved configuration, ordered non-audit training IDs,
normalization rows, RC/loss policy, and fixed epoch count. The only intended
difference is model/training randomness induced by the seed.

Each refit uses every row assigned `train_only` or `development` by the frozen
part split and uses `train_min_barcodes=1`. Audit rows are excluded by a
positive non-audit training allowlist. The refit process must not enumerate,
materialize, count, or load the complementary audit rows. Dataset, split,
allowlist, and final training-row SHA-256 values are mandatory provenance.
Target normalization is fit only on the complete non-audit training rows and
is stored with the checkpoint so predictions can be returned to raw
`log2_RNA_DNA` scale.

Final refits have:

- no validation dataset or validation loader;
- no test/audit dataset or test/audit loader;
- no early stopping and no metric-based checkpoint choice;
- exactly the part-specific completed-epoch budget above;
- training metrics only during fitting;
- `evaluate_test_after_fit=false`, no prediction export from an audit split,
  and `n_test=0` at completion;
- the final epoch state, not a development-best checkpoint, as the retained
  model;
- intentional retention of the portable model archive and final checkpoint.

The final-refit W&B project for each part is:

```text
<part>__bashor_in_house__dedup_exact_v1__audit_refit
```

The entity remains
`minhangxu1998-baylor-college-of-medicine`. The live entity and project must be
validated before fitting. Required run identity includes part, full selected
config, seed, fixed epoch count, RC/loss policy, non-audit row hash, dataset and
split hashes, graph/model class, training regime, and source Stage 2/3 run IDs.

## Refit Reconciliation And Checkpoint Allowlist Gate

Audit scoring remains disabled until all of the following pass:

1. exactly 15 completed registry records exist, one for each frozen
   `(part, model_seed)` cell, with no second completed attempt;
2. every record has `n_test=0`, blank audit/test metrics, no audit prediction
   path, the expected final epoch, and the exact frozen policy fields;
3. the three seeds within a part have identical dataset, split, non-audit
   allowlist, final training-row, normalization-row, and input-policy hashes;
4. every retained portable archive and final checkpoint exists and can be
   loaded without constructing a data loader;
5. an immutable 15-row checkpoint allowlist is written and hashed.

Each allowlist row must contain at least the part, full `base_config_id`, RC and
loss modes, architecture, graph class, transfer fields where applicable,
model seed, fixed completed epochs, W&B run ID/URL, source run IDs, dataset and
split hashes, non-audit allowlist/training/normalization hashes, local archive
path and SHA-256, final checkpoint path and SHA-256, code commit, and completion
status. The scorer must require the expected allowlist SHA-256 and reject an
extra, missing, duplicate, changed, or nonallowlisted checkpoint.

## One-Time Audit Scoring Contract

Audit access must be a separate program and process from refit training. It
requires an explicit confirmation flag plus the frozen protocol and checkpoint
allowlist hashes. It may not train, update, calibrate, early-stop, select, or
write model weights. It loads the existing frozen audit assignment for each
part once, verifies the frozen dataset/split hashes, and obtains one original-
orientation raw prediction per allowlisted seed and audit construct.

For each part, convert every seed's output to raw `log2_RNA_DNA` using only that
refit's stored non-audit normalization parameters. Align the three seed tables
by frozen construct ID and assert identical targets and barcode metadata. The
primary audit predictor is the construct-wise arithmetic mean of the three
raw prediction values:

```text
prediction_primary_i = mean(prediction_raw_i_seed1701,
                            prediction_raw_i_seed1702,
                            prediction_raw_i_seed1703)
```

Do not average neural weights. Do not choose a seed from audit performance.
Report every individual seed as a secondary robustness result. Seed 1701 is
the canonical neural checkpoint for later single-checkpoint CRE integration;
this operational designation is fixed independently of audit performance and
does not replace the three-seed mean as the primary audit predictor.

The scorer writes immutable per-seed and primary-ensemble prediction tables,
metric tables, provenance, an artifact index, and SHA-256 values. The analysis
notebook consumes those frozen scorer products; it must not contain or invoke
audit-loader logic.

## Frozen Metrics And Calibration Reporting

For the three-seed mean and separately for seeds 1701, 1702, and 1703, report
the audit `n` and raw-scale:

- Pearson correlation;
- Spearman correlation;
- RMSE;
- coefficient-of-determination R2,
  `1 - sum((observed - predicted)^2) / sum((observed - mean(observed))^2)`;
- mean prediction bias, `mean(predicted - observed)`;
- calibration slope and intercept from
  `observed = intercept + slope * predicted`.

Pearson is the primary association metric. Spearman, RMSE, and COD R2 are
secondary performance metrics. Bias and the observed-on-prediction slope and
intercept are calibration diagnostics only. Raw predictions remain the sole
primary predictor. No affine correction may be fit on audit targets, and no
audit-fitted correction may be used for any reported primary or secondary
performance result. Any later OOF-only calibration analysis must remain
clearly separate and cannot replace this frozen audit report.

## Frozen Intron Audit Reporting

Use the existing natural 265-row Intron audit and exactly the same three-seed
predictions for every view. Do not construct a balanced replacement subset.
The three deterministic inferred-mask sensitivity categories retain their
frozen precedence and names:

1. `mask1_specific`;
2. `mask2_not_mask1`;
3. `mask3_residual`.

These are inferred sequence-mask categories, not verified synthesis-pool or
sublibrary membership. For the primary ensemble and each seed, report:

- natural-mixture pooled metrics;
- pooled within-inferred-stratum-centered Pearson, obtained by subtracting the
  corresponding audit-stratum mean from observed and predicted values before
  pooling;
- macro mean and minimum of the three stratum Pearson values;
- per-stratum `n`, observed and prediction means, bias, MAE, RMSE, Pearson,
  Spearman, COD R2, and raw-scale calibration slope/intercept;
- the optional equal-stratum sensitivity estimate with its effective sample
  size, clearly labeled nonprimary;
- natural audit summaries at barcode cutoffs `n_barcodes >= 8`, `>= 10`, and
  `>= 12`.

Every category and cutoff row must show `n`. Suppress Pearson when `n < 30` or
when either the observed or predicted values have zero variance, and record
the reason rather than substituting a number. The natural pooled result is the
deployment-mixture estimate. The centered and per-stratum results diagnose how
much apparent pooled performance depends on inferred-family mean offsets; they
do not redefine the audit population.

## Claims, Audit Finality, And Retry Rules

This is a confirmatory generalization report for five already-frozen
development policies. Audit performance may strengthen or limit the claim for
a part, but it cannot:

- change its config, architecture, RC mode, loss mode, epoch count, seed, or
  calibration rule;
- select the best-performing audit seed;
- reopen Stage 1, Stage 2, targeted 3'UTR HPO, or Stage 3;
- add an architecture-diverse finalist after audit visibility;
- alter the development-only Stage 4 design to chase an audit result.

Only exact technical retries are permitted. A retry requires a documented
hardware, process, serialization, or integrity failure and must reuse the same
command, code, protocol hash, allowlist hash, checkpoint hashes, dataset/split
hashes, ordering, and output schema. Failed and partial attempts remain in the
provenance log. After any audit value is visible, a retry cannot change a model
or analysis decision and cannot be used to choose among attempts. Scientific
disappointment, seed variance, or a low metric is not a retry condition.

## Audit-Isolation Incident Record

After Stage 3 selection and all five policies had already been frozen, a broad
read-only repository `rg` search on 2026-07-16 inadvertently traversed a
checked-in minified split-manifest file and surfaced assignment content in tool
output. No data module or audit loader was instantiated; no audit targets,
model predictions, performance metrics, or inferred audit-stratum counts were
accessed or computed; and no development selection or protocol decision was
changed in response. No further audit-file inspection occurred during this
amendment. This incident is recorded before the authorized scorer is built or
run and must be retained with the final audit provenance.

## Authorization Boundary

This amendment authorizes implementation, unit/static validation, a dry-run
15-refit manifest, final-refit execution, checkpoint reconciliation, and the
separately confirmed one-time audit scoring described above. It does not
authorize scoring alternative arms, exploratory audit slicing beyond the
predeclared reports, audit-based calibration, or any audit-driven retraining.
