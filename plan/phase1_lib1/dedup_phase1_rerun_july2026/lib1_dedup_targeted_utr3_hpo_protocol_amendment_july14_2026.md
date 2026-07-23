# Lib1 Dedup Targeted 3'UTR HPO Protocol Amendment

**Frozen:** 2026-07-14
**Campaign:** `lib1_dedup_phase1_rerun_july2026`
**Stage:** `targeted_utr3_hpo`
**Status:** dry-run design frozen; no training cell or audit loader launched

This amendment is the binding design record for the bounded 3'UTR
UTRBassetVL search approved after the Stage 2 analysis. It supplements the
full campaign plan and the pre-Stage-2 amendment. It does not authorize the
full campaign, Stage 3 weighted-loss runs, or frozen-audit access.

## Frozen Scientific Scope

The search asks whether optimizer and regularization settings near and above
the historical learning-rate ceiling improve the fixed leading UTRBassetVL
architecture. It does not search a new architecture, target, split, seed,
loss, scheduler, epoch budget, or data policy.

The architecture anchor is Stage 2 UTRBassetVL base configuration
`basecfg_86969bcf79247695d2c27ce1466d4eab2373e5e1f3645da99f24ebf4c59c0fbe`
from historical run `utc3cqzn`. Its convolutional and linear architecture,
batch size 64, AdamW beta values, epsilon, lack of scheduler, 220 maximum
epochs, 25 minimum epochs, patience 45, FP32 precision, and model seed 1701
remain fixed. Only the three dimensions below change.

## Exact 24-Configuration Search Space

The manifest is the complete Cartesian product:

| Dimension | Exact values |
|---|---|
| AdamW learning rate | `0.001`, `0.002`, `0.004`, `0.006` |
| AdamW weight decay | `0.0001`, `0.0007`, `0.003` |
| UTRBassetVL shared dropout | `0.35`, `0.50` |

This gives `4 x 3 x 2 = 24` new base configurations. The learning-rate ceiling
is expanded from the historical `0.002` boundary to `0.006`. The values are a
fixed grid, not W&B Bayesian proposals. No value may be inserted, removed, or
adapted after observing a partial result without a new dated amendment and a
new manifest tag.

The canonical config order is learning rate ascending, then weight decay
ascending, then dropout ascending. Each config receives a content-derived
`base_config_id` from its complete base identity, not from its grid position.

## Screening, Promotion, And Cell Accounting

There is no partial-fold screening or promotion stage. All 24 configurations
receive the complete five-fold, paired-RC development evaluation:

```text
24 new configs x 5 development folds x 2 RC modes = 240 training cells
```

Every base configuration therefore produces two complete OOF arms and one
complete paired-RC configuration. The dry-run manifest contains exactly 240
new rows, 120 RC pairs, 48 complete new OOF arms, and no reuse rows. The 200
existing Stage 2 3'UTR cells are comparison evidence, not part of the 240-cell
launch accounting:

```text
10 historical UTRBassetVL configs x 5 folds x 2 RC = 100 existing cells
10 Stage 2 ResNet1D configs       x 5 folds x 2 RC = 100 existing cells
```

The no-screening decision is deliberate. With only 525 high-barcode
development constructs and material fold variability in the leading Stage 2
arm, a one- or two-fold promotion gate would be a noisy estimand and would
deny non-promoted candidates a primary pooled OOF score.

## Frozen Data And Training Contract

- Dataset: the existing deduplicated modal-100 3'UTR learn-ready product,
  SHA256 `1bd0f70655cbbc3f47be40b2bb50cc641430a6741145ae85cb27506d512f7cc0`.
- Split manifest: `lib1_utr3_dedup_exact_v1_split_seed20260709`, SHA256
  `c7a5799b8e0c5b92a0041822a6bc5d0d9513a39d97f28061a5d46183ef998e1a`.
- Target: raw `log2_RNA_DNA = log2(RNA_bc_counts_sum / DNA_bc_counts_sum)`;
  training-fold normalization remains enabled.
- Training rows: non-audit rows with `n_barcodes >= 1` under the canonical
  fold assignment.
- Validation estimand: the same 525 development high-barcode constructs,
  each predicted once OOF in its original orientation.
- Loss: unweighted MSE only.
- RC: deterministic training-only augmentation, evaluated as paired off/on
  arms; validation remains original orientation.
- Model seed: 1701.
- Checkpoint/early stopping monitor: fold-validation Pearson, mode `max`.
- Per-epoch evaluation: `train` and `val` only; prediction export: `val` only.

## OOF Selection And Comparison Rule

An arm is eligible only when all five fold cells complete, each validation
construct appears exactly once in the pooled predictions, construct IDs match
the frozen development assignments, and no audit ID appears. Failed or
incomplete arms are reported but are not ranked.

The primary score is Pearson correlation between raw `log2_RNA_DNA` and
`prediction_raw` pooled over all 525 OOF rows. Report beside it pooled
Spearman, RMSE, COD R2, raw prediction bias, each fold's Pearson, mean/SD/min
fold Pearson, best epochs, parameter count, and training time.

For the one-standard-error set, estimate the standard error of the numerically
best eligible arm's pooled Pearson with 10,000 fold-stratified construct
bootstrap resamples using RNG seed `20260714`. A candidate is in the set when
its pooled Pearson is at least `best_pearson - SE(best_pearson)`. Select the
preferred targeted-HPO arm from this set by the following deterministic order:

1. highest minimum fold Pearson;
2. lowest pooled RMSE;
3. highest pooled COD R2;
4. lower learning rate, then lower weight decay, then higher dropout;
5. lexicographically smallest full `base_config_id`.

The same rule is applied to all 48 new `(base_config_id, rc_mode)` arms. For
Stage 3 deliberation, append the existing 20 UTRBassetVL and 20 ResNet1D arms
using their original Stage 2 identities and provenance, recompute the same
metrics from the frozen Stage 2 OOF products, and label the portfolio source
as `targeted_20260714`, `stage2_utrbasset_challenger`, or
`stage2_resnet_core`. Never overwrite or relabel the existing identities.

This rule selects a preferred arm and supplies a ranked evidence table; it
does not by itself freeze the five 3'UTR Stage 3 base configurations. The
Stage 3 five must be named in a later immutable full-ID selection record.

## W&B And Local Organization

- Entity: `minhangxu1998-baylor-college-of-medicine`
- Project: `utr3__bashor_in_house__dedup_exact_v1__targeted_hpo_development`
- Group: `lib1_dedup_phase1_rerun_july2026__targeted_utr3_hpo__full_oof_rc`
- Job type: `targeted_utr3_hpo_cell`
- Run name:
  `lib1_dedup_utr3_targeted_hpo_july2026__cfgNN__<base-sha16>__foldF__rc_<off|on>`
- Local root:
  `src/learn/outputs/hpo_runs/lib1_dedup_utr3_targeted_hpo_july2026/<base_config_id>/fold_F/rc_<off|on>`
- Dry-run manifests and reports:
  `src/learn/outputs/hpo_manifests/lib1_dedup_utr3_targeted_hpo_july2026__*`

W&B is used only for fixed-grid run logging; there is no W&B sweep controller
and no adaptive proposal process.

## Audit-Isolation And Launch Gates

Generation and validation are metadata/file operations and must not import or
construct a DataModule. The independent verifier must fail closed unless all
of the following are true:

1. every row uses the frozen dataset and split hashes;
2. `evaluate_test_after_fit=false`;
3. `epoch_eval_splits` is exactly `train val`;
4. `prediction_splits` is exactly `val`;
5. artifact retention is `none` and no published checkpoint is requested;
6. the command contains no `test` prediction/evaluation split and no audit
   identifier or audit row list;
7. every config has exactly five folds and both RC states;
8. W&B entity/project/group/job-type and all run names are exact and unique;
9. the dry-run report records `audit_loader_instantiated=false`,
   `audit_ids_materialized=false`, and `commands_executed=0`.

The canonical DataModule option named `test_min_barcodes=8` is retained as
part of the established high-barcode data contract; it does not authorize or
instantiate the test loader. The launch path remains prohibited from calling
`trainer.test`, requesting test predictions, or evaluating the audit split.

Only a one-row pilot command is handed off here. The full 240-row campaign
requires a separate user launch decision after reviewing the validated dry
run. Stage 3 manifest generation remains blocked until the targeted search is
complete and analyzed and the remaining cross-part decisions below are
frozen.

## Remaining Pre-Stage-3 Work Outside 3'UTR

For the other CRE parts, Stage 2 training itself is complete. Before Stage 3
implementation:

- freeze the numerical threshold for "no material" RMSE or COD-R2 degradation
  used by both the RC and weighted-loss gates;
- apply and record the one-standard-error/simple-model rule to the provisional
  selections;
- decide whether 5'UTR's fifth slot remains UTRBassetVL `bee0f2b5` or becomes
  the near-tied architecture-diverse ResNet1D `ffd49926`;
- decide whether Enhancer's five weighted-loss candidates are the pure top
  five transfer policies or retain one explicitly labeled scratch anchor;
- implement and test weights-required barcode-weighted MSE in
  `CNNBassetBranchedScopedTransfer`, preserving source-head loading,
  warm-up/unfreezing, optimizer groups, RC/fold pairing, and rejecting a
  two-tensor batch in a run labeled weighted;
- verify the scratch and UTRBassetVL weighted paths retain the same explicit
  weighted arithmetic and missing-weight guards;
- record parameter-count/model-size evidence consistently for scratch and
  transfer candidates; and
- write the immutable full-ID Stage 3 selection and launch manifest only after
  all decisions above are signed off.

The audit remains unavailable throughout these steps.
