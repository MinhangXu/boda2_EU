# Lib1 Dedup Phase 1 Stage 1 Implementation Checks (July 2026)

> **Role:** Technical Stage 1 audit appendix. For the conceptual Stage 1 to
> Stage 2 explanation, start with
> `lib1_dedup_stage1_to_stage2_reader_guide_july2026.md`. Do not use this file
> as the primary Stage 2 decision document.

This is the implementation and verification companion to
`lib1_dedup_phase1_hpo_rerun_plan_july2026.md`. The plan defines the scientific
contract; this file records how Stage 1 implements that contract and what was
checked before the overnight replay.

## Stage 1 Contract

Every broad-replay row uses the canonical exact-dedup log2 target, development
fold 0, model seed 1701, reverse complements off, unweighted mean MSE, and
FP32. Per-epoch train and validation histories are logged. Validation
predictions and compact provenance are retained; the frozen audit split is
unavailable to training, test evaluation is disabled, and model/checkpoint
retention is `none`.

The replay is exact-config replay, not a new factorial sweep. A
`base_config_id` hashes the resolved model, optimizer, active scheduler, batch,
trainer, checkpoint-monitor, and early-stopping settings from a completed June
run. Data paths, historical seed/split, old target, RC, and other campaign
fields are deliberately replaced by the fixed Stage 1 contract.

## Canonical Data and Frozen Splits

`PREPARE_DATASET=1` performs these deterministic preparation steps before
manifest resolution:

1. Build the five exact-dedup learn-ready TSVs and their paired pre-dedup
   calibration TSVs from the declared external sources.
2. Recompute the exact target as `log2(sum RNA barcode counts / sum DNA barcode
   counts)` with no pseudocount, apply the declared part-specific length policy,
   assign stable construct IDs, and verify source/output hashes and row counts.
3. Generate the frozen audit assignment and five disjoint development folds,
   plus companion pre-dedup split manifests using the same construct IDs.
4. Resolve only completed historical configurations into the immutable replay
   manifest. Re-running preparation is intended to reproduce the same files and
   hashes; it does not augment data or start training.

| Part | Dedup rows | High-barcode rows | Fold-0 train | Fold-0 validation | Frozen audit | Split SHA-256 prefix |
|---|---:|---:|---:|---:|---:|---|
| Enhancer | 4,787 | 1,229 | 4,341 | 196 | 250 | `ebbe07daab0a` |
| Promoter | 7,893 | 1,931 | 7,198 | 309 | 386 | `feb2a26d7f57` |
| Intron | 7,848 | 1,326 | 7,370 | 213 | 265 | `4cdda8008e4a` |
| 3'UTR | 6,845 | 775 | 6,490 | 105 | 250 | `c7a5799b8e0c` |
| 5'UTR | 8,331 | 1,797 | 7,684 | 288 | 359 | `038b7416f254` |

## Resolved Replay Inventory

The generated manifest SHA-256 is
`3ddfe712a45b9e091c7f773afae9a0786b8b250754e9e1bf8c3619b7811394c1`.
It contains 885 dedup exact-replay rows and 25 separately labeled pre-dedup
calibration rows.

| Part / architecture | Completed broad HPO | Completed June outer bases | Stage 1 exact rows |
|---|---:|---:|---:|
| Enhancer / ResNet1D | 128 | 0 | 128 |
| Promoter / PromoterBassetVL | 128 | 30 | 158 |
| Intron / ResNet1D | 126 | 30 | 156 |
| 3'UTR / ResNet1D | 127 | 30 | 157 |
| 5'UTR / ResNet1D | 128 | 30 | 158 |
| 5'UTR / UTRBassetVL | 128 | 0 | 128 |
| **Total** | **765** | **120** | **885** |

The June outer rows include already-completed `local_variant` and
`narrow_prior` configurations. Those labels do not disqualify a configuration;
completion does. Generated candidates that never completed are excluded.
There was no trainer-inclusive `base_config_id` overlap between the 765 broad
rows and 120 outer bases, so all 885 remain distinct.

The tested hyperparameters are the exact joint combinations from those runs,
not all cross-products of the values below:

| Lane | Main resolved variation |
|---|---|
| ResNet1D lanes | Batch 64/128/256; Adam or AdamW; no scheduler or cosine warm restarts; continuous learning rate, weight decay, and Adam betas; stem/head widths and kernels; block kernels; dropout; batch norm |
| PromoterBassetVL | Batch 64/128/256; AdamW; no scheduler; continuous learning rate/weight decay; convolution widths/kernels; 1 or 2 linear layers; linear width/dropout; batch norm |
| UTRBassetVL | Batch 64/128/256; Adam or AdamW; no scheduler or cosine warm restarts; continuous learning rate/weight decay and Adam betas; convolution widths/kernels; 1 or 2 linear layers; linear width/dropout; batch norm |

The full per-row values are preserved in `base_identity` and
`resolved_historical_config` in the JSONL manifest; this is the authoritative
hyperparameter record.

## Calibration Switch

`INCLUDE_CALIBRATION=0` (the default) selects only the 885 dedup replay rows.
`INCLUDE_CALIBRATION=1` also selects 25 pre-dedup calibration mates: five
predeclared validation-ranked configs per part. Each mate is retrained on the
paired pre-dedup product with the same current fold-0 IDs, seed 1701, RC off,
unweighted MSE, and trainer settings. Its dedup mate is already one of the 885,
so enabling the switch adds 25 runs, not 50.

This is the plan's Direct Data-Product Calibration. It estimates the data
product change under a controlled paired retraining comparison. It is not a
mere reevaluation of an old model, and it is not a pure comparison against the
historical W&B score because those old runs used different seeds/splits. The
pre-dedup arm is diagnostic only and cannot enter new-model selection.

## W&B and History Checks

The launcher forces entity
`minhangxu1998-baylor-college-of-medicine`, `WANDB_MODE=online`, the dedicated
`*__dedup_exact_v1__*__exact_replay` projects, and campaign group
`lib1_dedup_phase1_rerun_july2026__stage1_exact_replay`. Here `online` only
means immediate W&B cloud synchronization, not online/continual learning. The
local cache root is `src/learn/wandb/`.

The two-part pilot exposed that the clean best-checkpoint train evaluation was
being appended as though it were one more chronological epoch. That produced a
misleading last jump in W&B charts. The evaluation remains in the run summary
under canonical `train_*` and explicit `best_checkpoint_train_*` keys, but is
no longer appended to history. Per-epoch charts now contain only epoch metrics.

## Pre-launch Safety and Verification

- Dataset and split paths/hashes are checked before each command.
- The manifest/command contract is checked, and exact row completion markers
  are bound to immutable row fingerprints.
- The launcher lock prevents two orchestrators from owning one queue.
- The audit/test split is not exposed and `evaluate_test_after_fit=false`.
- No W&B sweep is created; this is a fixed local manifest queue.
- GPU workers, failure markers, monitoring TSV, disk thresholds, and resumable
  skip-completed behavior are implemented by the orchestrator.
- Unit coverage checks deterministic data/splits, manifest resolution and
  determinism, W&B identity/history behavior, test exclusion, checkpoint
  pruning, and process-safe registry writes.
- The online pilot verifier checks finished state, expected entity/project,
  train/validation and learning-rate histories, absence of test metrics and
  model artifacts, validation predictions, and provenance.

On 2026-07-10, 13 focused data/split and training-contract tests passed. The
cloud verifier also passed for manifest rows 1, 600, and 601 (the attached
Enhancer and 5'UTR pilot runs), including history, prediction, and provenance
checks.

## Commands

Dry preparation/command audit:

```bash
PREPARE_DATASET=1 DRY_RUN=1 PILOT=1 GPU_LIST="0" MAX_PARALLEL=1 \
  bash src/learn/launch/lib1_dedup_phase1_exact_replay_orchestrator.sh
```

Dedup-only overnight queue (completed pilot rows are skipped by fingerprint):

```bash
WANDB_ENTITY="minhangxu1998-baylor-college-of-medicine" \
BODA_WANDB_ENTITY="minhangxu1998-baylor-college-of-medicine" \
GPU_LIST="0 1 2 3" MAX_PARALLEL=4 PREPARE_DATASET=1 INCLUDE_CALIBRATION=0 \
  bash src/learn/launch/lib1_dedup_phase1_exact_replay_orchestrator.sh
```

Implementation entry points are
`src/learn/prepare_lib1_dedup_exact_datasets.py`,
`src/learn/generate_lib1_dedup_split_manifests.py`,
`src/learn/generate_lib1_dedup_exact_replay_manifest.py`,
`src/learn/launch/lib1_dedup_phase1_exact_replay_orchestrator.sh`, and
`src/learn/verify_lib1_dedup_stage1_pilot.py`.
