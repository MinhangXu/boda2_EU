# Lib1 Deduplicated Phase 1 HPO Rerun Plan

> **Navigation:** Start with
> `lib1_dedup_stage1_to_stage2_reader_guide_july2026.md`. This file is the
> formal, comprehensive scientific contract and is intentionally detailed.
> Stage 1 implementation evidence is in
> `lib1_dedup_phase1_stage1_implementation_checks_july2026.md`; pre-Stage-2
> changes are in `lib1_dedup_pre_stage2_protocol_amendment_july2026.md`. The
> dedicated Intron-strata diagnostic is
> `tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/02_intron_inferred_mask_strata_analysis.ipynb`.

Generated: 2026-07-09

Status: Stages 1 and 2 are complete, and Stage 2 analysis is underway. This
remains the formal full-campaign contract for the five-part deduplicated Lib1
rerun.

Execution update (2026-07-13): Stage 1 completed 885 exact-dedup replay rows
plus 25 pre-dedup diagnostic mates. The linked amendment is frozen at
Enhancer `N=2`, 3'UTR UTRBassetVL `K=10`, and inferred Intron masks as
sensitivity labels. All 660 Stage 2 analysis cells are complete: 50 verified
Stage 1 reuse cells plus 610 new launches, comprising 132 complete five-fold
OOF arms and 66 complete paired-RC configs. Stage 2 analysis is underway;
weighted-loss and frozen-audit evaluation have not launched.

Companion context:

- `plan/phase1_lib1/lib1_sweep_workflow_relationships_june2026.md`
- `plan/phase1_lib1/lib1_outer_seed_prior_hpo_plan_june2026.md`
- `plan/phase1_lib1/learn/lib1_barcode_weighted_loss_plan_june2026.md`
- `plan/phase1_lib1/lib1_barcode_threshold_downsampling_plan_june2026.md`
- `plan/repo_hygiene/barcode_level_dedup_update_july6_2026.md`
- `tutorials/lib1_tasks/barcode_level/variant_level_redo_jul7_2026/lib1_dedup_followup2_analysis_jul7_2026.ipynb`



## Decision Summary

The exact-barcode deduplication effect appears real but modest enough that it
does not, by itself, invalidate every architecture conclusion from June. It
does invalidate the old target values and makes the old model-selection chain
non-canonical. This is a good opportunity to rerun the parts of Phase 1 that
answer the most useful forward-looking questions.

Recommended rerun:

```text
canonical five-part dedup data and frozen split manifests
  -> unweighted broad exact-config replay on development fold 0
  -> promote exact, already-trained base configurations only
  -> five-fold paired RC screen with unweighted MSE
  -> paired weighted-loss add-on for the top RC-screened configs
  -> validation-selected final audit-test evaluation
  -> top-three nested downsampling learning curves
```

The main design changes from June are:

1. Use deduplicated `log2(RNA/DNA)` for all five single-part targets.
2. Keep the established part-specific length branches for the main rerun.
3. Do not ask W&B Bayesian search to invent new combinations. Replay exact
   model/optimizer/trainer hyperparameters that were already trained before
   deduplication.
4. Define a `base_config_id` that excludes data generation, split/fold, model
   seed, RC, and loss mode. Those are controlled experimental factors rather
   than model hyperparameters.
5. Keep previously trained June `local_variant` and `narrow_prior` configs
   eligible. Do not generate additional local variants or narrow-prior samples
   in this rerun.
6. Run broad replay with unweighted MSE and RC off, then assess RC on/off as a
   paired factor across narrowed configs and identical development folds.
7. Treat loss mode as another paired outer-loop factor on the top RC-screened
   exact configs.
8. Select configs, RC policy, and loss policy from cross-fold validation only.
9. Keep a frozen high-barcode audit test set out of replay and outer selection.
10. Redo sample-size learning curves, but do not automatically redo the exact
    barcode-bin or barcode-threshold factorial experiments.

The original draft's phrase "reuse the June hyperparameter supports" would
have allowed a new Bayesian sweep to propose unseen combinations. That is not
the updated recommendation. The updated broad stage is a fixed replay
manifest, not a new Bayesian sweep.



## What The Quick Dedup Rerun Established

The July focused rerun used 60 matched source hyperparameter sets across
Promoter/5'UTR and ResNet1D/BassetVL, with unweighted and barcode-weighted
dedup arms.

Across all 60 matched sets:


| Contrast                             | Mean test Pearson delta | Positive pairs |
| ------------------------------------ | ----------------------- | -------------- |
| Dedup + MSE minus pre-dedup          | +0.0389                 | 55/60          |
| Dedup + weighted MSE minus pre-dedup | +0.0502                 | 56/60          |
| Weighted minus unweighted on dedup   | +0.0113                 | 41/60          |


The strongest dedup effect was Promoter. The weighted-minus-unweighted effect
was smaller and architecture dependent. Validation changes were close to zero
overall, while test changes favored the deduplicated products.

This supports the following interpretation:

- The deduplicated data products produce more favorable heldout Pearson for
matched configurations.
- The sequence membership and split membership were unchanged in the focused
comparison, but the aggregate target values changed. Therefore this is a
data-product effect, not clean evidence that changing training labels alone
improved a model evaluated against an unchanged truth set.
- A single 250-row validation/test split is too small to decide whether a
config or weighted-loss policy generalizes robustly.
- The result is large enough to justify rebuilding the model-selection chain,
but not large enough to justify discarding all useful architecture priors.



## Scientific Questions

The rerun should answer three primary questions.

### Question 1: Robust Model And Loss Selection

For each of Enhancer, Promoter, 5'UTR, Intron, and 3'UTR:

```text
Which already-observed broad-HPO configuration is most robust across
high-barcode development folds, and does barcode-weighted MSE improve that
configuration relative to ordinary MSE?
```

### Question 2: Value Of RC Training Augmentation

```text
For the same base config, fold, model seed, and unweighted loss, does adding
the reverse-complement copy of every training sequence improve original-
orientation heldout performance, and is that effect stable by CRE part?
```

RC should be reported as a training augmentation effect. It is not test-time
augmentation in the current implementation.


### Question 3: Marginal Value Of More Variants

After robust configs are selected:

```text
How much heldout performance is gained as the number of deduplicated training
variants increases, and where does the learning curve begin to saturate?
```

## RC Implementation And Interpretation

The current implementation in `boda/data/bashor_datamodule.py` applies RC only
to the training dataset:

```text
training dataset length = 2 x number of training constructs
even item = original sequence
odd item = reverse complement of the same sequence
target and sample weight = unchanged
```

The validation dataset, test dataset, and deterministic train-evaluation
dataset use only the original sequence orientation. RC is therefore a
deterministic paired augmentation, not a random 50% transform and not
test-time averaging.

Assigning the same target to an RC sequence asserts orientation invariance.
That is biologically more plausible for some enhancer contexts than for
Promoter, UTR, or Intron. A predictive gain may reflect useful regularization
rather than true biological equivalence, so report the effect by part and do
not turn RC on globally from a pooled average alone.

The June broad configs were inconsistent by part:

- Promoter, Intron, 5'UTR, and Enhancer swept `RC=false/true` alongside other
  variables.
- The standardized 3'UTR ResNet1D broad config fixed RC off.
- The June outer-seed, weighted-loss, barcode-bin, and threshold runs fixed RC
  off for all included parts.

Because the broad HPO did not pair RC states for every identical config/split,
its RC distribution is suggestive rather than a controlled RC estimate. The
new outer RC screen supplies the missing paired comparison. Directionality is
a biological reason for caution in Promoter/UTR/Intron, not a reason to omit
the measurement.



## Non-Goals

Do not mix these questions into this campaign:

- multi-part modeling or pML299 normalization;
- barcode-level heteroscedastic/two-head modeling;
- a log2-versus-log10 target ablation;
- a new architecture family;
- a fresh exact-barcode-bin factorial;
- a fresh barcode-threshold factorial;
- a search for new weighted-loss formulas or cap values.

Those remain valid follow-ups, but each adds another experimental factor.

## Canonical Data Contract



### Target

Use one target definition for all five parts:

```text
target = log2(RNA_bc_counts_sum / DNA_bc_counts_sum)
```

Keep:

```text
normalize = true
barcode_column = n_barcodes
train_min_barcodes = 1
heldout_min_barcodes = 8
```

With training-set z-score normalization enabled, log2 and log10 targets differ
only by a positive constant scale and become mathematically equivalent after
standardization. Log2 is not universally more interpretable than log10; it is
convenient for biological fold-change language because `+1` means two-fold
higher and `-1` means one-half, whereas a log10 unit means ten-fold. Log2 also
matches the current single-part prep scripts.

### Aggregate Zero Counts And Pseudocount Decision

The 2026-07-09 audit of all five `.dedup_exact.csv` variant tables found no
construct with zero aggregate DNA or zero aggregate RNA. The result remains
zero after applying every June length branch:

| Part | June-branch rows | `DNA_bc_counts_sum == 0` | `RNA_bc_counts_sum == 0` |
|---|---:|---:|---:|
| Enhancer | 4,787 valid DNA rows | 0 | 0 |
| Promoter | 7,893 | 0 | 0 |
| Intron modal-80 | 7,848 | 0 | 0 |
| 3'UTR modal-100 | 6,845 | 0 | 0 |
| 5'UTR modal-50 | 8,331 | 0 | 0 |

Therefore, do not add a pseudocount to the construct-level target in this
rerun. Adding one would change every low-count target and shrink ratios toward
one even though no aggregate ratio is undefined. It would introduce another
data-policy change while the experiment is meant to isolate deduplication.

Barcode-level rows are different:

| Part | Dedup barcode rows | Either DNA or RNA is zero | Both are zero |
|---|---:|---:|---:|
| Enhancer | 33,940 | 13,536 | 7,505 |
| Promoter | 47,036 | 12,178 | 11,342 |
| Intron | 39,788 | 10,006 | 8,436 |
| 3'UTR | 30,260 | 8,532 | 7,268 |
| 5'UTR | 46,271 | 11,206 | 10,044 |

That is why a target such as
`log2((RNA + alpha_R) / (DNA + alpha_D))` is relevant to Follow-Up 3 but not
needed for this aggregate-target HPO. In Follow-Up 3, compare explicit
pseudocount/shrinkage policies or a Poisson/negative-binomial count likelihood;
do not silently inherit a pseudocount into this rerun.

### Meaning Of "All Variants"

For this rerun, "all variants" should mean all eligible variants in the
established modeling branch, including low-barcode training variants. Preserve
the June length policies so the dedup change remains the main data change:


| Part     | Main length policy                    | Current modeled rows | Current `n_barcodes >= 8` |
| -------- | ------------------------------------- | -------------------- | ------------------------- |
| Enhancer | no-flank, valid lengths padded to 216 | 4,787                | 1,229                     |
| Promoter | all valid 41-51 nt, padded to 51      | 7,893                | 1,931                     |
| Intron   | modal 80 nt                           | 7,848                | 1,326                     |
| 3'UTR    | modal 100 nt                          | 6,845                | 775                       |
| 5'UTR    | modal 50 nt                           | 8,331                | 1,797                     |

The dedup products confirm that the expected biological lengths are the modes:

| Part     | Valid rows | Modal length | Rows at mode | Percent at mode | Valid length range |
| -------- | ---------- | ------------ | ------------ | --------------- | ------------------ |
| Enhancer | 4,787      | 200          | 4,030        | 84.19%          | 76-211             |
| Promoter | 7,893      | 50           | 7,774        | 98.49%          | 41-51              |
| Intron   | 8,088      | 80           | 7,848        | 97.03%          | 42-82              |
| 3'UTR    | 7,257      | 100          | 6,845        | 94.32%          | 45-101             |
| 5'UTR    | 8,460      | 50           | 8,331        | 98.48%          | 40-51              |

The 216-nt Enhancer width is not a biological claim that enhancers are 216 nt.
The original June config documents 216 as the smallest Basset-compatible input
length above the observed valid maximum of 211 nt. It allows BassetVL and
ResNet1D to share the same exact-replay input representation while retaining
the 757 valid non-200 rows. Restricting to modal 200 nt would discard 15.81% of
the valid Enhancer variants and would be a new length-policy experiment.

Likewise, the 51-nt Promoter width is not a claim that the biological mode is
51 nt. It is the observed valid maximum and retains the 119 non-50 rows (1.51%)
by neutral padding. Intron, 3'UTR, and 5'UTR followed the stricter June policy
of retaining only their modal 80/100/50 nt rows with no padding.

Preserve 216 and 51 for the exact replay because changing width or filtering at
the same time as deduplication would weaken the direct comparison. If the
scientific question later shifts to strictly canonical-length constructs, add
separately named `enhancer_modal200` and `promoter_modal50` branches and do not
mix their results into the dedup-only comparison.


Literal all-length Intron/3'UTR/5'UTR training would be a separate data-policy
change. If desired, add it later as an explicitly named all-valid branch.

"All-length" does not mean using `parts_concatenated` as model input. The
single-part model still consumes only `Enhancer`, `Promoter`, `Intron`,
`ThreePrime`, or `FivePrime`. The `parts_concatenated` string, including its
`x` placeholders for absent parts, is used as a stable construct identifier.
All-length would mean retaining non-modal lengths from one part column and
padding that same part sequence.

One Enhancer row has `Enhancer=x`, 5,842 reported barcodes, and positive
aggregate counts. It is not valid DNA and must be excluded from the sequence
model or explicitly documented as a control. The valid Enhancer counts above
exclude it; the old 4,788-row learn-ready table retained it as `X`.

### Required Five-Part Products

The external deduplicated variant products already exist for all five parts
under:

```text
$BODA_WORK_ROOT/opt_EU_learn_n_design/MattLee_lib1/
  single_part_variant_level/*.dedup_exact.csv
```

Before HPO, create one canonical single-output learn-ready TSV and metadata
record per part. Every metadata record must include:

- external source path and SHA256;
- dedup manifest path and SHA256;
- output path and SHA256;
- sequence column and length policy;
- target column and exact target formula;
- row count and high-barcode count;
- `dedup_policy=exact_barcode_row_dedup_v1`;
- data-generation ID, proposed `lib1_single_part_dedup_exact_v1`.

SHA256 is a cryptographic file fingerprint. Identical bytes produce the same
64-character digest; a changed byte produces a different digest. It verifies
that a run used the intended immutable file, but it does not prove that the
file is biologically or statistically correct.

Current gap: Promoter and 5'UTR have canonical dedup single-output TSVs.
Intron and 3'UTR still default to pre-dedup source files, and the Enhancer prep
script still defaults to the old source and old shifted-log10 target. The
two-head dedup tables do not replace the missing single-output products.

This gap is expected: the quick July variant-level rerun intentionally prepared
only Promoter and 5'UTR. It is unfinished five-part standardization, not
evidence that the other dedup variant products are missing.

The two-head tables are the parallel Follow-Up 3 mean/spread experiment. They
combine the dedup variant sequence table with barcode-level summaries and
construct targets such as pseudocounted mean expression and barcode spread.
They are related to the barcode-level uncertainty question, but they are not
the canonical one-head target `log2(sum RNA / sum DNA)` required here.

## Split Redesign

The old outer-seed run reduced split luck, but independent random 250-row
splits can overlap and should not be treated as five independent test sets.
Use explicit, stable split manifests in the rerun.

### Stable IDs

Assign splits from a stable construct ID, preferably `parts_concatenated`, and
record the sequence as an audit field. Do not make the split depend on current
TSV row order. Assert that a construct cannot appear in more than one split.

### Frozen Audit Test

Before broad replay, freeze a high-barcode audit test set for each part. A useful
part-aware rule is:

```text
audit_test_n = min(400, max(250, round(0.20 * n_high_barcode)))
```

Using the current counts gives approximately:


| Part     | Proposed frozen audit test N | Remaining high-barcode development N |
| -------- | ---------------------------- | ------------------------------------ |
| Enhancer | 250                          | 979                                  |
| Promoter | 386                          | 1,545                                |
| Intron   | 265                          | 1,061                                |
| 3'UTR    | 250                          | 525                                  |
| 5'UTR    | 359                          | 1,438                                |


The exact assignments, counts, and hashes must be written before launching any
new run. The audit test must not be evaluated during broad replay or outer-fold
selection.

### Five Development Folds

Partition the remaining high-barcode development rows into five deterministic,
non-overlapping folds. For outer fold `k`:

```text
validation = development fold k
training = all low-barcode rows + all other development folds
audit test = unavailable to the training command
```

This gives every development high-barcode row one out-of-fold (OOF) prediction
and uses the full development pool for robust config/loss selection. OOF means
that each row is predicted by the one fold model for which that row was held
out of training. Concatenate those once-held-out predictions across all five
folds, then compute the pooled OOF metrics. All configs and both loss modes must
use the exact same fold assignment.

For 3'UTR, five folds contain only about 105 validation rows each. Keep all
five folds so every one of the 525 development-HQ rows receives one OOF
prediction, but make pooled OOF Pearson across all 525 rows the primary score.
Use the five per-fold Pearson values as stability diagnostics, not as five
high-precision independent estimates. If rankings are unstable, predeclare a
three-fold sensitivity analysis (about 175 validation rows per fold) before
opening the audit test.

### Do Validation And Test Need To Be Larger?

There is no separate test set inside each development fold. Each fold supplies
validation for early stopping and config/policy selection; across five folds,
every development-HQ row is used once for OOF validation. This increases the
effective validation evidence without permanently removing a larger fixed
validation set from training.

The frozen audit test is the only test set in the new selection pipeline. Its
part-aware size is already larger than 250 where HQ support allows. For 3'UTR,
keeping 250 audit rows is a compromise: reducing it would weaken the final
test, while increasing it would leave too few development-HQ rows. Pooled OOF
validation plus the 250-row audit test is preferable to five overlapping
250/250 random val/test splits.

### Final Refit

After selection, use the median best epoch from the five development folds as
a fixed epoch budget, refit on every non-audit row, and evaluate the frozen
audit test once. This avoids holding out another early-stopping subset during
the final fit.

The selected object is one base hyperparameter config ranked from its five OOF
runs. The median is taken over that config's five best epochs; it is not a
"median hyperparameter config." Fold models checkpoint and early-stop on
validation Pearson. The final all-development refit has no validation fold and
therefore trains for the fixed median epoch budget before reporting audit-test
Pearson and the secondary metrics.

### Metric-History Policy

Preserve learning-dynamics visibility without repeatedly exposing the audit
test:

- Stage 1 exact replay logs train and fold-validation metrics at every
  validation epoch; it receives no audit-test loader.
- Stage 2 RC and Stage 3 loss runs also log train and fold-validation metrics
  at every validation epoch; they receive no audit-test loader.
- The final all-development refit logs train metrics by epoch. It has no
  validation fold because every non-audit row is used for training.
- The final refit evaluates the frozen audit test once after its fixed epoch
  budget. Do not log audit-test performance every epoch, even in this final
  stage, because those curves would invite test-set model or epoch selection.

For test-like learning curves during development, use the five OOF validation
histories and predictions. Historical train/validation/test-per-epoch logging
can remain in old runs, but it is intentionally not the contract for this new
frozen-audit campaign.

## Architecture Scope

The dedup effect is not large enough to justify blindly repeating every old
architecture sweep. Use the old architecture evidence as a prior and reopen
only the unresolved comparison.

Recommended historical source lanes for exact replay:


| Part     | Architecture lane         | Reason                                                                                         |
| -------- | ------------------------- | ---------------------------------------------------------------------------------------------- |
| Enhancer | ResNet1D                  | June standardized HPO favored ResNet1D; scratch signal remained modest.                        |
| Promoter | PromoterBassetVL          | June HPO and the dedup follow-up both favor BassetVL over ResNet1D.                            |
| Intron   | ResNet1D                  | Strong June signal and the only standardized in-house architecture.                            |
| 3'UTR    | ResNet1D                  | Full standardized sweep and June outer run promoted ResNet1D.                                  |
| 5'UTR    | ResNet1D and UTR_BassetVL | Prior validation was nearly tied and the dedup follow-up did not resolve architecture cleanly. |


This is six source lanes. The implementation must resolve the completed prior
runs, drop duplicate `base_config_id` values after excluding seed/RC/loss/data
fields, and report the exact replay count before launch. It is not a request to
create six new 128-trial Bayesian sweeps.

The earlier optional-challenger sketch is superseded by the frozen amendment.
Stage 2 adds only two bounded routes: 60 Enhancer BODA2/Malinois transfer cells
from two source heads crossed with three scopes, and 100 3'UTR UTRBassetVL
cells from ten exact completed historical configs. It does not add an Enhancer
scratch BassetVL or Promoter ResNet lane, and it does not sample a new search
space.

Previously generated June outer configs are eligible if they actually
completed training. This matters because `local_variant` supplied four of the
top five June outer configs for both Promoter and Intron, and one
`narrow_prior` config entered the Promoter and 3'UTR top five. The restriction
is against generating new variants now, not a claim that the old generated
configs failed.

## Stage 0: Repository And Data Readiness

Acceptance gates before any GPU launch:

1. Five dedup single-output learn-ready TSVs exist and pass schema checks.
2. Every selection/replay config resolves to `log2_RNA_DNA` and the dedup
  data-generation ID; only explicitly labeled calibration mates may resolve to
  `pre_dedup_v0`.
3. Five split manifests exist with stable construct IDs and no overlap.
4. Dataset and split hashes are logged in a two-run CPU/GPU smoke test.
5. RC and loss arms contain the same construct IDs for a matched config/fold
  pair; RC-on doubles only the training examples.
6. Weighted MSE is hand-checked against
  `sum(w * squared_error) / sum(w)`.
7. W&B entity preflight resolves to
  `minhangxu1998-baylor-college-of-medicine`.
8. A run can complete without retaining a permanent model artifact.
9. Replay/outer smoke runs produce no test metric at any epoch or after fit.



## Stage 1: Broad Exact-Config Replay

### What Counts As An Exact Config

Define `base_config_id` from the resolved architecture, model hyperparameters,
optimizer, scheduler, batch size, and trainer/early-stopping settings. Exclude:

```text
data generation
split seed or development fold
model seed
use_reverse_complements
graph/loss module and barcode weighting
W&B/project/output paths
```

Those excluded values are experimental factors crossed around the same base
config. This definition lets the campaign say both "the model hyperparameters
were already tried" and "RC/loss are now evaluated in controlled pairs."

Candidate configs may come from completed June broad HPO or completed June
outer runs. A config originally labeled `local_variant` or `narrow_prior` is
eligible because it has already been trained. A generated config that never
completed is not eligible.

### Selection Replay

Create a fixed manifest containing every unique eligible `base_config_id` from
the approved six architecture lanes. This is not a W&B Bayesian sweep.

Run every row with:

```text
data_generation = dedup_exact_v1
loss_mode = unweighted_mse
graph_module = CNNBasicTraining
model_seed = 1701
development_fold = 0
use_reverse_complements = false
train_min_barcodes = 1
precision = 32
audit test = unavailable
```

Log train and validation dynamics and final validation metrics. Save the exact
source run ID(s), resolved historical config snapshot, and the fields removed
when constructing `base_config_id`. Do not permanently archive every model.

### Direct Data-Product Calibration

Reusing the same base config is useful historical context, but comparing a new
fold-0 result directly with an old run that used a different split/model seed
is not a pure dedup effect.

For a clean data comparison, preselect five configs per part from historical
validation evidence and run this pair on the same current fold:

```text
for part in PARTS:
    for config in PREDECLARED_TOP5[part]:
        for data_generation in [pre_dedup_v0, dedup_exact_v1]:
            train(
                base_config=config,
                fold=0,
                model_seed=1701,
                rc=False,
                loss="unweighted_mse",
            )
```

The dedup member is already present in the broad replay, so this adds only 25
pre-dedup runs. Treat these deltas as a data-product diagnostic; do not let the
pre-dedup arm enter new-model selection. The July Promoter/5'UTR top-15 study
remains supporting evidence but used the legacy split contract.

### Promotion To The RC Screen

Promote 10 exact base configs per part:

- 6 highest fold-0 validation-Pearson configs;
- 2 diverse representatives from the top validation quartile;
- 2 strong validation configs with complementary RMSE/COD R2 or architecture
  evidence.

Do not use historical or current test metrics. Do not jitter parameters or
sample a new prior. For 5'UTR, select 10 total across both architectures and
retain architecture representation when the leaders are within 0.01 Pearson
or have overlapping paired prediction-bootstrap intervals.

## Stage 2: Paired Five-Fold RC Screen

Pre-launch amendment: apply
`lib1_dedup_pre_stage2_protocol_amendment_july2026.md` for the Intron inferred
mask-stratum robustness estimands and the bounded Enhancer/3'UTR challenger
lanes. The 500-cell core design below remains intact; the frozen `N=2`, `K=10`
amendment adds 160 separately labeled challenger cells and does not open the
frozen audit set.

Use unweighted MSE to isolate the augmentation question:

```text
5 parts
x 10 exact base configs per part
x 5 development folds
x 2 RC modes
= 500 analysis cells
```

There are 10 promoted configs **per part**, not 10 across the campaign. For one
part, `10 configs x 5 folds x 2 RC modes = 100` cells. Across five parts,
`5 x 100 = 500` cells.

Stage 1 has already trained the fold-0, RC-off, unweighted cell for every
promoted config. Therefore, reuse `5 parts x 10 configs = 50` Stage 1 runs and
launch only 450 new Stage 2 runs. Reuse is allowed only when the dataset,
split, selected-row, normalization, base-config, model-seed, and trainer-policy
hashes match and the Stage 1 run exported the required validation predictions
and history. Otherwise, regenerate that cell and record why it was not reused.

The frozen bounded lanes add 60 Enhancer transfer cells
(`2 heads x 3 scopes x 5 folds x 2 RC`) and 100 3'UTR UTRBassetVL cells
(`10 configs x 5 folds x 2 RC`). Thus Stage 2 has **660 analysis cells and 610
new launches**.

Use `analysis_lane` only as a candidate-origin/reporting label, never as a
quality rank. Its allowed values are `core_scratch`,
`enhancer_transfer_challenger`, and `utr3_utrbasset_challenger`. A challenger
may win the route comparison; the label prevents it from being misrepresented
as one of the 50 Stage-1-selected scratch configs.

Hold paired within every `(part, base_config_id, development_fold)`:

- architecture and all model/optimizer/trainer hyperparameters;
- model seed `1701`;
- training and validation row IDs;
- target-normalization source rows;
- loss module and unweighted MSE;
- epoch/patience policy.

Change only `use_reverse_complements=false/true` and provenance/output labels.
Validation remains original-orientation in both arms.

For every `(part, config, RC arm)`, concatenate the five validation prediction
exports and assert exactly one held-out prediction per development construct:
979 Enhancer, 1,545 Promoter, 1,061 Intron, 525 3'UTR, and 1,438 5'UTR.
The number 1,061 is specific to Intron. Pooled five-fold OOF Pearson remains
primary. On raw `log2_RNA_DNA`/`prediction_raw`, Intron additionally reports
within-stratum-centered, macro-stratum, minimum-stratum, and per-stratum
calibration metrics. RC off/on must contain identical construct IDs and raw
targets before paired comparison. Inferred masks are sensitivity labels only,
and no frozen-audit loader may be instantiated or scored in Stage 2.

## Stage 3: Paired Weighted-Loss Add-On And Selection

### Complete The RC By Loss Factorial On The Top Five

Rank the Stage 2 configs by pooled OOF validation, allowing each config's RC
state to be chosen by validation. Promote five base configs per part while
retaining near-tied/diverse configurations.

The unweighted RC-off/on rows already exist. Add weighted rows for both RC
states:

```text
5 parts
x 5 exact base configs per part
x 5 development folds
x 2 RC modes
x 1 new weighted-loss arm
= 250 additional runs
```

For one part, `5 configs x 5 folds x 2 RC modes = 50` new weighted runs. Across
five parts, `5 x 50 = 250` new runs. The complete top-five factorial contains
500 result cells:

```text
5 parts x 5 configs x 5 folds x 2 RC modes x 2 loss modes = 500 cells
```

However, its 250 unweighted cells are reused from Stage 2. Stage 3 launches
only the 250 missing barcode-weighted mates; it does not rerun those unweighted
cells.

Together with the reused unweighted rows, the top-five analysis has a complete
`RC x loss` factorial:

```text
RC off + unweighted MSE
RC on  + unweighted MSE
RC off + barcode-weighted MSE
RC on  + barcode-weighted MSE
```

### Meaning Of A Paired Outer-Loop Factor

Loss mode is not randomly sampled by HPO. Every selected config/fold/RC cell
gets an otherwise identical unweighted and weighted run:

```python
for part in parts:
    for config in exact_top5[part]:
        for fold in development_folds:
            for rc in (False, True):
                common = {
                    "part": part,
                    "base_config": config,
                    "fold": fold,
                    "model_seed": 1701,
                    "rc": rc,
                    "train_row_ids": split_manifest.train_ids(fold),
                    "val_row_ids": split_manifest.val_ids(fold),
                }
                unweighted = train(**common, loss="mse")
                weighted = train(**common, loss="barcode_weighted_mse")
                record_pair(
                    weighted.val_pearson - unweighted.val_pearson,
                    pair_id=(part, config.id, fold, rc),
                )
```

The same logic defines the RC pair within each config/fold/loss cell. Pairing
removes variation from config and split before estimating the RC or loss
effect. "Outer loop" means RC and loss enumerate fixed arms around the tried
base config; neither is a Bayesian HPO proposal.

### Config And Policy Ranking

Primary ranking unit:

```text
(part, architecture, base_config_id, rc_mode, loss_mode)
```

Primary score:

```text
pooled OOF validation Pearson across all development-HQ rows
```

Also report:

- mean, standard deviation, and minimum of the fold Pearson values;
- median and 20th-percentile fold Pearson;
- validation Spearman, COD R2, RMSE, and best epoch;
- paired RC-on-minus-RC-off deltas by fold and loss;
- paired weighted-minus-unweighted deltas by fold and RC;
- the RC-by-loss interaction delta;
- model size and training time.

Use a one-standard-error rule when configs are effectively tied: prefer the
lower-variance or simpler config rather than the numerically largest mean.

Recommended rule for making RC-on the downstream default for a part:

```text
mean paired validation-Pearson delta >= 0.005
and positive delta in at least 4 of 5 folds
and no material RMSE or COD-R2 degradation
```

Apply the same predeclared rule to weighted-minus-unweighted loss. If a policy
does not meet the rule, keep RC off or unweighted MSE as the conservative
default. Report config-specific heterogeneity even when the part-level default
is off.

### Validation-Selected Frozen Audit Evaluation

For each part, select one base config from OOF validation and pre-register its
four RC/loss arms. For each arm, take the median of its five fold-specific best
epochs, refit on every non-audit row for that fixed number of epochs, and run
three model seeds `[1701, 1702, 1703]`:

```text
5 parts x 4 RC/loss arms x 3 model seeds = 60 runs
```

This is the precise meaning of "validation-selected final audit-test
evaluation":

1. Fold models checkpoint and early-stop on fold validation Pearson.
2. Config/RC/loss selection uses pooled OOF validation only.
3. Final refits do not early-stop because all non-audit rows are used for
   training; they use the median selected epoch budget.
4. The frozen audit test is evaluated once per final refit, reporting Pearson,
   Spearman, COD R2, and RMSE.

Do not use audit-test results to return to HPO and choose another config. An
optional architecture-diverse 5'UTR finalist must be declared before opening
the audit test and budgeted separately.

## Stage 4: Dedup Learning Curves

After Stage 3, choose up to three base configs per part from outer-fold
validation. Include the winner, a one-standard-error runner-up, and an
architecture-diverse 5'UTR config when justified.

Fix one RC policy and one loss policy per part before launching downsampling.
Use all eligible dedup training variants (`train_min_barcodes=1`) and nested
random subsets:

```text
N = [100, 250, 500, 1000, 2000, 3500, full]
```

Primary design:

```text
5 parts
x 3 configs
x 5 development folds
x 7 size arms
= 525 runs
```

For each `(part, fold, downsample_seed)`, create one deterministic permutation
of the eligible training pool and use prefixes for every N. Reuse the same
selected row IDs across configs. Record selected-row hashes and barcode/target
composition for every arm.

Primary outputs:

- heldout Pearson versus actual N;
- paired delta between adjacent N values;
- gain per additional 1,000 variants;
- N needed to reach 90% and 95% of full-data performance;
- fitted saturation curve with uncertainty across configs/folds;
- train/validation gap and best epoch versus N.

This stage answers sample quantity. It intentionally does not cross N with
barcode thresholds, RC modes, or both loss modes.

## Approximate Run Budget

Let `N_exact` be the number of unique eligible `base_config_id` rows resolved
from the six historical source lanes plus already-trained June outer configs.
The generator must print this count before launch; it is expected to be on the
order of 800-850, not assumed to equal six times 128.

| Stage | Runs |
|---|---:|
| Broad dedup exact-config replay | `N_exact` |
| Pre-dedup calibration mates | 25 |
| Ten-config paired five-fold RC screen | 500 cells; 450 new after Stage 1 reuse |
| Frozen bounded Stage 2 challengers | 160 cells; all new |
| Top-five weighted add-on across both RC states | 250 |
| Frozen audit finalists | 60 |
| Downsampling learning curves | 525 |
| Total new launches through downsampling | `N_exact + 1,470` |

If `N_exact` is 800-850, the full total with the frozen challenger lanes is
approximately 2,270-2,320 new launches. The Stage 2 analysis table contains
all 660 cells.


The following historical compute-reduction order is not active under the
frozen 660-cell manifest. Any use would require a documented amendment before
regeneration:

1. Keep all five development folds.
2. Reduce the RC screen from 10 to 8 configs per part, making it 400 cells and
   360 new launches after reusing 40 Stage 1 cells.
3. Reduce the weighted add-on from 5 to 3 configs per part, making it 150 new
   runs.
4. If broad replay remains too large, select a predeclared validation-ranked
   and hyperparameter-diverse exact subset from each source lane. Do not replace
   omitted exact configs with new Bayesian proposals.
5. Reduce downsampling configs from three to two.
6. Do not reduce to one split/fold merely to retain more near-duplicate replay
   configs.



## W&B And Provenance Contract

Proposed campaign ID:

```text
lib1_dedup_phase1_rerun_july2026
```

Use explicit project names that cannot be confused with pre-dedup HPO, for
example:

```text
<part>__bashor_in_house__dedup_exact_v1__scratch__<architecture>__exact_replay
<part>__bashor_in_house__dedup_exact_v1__outer_fold__paired_rc
<part>__bashor_in_house__dedup_exact_v1__outer_fold__paired_rc_loss
<part>__bashor_in_house__dedup_exact_v1__audit_refit
<part>__bashor_in_house__dedup_exact_v1__downsampling
```

Every run must log structured fields for:

- campaign ID and stage;
- data-generation ID and dataset SHA256;
- split-manifest ID/SHA256 and development fold;
- base config ID and all historical source run IDs;
- architecture and model seed;
- loss mode and barcode-weight parameters;
- target definition and length policy;
- selected-row hash for downsampling;
- Git commit and hostname;
- resolved W&B entity/project.

Add a launcher preflight that prints and validates the entity before the first
job. The expected entity is:

```text
minhangxu1998-baylor-college-of-medicine
```

This must be enforced inside the Python training entry point, not only supplied
as a shell default. The current `train_wandb_log.py` constructs `WandbLogger`
without an `entity` argument, so W&B can resolve the run through ambient login
or environment state. The quick July launcher also preserved an already-set
`WANDB_ENTITY`, which explains how the Promoter/5'UTR rerun could land in
`mlee228-rice-university-org`.

Add an explicit `--wandb_entity` argument, pass it as
`WandbLogger(entity=..., project=...)`, and immediately after initialization
assert that `wandb.run.entity` exactly equals the expected entity. Abort before
`trainer.fit` on a mismatch. The orchestrator must not silently prefer an
ambient `WANDB_ENTITY` over the campaign value, and every manifest row must
record the expected entity. A one-run pilot must print the resolved entity,
project, and run URL before the full launch.



## Repository Update Plan



### 1. Data Preparation And Manifest

Update these scripts to use `.dedup_exact.csv` inputs and a common metadata
contract:

```text
src/learn/prepare_lib1_enhancer_fastqs1_5_dataset.py
src/learn/prepare_lib1_promoter_inhouse_dataset.py
src/learn/prepare_lib1_intron_inhouse_dataset.py
src/learn/prepare_lib1_threeprime_inhouse_dataset.py
src/learn/prepare_lib1_fiveprime_inhouse_dataset.py
```

Add a tracked compact manifest, proposed:

```text
src/learn/data_manifests/lib1_single_part_dedup_exact_v1.json
```

Do not rely on metadata only inside `src/learn/derived_data/`; that directory
is Git-ignored.

### 2. Explicit Split Support

Extend `boda/data/bashor_datamodule.py` with backward-compatible arguments
such as:

```text
split_manifest_path
split_fold
split_id_column
```

Add a generator, proposed:

```text
src/learn/generate_lib1_dedup_split_manifests.py
```

The data module should verify source hash, ID coverage, split exclusivity,
expected counts, and the selected-row hash.

### 2A. Audit-Test Evaluation Gate

`src/learn/train_wandb_log.py` currently calls `trainer.test(...)` after every
fit whenever a test loader exists, even if test was omitted from
`epoch_eval_splits`. Add a backward-compatible control such as:

```text
evaluate_test_after_fit = true | false
```

Set it to `false` for exact replay, RC screen, and weighted outer runs. Set it
to `true` only for the frozen audit refits. Use `epoch_eval_splits=[train,val]`
for replay/outer runs and ensure their split manifest does not expose audit
rows through another loader name.

The end-of-fit train summary should use `train_eval_dataloader()` when it is
available. The current helper uses `train_dataloader()`, which includes both
original and RC examples in an RC-on run and makes its final train metric
incomparable to RC-off. Per-epoch train diagnostics already use the clean
train-evaluation loader.

### 3. Versioned Replay Config Snapshots

Create explicitly versioned dedup config files rather than silently reusing
historical config names. Historical runs currently point to config paths that
have begun changing in place, which weakens reproducibility.

Use names containing `dedup_exact_v1`, and preserve/export the old resolved
config snapshots before cleaning W&B cache directories.

### 4. Exact-Replay Manifest And Orchestration

Add:

```text
src/learn/generate_lib1_dedup_exact_replay_manifest.py
src/learn/launch/lib1_dedup_phase1_exact_replay_orchestrator.sh
```

The generator should read completed historical run/config exports, construct
and deduplicate `base_config_id`, include already-trained June outer configs,
and emit the fold-0 unweighted RC-off commands plus the 25 pre-dedup
calibration mates. It must not call W&B sweep creation.

Reuse the current global manifest-worker pattern, but make the dedicated
campaign wrapper:

- prepares all five datasets once;
- resolves only the approved six historical architecture source lanes;
- fixes fold/model-seed/RC/loss policy;
- records campaign and dataset/split IDs;
- supports a one-run-per-lane pilot;
- fails on a W&B entity mismatch.



### 5. Paired RC/Loss Outer Manifests

Add a new generator rather than mutating the June generator:

```text
src/learn/generate_lib1_dedup_stage2_manifest.py
src/learn/verify_lib1_dedup_stage2_manifest.py
src/learn/launch/lib1_dedup_stage2_orchestrator.sh
src/analysis/lib1_dedup_stage2_analysis.py
```

Unlike `generate_lib1_outer_seed_prior_hpo_manifest.py`, this generator should
accept only already-trained base configs or the two frozen bounded challenger
routes. It emits the 660-cell Stage 2 analysis table, marks its 50 matching
Stage 1 rows as reused with source run IDs, and emits commands only for the
remaining 610 rows. The 250-row weighted add-on is a later, separately frozen
Stage 3 product; it is not launched by Stage 2. Reject an RC or weighted row
without its exact mate and shared data-row hash.

Reuse the global queue in
`src/learn/launch/lib1_inhouse_outer_seed_prior_orchestrator.sh`, generalized
to a manifest path/tag supplied by the caller.

### 6. Artifact Retention Controls

`src/learn/train_wandb_log.py` currently saves a portable model archive for
every completed run. Add an explicit retention mode, for example:

```text
artifact_retention = none | selected | all
```

Broad replay and outer evaluation should use `none` after best-epoch metrics and
predictions are exported. Final audit refits should use `all`. Keep temporary
Lightning checkpoints only until end-of-run best-checkpoint evaluation.

Retention must control both local saving and Lightning's W&B model logging.
The current training entry point hard-codes `WandbLogger(log_model=True)`, which
can upload and cache a checkpoint even if the portable local archive is later
deleted. Use `log_model=False` for `artifact_retention=none` and enable model
logging only for a retention mode that intentionally keeps the model. A smoke
test must confirm that a `none` run leaves metrics, predictions, provenance,
and history but no local archive, published checkpoint copy, Lightning
checkpoint, or W&B model artifact.

Also disable progress bars in manifest workers or strip carriage-return
updates before writing logs. Historical per-row progress logs are several MB
each and provide little provenance value.

### 7. Registry And Analysis Outputs

Keep `runs.csv` append-only, but add structured campaign fields either as
backward-compatible columns or a companion campaign-run table. Do not encode
all new provenance only in free-text notes.

Create analysis notebooks under a new directory, proposed:

```text
tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/
```

Recommended notebooks:

```text
01_exact_replay_selection.ipynb
02_intron_inferred_mask_strata_analysis.ipynb
03_stage2_paired_rc_analysis.ipynb
04_stage3_paired_rc_loss_analysis.ipynb
05_audit_test_report.ipynb
06_downsampling_learning_curves.ipynb
```

Reuse `src/analysis/hpo_results_eval_utils.py` and the corrected
`src/learn/export_wandb_history.py`. Export compact run/config/history tables
before removing local W&B caches.

Raw `wandb_history_exports/` and timestamped `launch/generated/` command trees
are currently local generated state. Add explicit Git-ignore rules for those
roots, preserve them through the cache migration, and commit only curated
summary tables plus the scripts/manifests that regenerate them. This avoids
accidentally adding tens of MB of per-run history while keeping the evidence
available locally.

### 8. Tests

Add focused tests for:

- five-part dedup prep and target formula;
- stable-ID split reproduction and no leakage;
- frozen audit rows never entering broad/outer training;
- `evaluate_test_after_fit=false` prevents all test-loader execution;
- the invalid Enhancer `x` control is excluded or explicitly handled;
- RC doubles only the training dataset and leaves val/test/train-eval in the
  original orientation;
- matched construct IDs across paired RC and loss arms;
- weighted-MSE arithmetic and missing-weight failure;
- nested downsampling prefixes;
- manifest uniqueness and exact expected run counts;
- retention mode deleting transient checkpoints without deleting selected
final artifacts.



## Storage Audit And Cleanup Plan

No cache was deleted while writing this plan. Current relevant usage on
`10.66.4.13` is:


| Path/layer                                        | Size           | Recommendation                                                                                                                  |
| ------------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `src/learn/wandb/run-*`                           | 28 GB          | Delete after remote-sync/config-history export verification.                                                                    |
| `src/learn/wandb/sweep-*`                         | 11 MB          | Keep until resolved source configs are exported.                                                                                |
| `src/learn/wandb/offline-run-*`                   | 304 KB, 3 runs | Do not delete; inspect or sync first.                                                                                           |
| `src/learn/outputs/hpo_runs/status/*/logs`        | 3.7 GB         | Safe first cleanup after preserving a small failure summary; all six historical manifests show complete rows and zero failures. |
| Pre-dedup five-part scratch `local_artifacts`     | about 6.8 GB   | Delete non-promoted checkpoints after config/history export; retain named comparison checkpoints only.                          |
| June outer-seed `local_artifacts`                 | 2.5 GB         | Retain selected checkpoints/predictions, delete the rest after the June analysis is frozen.                                     |
| June weighted-loss `local_artifacts`              | 221 MB         | Delete after compact paired tables and any selected checkpoints are retained.                                                   |
| June exact-bin `local_artifacts`                  | 1.1 GB         | Delete after analysis tables are frozen unless checkpoint-level re-evaluation is planned.                                       |
| June threshold-downsampling `local_artifacts`     | 6.6 GB         | Delete non-promoted checkpoints after preserving learning-curve tables and selected-row provenance.                             |
| June n=1 full/downsample `local_artifacts`        | about 454 MB   | Delete after confirming these exploratory branches are not being resumed.                                                       |
| July Promoter/5'UTR dedup follow-up artifacts     | 314 MB         | Keep through this rerun design checkpoint; later retain only selected representatives.                                          |
| `src/learn/outputs/hpo_runs` retained checkpoints | 7.6 GB total   | Prune per-run checkpoint copies after a selected-checkpoint allowlist is created.                                               |
| `src/learn/run_registry/wandb_history_exports`    | 62 MB          | Keep; this is compact evidence needed before W&B cache deletion.                                                                |
| `src/learn/outputs/hpo_analyses`                  | 32 MB          | Keep; these are compact derived evidence tables.                                                                                |
| `src/wandb`                                       | 21 GB          | Separate legacy cache; audit and delete synced run dirs in a separate cleanup pass.                                             |


### Cleanup Timing Decision

Historical cleanup is deferred until after the dedup multi-stage rerun. On
2026-07-09, `/home` had about 313 GB free, was 56% utilized, and had 96% of its
inodes free. Current major active footprints were approximately 28 GB in
`src/learn/wandb`, 20 GB in `src/learn/local_artifacts`, and 8 GB in
`src/learn/outputs`. This is enough operational runway, and retaining the June
cache until exact source configs and histories have been exported reduces
reproducibility risk.

Deferral depends on the new retention controls being implemented before the
full Stage 1 launch. Broad replay and outer runs must use
`artifact_retention=none`, disable W&B model logging, prune transient
checkpoints, and avoid verbose progress logs. Record `df -h` and campaign
directory sizes at every stage boundary. Pause for cleanup if `/home` reaches
80% utilization or falls below 150 GB free, or if `/` falls below 20 GB free
for temporary training state. Cleanup may also begin sooner after compact
config/history/prediction exports are proven sufficient, but it is not a
prerequisite for implementation or Stage 1.




### Safe Cleanup Order

1. Freeze compact evidence:
  - run/config tables;
  - per-epoch histories needed by notebooks;
  - split assignments and prediction tables;
  - selected checkpoint allowlist;
  - manifest completion summaries.
2. Inspect/sync the three `offline-run-*` directories.
3. Verify the relevant W&B projects and run counts remotely.
4. Delete completed manifest progress logs, preserving compact summaries.
5. Delete synced `src/learn/wandb/run-*` directories, but keep `sweep-*` until
  config snapshots are exported.
6. Delete non-selected local model archives and duplicate published
  checkpoints by campaign.
7. Audit `src/wandb` separately because it includes non-Lib1 work.
8. Re-run analysis notebooks from compact exports to prove the cleanup did not
  remove an undeclared dependency.

Immediate low-risk recovery is about 3.7 GB from completed status logs.
Conditional recovery is another 28 GB from synced `src/learn/wandb/run-*`
cache. A selected-checkpoint cleanup of superseded June artifacts can recover
roughly another 15-20 GB, depending on how many comparison checkpoints are
retained.

Add a dry-run cleanup utility rather than relying on ad hoc `rm` commands:

```text
src/learn/cleanup_run_cache.py \
  --campaign <campaign_id> \
  --keep-manifest <selected_checkpoint_allowlist.csv> \
  --dry-run
```

The utility should refuse paths outside allowlisted cache roots and print
expected reclaimed bytes before deletion.

## Pilot And Launch Gates



### Data/Split Pilot

- regenerate all five learn-ready TSVs;
- generate split manifests twice and verify identical hashes;
- inspect target/barcode distributions by train, development fold, and audit
test;
- confirm no construct overlap.



### Training Pilot

Run an exact-replay pilot first:

```text
1 part x 2 configs x fold 0 x RC off x unweighted = 2 runs
```

Then run an outer pairing pilot:

```text
1 part x 2 configs x 2 folds x 2 RC modes x 2 loss modes = 16 runs
```

Acceptance:

- W&B entity/project/campaign fields are correct;
- paired RC and loss arms have identical row hashes;
- train/validation histories are present;
- audit-test metrics are absent;
- model artifacts follow the requested retention mode;
- resume markers work after an interrupted worker;
- registry rows contain dataset/split/config provenance.



### Full Launch Order

1. Generate the exact-replay manifest and review `N_exact`, source run IDs,
   deduplication rules, and resolved commands.
2. Run the two-row exact-replay pilot.
3. Launch the full dedup exact-config replay plus 25 pre-dedup calibration
   mates.
4. Freeze and review the 10 exact RC-screen configs per part.
5. Generate the 660-cell Stage 2 analysis manifest, verify the 50 Stage 1 reuse
   links and the 160 separately labeled challenger cells, then pilot and launch
   its 610 new rows.
6. Freeze and review the top five configs per part from pooled OOF validation.
7. Generate, pilot, and launch the 250-row weighted add-on.
8. Freeze config, RC, loss, and epoch selection before audit evaluation.
9. Run the 60 finalist audit refits.
10. Review the learning-curve shortlist and launch downsampling.

### Screen Session Launch Contract

The implementation agent should create this Stage 1 launcher:

```text
src/learn/launch/lib1_dedup_phase1_exact_replay_orchestrator.sh
```

After that file and its manifest generator pass dry-run review, start an
attached screen session:

```bash
cd "$(git rev-parse --show-toplevel)"
screen -S lib1_dedup_exact_replay
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate boda_env

WANDB_ENTITY="minhangxu1998-baylor-college-of-medicine" \
BODA_WANDB_ENTITY="minhangxu1998-baylor-college-of-medicine" \
GPU_LIST="0 1 2 3" MAX_PARALLEL=4 PREPARE_DATASET=1 \
  bash src/learn/launch/lib1_dedup_phase1_exact_replay_orchestrator.sh
```

Detach with `Ctrl-a d` and reattach with:

```bash
screen -r lib1_dedup_exact_replay
```

Do not run this command before the launcher exists. Its default action should
be Stage 1 exact replay only; it must not automatically continue into RC,
weighted loss, audit evaluation, or downsampling without frozen selection
files between stages.

No additional W&B project information is currently needed. The host already
has the intended entity in `_wandb_helpers.sh`; the launcher must still force
and validate both entity variables above. If `wandb status` reports no API key,
authenticate locally with `wandb login`; do not put the API key in a script,
manifest, plan, or chat message.



## Stop Conditions

Pause the campaign if:

- any non-calibration config resolves to a pre-dedup source or non-log2 target;
- a construct crosses train/validation/audit boundaries;
- paired RC/loss arms have different construct-row hashes;
- target normalization uses validation or audit rows;
- broad replay logs or selects on audit-test metrics;
- a promoted outer base config was never completed in a historical run;
- W&B resolves to the collaborator entity instead of the expected entity;
- per-run model artifacts or logs begin consuming storage contrary to the
retention policy;
- any part has too few development HQ rows for stable five-fold validation,
in which case fold count/partition size should be revisited before launch.



## Recommended Defaults To Approve

Unless later evidence changes them, use:

```text
target = log2_RNA_DNA
data_generation = lib1_single_part_dedup_exact_v1
primary architecture lanes = 6
stage1 = fixed exact-config replay, not Bayesian HPO
RC-screen configs per part = 10
weighted-add-on configs per part = 5
development folds = 5
model_seed during replay/outer = 1701
final model seeds = [1701, 1702, 1703]
stage1 RC = false
outer RC arms = [false, true]
outer loss arms = unweighted MSE and clipped-log barcode-weighted MSE
barcode_weight_cap = 8.0
barcode_weight_min = 0.1
downsampling configs per part = 3
downsampling N = [100, 250, 500, 1000, 2000, 3500, full]
```
