# Lib1 Dedup Pre-Stage-2 Protocol Amendment (July 2026)

> **Role:** Formal amendment record. Read the plain-language rationale in
> `lib1_dedup_stage1_to_stage2_reader_guide_july2026.md` first; return here to
> verify the exact Intron estimands, challenger definitions, and run budget.

Status: **frozen amendment; Stage 2 execution complete**. On 2026-07-12, the
collaborator approved Enhancer `N=2`, 3'UTR UTRBassetVL `K=10`, and the use of
inferred Intron masks as sensitivity labels. All 660 analysis cells completed
on 2026-07-13, and Stage 2 analysis is underway. No frozen-audit evaluation
has launched.

This document amends only the Intron diagnostic estimands and the bounded
Enhancer/3'UTR challenger scope in
`lib1_dedup_phase1_hpo_rerun_plan_july2026.md`. The Stage 1 result product,
five development folds, frozen audit set, model seed, unweighted loss, and
paired RC design remain unchanged.

## Rationale

Stage 1 completed 885 dedup exact-config replays and 25 pre-dedup diagnostic
mates. Post-run review identified two issues that should be declared before
Stage 2:

1. The Intron library was designed from three collaborator-supplied 80-nt
   IUPAC masks, but exact synthesis-pool membership is unavailable.
2. Enhancer and 3'UTR scratch performance is modest and each has an isolated
   fold-0 leader. The repository already contains stronger prior information:
   an Enhancer BODA2/Malinois transfer route and completed 3'UTR UTRBassetVL
   configurations.

This amendment is not a response to audit performance; the audit set remains
unopened.

## Frozen Approval Record

The following rules are the implementation contract for Stage 2:

1. Keep pooled five-fold OOF Pearson primary so Stage 1 reuse and the intended
   mixed-library estimand remain intact.
2. Concatenate exactly one held-out prediction for every development construct
   in each `(config, RC arm)`. Intron has exactly **1,061** such constructs;
   `1,061` is not the all-part total.
3. On raw `log2_RNA_DNA` and `prediction_raw`, compute OOF
   within-stratum-centered, macro-stratum, minimum-stratum, and per-stratum
   calibration metrics for Intron.
4. Compare RC off/on as paired predictions on identical construct IDs.
5. Treat the deterministic Intron mask strata as sensitivity categories, not
   true synthesis-subset membership.
6. Do not instantiate or score a frozen-audit loader until the final audit
   stage.

The development-set row counts that each config/RC arm must cover exactly once
are:

| Part | Development constructs in five-fold OOF product |
|---|---:|
| Enhancer | 979 |
| Promoter | 1,545 |
| Intron | 1,061 |
| 3'UTR | 525 |
| 5'UTR | 1,438 |

The approved campaign sizes are fixed at Enhancer `N=2` and 3'UTR `K=10`.
Together with the 500-cell scratch core, this gives 660 analysis cells: 50
verified Stage 1 reuse cells plus 610 new launches.

## Intron Mask Strata Are Not Ground-Truth Subset Labels

The supplied masks are:

```text
mask 1: GTRHKHNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNYHYNYYYYYYYYYYYYYYYYYNYAG
mask 2: GTNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNAG
mask 3: NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
```

They are nested sequence spaces: every mask-1 match also matches mask 2, and
every 80-nt A/C/G/T sequence matches mask 3. Sequence alone therefore cannot
recover the original oligo-pool membership. All analysis and tables must use
the term **inferred mask stratum**, never original subset, synthesis pool, or
ground-truth class.

Use deterministic most-specific-first precedence:

```text
inferred_mask1 = matches mask 1
inferred_mask2 = matches mask 2 and not mask 1
inferred_mask3_residual = all remaining exact-80 sequences
```

If true pool membership becomes available later, rerun the diagnostic tables
with that field and preserve this inferred analysis as a sensitivity check.

### Observed Non-Audit Stratum Structure

| Inferred stratum | Constructs | Median barcodes | HQ constructs | Mean log2 target |
|---|---:|---:|---:|---:|
| mask 1 | 2,601 | 4 | 374 | 2.769 |
| mask 2, not mask 1 | 2,664 | 4 | 365 | 1.996 |
| mask 3 residual | 2,318 | 4 | 322 | 1.893 |

This descriptive table excludes the 265 frozen audit rows. Barcode
distributions are similar, while target distributions are shifted but
overlapping.

All model metrics below are from the 213 fold-0 validation predictions of
Stage 1 leader `zho9ew6n` (81/60/72 rows by stratum), not from training and not
from five-fold OOF predictions:

```text
pooled Pearson                         0.778
pooled within-stratum-centered Pearson 0.472
mask-1 Pearson                         0.575
mask-2-not-1 Pearson                   0.488
mask-3-residual Pearson                0.212
fold-0-training-fitted mean baseline   0.703
validation/oracle mean baseline        0.703
```

Within-stratum centering subtracts the validation target mean and prediction
mean separately inside each inferred stratum before pooling the 213 residual
pairs for Pearson. It removes credit for the between-stratum offsets.

The validation/oracle baseline uses the validation stratum target means and is
only an explanatory decomposition. The honest baseline estimates those means
from the fold-0 training rows and obtains Pearson `0.7026`, essentially the
same as the oracle `0.7029`. Approximately 49.4% of validation target variance
is between stratum means and 50.6% is within strata; this describes target
variation, not model accuracy. These results support—but do not prove—the
hypothesis that the epoch-1 checkpoint rapidly learns design-family offsets.
Per-epoch stratum predictions would be required for a mechanistic claim.

See the executed
[`02_intron_inferred_mask_strata_analysis.ipynb`](../../../tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/02_intron_inferred_mask_strata_analysis.ipynb)
for formulas, target-distribution plots, provenance assertions, and the
Stage 2 OOF contract.

### Split Decision

Do not regenerate the five frozen development folds. Their inferred-stratum
validation counts are sufficiently populated:

| Fold | mask 1 | mask 2, not 1 | mask 3 residual |
|---:|---:|---:|---:|
| 0 | 81 | 60 | 72 |
| 1 | 70 | 85 | 57 |
| 2 | 73 | 81 | 58 |
| 3 | 70 | 69 | 73 |
| 4 | 80 | 70 | 62 |

The same assignments remain paired across RC modes. Do not inspect inferred
audit-stratum performance until the final audit stage.

## Intron Stage 2 Estimands And Selection

Keep pooled OOF validation Pearson as the primary score. This preserves the
original plan, the intended mixed-library deployment estimand, and reuse of
the ten matching Stage 1 fold-0/RC-off cells.

Mandatory Intron robustness outputs are:

- pooled OOF Pearson, Spearman, RMSE, and COD R2;
- pooled within-inferred-stratum-centered Pearson;
- macro mean of the three inferred-stratum Pearsons;
- minimum inferred-stratum Pearson;
- per-stratum sample count, target/prediction means, bias, RMSE, MAE, Pearson,
  Spearman, COD R2, and raw-scale calibration slope/intercept;
- the fold-training-fitted stratum-mean baseline, with any
  validation/oracle explanatory baseline labeled separately;
- best epoch and learning curves for pooled and stratum metrics.

Build each OOF table by concatenating the five validation exports and require
one—and only one—prediction for each of the 1,061 Intron development construct
IDs. The raw calibration regression is
`log2_RNA_DNA = intercept + slope * prediction_raw`. RC comparison is valid
only after asserting that the RC-off and RC-on OOF construct-ID sets and raw
targets are identical.

Amend the ten-config Intron nomination composition to:

```text
6 highest pooled fold-0 validation-Pearson configs
2 maximin hyperparameter-diverse configs from the top pooled quartile
1 remaining top-quartile config with highest within-stratum-centered Pearson
1 remaining top-quartile config with highest minimum-stratum Pearson
```

The inferred metrics are robustness guards, not claims about true experimental
subsets. For a near tie in pooled OOF Pearson (absolute difference <= 0.01),
prefer the config with higher within-centered and minimum-stratum performance.

An RC-on policy may become the Intron default only if it meets the original
pooled criterion and also has nonnegative mean paired change in
within-stratum-centered Pearson with no more than two of five negative folds.

## Core Stage 2 Config Count

Retain ten core exact scratch configs per part. The number is justified as a
fixed screening budget, not an asserted universal optimum. Publication
analysis must report sensitivity of the config/policy conclusions to the top
6, top 10, and top 15 validation-ranked configs where available.

Do not add newly sampled scratch configurations before Stage 2. Stage 1 shows
broad plateaus for Promoter, Intron, and 5'UTR. Enhancer and 3'UTR have isolated
leaders, but searching the same fold again would increase model-selection
overfitting risk before fold stability is known.

## Bounded Challenger Lanes

Challengers supplement the 50-config core and are reported separately. They
do not retroactively enter the 885-row Stage 1 population or replace its
results.

`analysis_lane` is a provenance/reporting-stratum label, not a quality grade
or ranking. In particular, `challenger` does **not** mean inferior: a
challenger may outperform the core and become the selected route. It means
only that the candidate came from a route not screened in the Stage 1 scratch
population, so it must not be silently counted as Stage 1-selected evidence.
The only allowed values are:

```text
core_scratch
enhancer_transfer_challenger
utr3_utrbasset_challenger
```

Manifest, W&B, registry, prediction, and result rows must carry this field.
Challenger rows additionally carry their exact policy/base-config ID,
initialization, architecture or source head, unfreezing scope where applicable,
input policy, fold, RC mode, and RC-pair ID.

Use one Stage 2 W&B project per part:

```text
<part>__bashor_in_house__dedup_exact_v1__stage2_development
```

Within each part project, use `analysis_lane` as the W&B group. This keeps all
directly comparable routes in the correct part/data-stage location while
making core versus challenger provenance visible and filterable.

### Enhancer Transfer Challenger

Use the canonical BODA2/Malinois initialization already exercised by the
repository's Lib1 Enhancer fine-tuning workflows. The canonical evidence is
the May 15 HQ8 analysis of 135 fixed policies across eight seeds. It ranked
policies by **mean validation Pearson across seeds**, not by test performance.
That study varied source head, unfreezing scope, and training threshold/size;
it fixed the LR/optimizer/trainer recipe. Earlier completed transfer work
covered only a modest `2 x 2` head/backbone-LR grid. It is therefore broader
than one lucky run but not a completed current-protocol broad transfer-
optimizer sweep.
Its overall winner was:

```text
BassetBranched pretrained model; HepG2 source head
train threshold 1; full eligible pool
head learning rate 5e-4; backbone learning rate 1e-4
AdamW; weight decay 1e-4; batch size 256; no scheduler
two branch/output warm-up epochs; then full unfreezing
unweighted MSE; historical run used RC augmentation
mean validation Pearson 0.5144 (SE 0.0095; 8 seeds)
```

Do not run a new broad transfer Bayesian sweep solely because of deduplication.
The five scratch-Enhancer calibration pairs had mean dedup-minus-pre-dedup
validation Pearson `-0.014`, range `-0.047` to `+0.005`, and rank Spearman
`0.90`. This is only indirect transfer evidence, so use a bounded source-head
sensitivity portfolio.

The frozen choice is `N=2` policies sharing the selected
optimizer/LR/trainer recipe:

```text
policy A = HepG2 source head
policy B = K562 source head
```

For threshold 1/full pool/full scope, historical validation Pearson was
`0.5144 +/- 0.0095` for HepG2 and `0.5110 +/- 0.0125` for K562. K562 was
slightly ahead under the same data policy for the two narrower scopes. Cross
each approved source-head policy with three previously studied adaptation
scopes:

```text
branched_only = pretrained branch plus output parameters
conv3_plus    = conv3, linear, branch, and output parameters
full          = every parameter
```

The corresponding historical HepG2 mean validation Pearsons were 0.4806,
0.5042, and 0.5144. Cross the resulting fully specified transfer configs with
the same five development folds and RC off/on using the exact-dedup target,
model seed 1701, original-orientation validation, and no audit loader:

```text
N policies x 3 scopes x 5 folds x 2 RC modes = 30N challenger cells
approved N=2                                         = 60 cells
```

Here scope is already part of a fully specified transfer config. `N=2` refers
to two source-head policies—HepG2 and K562—each crossed with every scope. It
does not mean two arbitrary training runs or two scopes.

Do not count a lower-backbone-LR sensitivity as a full third policy across all
scopes. If later justified, backbone LR `1e-5` under HepG2 adds only
`conv3_plus` and `full` (20 cells), because backbone LR is inactive for
`branched_only`.

The downstream RC arm is a controlled fine-tuning factor even if the source
pretraining procedure used augmentation. Stage 2 must use validation Pearson
for checkpointing, even though the historical implementation restored by
validation MSE. The historical input is 600 nt with fixed MPRA flanks, whereas
the scratch core uses a 216-nt neutral-padded representation; freeze and label
that transfer-specific input policy rather than claiming identical inputs.
Historical pre-dedup/log10/random-HQ8 scores are prior evidence only.

### 3'UTR Architecture Challenger

Do not start a new scratch sweep. Use only the 128 completed configurations
from the standardized full-orchestrator June sweep `nhoh1zuw` in project
`utr3__bashor_in_house__threeprime_modal100__scratch__utr_bassetvl_fp32`.
Do not mix in the earlier 32-run sweep or the later eight-policy replicated
factorial. Resolve `K` exact UTRBassetVL base configs from that source. The
frozen choice is `K=10`:

```text
5 historical validation leaders
5 maximin hyperparameter-diverse representatives from the historical top quartile
```

"Resolve" means load each completed sweep YAML plus its authoritative run
config, fill defaults, construct the trainer-inclusive base identity, remove
data/split/model-seed/RC/loss/output fields, hash and deduplicate the identities,
then freeze the selected base IDs, source run IDs, full snapshots, and hashes.
The deterministic resolver froze these exact source runs and normalized base
config identities:

| Selection reason | Source run | Frozen base config ID | Historical rank |
|---|---|---|---:|
| leader 1 | `utc3cqzn` | `basecfg_86969bcf79247695d2c27ce1466d4eab2373e5e1f3645da99f24ebf4c59c0fbe` | 1 |
| leader 2 | `r8gx494e` | `basecfg_6b80f0ea2299b4022a3de4766d8b773ac05c788bff651c5abe613d4f90ee6942` | 2 |
| leader 3 | `dx4cw1l9` | `basecfg_acad344836c3bebfa5d9e9494a849554170fd4d47038109355e6412963b07e19` | 3 |
| leader 4 | `11g559xo` | `basecfg_3a58569da5db177022c965a78461075f25af834e799bf907583e42ab6b15c817` | 4 |
| leader 5 | `v0xdcm0y` | `basecfg_4c3c7c4733d1d0a0fb74b5640cd427c87722d26c003988c8dd1c0d22d2a7e2e7` | 5 |
| top-quartile maximin 1 | `h5hkkd86` | `basecfg_b9591779c2fde4fb3b15c10bc107bc65c8975f7875adcbda70c4a3378247355b` | 29 |
| top-quartile maximin 2 | `okhto5as` | `basecfg_4f095c0ca69caf8c2283bafaf5335aff0a14055290375f2a78a2d43d3c99094d` | 22 |
| top-quartile maximin 3 | `9kneglhi` | `basecfg_8b0c2f56d32964a95bc54cef6096d49d6e2edf1292993f8a7d018a5fc88616b8` | 15 |
| top-quartile maximin 4 | `jfzrac53` | `basecfg_5c703c80c70c38b50f7b85bc462e75c7158bfad29852a346555ada3f493c46f8` | 13 |
| top-quartile maximin 5 | `zwf5cj86` | `basecfg_09d1c2648318dc9d476a10f92a7ef27a6e22d11e55d8360cae0af1901475ed46` | 16 |

The ordered-selection digest is
`b5f3e773496a72759d9df4b6c9010f8fbc0e6bac712126843135915b8e6996ef`.
Historical ranks and validation scores are selection priors, not July
performance estimates.

Replay those exact configs on the dedup product across five folds and both RC
modes:

```text
K configs x 5 folds x 2 RC modes = 10K challenger cells
approved K=10                              = 100 cells
```

Keep Hani 240-nt 3'UTR transfer outside this amendment because the in-house
insert is exact 100 nt. Padding/context placement must be resolved in a
separate feasibility protocol before that pretrained model is comparable.

### Revised Stage 2 Accounting

```text
core analysis:       5 parts x 10 configs x 5 folds x 2 RC = 500 cells
Enhancer challenger: N policies x 3 scopes x 5 folds x 2 RC = 30N cells
3'UTR challenger:    K configs x 5 folds x 2 RC              = 10K cells
total analysis table                                = 500 + 30N + 10K
```

The core reuses 50 Stage 1 cells and launches 450 new cells. Challengers have
no matching Stage 1 cells. The frozen `N=2`, `K=10` decision therefore gives
660 analysis cells and 610 new launches.

## Gate For Any Later Targeted HPO

After Stage 2, approve a new targeted search only when at least one condition
is documented from pooled OOF validation:

- the same low-learning-rate or low-weight-decay boundary trend repeats across
  folds;
- rankings are unstable and none of the core configs is consistently
  competitive;
- a challenger architecture wins reproducibly but its existing configs show
  systematic undertraining or boundary saturation;
- learning histories indicate an optimizer/trainer deficiency rather than
  irreducible validation noise.

If triggered, use a bounded 20-30-config targeted search for the affected part,
not another broad replay. Freeze its search space before launch and keep the
audit set unavailable.

## Calibration Diagnostic

The 25 pre-dedup mates are sufficient for the local diagnostic purpose. Report
the result descriptively as a modest, part-dependent change in fitted model
behavior. Do not make a universal dedup-improves-performance claim, and do not
add more calibration runs unless this comparison becomes a publication
endpoint.

## Reproducible Evidence

The implementation and generated evidence live in:

```text
src/analysis/lib1_dedup_stage1_analysis.py
src/analysis/lib1_dedup_stage2_analysis.py
src/learn/generate_lib1_dedup_stage2_manifest.py
src/learn/verify_lib1_dedup_stage2_manifest.py
src/learn/launch/lib1_dedup_stage2_orchestrator.sh
src/learn/data_manifests/splits/lib1_enhancer_dedup_exact_v1_transfer_mpra600_split.json
src/learn/outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl
src/learn/outputs/hpo_manifests/lib1_dedup_stage2_july2026__run_manifest.jsonl
src/learn/outputs/hpo_manifests/lib1_dedup_stage2_july2026__stage1_reuse_cells.jsonl
src/learn/outputs/hpo_manifests/lib1_dedup_stage2_july2026__utr3_utrbassetvl_selected_configs.jsonl
src/learn/outputs/hpo_manifests/lib1_dedup_stage2_july2026__summary.json
tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/01_exact_replay_selection.ipynb
tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/02_intron_inferred_mask_strata_analysis.ipynb
tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/outputs/
```

The generated analysis manifest is the complete 660-cell table. The run
manifest contains only the 610 new launches; the separate reuse manifest binds
the other 50 cells to completed Stage 1 evidence. Regenerate them only through
the deterministic generator and review the summary hashes before launch.
