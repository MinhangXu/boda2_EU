# Start Here: Lib1 Dedup Stage 1 to Stage 2 Reader Guide

Updated: 2026-07-13

This is the human-readable guide to the frozen Stage 2 design. It is the first
substantive document to read after the campaign `README.md`. It explains the
terminology, what Stage 1 did, what is core versus challenger evidence, the
Enhancer transfer result, the 3'UTR architecture gap, the Intron masks, and the
conditions that would justify future HPO. The 2026-07-12 choices are `N=2`,
`K=10`, and inferred masks as sensitivity labels. All 660 Stage 2 analysis
cells completed on 2026-07-13, Stage 2 analysis is underway, and frozen-audit
evaluation has not launched.

## Document Map

Read in this order:

1. **This guide** — conceptual understanding and current decisions.
2. `lib1_dedup_phase1_hpo_rerun_plan_july2026.md` — formal full-campaign
   scientific contract.
3. `lib1_dedup_phase1_stage1_implementation_checks_july2026.md` — technical
   audit of data, manifests, W&B, and the completed Stage 1 launch.
4. `lib1_dedup_pre_stage2_protocol_amendment_july2026.md` — frozen formal
   amendment made after Stage 1, including exact challenger identities and OOF
   requirements.
5. `tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/01_exact_replay_selection.ipynb`
   — executable Stage 1 analysis and candidate tables.
6. [`02_intron_inferred_mask_strata_analysis.ipynb`](../../../tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/02_intron_inferred_mask_strata_analysis.ipynb)
   — dedicated fold-0 Intron-strata explanation, plots, and Stage 2 metric
   contract.

The long plan is deliberately comprehensive. Use this guide for discussion;
consult the other files when checking exact rules or provenance.

## The Campaign In One Picture

```text
Stage 1 (finished)
885 exact scratch configs on one fold, RC off, unweighted MSE
  -> compare configs on one common dedup protocol
  -> nominate 10 scratch configs per part

Stage 2 core (next)
5 parts x 10 configs x 5 folds x 2 RC modes = 500 cells
  -> test fold stability
  -> estimate RC-on minus RC-off in matched pairs

Bounded challengers (separate from core)
Enhancer: N transfer policies x 3 scopes x 5 folds x 2 RC    = 30N cells
3'UTR: K historical UTRBassetVL configs x 5 folds x 2 RC     = 10K cells

Frozen challengers: N=2 and K=10                             = 160 cells

Complete Stage 2 analysis table                              = 660 cells
Stage 1 cells reused / new launches                          = 50 / 610

Later
weighted loss -> frozen audit -> learning curves
```

## Vocabulary: Config, Cell, Core, And Challenger

### Base config

A base config is one complete model/training recipe before experimental factors
are crossed around it. It includes architecture, layer widths and kernels,
optimizer, learning rates, scheduler, batch size, and trainer/early-stop
settings.

For a transfer model, unfreezing scope is also part of the config because it
changes which parameters are trained.

### Cell

A cell is one actual training job:

```text
one base config
x one development-fold ID k from {0, 1, 2, 3, 4}
x one RC state
x one loss state
x one fixed model seed
= one cell/run
```

For the cell using fold `k`, that fold is the high-quality validation set.
Training uses the low-barcode/train-only rows plus the other four high-quality
development folds. The frozen audit rows are absent. Crossing a config/RC arm
with all five fold IDs makes every development construct held out exactly once;
concatenating those five validation predictions later produces OOF predictions.

### Core config

The core is the primary, predeclared Stage 2 experiment:

- 10 scratch configs selected per part from Stage 1;
- 50 total configs;
- every one was trained in Stage 1 on fold 0, RC off, unweighted MSE;
- every one is evaluated on five folds and both RC states in Stage 2.

Thus:

```text
50 configs x 5 folds x 2 RC states = 500 core cells
```

Stage 1 already supplies one matching cell per config:

```text
50 existing fold-0/RC-off cells
500 total - 50 reusable = 450 new core runs
```

Yes: all 500 core analysis cells originate from the 50 configs selected from
the Stage 1 broad exact-config replay. The 25 pre-dedup calibration runs are
not core configs.

### Challenger config

A challenger answers a route or architecture question that Stage 1 deliberately
did not include. Challengers are:

- separately labeled;
- not silently mixed into the core population;
- evaluated with the same current folds, target, seed, and RC pairing;
- supplementary evidence, not a retroactive rewrite of Stage 1.

"Separately labeled" is computational and analytical, not merely a plot
caption. The primary field is:

```text
analysis_lane = core_scratch | enhancer_transfer_challenger | utr3_utrbasset_challenger
```

This says **where the candidate came from and which claim it belongs to**. It
does not rank the candidate, and `challenger` does not mean inferior. A
challenger can beat the scratch core and become the selected modeling route;
the label prevents us from later claiming it was one of the 50 configs selected
by the Stage 1 scratch replay.

Supporting fields make each row self-describing:

```text
challenger_policy_id
initialization
source_head or architecture
scope
input_policy
fold, rc_mode, and rc_pair_id
```

They remain directly comparable with the appropriate core part under the same
target/folds/seed/RC design. They are shown both in lane-specific summaries and
in an explicitly labeled route comparison, never silently mixed into the
Stage-1-selected scratch ranking or reuse claim.

## Enhancer Transfer Learning: What The Most Recent Evidence Says

### Which run family is canonical?

The canonical decision surface is the May 15, 2026 HQ8 multiseed HPO analysis:

```text
135 fixed configurations x 8 seeds = 1,080 completed runs
```

Your interpretation is mostly right: this transfer study had fewer search axes
than the scratch architecture HPO. The 135 configs varied three pretrained
cell-type heads, three unfreezing scopes, and training threshold/size. They
held the head LR (`5e-4`), backbone LR (`1e-4`), AdamW/weight decay, batch size,
scheduler, and warm-up fixed. Earlier completed work covered only a modest
`2 x 2` head/backbone-LR grid; the repository does not contain a completed
current-protocol broad transfer-optimizer sweep.

This concerns the number of hyperparameters searched, not necessarily the
number of model parameters trained. `full` fine-tuning updates the entire
pretrained model; `branched_only` and `conv3_plus` update progressively fewer
layers.

The later May 18 run disabled early stopping for a narrow K562/branched-only
diagnostic. Validation performance degraded at late epochs, so it supports
keeping early stopping; it is not a newer winner.

### How was the winner chosen?

For every fixed policy, the analysis averaged validation Pearson across eight
seeds. It selected the highest **mean validation Pearson**. Test metrics were
shown only after selection and were not used to rank configs.

The winner was:

| Field | Selected value |
|---|---|
| Source model | BODA2/Malinois BassetBranched |
| Pretrained output head | HepG2 |
| Fine-tuning scope | full |
| Train eligibility | barcode threshold 1, full pool (~4,295 rows) |
| Head learning rate | `5e-4` |
| Backbone learning rate | `1e-4` |
| Optimizer | AdamW |
| Weight decay | `1e-4` |
| Batch size | 256 |
| Scheduler | none |
| Warm-up | 2 epochs training branch/output before deeper unfreezing |
| Loss | unweighted MSE |
| Mean validation Pearson | 0.5144 (SE 0.0095; 8 seeds) |
| Historical diagnostic test Pearson | 0.5066 (SE 0.0229) |

Under the same HepG2/head/LR/full-pool policy, scope results were:

| Scope | What trains | Mean validation Pearson |
|---|---|---:|
| `branched_only` | branch and output | 0.4806 |
| `conv3_plus` | conv3, linear, branch, output | 0.5042 |
| `full` | all parameters | 0.5144 |

`branched_only` is not literal head-only. It includes the pretrained branch
and output layers.

### Do we need a new broad transfer HPO after deduplication?

Not on the current evidence. The five matched scratch-Enhancer calibration
pairs had mean dedup-minus-pre-dedup validation Pearson `-0.014`, range
`-0.047` to `+0.005`, and pre/dedup rank Spearman `0.90`. This is a small
sample and is not proof that transfer-policy rankings are unchanged, but it
supports a bounded sensitivity portfolio rather than hundreds of new jobs.

The frozen choice is `N=2`, using the same optimizer/LR/trainer policy with two
historically near-tied source heads:

```text
policy A = HepG2 head, the overall historical winner
policy B = K562 head, a source-head sensitivity check
```

At threshold 1/full pool/full scope, historical validation Pearson was
`0.5144 +/- 0.0095` for HepG2 and `0.5110 +/- 0.0125` for K562. K562 was
slightly ahead under the same data policy for `branched_only` and
`conv3_plus`, so the source-head choice is unresolved within historical
uncertainty.

### Correct cell formula

The three scopes are already three fully specified transfer configs. If `N`
means an additional number of heads or LR policies crossed at every scope,
then the formula would be:

```text
N policies x 3 scopes x 5 folds x 2 RC modes = 30N cells
```

Thus the approved bounded lane is:

```text
N=2 source-head policies x 3 scopes x 5 folds x 2 RC modes = 60 cells
```

This is not a broad HPO: it asks whether the historical head choice and scope
survive the new data/split protocol. An optional later sensitivity would add
backbone LR `1e-5` only for `conv3_plus` and `full` under HepG2, adding 20
unique cells. It should not be naively counted as another all-scope policy,
because backbone LR has no effect when only branch/output parameters train.

### Why must it be rerun?

The old transfer score is prior evidence, not directly comparable with Stage 1:

- old data were pre-dedup;
- old target was log10 ratio followed by train-only standardization;
- splits were seed-specific random HQ8 validation/test sets;
- input was a 600-nt BODA representation with fixed MPRA flanks;
- historical checkpoint restore used validation MSE, although configs were
  ranked by validation Pearson.

The challenger must use current exact-dedup targets, frozen folds, no audit
loader, model seed 1701, and validation-Pearson checkpointing. The 600-nt flank
representation is architecture-required and must be labeled; it is not the
same representation as the 216-nt neutral-padded scratch input.

Evidence:

- `tutorials/lib1_tasks/fine_tuning/enhancer_finetune_w_boda_pretrain/may15_2026_hq8_multiseed_hpo_analysis.ipynb`
- `src/finetune/learning_curve/lib1_enhancer_threshold_hq8_random_mixed_b2_allheads_8seed_absgrid_may2026/analysis/overall_mean_validation_selected_config.csv`
- `src/finetune/learning_curve/lib1_enhancer_branched_only_k562_hq8_4seed_noearly_250epoch_may2026/analysis/branched_only_noearly_best_vs_final_summary.csv`

## 3'UTR UTRBassetVL: Why Stage 1 Did Not Cover It

Stage 1 intentionally approved only the 3'UTR ResNet1D source lane. This was a
prior architecture-scope decision: June standardized evidence favored ResNet,
and the plan prioritized fold replication over replaying every architecture.
The generator therefore hard-coded 3'UTR ResNet only.

The Stage 1 3'UTR core source contains:

```text
127 completed broad ResNet configs
+ 30 completed June outer ResNet configs
= 157 Stage 1 ResNet replays
```

There are zero 3'UTR UTRBassetVL rows in Stage 1. Therefore any proposed
UTRBassetVL challengers do not overlap the 10 ResNet core configs.

### What completed UTRBasset evidence exists?

| Historical source | Runs | Unique configs | Role |
|---|---:|---:|---|
| earlier broad sweep `k0g0eh19` | 32 | 32 | early diagnostic |
| standardized broad sweep `nhoh1zuw` | 128 | 128 | clean challenger source pool |
| focused factorial `wt68uekw` | 128 | 8 | replicated diagnostic grid |

All are old/pre-dedup 100-nt in-house runs. None has been trained on the July
dedup target/current folds.

### What does "resolve K exact configs" mean?

It does not mean invent new hyperparameters. It means:

1. Take the 128 completed `nhoh1zuw` run records.
2. Load each sweep YAML and the authoritative full run config.
3. Fill historical defaults so every recipe is complete.
4. Keep model, optimizer, scheduler, batch, and trainer fields.
5. Remove data path, old split, model seed, RC, loss, and logging/output fields.
6. Hash that normalized recipe into `base_config_id` and verify 128 unique IDs.
7. Select a predeclared number of historical validation leaders.
8. Select the remaining configs by maximin diversity within the historical top
   quartile.
9. Freeze all `K` IDs, source run IDs, snapshots, and hashes before launch.

The frozen `K=10` set contains five historical validation leaders and five
maximin-diverse top-quartile representatives:

```text
leaders:  utc3cqzn, r8gx494e, dx4cw1l9, 11g559xo, v0xdcm0y
maximin:  h5hkkd86, okhto5as, 9kneglhi, jfzrac53, zwf5cj86
```

Its ordered-selection digest is
`b5f3e773496a72759d9df4b6c9010f8fbc0e6bac712126843135915b8e6996ef`.
The amendment and generated selected-config artifact contain the full normalized
base-config IDs and snapshots. The old validation numbers are ranking priors
only; they are not July performance estimates.

The general cost is:

```text
K UTRBassetVL configs x 5 folds x 2 RC modes = 10K new cells
K=10 -> 100 cells (approved)
```

We are not adding public Hani 3'UTR transfer because its 240-nt input is not
directly compatible with the exact 100-nt in-house inserts.

## Intron Masks From First Principles

The executed visual analysis is
[`02_intron_inferred_mask_strata_analysis.ipynb`](../../../tutorials/lib1_tasks/pretrain_CRE_inhouse_data/dedup_phase1_rerun_july2026/02_intron_inferred_mask_strata_analysis.ipynb).

### What do IUPAC letters mean?

| Symbol | Allowed DNA base(s) |
|---|---|
| `A`, `C`, `G`, `T` | that exact base |
| `N` | A, C, G, or T |
| `R` | A or G |
| `Y` | C or T |
| `H` | A, C, or T |
| `K` | G or T |

A construct matches a mask when all 80 positions satisfy the allowed letter
at that position.

For mask 2:

```text
GT + 76 arbitrary bases + AG
```

the construct must start `GT` and end `AG`; every middle base is allowed.

For mask 1, the beginning and a long pyrimidine-rich region near the end are
constrained. For example, this observed construct matches mask 1:

```text
GTAAGAAAAACTGGGGTGCCTTGCCGGCGAGGTCCGCCGGGATTCTTGTCCAGTACTTACCTTTTTTTCTCCTCCCGCAG
```

It starts with bases allowed by `GTRHKH`, obeys the later `Y/H` constraints,
and ends in bases allowed by `NYAG`.

### Why can we match masks without a membership map?

We can test **sequence compatibility** with each mask. We cannot recover the
true synthesis-pool membership because the masks are nested:

```text
every mask-1-compatible sequence is also mask-2-compatible
every 80-nt sequence is mask-3-compatible
```

An unconstrained mask-3 oligo can also happen by chance to start `GT` and end
`AG`. Therefore our labels are explicitly inferred sequence strata, assigned
most-specific-first:

```text
stratum 1 = compatible with mask 1
stratum 2 = compatible with mask 2 but not mask 1
stratum 3 = residual exact-80 sequences
```

These are useful sequence categories. They are not claims about which pool a
construct physically came from.

### Observed non-audit dataset structure

| Inferred mask stratum | Constructs | Median barcodes | HQ constructs | Mean log2 target |
|---|---:|---:|---:|---:|
| mask 1 compatible | 2,601 | 4 | 374 | 2.769 |
| mask 2 but not mask 1 | 2,664 | 4 | 365 | 1.996 |
| residual | 2,318 | 4 | 322 | 1.893 |

This table uses 7,583 non-audit constructs. The barcode distributions are
almost the same; the major difference is the shifted, overlapping activity
distributions. The earlier all-7,848 structural table included 265 frozen
audit rows descriptively. The dedicated notebook excludes those targets so
its scope cannot be confused with model evaluation.

### Why do we care?

All following values come from the Stage 1 leader `zho9ew6n` on its 213
held-out **fold-0 validation** predictions: 81 Mask-1-compatible, 60
Mask-2-not-1, and 72 residual. They are not training metrics and are not yet
five-fold OOF metrics.

```text
pooled validation Pearson             0.778
mask-1-compatible Pearson             0.575
mask-2-not-1 Pearson                  0.488
residual Pearson                      0.212
within-stratum-centered Pearson       0.472
training-fitted stratum-mean baseline 0.703
validation/oracle explanatory baseline 0.703
```

#### Within-stratum-centered Pearson

For each validation row, subtract its stratum's validation mean from the
observed target and separately subtract the stratum's mean prediction from the
model prediction. Pearson correlation across those 213 residual pairs is
`0.472`. This removes credit for merely placing the three strata at different
average activity levels. It asks whether a construct predicted above its own
stratum average is also observed above its own stratum average. It is not the
simple average of the three per-stratum correlations.

#### Stratum-mean-only baselines

The old `0.703` label needs qualification. The explanatory/oracle version
assigns each validation row its own validation stratum's observed mean. Because
those means use validation targets, it explains the decomposition but is not a
deployable predictor. The honest counterpart estimates the three means from
the 7,370 fold-0 training rows and applies them to validation; its Pearson is
`0.7026`, almost identical to the oracle value `0.7029`.

#### What the 49% statement means

On fold-0 validation, raw log2 target means are `2.874`, `2.043`, and `1.901`
for the three strata. The total target spread can be decomposed into spread of
those three means around the grand mean plus construct-to-construct spread
inside each stratum:

```text
total variance = between-stratum variance + within-stratum variance
       100.0%  =            49.4%       +          50.6%
```

This is not “49.4% of model accuracy.” It means that replacing every target
with one of only three stratum averages retains almost half of the target's
total squared variation. Correspondingly, `sqrt(0.494) = 0.703`, the
validation/oracle mean-only correlation.

A CNN does not need a supplied stratum column. Sequence filters can recognize
the `GT...AG` boundaries and the constrained/pyrimidine-rich Mask-1 pattern;
later layers can associate those recognizable features with three output
offsets. The leader's mean predictions (`3.020`, `2.050`, `1.940`) closely
track the three observed means. This can create a high pooled score quickly.
The epoch-1 mechanism remains a hypothesis, however, because Stage 1 retained
only final-checkpoint—not per-epoch stratum—predictions.

This is not train/validation leakage: every fold contains all three strata.
It is an estimand issue. Pooled Pearson answers, "Can the model rank the mixed
library?" It does not by itself answer, "Can the model rank two sequences from
the same design family?"

If we ignore the strata:

- pooled performance can be mistaken for within-family sequence understanding;
- config ranking can favor fast design-family classification;
- an RC benefit could be driven by only one stratum;
- a publication could overstate generalization to new intron designs.

We can still proceed. Pooled OOF Pearson remains primary, preserving Stage 1
reuse, while Stage 2 must also report within-stratum-centered, macro-stratum,
minimum-stratum, and per-stratum calibration metrics. The two complementary
Intron candidate slots now explicitly preserve within/minimum-stratum evidence.

### Frozen Intron OOF and RC contract

For each Intron config and each RC arm, concatenate the five held-out-fold
exports into one table with exactly one row for each of the **1,061** Intron
development constructs. `1,061` is Intron-specific, not the number expected
for every part. The corresponding exact OOF counts are Enhancer 979, Promoter
1,545, 3'UTR 525, and 5'UTR 1,438.

Use the raw columns `log2_RNA_DNA` and `prediction_raw`. Pooled five-fold OOF
Pearson is primary. For Intron, also compute OOF within-stratum-centered,
macro-stratum, minimum-stratum, and per-stratum calibration metrics. Before
computing any RC effect, assert that RC off/on contain identical construct IDs
and raw targets, then compare their paired predictions/errors. The inferred
masks are accepted sensitivity labels only. No frozen-audit loader may be
instantiated or scored until the final audit stage.

## Optimization Terms In Plain Language

### Search boundary

A hyperparameter search samples inside a declared range. For example:

```text
3'UTR learning rate range: 3e-5 to about 1.1e-3
```

`3e-5` and `1.1e-3` are the lower and upper boundaries.

### Boundary trend

A boundary trend means performance tends to improve as a parameter moves
toward one edge. In Stage 1, several top Enhancer/3'UTR configs used learning
rates or weight decay near the low edge.

Yes, this can justify expanding the search below the old lower bound—but not
from fold 0 alone. At this point it is a hypothesis that Stage 2 is designed to
check.

### "The trend repeats across folds"

One fold can produce a lucky winner. We call it repeated only if, on folds
1–4 as well as fold 0:

- low-edge configs repeatedly outrank middle/high values;
- the relationship has the same direction;
- it is not explained by one failed or collapsed config.

Only then is there evidence that the useful region may extend below the old
boundary. If confirmed, extend only the implicated parameter edge in a bounded
20–30-config targeted search; do not reopen the entire broad search space.

### Boundary saturation

Boundary saturation is stronger than one boundary winner. It means many of
the best configs pile up at the lowest/highest allowed value, with few good
interior configs. This suggests the search may have truncated the optimum.

### Optimizer/trainer deficiency versus validation noise

Evidence more consistent with an optimizer/trainer problem:

- train and validation metrics are still improving at the epoch limit;
- many best epochs occur at the maximum allowed epoch;
- loss oscillates or diverges in a repeatable LR-dependent way;
- many configs are dead/constant, while a coherent LR/scheduler change fixes them;
- both training and validation remain poor, suggesting underfitting.

Evidence more consistent with finite-sample/measurement noise:

- training fit is strong but validation fluctuates around a stable plateau;
- many materially different configs have overlapping bootstrap intervals;
- ranking changes across folds without a repeatable hyperparameter direction;
- residual error is concentrated in lower-reliability targets or small strata;
- more epochs improve training but not held-out performance.

Neither can be proven from one curve. Fold replication, seeds, barcode-quality
analysis, and learning histories together distinguish the explanations.

The May 18 Enhancer no-early-stop diagnostic is a concrete trainer example:
training continued improving while validation/test performance degraded after
their best epochs. That supports early stopping; it does not support simply
training longer.

## When Would We Run More HPO?

Do not run another broad scratch sweep before Stage 2. Approve a bounded
20–30-config targeted search for a part only if Stage 2 shows at least one of:

1. A low/high boundary relationship repeats across folds.
2. Existing rankings are unstable and no config is consistently competitive.
3. A challenger architecture wins reproducibly but its configs show a coherent
   undertraining/boundary problem.
4. Learning histories identify a fixable optimizer/trainer limitation rather
   than a held-out plateau.

The frozen audit remains unavailable throughout that decision.

## Current Run Accounting

```text
core:                  500 cells, 450 new
Enhancer transfer:     30N cells, all new
3'UTR UTRBassetVL:     10K cells, all new
-------------------------------------------------
total analysis:        500 + 30N + 10K
new launches:          450 + 30N + 10K
```

The frozen `N=2`, `K=10` design gives exactly 660 analysis cells and 610 new
launches:

```text
core scratch:                  500 analysis cells, 450 new
Enhancer transfer challenger:  60 analysis cells,  60 new
3'UTR UTRBassetVL challenger: 100 analysis cells, 100 new
----------------------------------------------------------
total:                         660 analysis cells, 610 new
```

The challengers are valuable but separately labeled. If compute must be cut,
protect the five-fold core first; reduce challenger breadth before reducing
fold coverage.

## What "No Audit Loader" Means

An audit loader is the PyTorch/DataModule data loader containing the frozen
high-barcode audit rows. “No audit loader in Stage 2” is stronger than “do not
look at an audit metric”: each selection job receives only its training loader
and the selected development-fold validation loader. Audit rows are not passed
to the model, so per-epoch or post-fit audit predictions cannot occur.

The launcher must therefore keep `evaluate_test_after_fit=false`, request only
train/validation epoch metrics, and export only validation predictions. An
audit loader is created only later for the one-time final audit after the
config, RC, loss, and epoch policies are frozen. Descriptive inspection of a
raw dataset table is not itself a model audit loader; the new Intron notebook
nevertheless excludes audit targets from its distribution panels to keep the
scopes unambiguous.

## Implementation And Frozen Products

The deterministic Stage 2 implementation is organized as follows:

```text
src/learn/generate_lib1_dedup_stage2_manifest.py
  -> src/learn/outputs/hpo_manifests/lib1_dedup_stage2_july2026__analysis_manifest.jsonl
  -> src/learn/outputs/hpo_manifests/lib1_dedup_stage2_july2026__run_manifest.jsonl
  -> src/learn/outputs/hpo_manifests/lib1_dedup_stage2_july2026__stage1_reuse_cells.jsonl
  -> src/learn/outputs/hpo_manifests/lib1_dedup_stage2_july2026__utr3_utrbassetvl_selected_configs.jsonl
  -> src/learn/outputs/hpo_manifests/lib1_dedup_stage2_july2026__summary.json

src/learn/verify_lib1_dedup_stage2_manifest.py
  -> checks cell accounting, pair structure, frozen split/input policies, and
     absence of audit/test evaluation before launch

src/learn/launch/lib1_dedup_stage2_orchestrator.sh
  -> launches only the 610 new rows from the hash-locked run manifest

src/analysis/lib1_dedup_stage2_analysis.py
  -> assembles exact OOF products, computes the primary/robustness metrics,
     and performs construct-paired RC comparisons after runs finish
```

Direct links: [generator](../../../src/learn/generate_lib1_dedup_stage2_manifest.py),
[verifier](../../../src/learn/verify_lib1_dedup_stage2_manifest.py),
[orchestrator](../../../src/learn/launch/lib1_dedup_stage2_orchestrator.sh),
[analysis](../../../src/analysis/lib1_dedup_stage2_analysis.py), and
[frozen summary](../../../src/learn/outputs/hpo_manifests/lib1_dedup_stage2_july2026__summary.json).

The Enhancer transfer lane uses the derivative, assignment-preserving
`src/learn/data_manifests/splits/lib1_enhancer_dedup_exact_v1_transfer_mpra600_split.json`.
It changes only the architecture-required input representation to the frozen
600-nt MPRA-flank policy; fold/development/audit assignments remain bound to
the canonical Enhancer split.

## Frozen Pre-Launch Checklist

The 2026-07-12 review resolved the decisions required to implement Stage 2:

- [x] Confirm the distinction between a base config and a training cell.
- [x] Confirm 10 core scratch configs per part.
- [x] Approve bounded Enhancer `N=2`: HepG2 and K562, without opening a broad
      transfer sweep.
- [x] Accept UTRBassetVL as a separately reported 3'UTR architecture
      challenger rather than part of the ResNet core.
- [x] Freeze `K=10` for 3'UTR UTRBassetVL: five leaders plus five maximin
      representatives with the recorded selection digest.
- [x] Approve the explicit `analysis_lane` manifest, W&B, and result labels as
      provenance/reporting categories, not quality ranks.
- [x] Accept inferred Intron mask strata as sensitivity labels, not true pool
      membership.
- [x] Confirm pooled Pearson remains primary for Intron, with mandatory
      within/macro/minimum-stratum reporting.
- [x] Treat the dedicated notebook's fold-0 result as diagnostic; replace it
      with the specified five-fold OOF product for Stage 2 decisions.
- [x] Require identical construct-paired RC comparisons on raw predictions.
- [x] Confirm Stage 2 does not instantiate or score an audit loader.
- [x] Keep weighted-loss Stage 3 outside the Stage 2 launcher.

This is enough scientific specification to implement and dry-run Stage 2. It
does not authorize opening the audit data or automatically starting Stage 3.
