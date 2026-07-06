# Lib1 Barcode-Threshold Downsampling Plan

Generated: 2026-06-17

Status: implementation handoff plan for a non-enhancer Lib1 learning-curve
follow-up.

## Short Answer

Yes, keep an outer split seed in this experiment.

Use the same 5 split seeds as the outer-seed prior run:

```text
split_seed = [101, 202, 303, 404, 505]
```

Also randomize the downsampled training subsets, but do it reproducibly and
preferably as nested subsets within each `part x config x split_seed x barcode
threshold` pool. Do not take the first N rows from file order.

Recommended primary design:

```text
parts = Promoter, Intron, 3UTR, 5UTR
configs = top 5 robust outer-seed configs per part
barcode thresholds = train_min_barcodes in [1, 2, 3]
training sizes = 100, 500, 1500, 2500, 3500, full
loss = unweighted CNNBasicTraining
eval logging = train, val, test each epoch
heldout policy = val/test from n_barcodes >= 8, 250 val + 250 test
split seeds = 5 outer split seeds
model seed = fixed 1701 unless a later confirmation needs model-seed repeats
```

One important feasibility caveat:

- `3UTR` with `train_min_barcodes=3` has only 3,484 eligible train rows after
  the 250/250 high-barcode heldout split, so an exact `N=3500` arm is
  infeasible there. Treat `full` as the large-size arm for that cell and omit
  exact `N=3500`, or explicitly mark it as skipped.

With one downsample subset seed per cell, this is about 1,775 runs:

```text
4 parts x 5 configs x 5 split seeds x 3 thresholds x 6 size arms = 1,800
minus 25 skipped runs for 3UTR 3+ N=3500 = 1,775
```

## Context And Prior Results

Relevant notebooks:

- `tutorials/lib1_tasks/pretrain_CRE_inhouse_data/outer_seed_prior_lib1_orchestrator_jun15.ipynb`
  - Analyzes the robust outer split-seed HPO.
  - Source for top config selection.
- `tutorials/lib1_tasks/pretrain_CRE_inhouse_data/lib1_barcode_bin_matched_n1000_analysis_june2026.ipynb`
  - Analyzes exact barcode-bin training pools at matched `N=1000`.
  - Main result: decreasing barcode count hurts generalization in Promoter,
    3UTR, and 5UTR, while Intron is almost flat across barcode bins.
- `tutorials/lib1_tasks/pretrain_CRE_inhouse_data/lib1_inhouse_scratch_hpo_seed_split_diagnostics_june2026.ipynb`
  - Older seed/split diagnostics and residual-vs-barcode checks.

Relevant local result artifacts:

- `src/learn/outputs/hpo_analyses/lib1_outer_seed_prior_no_rc_june2026/outer_seed_config_summary.csv`
- `src/learn/outputs/hpo_analyses/lib1_outer_seed_prior_no_rc_june2026/outer_seed_run_table.csv`
- `src/learn/outputs/hpo_analyses/lib1_barcode_bin_accounting_june2026/train_pool_exact_barcode_counts_by_split_seed.csv`
- `src/learn/outputs/hpo_analyses/lib1_outer_seed_selected_barcode_weighted_june2026/paired_weighted_vs_unweighted_part_summary.csv`
- `src/learn/outputs/hpo_analyses/lib1_outer_seed_selected_barcode_weighted_june2026/paired_weighted_vs_unweighted_overall_summary.csv`

Weighted-loss context:

| Scope | Val delta | Test delta | Read |
|---|---:|---:|---|
| Overall | +0.0042 | +0.0016 | Tiny positive, CI crosses zero |
| Promoter | -0.0026 | +0.0055 | Basically neutral |
| Intron | +0.0064 | +0.0042 | Mildly positive |
| 3UTR | +0.0086 | -0.0123 | Val gain does not generalize cleanly |
| 5UTR | +0.0043 | +0.0089 | Best-looking part |

Takeaway: weighted loss is not the right variable to mix into this next
experiment. Keep loss unweighted so the run answers a clean sample-size and
barcode-threshold question.

## Scientific Question

The exact-bin run asked:

```text
If I train on exactly one barcode-count range at matched N=1000,
how much signal transfers to the n>=8 heldout set?
```

This threshold downsampling run should ask a different operational question:

```text
At matched training sizes, what minimum barcode-count threshold gives the best
heldout generalization, and how quickly does each threshold saturate as N grows?
```

This directly maps to future training policy:

- Should full training include every eligible row (`1+`)?
- Should we drop `n=1` rows and train on `2+`?
- Should we drop `n=1` and `n=2` rows and train on `3+`?
- Is Intron genuinely insensitive to barcode count, or did the matched-bin run
  look flat because of the chosen configs/N?
- Does the best threshold depend on training set size?

## What This Run Can And Cannot Claim

Can claim:

- Learning curves for `1+`, `2+`, and `3+` barcode thresholds.
- Matched-size comparisons of threshold policies at `N=100`, `500`, `1500`,
  `2500`, and usually `3500`.
- Operational value of keeping versus filtering low-barcode variants.
- Whether the `full` pool improves over matched fixed-N arms.
- Whether Intron remains a negative/control case where barcode threshold does
  not matter much.

Cannot claim by itself:

- Pure causal effect of barcode count. Barcode count may correlate with
  sequence class, GC, length, expression range, synthesis/assay recovery, or
  other library properties.
- Marginal value of adding only `n=1` rows to a fixed high-barcode base. A
  threshold policy changes both barcode quality and pool composition.
- Whether weighted loss rescues low-barcode examples. This run is intentionally
  unweighted.

If the marginal-addition question becomes the priority, run a separate additive
design later:

```text
HQ/high-threshold base + low-bin additions, matched total N,
with matched high-threshold-only controls
```

## Config Selection

Use top 5 configs per part from
`outer_seed_config_summary.csv`, ranked primarily by `rank_val_mean` from the
outer split-seed experiment. Test metrics are diagnostic only and should not be
used to choose configs.

Recommended selected configs:

| Part | Configs |
|---|---|
| Promoter | `promoter_cfg011`, `promoter_cfg029`, `promoter_cfg014`, `promoter_cfg018`, `promoter_cfg013` |
| Intron | `intron_cfg011`, `intron_cfg013`, `intron_cfg009`, `intron_cfg014`, `intron_cfg003` |
| 3UTR | `utr3_cfg001`, `utr3_cfg009`, `utr3_cfg003`, `utr3_cfg022`, `utr3_cfg011` |
| 5UTR | `utr5_cfg007`, `utr5_cfg005`, `utr5_cfg015`, `utr5_cfg008`, `utr5_cfg019` |

Why top 5 rather than top 3:

- The threshold/downsampling effect might interact with hyperparameters.
- The previous exact-bin run used 3 configs and was useful, but some bin effects
  looked noisy, especially 3UTR.
- Top 5 gives a better config-level uncertainty estimate while still avoiding a
  fresh HPO.

Critical note: these configs were selected under the original full eligible
training pool, not under every threshold/N condition. That is acceptable here
because the question is not "which hyperparameters are globally optimal for
every N"; it is "under robust existing configs, how does threshold policy
change learning curves?" If this run shows a strong threshold-specific winner,
a later narrow HPO at that policy may be warranted.

## Barcode Thresholds And Training Sizes

Use threshold pools:

| Label | Data module settings | Interpretation |
|---|---|---|
| `bc_ge1` | `train_min_barcodes=1`, no max | all eligible rows |
| `bc_ge2` | `train_min_barcodes=2`, no max | drop `n=1` rows |
| `bc_ge3` | `train_min_barcodes=3`, no max | drop `n=1` and `n=2` rows |

Use size arms:

```text
train_size_n in [100, 500, 1500, 2500, 3500]
full = train_size_n unset / None and train_size_frac=1.0
```

Feasibility after high-barcode val/test removal:

| Part | Threshold | Min eligible train rows across split seeds | N=3500 feasible? |
|---|---:|---:|---|
| Promoter | 1+ | 7,393 | yes |
| Promoter | 2+ | 6,311 | yes |
| Promoter | 3+ | 5,141 | yes |
| Intron | 1+ | 7,348 | yes |
| Intron | 2+ | 6,073 | yes |
| Intron | 3+ | 4,777 | yes |
| 3UTR | 1+ | 6,457 | yes |
| 3UTR | 2+ | 4,847 | yes |
| 3UTR | 3+ | 3,484 | no |
| 5UTR | 1+ | 7,831 | yes |
| 5UTR | 2+ | 6,552 | yes |
| 5UTR | 3+ | 5,261 | yes |

Implementation rule:

- Generate exact-N rows only when the threshold pool has at least N rows for
  every split seed.
- Generate a separate `full` row for every threshold.
- Do not silently clip `N=3500` to full. That creates a duplicate or
  near-duplicate condition with ambiguous labels.

## Outer Seed Policy

Keep outer split seeds.

Rationale:

- The previous outer-seed run was built precisely because a good config on one
  high-barcode val/test split may not be good on another.
- This experiment still evaluates on high-barcode val/test sets, so split
  difficulty remains a major variance source.
- Threshold pools are computed after removing the split-specific heldout rows,
  so the training pool also changes slightly with `split_seed`.
- Without outer seeds, a learning curve could be driven by one easy or hard
  heldout split.

The primary analysis should aggregate paired by:

```text
part x config_id x split_seed x train_size_label
```

Then compare threshold policies within the same matched slice.

If compute becomes a problem, the least-bad reduction is:

1. keep all 5 split seeds;
2. reduce configs from top 5 to top 3;
3. keep the full threshold/size design.

Dropping split seeds is worse than dropping configs because this run is meant
to support robust generalization claims.

## Downsampling Subset Policy

Use randomized downsampling, but make it deterministic and nested.

Why not file order:

- TSV order can encode preparation artifacts.
- Taking the first N rows is not a random training subset and can bias learning
  curves.

Why not fully independent random subsets at each N:

- If `N=100` and `N=500` are unrelated samples, apparent gains can partly be
  subset luck.
- Independent subsets make the learning curve noisier and make paired deltas
  less interpretable.

Recommended implementation:

For each `part x split_seed x threshold x downsample_seed`, create one
deterministic random permutation of the eligible training pool:

```text
eligible_ids = rows after heldout removal and train_min_barcodes filter
perm = deterministic_shuffle(eligible_ids, downsample_seed)
N=100  uses perm[:100]
N=500  uses perm[:500]
N=1500 uses perm[:1500]
...
full   uses all eligible_ids
```

This makes smaller training sets nested inside larger training sets for the
same threshold/split/downsample seed. It is the cleanest learning-curve
construction.

Primary run:

```text
downsample_seed = one deterministic seed per split_seed/threshold,
or a fixed seed such as 91001 plus split/threshold offsets
```

Optional variance audit:

- Add 2 extra downsample seeds only for `N=100` and `N=500`.
- Do this only if early results show large small-N instability or if compute is
  available.
- Do not multiply the full 1,775-run design by 3 unless the compute budget is
  explicitly approved.

## Implementation Requirements

The current data module already supports:

- `train_min_barcodes`
- `train_max_barcodes`
- `train_size_n`
- `train_sampling_mode=random`

Before this run, add or verify a cleaner downsampling control:

```text
train_subsample_seed or downsample_seed
```

Recommended behavior:

- default to `split_seed` for backward compatibility;
- use deterministic shuffled-prefix sampling for nested learning curves;
- store the selected row identifiers or a stable hash of them in provenance;
- write the pre- and post-subsample barcode histogram into split summaries.

If the current implementation uses `pandas.DataFrame.sample(n=N,
random_state=seed)` independently for each N, do not assume nested subsets.
Implement an explicit permutation and prefix selection.

## Existing Code To Reuse

The implementation should not start from a blank orchestrator. Reuse the
manifest/global-queue pattern already working for the outer-seed and
barcode-bin runs.

Best launcher reuse:

- `src/learn/launch/lib1_inhouse_outer_seed_prior_orchestrator.sh`
  - global GPU queue;
  - accepts any manifest JSONL via `MANIFEST_JSONL`;
  - supports `DRY_RUN`, `GPU_LIST`, `MAX_ROWS`, `ROW_RANGE`, `PARTS`,
    `CONFIG_IDS`, `SPLIT_SEEDS`, and `SKIP_COMPLETED`;
  - writes `done/` markers and logs under `STATUS_DIR`.
- `src/learn/launch/lib1_inhouse_barcode_bin_matched_n1000_orchestrator.sh`
  - thin wrapper pattern for a derived manifest;
  - sets `MANIFEST_TAG`, `MANIFEST_JSONL`, `STATUS_DIR`, and
    `LAUNCH_NOTES`;
  - optionally runs the manifest generator first, then execs the global queue.

Recommended launcher shape:

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEARN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MANIFEST_TAG="${MANIFEST_TAG:-lib1_barcode_threshold_downsample_june2026}"
MANIFEST_JSONL="${MANIFEST_JSONL:-${LEARN_DIR}/outputs/hpo_manifests/${MANIFEST_TAG}__run_manifest.jsonl}"
STATUS_DIR="${STATUS_DIR:-${LEARN_DIR}/outputs/hpo_runs/status/${MANIFEST_TAG}}"
LAUNCH_NOTES="${LAUNCH_NOTES:-${MANIFEST_TAG}}"
GENERATE_MANIFEST="${GENERATE_MANIFEST:-0}"

export MANIFEST_TAG MANIFEST_JSONL STATUS_DIR LAUNCH_NOTES

if [[ "${GENERATE_MANIFEST}" == "1" || ! -f "${MANIFEST_JSONL}" ]]; then
  (
    cd "${LEARN_DIR}"
    python generate_lib1_barcode_threshold_downsampling_manifest.py --manifest-tag "${MANIFEST_TAG}"
  )
fi

exec bash "${SCRIPT_DIR}/lib1_inhouse_outer_seed_prior_orchestrator.sh"
```

Best manifest-generator reuse:

- `src/learn/generate_lib1_barcode_bin_matched_manifest.py`
  - closest template for this experiment;
  - imports `build_train_command`, `normalize_record_types`,
    `OUTER_SEED_SPLIT_SEEDS`, and `OUTER_SEED_MODEL_SEED`;
  - filters selected configs;
  - mutates the source config rows into derived-run rows;
  - validates feasibility;
  - writes CSV, JSON, JSONL, selected-config summary, feasibility CSV, and
    summary JSON.
- `src/learn/generate_lib1_outer_seed_prior_hpo_manifest.py`
  - source for `build_train_command`;
  - `TRAIN_COMMAND_KEYS` already includes `train_min_barcodes`,
    `train_max_barcodes`, and `train_size_n`;
  - if adding `train_subsample_seed` or `downsample_seed` as a data-module arg,
    add that key to `FIXED_TRAIN_KEYS`, `TRAIN_COMMAND_KEYS`, and
    `INT_VALUE_KEYS`.

Recommended source rows for this new generator:

- Prefer
  `src/learn/outputs/hpo_manifests/lib1_outer_seed_prior_no_rc_june2026__run_manifest.csv`
  as the source manifest, filtered to the selected top-5 configs and all 5
  split seeds.
- Do not rely on
  `lib1_outer_seed_selected_barcode_weighted_june2026__selected_unweighted_baseline.csv`
  for this run, because that file contains the previous top-3 weighted-loss
  follow-up baseline rather than the desired top-5 threshold-learning-curve
  shortlist.

Expected generator loop:

```text
for source_row in selected outer-seed source rows:
    for threshold in [1+, 2+, 3+]:
        for size_label in [n100, n500, n1500, n2500, n3500, full]:
            skip exact-N rows where eligible pool < N
            build derived row
            set graph_module=CNNBasicTraining
            set barcode_weighting=false
            set train_min_barcodes=threshold
            set train_max_barcodes=None
            set train_size_n=N for exact-N rows
            omit/None train_size_n for full rows
            set train_size_frac=1.0
            set train_sampling_mode=random
            set downsample_seed/train_subsample_seed
            build train_command
```

Run-name fields to include in the manifest:

- `barcode_threshold`
- `barcode_threshold_label`
- `train_size_label`
- `train_size_n_requested`
- `downsample_seed` or `train_subsample_seed`
- `available_train_rows`
- `is_full_train_pool`

The global queue dry-run display will not show these extra fields unless the
launcher helper is extended, but that is okay as long as they are present in
the manifest CSV/JSONL and encoded in `planned_run_name`.

Suggested new files for the implementation agent:

- `src/learn/generate_lib1_barcode_threshold_downsampling_manifest.py`
- `src/learn/launch/lib1_inhouse_barcode_threshold_downsampling_orchestrator.sh`
- optionally a small analysis notebook after the run:
  `tutorials/lib1_tasks/pretrain_CRE_inhouse_data/lib1_barcode_threshold_downsampling_analysis_june2026.ipynb`

Reuse existing command/manifest helpers where possible:

- `src/learn/generate_lib1_outer_seed_prior_hpo_manifest.py`
- `src/learn/generate_lib1_barcode_bin_matched_manifest.py`
- `src/learn/launch/lib1_inhouse_outer_seed_prior_orchestrator.sh`

Training settings to preserve:

```text
graph_module = CNNBasicTraining
barcode_weighting = false
epoch_eval_splits = ['train', 'val', 'test']
checkpoint_monitor = val_pearson
stopping_mode = max
test_min_barcodes = 8
val_size_within_hq = 250
test_size_within_hq = 250
model_seed = 1701
use_reverse_complements = false
```

The run name should encode:

```text
manifest tag
part
config_id
threshold label
train size label
split_seed
downsample_seed
```

Example:

```text
lib1_barcode_threshold_downsample_june2026__utr5__utr5_cfg007__bc_ge2__n1500__seed303__ds91002
```

## Analysis Plan

Primary metric:

- `test_pearson` on the high-barcode heldout set.

Secondary metrics:

- `val_pearson`
- `test_spearman`
- `test_cod_r2`
- `test_mse`
- `train_pearson`
- `best_epoch`
- train/test generalization gap

Primary plots:

1. Learning curves by part and threshold:

```text
x = train_size_n, log-scaled or categorical with full as separate marker
y = test_pearson
line = threshold
points/error = paired config x split seed distribution
```

2. Matched threshold deltas at each N:

```text
2+ minus 1+
3+ minus 1+
3+ minus 2+
```

paired within:

```text
part x config_id x split_seed x train_size_label
```

3. Full-pool comparison:

```text
full 1+ vs full 2+ vs full 3+
```

Treat full separately from exact-N arms because full pool sizes differ by
threshold and part.

4. Intron negative/control panel:

Make the Intron panel explicit. If threshold curves are still flat, that
supports the interpretation that Intron barcode-count noise is not limiting
generalization in the same way.

Useful derived summaries:

- sample efficiency: smallest N that reaches 90 percent or 95 percent of the
  threshold's full-pool test Pearson;
- threshold winner rate: fraction of matched slices where each threshold wins
  at a given N;
- AUC over log(N) for each threshold, excluding full;
- full-pool delta: `full 2+ - full 1+`, `full 3+ - full 1+`;
- bootstrap CIs over matched config/split slices, not over individual heldout
  variants.

## Critical Checks To Include

1. Completion and manifest integrity:

   - all expected rows completed;
   - no duplicate run names;
   - skipped `3UTR 3+ N=3500` is intentional;
   - every exact-N row has exactly N training rows;
   - every full row records its actual training row count.

2. Split consistency:

   - same heldout seed policy as outer-seed prior run;
   - val/test row counts remain 250/250 from `n_barcodes >= 8`;
   - no heldout rows leak into training after threshold filtering.

3. Downsample reproducibility:

   - selected-row hash saved for every exact-N row;
   - nested subset property checked:

```text
selected_ids(N=100) subset selected_ids(N=500) subset selected_ids(N=1500)
```

4. Barcode composition audit:

   - for each threshold/N, report actual barcode-count distribution in the
     selected train set;
   - especially for `1+`, quantify how many `n=1` rows are actually sampled at
     each N.

5. Target distribution audit:

   - compare train target mean/std/quantiles across thresholds and N;
   - if threshold filtering changes target distribution, note that heldout
     performance may reflect both label quality and target/sequence
     distribution shift.

6. Sequence covariate audit:

   - at minimum GC content and sequence length after filtering/padding;
   - optional k-mer or simple nucleotide composition summaries if a threshold
     effect looks unexpectedly large.

7. Early stopping behavior:

   - low-N runs may stop earlier or overfit faster;
   - compare best_epoch and train/test gap across thresholds.

## Expected Interpretations

Possible outcomes:

1. `2+` or `3+` beats `1+` at matched N and in full.

   Interpretation: dropping low-barcode rows improves label quality enough to
   offset losing data. This would support a higher training threshold for that
   part, especially if the effect is stable across split seeds and configs.

2. `1+` wins at large N but loses at small N.

   Interpretation: low-barcode rows add useful sequence coverage when enough
   total data is available, but hurt or add variance in small training regimes.
   This would argue for threshold policy depending on available data size.

3. `1+`, `2+`, and `3+` are tied at matched N, but full `1+` wins.

   Interpretation: low-barcode rows are not worse at the same N, and the extra
   total data helps. This would argue against filtering for that part.

4. Intron stays flat.

   Interpretation: barcode count is less limiting for Intron under the current
   model/data setup. Treat Intron as an internal control rather than forcing the
   same conclusion across CRE regions.

5. Val gains do not match test gains.

   Interpretation: threshold policy may be over-tuned to a particular heldout
   split or small validation set. This is exactly why the outer split seed
   should stay in the run.

## Recommended Decision Rule After The Run

Do not pick thresholds by single best mean test Pearson alone.

For each part, prefer a threshold policy that:

1. has the best or near-best paired test Pearson;
2. has stable positive paired deltas across split seeds/configs;
3. does not create obvious train/heldout target distribution shift;
4. has acceptable sample efficiency at realistic N;
5. does not rely on only one config family or one split seed;
6. keeps interpretation consistent with the exact-bin and weighted-loss
   follow-up results.

For a production/default training policy, require:

```text
paired test delta > 0 in most matched slices
and no clear val/test disagreement
and no obvious target/sequence distribution confound
```

## Suggested Manifest Tag

```text
lib1_barcode_threshold_downsample_june2026
```

If adding extra small-N downsample repeats later:

```text
lib1_barcode_threshold_downsample_smalln_reps_june2026
```
