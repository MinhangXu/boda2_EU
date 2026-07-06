# Lib1 Outer-Seed Prior-Informed HPO Plan

Generated: 2026-06-14
Updated: 2026-06-14

This plan defines a narrower follow-up to the June 2026 Lib1 in-house scratch
HPO. It uses the finished 128-run-per-sweep campaign as a prior, but changes
the experimental design so `split_seed` becomes an outer loop rather than a
regular HPO parameter. The goal is to select part-specific model configs that
are robust across heldout splits before moving to training-set downsampling.

Companion math/evaluation note:

- `plan/phase1_lib1/eval/lib1_hpo_seed_eval_math_notes_june2026.md`

This run is only for:

- Promoter
- Intron
- 3 Prime UTR
- 5 Prime UTR

Enhancer is intentionally excluded from this follow-up.

## Core Idea

The previous orchestrator run was broad discovery. It explored hyperparameters,
RC, split seeds, and sometimes model seeds jointly. That found useful regions,
but it did not evaluate the same non-seed config across multiple heldout
splits. Therefore it cannot directly answer config-level split robustness.

The updated follow-up should do:

```text
for each part:
    choose 30 base hyperparameter configs from the previous HPO evidence
    for each split_seed in [101, 202, 303, 404, 505]:
        run that exact base config with RC off
```

This gives:

| Quantity | Count |
|---|---:|
| Base configs per part | 30 |
| RC arms per config | 1 |
| Runs per split seed per part | 30 |
| Split seeds | 5 |
| Runs per part | 150 |
| Parts | 4 |
| Total runs | 600 |

This design intentionally drops RC from the new run. That is a scope and budget
decision, not proof that RC is useless. The current RC plots suggest RC is not
uniformly beneficial, and this follow-up is now aimed at split-stable
hyperparameter selection. Intron ResNet1D still has some positive RC evidence,
so if intron is scientifically central and compute is available, a later small
paired RC confirmation can be run for the final intron shortlist.

## Architecture Choices

Use one architecture per part:

| Part | Architecture | Previous sweep used as prior |
|---|---|---|
| Promoter | `PromoterBassetVL` | `vi17zxcm` |
| Intron | `ResNet1DRegressor` | `5b0njbjz` |
| 3 Prime UTR | `ResNet1DRegressor` | `bnyvegba` |
| 5 Prime UTR | `ResNet1DRegressor` | `87uud4bc` |

These choices deliberately freeze the architecture dimension. The follow-up is
not trying to rerun the full architecture comparison; it is trying to choose
robust configs for one selected architecture per part.

## Seed Policy

Use the same split-seed set for every part:

```text
split_seed = [101, 202, 303, 404, 505]
```

Use a fixed model seed during this follow-up:

```text
model_seed = 1701
```

Rationale:

- `split_seed` is the outer loop because heldout-split difficulty is the main
  stochasticity we want to measure now.
- A fixed `model_seed` reduces run count and makes the split-seed comparison
  cleaner.
- Repeating shortlisted configs across additional `model_seed` values can be a
  later confirmation step if the top configs are close or training randomness
  appears large.

## Why A Manifest Rather Than Plain W&B Bayesian Sweeps?

There are two separate ideas here:

- the search policy: which hyperparameter configs get tried;
- the scheduler: how jobs get assigned to GPUs.

A plain W&B Bayesian sweep is good for broad discovery. W&B proposes one
configuration at a time from a search space. If `split_seed` is included as a
sweep parameter, W&B treats it like any other parameter. That means run A might
use config X with seed 101, while run B uses config Y with seed 202. You get
coverage, but not the same config repeated across all seeds.

This follow-up needs repeated configs:

```text
config_001 on seed 101
config_001 on seed 202
config_001 on seed 303
config_001 on seed 404
config_001 on seed 505
```

That is why a fixed manifest is preferred. A manifest is just a table where
each row is one exact training job. The manifest encodes the pairing explicitly:
same part, same `config_id`, same model hyperparameters, same `model_seed`,
different `split_seed`.

A global GPU queue is the scheduler for that manifest. It does not decide which
configs are good. It only keeps GPUs busy:

```text
one worker per GPU
worker claims the next unfinished manifest row
worker runs one training command on its assigned GPU
when done, worker claims another row
```

This is different from the current orchestrator's static part/sweep split,
where one group of GPUs can finish early while another group is still busy.

If we still want W&B sweep UI semantics, an alternative is a grid sweep over
`config_id` and `split_seed`, with a wrapper that resolves `config_id` to a
fixed hyperparameter dictionary. Plain W&B Bayesian search over all individual
hyperparameters is not the right primitive for this repeated-config design.

## How To Use The Previous HPO Without Retesting Bad Regions

Do not treat previous single-run winners as proven true winners. Treat them as
evidence for a prior over useful regions.

Use validation metrics only for config selection. Test metrics can be inspected
diagnostically, but should not decide which configs enter this new run.

Recommended selection pipeline per part:

1. Load completed runs from the selected prior sweep.
2. Drop invalid runs:
   - missing `val_pearson`,
   - invalid Pearson values outside `[-1, 1]`,
   - failed/incomplete status,
   - obvious metric explosions.
3. Deduplicate candidate configs ignoring:
   - `split_seed`,
   - `model_seed`,
   - `use_reverse_complements`.
4. Score prior configs by validation performance:
   - if the prior sweep used multiple split seeds, rank within each
     `split_seed` first, then use within-seed percentile or z-score;
   - if the prior sweep used only one split seed, use validation rank directly.
5. Select 30 base configs with a mix of exploitation and exploration:
   - 8 exact elite observed configs from the previous HPO,
   - 12 local variants jittered around elite/top-quartile ranges,
   - 10 broader samples from the narrowed prior, kept inside plausible ranges.
6. Duplicate every base config across the 5 split seeds.

This is Bayesian-flavored rather than a fully online Bayesian loop: previous
HPO results define a narrower prior, then the new run evaluates a fixed
manifest. That is preferable here because we need the same config evaluated
across all split seeds.

## Narrowed Prior By Part

These ranges are mined from the top validation region of the exact prior sweeps.
They are meant to guide candidate generation, not to be copied blindly.

### Promoter BassetVL

Previous sweep `vi17zxcm`:

- completed valid runs: 128
- val Pearson max/median: 0.425 / 0.388
- current max epochs / patience: 180 / 30
- best epoch q50/q90/max: 85 / 176 / 179
- top 20 validation configs were all RC off in the prior sweep

Top-region pattern:

| Parameter | Suggested narrowed support |
|---|---|
| `optimizer` | fix `AdamW` |
| `scheduler` | fix `"None"` |
| `lr` | log sample around `1e-4` to `1e-3` |
| `weight_decay` | log sample around `1e-5` to `2e-3` |
| `linear_dropout_p` | sample `0.38` to `0.65` |
| `conv1_channels` | sample roughly `48` to `128` |
| `conv2_channels` | sample roughly `40` to `128` |
| `conv3_channels` | sample roughly `24` to `96` |
| `conv1_kernel_size` | prefer `[5, 7, 9]`; keep few `11` exploratory |
| `conv2_kernel_size` | prefer `[7, 9]`; keep few `[3, 5]` exploratory |
| `conv3_kernel_size` | prefer `7`; keep few `[3, 5]` exploratory |
| `adaptive_pool_output_size` | prefer `12`; keep few `[6, 8]` exploratory |
| `n_linear_layers` | `[1, 2]`, with more weight on `2` |
| `linear_activation` | prefer `LeakyReLU`; keep `ELU`/`ReLU` exploratory |
| `batch_size` | include `[64, 128, 256]`, but prior top region favored `64` |

### Intron ResNet1D

Previous sweep `5b0njbjz`:

- completed valid runs: 126
- val Pearson max/median: 0.613 / 0.521
- current max epochs / patience: 180 / 35
- best epoch q50/q90/max: 12 / 70 / 172
- top 20 validation configs were mixed RC: 13 off, 7 on

Top-region pattern:

| Parameter | Suggested narrowed support |
|---|---|
| `optimizer` | keep `[Adam, AdamW]` |
| `scheduler` | strongly favor `CosineAnnealingWarmRestarts`; keep a few `"None"` |
| `lr` | log sample around `3e-5` to `4e-4` |
| `weight_decay` | log sample around `2e-6` to `2e-3` |
| `dropout_p` | sample `0.07` to `0.40` |
| `stem_channels` | sample roughly `36` to `96` |
| `head_hidden_channels` | sample roughly `32` to `160` |
| `stem_kernel_size` | prefer `[7, 9]`; keep few `[3, 5]` exploratory |
| `block_kernel_size` | prefer `[7, 9]`; keep few `[3, 5]` exploratory |
| `use_batch_norm` | mostly `false`; include a small number of `true` configs |
| `batch_size` | prefer `[128, 256]`, especially `256` |

### 3 Prime UTR ResNet1D

Previous sweep `bnyvegba`:

- completed valid runs: 127
- val Pearson max/median: 0.456 / 0.250
- current max epochs / patience: 160 / 30
- best epoch q50/q90/max: 42 / 123 / 157
- prior ResNet1D sweep had RC fixed off

Top-region pattern:

| Parameter | Suggested narrowed support |
|---|---|
| `optimizer` | favor `Adam`; keep some `AdamW` |
| `scheduler` | keep both `"None"` and `CosineAnnealingWarmRestarts` |
| `lr` | log sample around `3e-5` to `4e-4` |
| `weight_decay` | log sample around `1e-6` to `3e-5` |
| `dropout_p` | sample `0.08` to `0.40` |
| `stem_channels` | sample roughly `54` to `128` |
| `head_hidden_channels` | sample roughly `76` to `256` |
| `stem_kernel_size` | strongly favor `5`; include a few `7` |
| `block_kernel_size` | strongly favor `3`; include a few `5` exploratory |
| `use_batch_norm` | mostly `false`; include a small number of `true` configs |
| `batch_size` | include `[64, 128, 256]` |

### 5 Prime UTR ResNet1D

Previous sweep `87uud4bc`:

- completed valid runs: 128
- val Pearson max/median: 0.541 / 0.412
- current max epochs / patience: 180 / 35
- best epoch q50/q90/max: 22 / 129 / 179
- top 20 validation configs were mostly RC off, but the validation-best run was
  RC on

Top-region pattern:

| Parameter | Suggested narrowed support |
|---|---|
| `optimizer` | strongly favor `AdamW`; keep a few `Adam` |
| `scheduler` | favor `CosineAnnealingWarmRestarts`; keep a few `"None"` |
| `lr` | log sample around `3e-5` to `1.5e-4` |
| `weight_decay` | log sample around `1e-6` to `5e-4` |
| `dropout_p` | sample `0.16` to `0.40` |
| `stem_channels` | sample roughly `50` to `160` |
| `head_hidden_channels` | sample roughly `100` to `256` |
| `stem_kernel_size` | prefer `[5, 7]`; include very few `11` exploratory |
| `block_kernel_size` | prefer `[3, 5]` |
| `use_batch_norm` | mostly `false`; include a small number of `true` configs |
| `batch_size` | prefer `[128, 256]`, especially `256` |

## Epoch, Patience, And Diagnostic Logging

Current settings in the prior configs:

| Part | Architecture | Current `max_epochs` | Current `min_epochs` | Current `stopping_patience` | Prior best-epoch q50/q90/max |
|---|---|---:|---:|---:|---|
| Promoter | BassetVL | 180 | 20 | 30 | 85 / 176 / 179 |
| Intron | ResNet1D | 180 | 20 | 35 | 12 / 70 / 172 |
| 3 Prime UTR | ResNet1D | 160 | 20 | 30 | 42 / 123 / 157 |
| 5 Prime UTR | ResNet1D | 180 | 20 | 35 | 22 / 129 / 179 |

"Need more room" means some prior runs had their best validation epoch near
the current `max_epochs` cap. If a run's true optimum was after the cap, the
previous run may have been truncated. The proposed update gives those parts
more training time before hitting the hard stop:

| Part | Proposed `max_epochs` | Proposed `stopping_patience` | Why |
|---|---:|---:|---|
| Promoter | 220 | 35 | Many prior runs hit or nearly hit the 180-epoch cap. |
| Intron | 180 | 35 | Most runs stop early, but a few good configs peak late. |
| 3 Prime UTR | 180 | 30 | Current 160 cap may be tight for the top region. |
| 5 Prime UTR | 220 | 35 | Prior top runs include late best epochs near the 180 cap. |

Keep diagnostic epoch logging for now:

```yaml
epoch_eval_splits:
  value: [train, val, test]
```

This is more expensive than validation-only evaluation, but the train/val/test
curves have been useful for diagnosing barcode-count effects and overfitting.
The implementation agent should treat this as a conscious cost tradeoff. If
GPU throughput becomes the limiting factor, the first simplification would be
to log only `[val]` during the 600-run sweep and run post-hoc split-aware
evaluation on selected checkpoints.

## Scoring After The New Run

Primary model-selection score:

```text
mean validation Pearson across the 5 split seeds
```

Report for every `(part, config_id)` arm:

- mean validation Pearson,
- standard deviation across split seeds,
- min/max across split seeds,
- 95% CI across split seeds if useful,
- mean validation Spearman,
- mean validation COD R2,
- mean validation RMSE.

Test-set role:

- Do not select configs by test performance.
- Report test metrics for the validation-selected winner and runner-up.
- Use test paired bootstrap only as diagnostic evidence after validation-based
  selection.

## Manifest Generator

The first version of the manifest generator can live in
`tutorials/lib1_tasks/pretrain_CRE_inhouse_data/lib1_inhouse_scratch_hpo_seed_split_diagnostics_june2026.ipynb`.
That is appropriate because this is still an analysis-driven manifest: we want
to inspect the 30 base configs and the 8 exact elite prior runs before launching
anything.

The notebook generator should output:

- a base-config table: 30 rows per part, 120 rows total;
- an expanded run manifest: 30 configs x 5 split seeds x 4 parts = 600 rows;
- violin/scatter plots showing where the 8 exact elite configs sit relative to
  the previous HPO distribution for validation and test Pearson.

Once approved, an implementation agent can lift the generator into a script,
for example:

- `src/learn/generate_lib1_outer_seed_prior_hpo_manifest.py`

Manifest columns should include:

- `part`
- `architecture`
- `source_prior_sweep_id`
- `config_id`
- `config_source`: `exact_elite`, `local_variant`, or `narrow_prior`
- `source_run_id` for exact elite configs
- `split_seed`
- `model_seed`
- `use_reverse_complements=false`
- all fixed data-module fields
- all fixed model hyperparameters
- all fixed optimizer/trainer hyperparameters
- output/logging paths

## GPU Utilization Plan

Use a fixed manifest plus a global work queue rather than static per-part GPU
splits.

Recommended behavior:

```text
one worker per GPU
worker claims the next unfinished manifest row
worker runs one training command on its assigned GPU
when done, worker claims another row
```

Benefits:

- long promoter or UTR runs do not leave other GPUs idle,
- parts can be interleaved,
- failed rows can be resumed by status,
- the same orchestrator can later run downsampling manifests.

Additional GPU-efficiency settings:

- keep `num_workers: 8` unless dataloading contention appears,
- prefer larger `batch_size` where prior top configs supported it,
- keep `precision: 32` for comparability unless a pilot confirms mixed
  precision is stable for these small regression targets.

## Relationship To Downsampling

Do this outer-seed prior-informed run before downsampling. It should produce a
small set of validation-selected robust configs per part.

Then downsampling should use:

- the best validation-selected `config_id` per part,
- optionally the runner-up if it is close or biologically interesting,
- fixed heldout splits,
- nested train subsets where possible,
- original-variant downsampling first.

Downsampling should answer a different question:

```text
given a robust config, how much marginal gain do we get from more training rows?
```

It should not also be trying to resolve architecture and split-seed stability
from scratch.

## Immediate Next Actions

1. Finish the current HPO trust diagnostics notebook and decide whether any
   part should be excluded from the follow-up.
2. Add the notebook manifest generator that extracts prior runs and produces 30
   base configs per part.
3. Review the generated 30 configs per part manually before launching.
4. Implement the global-queue orchestrator.
5. Launch a pilot:
   - 1 part,
   - 2 base configs,
   - 2 split seeds,
   - total 4 runs.
6. If pilot logging, artifacts, and GPU scheduling are clean, launch the full
   600-run campaign.

Pilot dry-run command:

```bash
cd /home/minhang/synBio_AL/boda2_EU
python src/learn/generate_lib1_outer_seed_prior_hpo_manifest.py

DRY_RUN=1 GPU_LIST="0" PARTS="promoter" MAX_CONFIGS_PER_PART=2 \
  SPLIT_SEEDS="101 202" \
  bash src/learn/launch/lib1_inhouse_outer_seed_prior_orchestrator.sh
```

This selects Promoter only, the first two base configs for that part, and split
seeds 101 and 202, for 1 x 2 x 2 = 4 dry-run commands. Keep `DRY_RUN=1` until
the printed commands, W&B projects, and deterministic output paths have been
reviewed.

## Open Decisions

- Confirm `model_seed=1701` as the fixed seed for all parts.
- Confirm the fifth split seed: proposed `505`.
- Decide whether to use the proposed epoch/patience updates or keep exact prior
  trainer settings for maximum continuity.
- Decide whether to keep `[train, val, test]` epoch logging for the full
  campaign or switch to validation-only if throughput becomes the bottleneck.
