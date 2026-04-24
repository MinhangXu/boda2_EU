# `src/learn` Guide

This directory is the main training and HPO launcher layer for `boda2_EU`.

## What Lives Here

- `train_wandb_log.py`
  - canonical modern training entrypoint
  - use this for sweeps and most reproducible training runs
- `train.py`
  - older generic training entrypoint
- `previous_train.py`
  - older training variant, likely superseded
- `prepare_enhancer_single_head_dataset.py`
  - legacy helper for derived pan-cell enhancer targets (kept for provenance)
- `configs/`
  - hand-authored sweep configs organized by CRE family, target family, and model family
- `configs/README.md`
  - naming convention and comparison-oriented layout notes
- `launch/`
  - curated task-oriented scripts for creating sweeps and starting agents
- `derived_data/`
  - generated intermediate tables that are intentionally reused across runs
- `local_artifacts/`
  - saved model tarballs and other run outputs that you want to keep locally
- `run_registry/`
  - machine-readable best-run and sweep-launch bookkeeping
- `wandb/`
  - generated W&B run metadata cache plus local sweep/run logs
- shell launchers such as:
  - `deploy_wandb_agent_train.sh`
  - `fixed_utr_train.sh`

## Canonical Mental Model

Use `train_wandb_log.py` as the source of truth for:

- data module selection
- model module selection
- graph module selection
- artifact saving
- W&B logging
- sweep execution

The general training contract is:

1. choose `data_module`
2. choose `model_module`
3. choose `graph_module`
4. set task-specific data and architecture arguments
5. optimize on `epoch_end_val_pearson_r2` or another explicit checkpoint metric
6. save artifacts and record `model_saved_path`

## W&B Sweep Identity

Use `entity/project/sweep_id` as the source of truth for where a sweep lives on W&B.

Important distinction:

- top-level sweep YAML `entity` and `project` control where the sweep is created
- `parameters.logger_project` is task metadata logged with each run
- under sweep execution, `logger_project` should not be treated as the authoritative W&B project locator
- curated launchers now pass through `WANDB_SWEEP_ENTITY` and `WANDB_SWEEP_PROJECT` only when explicitly set, so YAML `entity/project` are the default source of truth

Every curated config in this repo uses the verbose project scheme
`<task_family>__<target_family>__<mode>__<model_family>` (e.g.
`utr3__hani_rna_activity__scratch__utr_bassetvl`). Both the top-level
`project:` and `parameters.logger_project.value` are set to this string
so the two never diverge. There is no longer a shared generic project
(`boda2_EU-src_learn`) — if a new config omits an explicit `project:`,
the launcher will fail loudly rather than write runs to an unnamed bucket.

The curated launchers now materialize sweep configs with explicit W&B placement and validate the returned sweep path. See `WANDB_SWEEP_WORKFLOW.md` for the full workflow and environment controls.

## Runs Manifest + Test-Set Evaluation Contract

Every `train_wandb_log.py` run now:

1. Trains with `trainer.fit` as before.
2. Loads the best checkpoint via `set_best`.
3. Calls `trainer.test(...)` to populate Pearson-R^2 / coefficient of
   determination / Pearson / Spearman / loss
   on the held-out split. `CNNBasicTraining.test_epoch_end` and
   `CNNWeightedRegressionTraining.test_epoch_end` both log the
   `test_pearson_r2`, `test_cod_r2`, `test_pearson`, `test_spearman`,
   `test_loss` keys consumed by the manifest below.
4. Runs a single inference pass over `data.train_dataloader()` and
   writes `train_pearson_r2`, `train_cod_r2`, `train_pearson`,
   `train_spearman`, `train_loss` to the W&B run summary.
5. Calls `save_model` which writes `torch_checkpoint.pt`,
   `provenance.json`, and a `.tar.gz` whose filename encodes the
   W&B project and run_id for trivial on-disk lookup.
6. Appends a row to `run_registry/runs.csv` (path overridable via
   `BODA_RUNS_CSV`) containing every field in `RUNS_CSV_COLUMNS` —
   including W&B entity/project/run_id/sweep_id, metric scalars,
   artifact path, git commit, hostname, and launch notes.

See `PRETRAINED_MODELS.md` for how to promote a row in `runs.csv` to
the curated `best_runs.csv` and for the `pretrained_registry.py`
lookup API used by downstream active-learning code.

## PILOT Mode

Set `PILOT=1` before any curated launcher to force a 1-agent / 1-run
smoke test regardless of the configured `NUM_AGENTS` / `NUM_RUNS`:

```bash
PILOT=1 ./launch/utr3_hani_utr_bassetvl_sweep.sh
```

The wrapper `launch/run_all_regions_pilot.sh` chains a pilot per CRE
region — good for verifying that the full config → train → test →
`runs.csv` chain is healthy before committing real HPO GPU time.

## Important Distinction: Source vs Generated State

Treat these as source material:

- `train_wandb_log.py`
- hand-authored YAML configs
- curated launch scripts

Treat these as generated metadata:

- `derived_data/`
- `local_artifacts/`
- `wandb/`

The `wandb/` directory is useful for provenance and run recovery, but it is not the place to hand-edit experiment definitions.

## Generated Directories And Lifecycle

The usual local state for a `train_wandb_log.py` run is split across a few places:

- `derived_data/`
  - reusable generated inputs
  - example: the combined single-head enhancer table created by `prepare_enhancer_single_head_dataset.py`
  - keep this when regeneration is slow or when you want reproducible HPO inputs
- `local_artifacts/`
  - long-lived local outputs produced near the end of successful training
  - typically contains saved model tarballs copied from the trainer scratch directory
  - this is the directory to keep if you want a local copy of trained models after a run finishes
- `wandb/`
  - local W&B cache for sweep agents and runs
  - each `run-*` directory usually contains:
    - `files/config.yaml`
    - `files/output.log`
    - `files/wandb-summary.json`
    - `logs/debug.log`
    - `logs/debug-internal.log`
  - each `sweep-*` directory contains local sweep-assignment config files generated for agent jobs
- `outputs/<task_family>/<target_family>/<model_or_variant>/...`
  - temporary trainer scratch space controlled by `default_root_dir`
  - Lightning checkpoints and transient files land here first
  - successful runs later bundle/copy the final payload into `artifact_path`
  - safe to prune when you no longer need intermediate checkpoints/logs

Practical workflow:

1. edit configs and launchers under `configs/` and `launch/`
2. let launchers create or reuse `derived_data/` inputs
3. monitor active and failed runs via `wandb/`
4. keep successful final model payloads in `local_artifacts/`
5. treat `outputs/...` as disposable scratch, not as the system of record

Rule of thumb:

- edit: `configs/`, `launch/`, training code
- inspect: `wandb/`, `run_registry/`
- keep: `local_artifacts/`, important `derived_data/`
- feel free to prune later: stale `wandb/run-*`, `wandb/sweep-*`, and temp scratch once you no longer need local debugging context

## Directory Placement Policy (Task Caches vs Final Artifacts)

Use this split consistently:

- `outputs/`
  - scratch/training-working directories and task cache folders
  - examples: `outputs/promoter/deboer_core/utr_bassetvl/bayes/`, `outputs/utr3/hani_rna_activity/utr_bassetvl/bayes/`
  - safe to prune when you no longer need intermediate checkpoints/logs
- `local_artifacts/`
  - final model payloads you intend to keep
  - examples: `local_artifacts/promoter/...`, `local_artifacts/utr3/...`, `local_artifacts/utr5/...`
  - default long-lived local storage for rerunnable model exports
- `wandb/`
  - W&B local cache (`run-*`, `sweep-*`, debug logs)
  - useful for debugging/recovery; safe to prune if cloud W&B is source of truth

For new in-house enhancer sweeps, follow the same pattern:

- scratch/root dir under `outputs/enhancer/bashor_in_house/...`
- artifact export under `local_artifacts/enhancer/bashor_in_house/...`

The helper script `cleanup_learn_state.sh` can prune generated state while preserving top-level directory scaffolding.

## Current Task Families

### Enhancer

Typical stack:

- data: `Lib1EnhancerDataModule` (in-house lib1 enhancer table with barcode-aware split controls)
- model: `BassetVL` or `ResNet1DRegressor`
- graph:
  - `CNNBasicTraining` for unweighted scratch regression
  - `CNNWeightedRegressionTraining` for barcode-weighted scratch regression

Current configs / launchers:

- `configs/enhancer/bashor_in_house/lib1_enhancer__scratch_basic__bayes.yml`
- `configs/enhancer/bashor_in_house/lib1_enhancer__scratch_weighted__bayes.yml`
- `launch/lib1_enhancer_scratch_compare_loss_modes.sh`
- `launch/lib1_enhancer_scratch_weighted_sweep.sh`
- `configs/enhancer/malinois_mpra/basset_branched/enhancer__malinois_mpra__basset_branched__transfer_baseline.yml`
- `launch/enhancer_malinois_basset_branched_baseline.sh`
- `configs/legacy/enhancer/malinois_mpra/basset_nonbranched/enhancer__malinois_mpra__basset_nonbranched__single_head_k562__bayes.yml`
- `configs/legacy/enhancer/malinois_mpra/basset_nonbranched/enhancer__malinois_mpra__basset_nonbranched__single_head_combined__bayes.yml`
- `launch/enhancer_malinois_basset_nonbranched_single_head_combined_sweep.sh` (legacy config path)

In-house lib1 scratch notes:

- target column: `RNA_DNA_Ratio_log10_scaled`
- sequence column: `Enhancers`
- key split controls in sweep configs:
  - `train_min_barcodes`
  - `test_min_barcodes`
  - `train_size_frac`
  - `val_frac_within_hq`
  - `test_frac_within_hq`
- output policy:
  - `default_root_dir` under `outputs/enhancer/bashor_in_house/...`
  - `artifact_path` under `local_artifacts/enhancer/bashor_in_house/...`

Historical note:

- the older combined single-head idea (`combined_activity_zmean`) is currently de-prioritized for near-term runs
- keep those configs for reproducibility, but prioritize in-house enhancer scratch sweeps first

### Promoter

Typical stack:

- data: `PromoterDataModule`
- model:
  - `UTR_BassetVL`
  - `BassetVL`
  - `ResNet1DRegressor`
- graph: `CNNBasicTraining`

Related configs / launchers:

- `configs/promoter/deboer_core/utr_bassetvl/promoter__deboer_core__utr_bassetvl__bayes.yml`
- `configs/promoter/deboer_core/bassetvl/promoter__deboer_core__bassetvl__bayes.yml`
- `configs/promoter/deboer_core/resnet1d/promoter__deboer_core__resnet1d__bayes.yml`
- `launch/promoter_deboer_utr_bassetvl_sweep.sh`
- `launch/promoter_deboer_compare_architectures.sh`

### 5'UTR polysome

Typical stack:

- data: `UTR_Polysome_MPRA_DataModule`
- model: `UTR_BassetVL`
- graph: `CNNBasicTraining`

Related files:

- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__bayes__egfp_1.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__bayes__egfp_2.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__bayes__mcherry_1.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__bayes__mcherry_2.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__egfp_1.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__egfp_2.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__mcherry_1.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__mcherry_2.yml`
- `launch/utr5_polysome_utr_bassetvl_sweep.sh`
- `launch/utr5_polysome_fixed_all.sh`
- `fixed_utr_train.sh`
- `tutorials/get_HPO_5utr_polysome.ipynb`

Use `launch/utr5_polysome_utr_bassetvl_sweep.sh` for a proper HPO sweep
(`LIBRARY=egfp_1|egfp_2|mcherry_1|mcherry_2`) and keep
`launch/utr5_polysome_fixed_all.sh` for fixed-parameter benchmark reruns.
This remains distinct from the Hani RNA activity workflow.

### 5'UTR Hani RNA activity

Typical stack:

- data: `HaniGoozardi_RNA_Activity_DataModule`
- model: `UTR_BassetVL`
- graph: `CNNBasicTraining`

Related config:

- `configs/utr5/hani_rna_activity/utr_bassetvl/utr5__hani_rna_activity__utr_bassetvl__bayes.yml`

### 3'UTR RNA activity

Typical stack:

- data:
  - `UTR3_RNA_Activity_DataModule` (current baseline bayes config)
  - `HaniGoozardi_RNA_Activity_DataModule` (focused historical config)
- model: `UTR_BassetVL`
- graph: `CNNBasicTraining`

Related configs:

- `configs/utr3/hani_rna_activity/utr_bassetvl/utr3__hani_rna_activity__utr_bassetvl__bayes.yml`
- `configs/utr3/hani_rna_activity/utr_bassetvl/utr3__hani_rna_activity__utr_bassetvl__focused_bayes__2025-06-16.yml`

## Run Recovery

If local artifacts are missing, use `wandb/` to recover:

- project name
- run timestamp
- metric values
- resolved hyperparameters
- intended `model_saved_path`

Useful fields in each run:

- `files/config.yaml`
- `files/output.log`
- `files/wandb-summary.json`
- `logs/debug.log`

For failed runs, `files/output.log` is usually the fastest place to confirm whether the job died:

- before W&B logger initialization
- during datamodule setup / split construction
- during model setup
- during fit / validation
- before artifact copyout

Notebook-friendly helpers now live in:

- `../analysis/hpo_results_eval_utils.py`

Best-known run summaries are being tracked in:

- `../plan/best_runs_snapshot.md`

## Current Config Layout

Authored configs now live under:

- `configs/enhancer/bashor_in_house/`
- `configs/enhancer/malinois_mpra/basset_branched/`
- `configs/legacy/enhancer/malinois_mpra/basset_nonbranched/`
- `configs/promoter/deboer_core/bassetvl/`
- `configs/promoter/deboer_core/resnet1d/`
- `configs/promoter/deboer_core/utr_bassetvl/`
- `configs/utr5/polysome/utr_bassetvl/`
- `configs/utr5/hani_rna_activity/utr_bassetvl/`
- `configs/utr3/hani_rna_activity/utr_bassetvl/`
- `configs/introns/placeholder/utr_bassetvl/` (template only; no data yet — see `configs/introns/README.md`)

This layout keeps model comparisons local to one biological task: add a sibling
model-family directory under the same target when you want an apples-to-apples
comparison.

## Launch Workflow

Preferred path for new work:

1. choose a config under `configs/`
2. launch it with the matching script under `launch/`
3. monitor the sweep in W&B
4. recover the best run via `wandb/`, `run_registry/`, and notebooks

Key docs:

- `launch/README.md`
- `run_registry/README.md`

Current task-oriented launchers:

- `launch/lib1_enhancer_scratch_compare_loss_modes.sh`
- `launch/lib1_enhancer_scratch_weighted_sweep.sh`
- `launch/enhancer_malinois_basset_branched_baseline.sh`
- `launch/enhancer_malinois_basset_nonbranched_single_head_combined_sweep.sh` (legacy / archived)
- `launch/promoter_deboer_compare_architectures.sh`
- `launch/promoter_deboer_utr_bassetvl_sweep.sh`
- `launch/utr3_hani_utr_bassetvl_sweep.sh`
- `launch/utr5_hani_utr_bassetvl_sweep.sh`
- `launch/utr5_polysome_utr_bassetvl_sweep.sh`
- `launch/utr5_polysome_fixed_all.sh`
- `launch/run_all_regions_pilot.sh` — orchestrates a 1-agent / 1-run pilot across every region (respects `PILOT=1`, `GPU_LIST`, and `REGIONS`)

## Near-Term Priorities

1. keep enhancer as the top reboot target
   - first run path: in-house lib1 scratch (`basic` or `weighted`) under `configs/enhancer/bashor_in_house/`
2. preserve the older 5'UTR polysome benchmark as a distinct task family
3. `run_registry/runs.csv` is now auto-populated; promote winning runs to
   `run_registry/best_runs.csv` and rely on `pretrained_registry.py`
   (see `PRETRAINED_MODELS.md`) for programmatic lookup
4. keep config naming comparison-friendly (verbose scheme
   `<region>__<target>__<mode>__<model>`) so additional model families
   can be evaluated side by side without touching W&B projects
5. onboard the introns data module (`configs/introns/` is reserved)
