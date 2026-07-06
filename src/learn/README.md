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
- `prepare_hani_utr5_lib1_lib2_phase3_dataset.py`
  - builds the Phase 3 Hani 5'UTR Lib1+Lib2 scratch-training table
  - preserves Lib1 folds, hash-splits Lib2 by sequence, and reserves Lib2 test
- `configs/`
  - hand-authored sweep configs organized by CRE family, target family, and model family
- `configs/README.md`
  - naming convention and comparison-oriented layout notes
- `launch/`
  - curated task-oriented scripts for creating sweeps and starting agents
- `derived_data/`
  - generated intermediate tables that are intentionally reused across runs
- `outputs/`
  - ignored generated training state, Lightning scratch directories, and
    per-W&B-project HPO run roots
- `local_artifacts/`
  - durable saved model tarballs exported by completed runs
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
   determination R^2 / Pearson / Spearman / MSE / loss
   on the held-out split. `CNNBasicTraining.test_epoch_end` and
   `CNNWeightedRegressionTraining.test_epoch_end` both log the
   `test_pearson_r2`, `test_cod_r2`, `test_pearson`, `test_spearman`,
   `test_mse`, `test_loss` keys consumed by the manifest below.
4. Runs a single inference pass over `data.train_dataloader()` and
   writes `train_pearson_r2`, `train_cod_r2`, `train_pearson`,
   `train_spearman`, `train_mse`, `train_loss` to the W&B run summary.
5. Calls `save_model` which writes `torch_checkpoint.pt`,
   `provenance.json`, and a `.tar.gz` whose filename encodes the
   W&B project and run_id for trivial on-disk lookup.
6. Appends a row to `run_registry/runs.csv` (path overridable via
   `BODA_RUNS_CSV`) containing every field in `RUNS_CSV_COLUMNS` —
   including W&B entity/project/run_id/sweep_id, metric scalars,
   artifact path, git commit, hostname, and launch notes.

The legacy `val_r2`, `test_r2`, and `train_r2` columns are Pearson correlation
squared for backward compatibility. New analysis should prefer the explicit
`*_pearson_r2` and `*_cod_r2` columns when distinguishing Pearson-squared from
coefficient-of-determination R^2.

See `PRETRAINED_MODELS.md` for how to promote a row in `runs.csv` to
the curated `best_runs.csv` and for the `pretrained_registry.py`
lookup API used by downstream active-learning code.

## PILOT Mode

Set `PILOT=1` before any curated launcher to force a 1-agent / 1-run
smoke test regardless of the configured `NUM_AGENTS` / `NUM_RUNS`:

```bash
PILOT=1 GPU_LIST="0" PARTS="enhancer" MODE=sequential \
  bash launch/lib1_inhouse_scratch_orchestrator.sh
```

The orchestrator can run a one-part pilot or the full standardized Lib1
in-house set. Use `DRY_RUN=1` first when checking GPU allocation without
creating W&B sweeps.

## Important Distinction: Source vs Generated State

Treat these as source material:

- `train_wandb_log.py`
- hand-authored YAML configs
- curated launch scripts

Treat these as generated metadata:

- `derived_data/`
- `outputs/`
- `local_artifacts/`
- `wandb/`

The `wandb/` directory is useful for provenance and run recovery, but it is not the place to hand-edit experiment definitions.

## Generated Directories And Lifecycle

The usual local state for a `train_wandb_log.py` run is split across a few places:

- `derived_data/`
  - reusable generated inputs
  - example: the combined single-head enhancer table created by `prepare_enhancer_single_head_dataset.py`
  - keep this when regeneration is slow or when you want reproducible HPO inputs
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
    and prune transient Lightning checkpoints by default
  - safe to prune when you no longer need intermediate checkpoints/logs
- `outputs/hpo_runs/by_project/<wandb_project_name>/`
  - per-W&B-project browsing layer for HPO sweeps
  - `best_checkpoint_model/<run_id>/` contains provenance/selection metadata
    and local symlinks to canonical artifacts when `best_checkpoint_dir` is
    configured
  - this is the home for directories named like
    `promoter__bashor_in_house__lib1_allvalid__scratch__promoter_bassetvl/`
  - do not leave those project-shaped directories directly under `src/learn/`
- `local_artifacts/<task_family>/<target_family>/<model_family>/...`
  - durable final model payloads produced near the end of successful training
  - contains `.tar.gz` archives copied from the trainer scratch directory via
    `artifact_path`
  - this is the directory to keep if you want local rerunnable model exports
    after temporary checkpoints are pruned

Practical workflow:

1. edit configs and launchers under `configs/` and `launch/`
2. let launchers create or reuse `derived_data/` inputs
3. monitor active and failed runs via `wandb/`
4. inspect per-project HPO run roots under `outputs/hpo_runs/by_project/`
5. keep successful final model payloads in `local_artifacts/`
6. treat `outputs/...` and `outputs/hpo_runs/...` as disposable generated
   state once W&B cloud history, `run_registry/`, and `local_artifacts/`
   contain the needed record

Rule of thumb:

- edit: `configs/`, `launch/`, training code
- inspect: `wandb/`, `run_registry/`, `outputs/hpo_runs/by_project/`
- keep: `local_artifacts/`, important `derived_data/`
- feel free to prune later: stale `wandb/run-*`, `wandb/sweep-*`, and temp scratch once you no longer need local debugging context

## Directory Placement Policy (Task Caches vs Final Artifacts)

Use this split consistently:

- `outputs/`
  - ignored generated run state
  - `outputs/<task_family>/...` is trainer scratch from `default_root_dir`
  - `outputs/hpo_runs/by_project/<wandb_project>/...` is the tidy home for
    project-shaped HPO run roots and `best_checkpoint_model/` convenience
    copies
  - safe to prune when you no longer need intermediate checkpoints/logs and
    have preserved the needed W&B/registry/artifact records
- `local_artifacts/`
  - final model payloads you intend to keep locally
  - examples: `local_artifacts/promoter/...`, `local_artifacts/utr3/...`,
    `local_artifacts/utr5/...`
  - default long-lived local storage for rerunnable model exports
- `wandb/`
  - W&B local cache (`run-*`, `sweep-*`, debug logs)
  - useful for debugging/recovery; safe to prune if cloud W&B is source of truth

For new standardized HPO sweeps, follow the same pattern:

- `default_root_dir` under `outputs/<task_family>/<target_family>/...`
- `best_checkpoint_dir` under
  `outputs/hpo_runs/by_project/<wandb_project>/best_checkpoint_model`
- `artifact_path` under `local_artifacts/<task_family>/<target_family>/...`

If a run creates `src/learn/<wandb_project>/`, treat that as a misplaced
generated project root. Move it into `outputs/hpo_runs/by_project/` or delete
it if the corresponding sweep/config has been retired.

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

- `configs/enhancer/bashor_in_house/resnet1d/lib1_enhancer_no_flank_hq8__scratch_resnet1d__bayes.yml`
- `configs/enhancer/bashor_in_house/bassetvl/lib1_enhancer_no_flank_hq8__scratch_bassetvl__bayes.yml`
- `launch/lib1_enhancer_no_flank_hq8_scratch_resnet1d_sweep.sh`
- `launch/lib1_enhancer_no_flank_hq8_scratch_bassetvl_sweep.sh`
- log2-target side test:
  - `configs/enhancer/bashor_in_house/resnet1d/lib1_enhancer_no_flank_hq8_log2target__scratch_resnet1d__bayes.yml`
  - `configs/enhancer/bashor_in_house/bassetvl/lib1_enhancer_no_flank_hq8_log2target__scratch_bassetvl__bayes.yml`
  - `launch/lib1_enhancer_no_flank_hq8_log2target_scratch_resnet1d_sweep.sh`
  - `launch/lib1_enhancer_no_flank_hq8_log2target_scratch_bassetvl_sweep.sh`
- `configs/enhancer/malinois_mpra/basset_branched/enhancer__malinois_mpra__basset_branched__transfer_baseline.yml`
- `launch/enhancer_malinois_basset_branched_baseline.sh`

In-house lib1 scratch notes:

- target column: `RNA_DNA_Ratio_log10_scaled`
- log2-target side-test column: `log2_RNA_DNA`, computed from the same
  `RNA/DNA` ratio as `log2(RNA/DNA)` for comparability with promoter, intron,
  3'UTR, and 5'UTR
- sequence column: `Enhancers`
- current no-flank HQ8 scratch HPO uses standardized Lib1 split controls and
  canonical W&B metric history logging
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

- old enhancer basic/weighted/FASTQ1-5 scratch sweeps and non-branched
  Malinois single-head sweeps were retired in the June 2026 cleanup

### Promoter

Typical stack:

- data: `PromoterDataModule`
- model:
  - `UTR_BassetVL`
  - `BassetVL`
  - `ResNet1DRegressor`
- graph: `CNNBasicTraining`

Related configs / launchers:

- `configs/promoter/bashor_in_house/resnet1d/lib1_promoter__scratch_resnet1d__bayes.yml`
- `configs/promoter/bashor_in_house/promoter_bassetvl/lib1_promoter__scratch_promoter_bassetvl__bayes.yml`
- `launch/lib1_promoter_scratch_resnet1d_sweep.sh`
- `launch/lib1_promoter_scratch_promoter_bassetvl_sweep.sh`

### 5'UTR polysome

Typical stack:

- data: `UTR_Polysome_MPRA_DataModule`
- model: `UTR_BassetVL`
- graph: `CNNBasicTraining`

Related files:

- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__egfp_1.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__egfp_2.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__mcherry_1.yml`
- `configs/utr5/polysome/utr_bassetvl/utr5__polysome__utr_bassetvl__fixed__mcherry_2.yml`
- `launch/utr5_polysome_fixed_all.sh`
- `fixed_utr_train.sh`
- `tutorials/get_HPO_5utr_polysome.ipynb`

The HPO sweep configs/launcher were retired in the June 2026 cleanup. Keep
`launch/utr5_polysome_fixed_all.sh` for fixed-parameter benchmark reruns. This
remains distinct from the Hani RNA activity workflow.

### 5'UTR Hani RNA activity

Typical stack:

- data: `UTR5_Branched_RNA_Activity_DataModule`
- model: `BassetBranched` or `ResNet1DRegressor`
- graph: `CNNBasicTraining`

Related configs / launchers:

- `configs/utr5/hani_rna_activity/basset_branched/utr5__hani_rna_activity__basset_branched__delta_aux_bayes.yml`
- `configs/utr5/hani_rna_activity/resnet1d/utr5__hani_rna_activity__resnet1d__cell_conditioned_delta_aux_bayes.yml`
- `launch/utr5_hani_basset_branched_delta_aux_sweep.sh`
- `launch/utr5_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh`

Phase 3 Lib1+Lib2 scratch HPO:

- table prep:
  `python src/learn/prepare_hani_utr5_lib1_lib2_phase3_dataset.py`
- derived table:
  `src/learn/derived_data/utr5/hani_rna_activity/5UTR_lib1_lib2_phase3_branched_observed_heads.csv`
- split policy:
  preserve Lib1 folds, aggregate Lib2 replicate rows by uppercased sequence
  and cell type, assign Lib2 train/val/test by deterministic sequence hash,
  and drop Lib1 rows that overlap Lib2 by sequence
- metrics:
  normal combined trainer metrics plus W&B summary keys named
  `eval_<fold>_<library>_*` for Lib1-only and Lib2-only monitoring

### 3'UTR RNA activity

Typical stack:

- data:
  - `UTR3_RNA_Activity_DataModule` (current baseline bayes config)
  - `HaniGoozardi_RNA_Activity_DataModule` (focused historical config)
- model: `BassetBranched` or `ResNet1DRegressor`
- graph: `CNNBasicTraining`

Related configs:

- `configs/utr3/hani_rna_activity/basset_branched/utr3__hani_rna_activity__basset_branched__delta_aux_bayes.yml`
- `configs/utr3/hani_rna_activity/resnet1d/utr3__hani_rna_activity__resnet1d__cell_conditioned_delta_aux_bayes.yml`

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

- `../../plan/learn/best_runs_snapshot.md`

## Current Config Layout

Authored configs now live under:

- `configs/enhancer/bashor_in_house/`
- `configs/enhancer/malinois_mpra/basset_branched/`
- `configs/promoter/bashor_in_house/`
- `configs/utr5/polysome/utr_bassetvl/`
- `configs/utr5/hani_rna_activity/`
- `configs/utr3/hani_rna_activity/`
- `configs/introns/bashor_in_house/`
- `configs/introns/seelig_2015/`
- `configs/introns/placeholder/utr_bassetvl/` (template only; see `configs/introns/README.md`)

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

- `launch/enhancer_malinois_basset_branched_baseline.sh`
- `launch/lib1_inhouse_scratch_orchestrator.sh`
- `launch/lib1_promoter_scratch_resnet1d_sweep.sh`
- `launch/lib1_promoter_scratch_promoter_bassetvl_sweep.sh`
- `launch/lib1_intron_scratch_resnet1d_sweep.sh`
- `launch/lib1_threeprime_scratch_resnet1d_sweep.sh`
- `launch/lib1_threeprime_scratch_utr_bassetvl_sweep.sh`
- `launch/lib1_fiveprime_scratch_resnet1d_sweep.sh`
- `launch/lib1_fiveprime_scratch_utr_bassetvl_sweep.sh`
- `launch/lib1_enhancer_no_flank_hq8_scratch_resnet1d_sweep.sh`
- `launch/lib1_enhancer_no_flank_hq8_scratch_bassetvl_sweep.sh`
- `launch/introns_seelig_a5ss_sd1_basset_branched_sweep.sh`
- `launch/utr3_hani_basset_branched_delta_aux_sweep.sh`
- `launch/utr3_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh`
- `launch/utr5_hani_basset_branched_delta_aux_sweep.sh`
- `launch/utr5_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh`
- `launch/utr_hani_resnet1d_cell_conditioned_delta_aux_sweeps.sh`
- `launch/utr5_polysome_fixed_all.sh`

## Near-Term Priorities

1. keep the standardized Lib1 in-house orchestrator as the main scratch-HPO surface
   - run path: `launch/lib1_inhouse_scratch_orchestrator.sh`
2. preserve the older 5'UTR polysome benchmark as a distinct task family
3. `run_registry/runs.csv` is now auto-populated; promote winning runs to
   `run_registry/best_runs.csv` and rely on `pretrained_registry.py`
   (see `PRETRAINED_MODELS.md`) for programmatic lookup
4. keep config naming comparison-friendly (verbose scheme
   `<region>__<target>__<mode>__<model>`) so additional model families
   can be evaluated side by side without touching W&B projects
5. prune stale generated outputs after exporting any run metadata needed for
   active decision notebooks
