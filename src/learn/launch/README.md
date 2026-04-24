# `src/learn/launch` Guide

This directory contains curated launch scripts that match the reorganized
`configs/` tree.

## Intended Workflow

1. choose a config from `src/learn/configs/...`
2. run the matching script in `src/learn/launch/`
3. let the script create a W&B sweep and launch agents
4. monitor the sweep on W&B
5. use `src/analysis/hpo_results_eval_utils.py` and `src/learn/run_registry/`
   to recover best runs and artifact paths

## Common Controls

All W&B sweep launchers support:

- `NUM_AGENTS`
- `NUM_RUNS`
- `GPU_LIST`
- `SWEEP_ID`
- `CREATE_SWEEP_ONLY=1`
- `LAUNCH_NOTES`
- `WANDB_SWEEP_ENTITY`
- `WANDB_SWEEP_PROJECT`
- `PILOT=1` — forces `NUM_AGENTS=1`, `NUM_RUNS=1`, a single GPU, and
  `LAUNCH_NOTES` defaulting to `pilot` for smoke-testing the full
  train → test → `runs.csv` chain

By default, launchers do not force a W&B location; sweep placement comes from
top-level `entity` and `project` in each YAML config (verbose scheme
`<task_family>__<target_family>__<mode>__<model_family>`). Set
`WANDB_SWEEP_ENTITY`/`WANDB_SWEEP_PROJECT` only when you want to override that
placement at launch time.

Launch metadata is propagated to every training process via
`BODA_CONFIG_PATH`, `BODA_TASK_FAMILY`, `BODA_TARGET_FAMILY`,
`BODA_COMPARISON_GROUP`, `BODA_LAUNCH_SCRIPT`, `BODA_SWEEP_PATH`,
`BODA_WANDB_ENTITY`, `BODA_WANDB_PROJECT`, `BODA_RUNS_CSV`, and
`BODA_LAUNCH_NOTES`. These land in `run_registry/runs.csv` automatically
— no extra bookkeeping is required in the launcher.

Examples:

```bash
cd /home/minhang/synBio_AL/boda2_EU/src/learn
bash launch/utr3_hani_utr_bassetvl_sweep.sh
```

```bash
cd /home/minhang/synBio_AL/boda2_EU/src/learn
NUM_AGENTS=4 NUM_RUNS=10 GPU_LIST="0 1 2 3" bash launch/promoter_deboer_utr_bassetvl_sweep.sh
```

```bash
cd /home/minhang/synBio_AL/boda2_EU/src/learn
CREATE_SWEEP_ONLY=1 bash launch/utr5_hani_utr_bassetvl_sweep.sh
```

```bash
cd /home/minhang/synBio_AL/boda2_EU/src/learn
GPU_POOL="0 1 2 3 4 5 6 7" bash launch/run_public_datasets_hpo_batch.sh
```

If `SWEEP_ID` is already known, set it before launching agents. Use:

- the full sweep path: `entity/project/sweep_id`

Sweep identity note:

- curated sweep YAMLs now carry explicit top-level `entity` and `project`
- the helper validates that the created sweep path matches those values
- `parameters.logger_project` is still useful experiment metadata, but it is not the authoritative W&B project locator under sweep execution
- see `../WANDB_SWEEP_WORKFLOW.md` for the full workflow

## Current Scripts

- `enhancer_malinois_basset_branched_baseline.sh`
- `enhancer_malinois_basset_nonbranched_single_head_combined_sweep.sh` (legacy config path)
- `lib1_enhancer_scratch_compare_loss_modes.sh`
- `lib1_enhancer_scratch_weighted_sweep.sh`
- `promoter_deboer_compare_architectures.sh`
- `promoter_deboer_utr_bassetvl_sweep.sh`
- `run_public_datasets_hpo_batch.sh` — launches the current promoter/UTR
  HPO batch across multiple GPUs, optionally in detached `screen` sessions
- `utr3_hani_utr_bassetvl_sweep.sh`
- `utr5_hani_utr_bassetvl_sweep.sh`
- `utr5_polysome_utr_bassetvl_sweep.sh`
- `utr5_polysome_fixed_all.sh`
- `run_all_regions_pilot.sh` — orchestrates a 1-agent / 1-run PILOT
  across every region sequentially. Accepts `REGIONS="<space list>"`
  to pick a subset and `DRY_RUN=1` to print commands without executing
  them.

Promoter architecture note:

- `promoter_deboer_compare_architectures.sh` accepts `MODE=utr_bassetvl`,
  `MODE=bassetvl`, or `MODE=resnet1d`
- `run_all_regions_pilot.sh` now uses that script for promoter smoke tests, so
  the umbrella pilot stays aligned with the individually debugged promoter modes
- all three modes share one `comparison_group` so their `runs.csv` rows can be
  compared directly after HPO

5'UTR polysome note:

- `utr5_polysome_utr_bassetvl_sweep.sh` accepts
  `LIBRARY=egfp_1|egfp_2|mcherry_1|mcherry_2`
- use it for actual HPO / pilot-HPO runs
- keep `utr5_polysome_fixed_all.sh` for fixed-parameter benchmark reruns

Enhancer combined-target note:

- `enhancer_malinois_basset_nonbranched_single_head_combined_sweep.sh` will
  build the derived pan-cell training table automatically if it does not exist
- it now points at `configs/legacy/.../basset_nonbranched/...` so the
  archived Malinois non-branched experiment stays runnable without looking like
  a current default path
- set `FORCE_REBUILD_DATASET=1` to regenerate that table before launching

`deploy_wandb_agent_train.sh` is now a legacy one-off example rather than the
preferred launch surface.
