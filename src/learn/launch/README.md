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

By default, launchers do not force a W&B location; sweep placement comes from
top-level `entity` and `project` in each YAML config. Set
`WANDB_SWEEP_ENTITY`/`WANDB_SWEEP_PROJECT` only when you want to override that
placement at launch time.

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

If `SWEEP_ID` is already known, set it before launching agents. Use:

- the full sweep path: `entity/project/sweep_id`

Sweep identity note:

- curated sweep YAMLs now carry explicit top-level `entity` and `project`
- the helper validates that the created sweep path matches those values
- `parameters.logger_project` is still useful experiment metadata, but it is not the authoritative W&B project locator under sweep execution
- see `../WANDB_SWEEP_WORKFLOW.md` for the full workflow

## Current Scripts

- `enhancer_malinois_basset_branched_baseline.sh`
- `enhancer_malinois_basset_nonbranched_single_head_k562_sweep.sh`
- `enhancer_malinois_basset_nonbranched_single_head_combined_sweep.sh`
- `promoter_deboer_utr_bassetvl_sweep.sh`
- `utr3_hani_utr_bassetvl_sweep.sh`
- `utr5_hani_utr_bassetvl_sweep.sh`
- `utr5_polysome_fixed_all.sh`

Enhancer combined-target note:

- `enhancer_malinois_basset_nonbranched_single_head_combined_sweep.sh` will
  build the derived pan-cell training table automatically if it does not exist
- set `FORCE_REBUILD_DATASET=1` to regenerate that table before launching

`deploy_wandb_agent_train.sh` is now a legacy one-off example rather than the
preferred launch surface.
