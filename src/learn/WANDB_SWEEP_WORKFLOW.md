# W&B Sweep Workflow

This document defines the canonical W&B workflow for `src/learn`.

## The Three Identifiers

Keep these separate:

- W&B sweep project:
  - the actual W&B location where the sweep and runs live
  - represented by `entity/project`
  - by default this comes from top-level YAML `entity` and `project`
- sweep id:
  - the short W&B id for one sweep, such as `qbj4v71s`
  - together with `entity/project`, this gives the full sweep path
- config `logger_project`:
  - a task-level config value passed to `train_wandb_log.py`
  - useful for grouping runs by experiment family inside cached metadata
  - not reliable as the actual W&B project under sweep execution

## Curated Launcher Contract

The curated launchers in `launch/` now follow this contract:

1. the sweep YAML contains explicit top-level `entity` and `project`
2. the launcher passes through `WANDB_SWEEP_ENTITY` and `WANDB_SWEEP_PROJECT` only when you explicitly set them
3. `launch/_wandb_helpers.sh` materializes the sweep config with those values before running `wandb sweep`
4. the helper parses the resulting full sweep path and validates that the resolved `entity/project` matches
5. the helper records the sweep launch in `run_registry/sweep_launches.csv`

This means new sweeps should no longer land in an implicit or accidental W&B project.

## Environment Controls

All curated launchers accept:

- `NUM_AGENTS`
- `NUM_RUNS`
- `GPU_LIST`
- `SWEEP_ID`
- `CREATE_SWEEP_ONLY=1`
- `LAUNCH_NOTES`
- `WANDB_SWEEP_ENTITY`
- `WANDB_SWEEP_PROJECT`

The last two override the YAML defaults when you intentionally want to place a sweep in a different W&B location.

## Reusing An Existing Sweep

If a sweep already exists, pass the full sweep path:

```bash
SWEEP_ID="minhangxu1998-baylor-college-of-medicine/boda2_EU-src_learn/qbj4v71s" \
bash launch/enhancer_malinois_basset_nonbranched_single_head_combined_sweep.sh
```

For an agent-only attach flow, use `deploy_wandb_agent_train.sh` with a full `SWEEP_ID`.

## Notebook And Recovery Guidance

When recovering best runs:

- use the full W&B project path plus sweep id to query W&B
- use local `wandb/run-*` cache plus `wandb/sweep-*` files to map cached runs back to sweep membership
- treat `logger_project` as metadata, not as the authoritative W&B project locator

Useful files:

- `launch/_wandb_helpers.sh`
- `run_registry/sweep_launches.csv`
- `wandb/run-*/files/config.yaml`
- `wandb/run-*/files/output.log`
- `wandb/run-*/files/wandb-summary.json`

## Practical Rule

If you are asking "where does this sweep live on W&B?", use:

- top-level YAML `entity`
- top-level YAML `project`
- recorded `sweep_id`

If you are asking "what experiment family was this run meant to belong to?", use:

- `parameters.logger_project`
