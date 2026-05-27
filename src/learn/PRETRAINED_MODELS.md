# Pretrained Models — Registry and Lookup

This document describes how the per-region pretrained models are organized,
how they get registered, and how downstream active-learning code resolves
"the best model for region X" without hard-coded paths.

## Registry files

Both registries live under `src/learn/run_registry/`:

| File | Writer | Purpose |
| --- | --- | --- |
| `runs.csv` | `train_wandb_log.py` (automatic, one row per finished run) | Append-only log of every training run with full W&B identifiers, best-checkpoint metric, test/val/train R2/Pearson/Spearman/loss, and the saved `.tar.gz` path. |
| `best_runs.csv` | Manually curated | Canonical "this is the model we stand behind" entry per `task_family`/`target_family`/`comparison_group`. Preferred by default in `resolve_pretrained`. |
| `sweep_launches.csv` | `launch/_wandb_helpers.sh` (one row per `wandb sweep`) | Historical trail of sweep dispatches (entity/project/sweep_id, launcher, GPU list). Not consulted when resolving a model, but useful to re-run or extend a sweep. |

## How a run gets registered

1. The launcher shell script exports `BODA_CONFIG_PATH`, `BODA_TASK_FAMILY`,
   `BODA_TARGET_FAMILY`, `BODA_COMPARISON_GROUP`, `BODA_LAUNCH_SCRIPT`,
   `BODA_SWEEP_PATH`, `BODA_WANDB_ENTITY`, `BODA_WANDB_PROJECT`,
   `BODA_RUNS_CSV`, and `BODA_LAUNCH_NOTES` before starting `wandb agent`.
2. Every `wandb agent` spawns a Python `train_wandb_log.py` subprocess that
   inherits those env vars.
3. After `trainer.fit`, `set_best` restores the best checkpoint, then
   `trainer.test` is invoked and a train-set inference pass is run to
   populate `test_*` and `train_*` metrics on the W&B run summary.
4. `save_model` writes:
     * `torch_checkpoint.pt` (model state + hparams, same as before),
     * `provenance.json` (flat record of every field in
       `RUNS_CSV_COLUMNS`, including W&B entity/project/run_id/sweep_id,
       metrics, artifact path, git commit, hostname, and launch notes),
     * `model_artifacts__<project>__<run_id>__<timestamp>.tar.gz` — the
       filename itself encodes enough to reconstruct the W&B run.
5. `append_runs_csv_row` appends one row to `run_registry/runs.csv` (or
   the path in `BODA_RUNS_CSV`) with the same provenance dict.

## Artifact and checkpoint layout

There are three related outputs, with different jobs:

| Location | What it is | When to use it |
| --- | --- | --- |
| `local_artifacts/**/*.tar.gz` | Portable model bundle written after the best checkpoint is restored. Contains `artifacts/torch_checkpoint.pt` and `artifacts/provenance.json`. | Default source for transfer learning, inference, and registry promotion. |
| `<wandb_project>/<run_id>/checkpoints/*.ckpt` | Raw PyTorch Lightning checkpoint produced during training. | Resume/debug a Lightning training run, inspect callback state, or recover when artifact export failed. |
| `<wandb_project>/best_checkpoint_model/<run_id>/` | Optional clean mirror for humans, enabled by `--best_checkpoint_dir`. Contains the portable checkpoint plus provenance and selection metadata, and copies `lightning_best.ckpt` when local. | Quick browsing and handoff without hunting through long artifact names or raw run folders. |

The `.ckpt` files are useful, but they are not the main pretrained-model
handoff format. `train_wandb_log.py` calls `set_best` before `save_model`,
so the `torch_checkpoint.pt` inside each saved artifact already contains
the best-checkpoint model weights rather than the final epoch by accident.

For downstream code, prefer:

```text
run_registry/best_runs.csv -> model_saved_path/artifact_path -> local_artifacts/*.tar.gz
```

The `best_checkpoint_model/` directory is just a predictable browsing layer.
It does not replace `local_artifacts/` or the registry.

## Promoting a run to the "best" list

1. Inspect `runs.csv` (or the W&B UI) to pick the winning run per region.
2. Append one CSV row to `best_runs.csv` with `registry_status=current`.
   Minimum required columns are `task_family`, `target_family`,
   `comparison_group`, `run_id`, `artifact_path` (or `model_saved_path`),
   `metric_name`, `metric_value`, `config_path`, `launch_script`,
   `model_module`, and `graph_module`.
3. Optionally set the previous row for that region to
   `registry_status=superseded` so history is preserved.

## Looking up a model in code

```python
from learn.pretrained_registry import resolve_pretrained

rec = resolve_pretrained("utr3", target_family="hani_rna_activity")
if rec is None:
    raise RuntimeError("No pretrained 3'UTR model registered yet.")

print(rec.best_artifact())  # absolute path to the .tar.gz
print(rec.wandb_project, rec.run_id, rec.metric_name, rec.metric_value)
```

Set `prefer="latest"` to always take the most recent `runs.csv` entry
even when `best_runs.csv` has a curated row — useful during pilot
iteration.

To load the model into memory:

```python
from boda.common.utils import unpack_artifact, model_fn

unpack_artifact(rec.best_artifact(), "/tmp/bundle")
model = model_fn("/tmp/bundle/artifacts")
```

## CLI

```bash
# List every task_family the registries know about.
python src/learn/pretrained_registry.py --list-regions

# Print a status table across regions.
python src/learn/pretrained_registry.py --summary

# Resolve the curated best model for a region.
python src/learn/pretrained_registry.py --task-family utr5 --target-family hani_rna_activity
```

## Current state (as of repo reorg)

See `../../plan/learn/hpo_repo_reboot_plan.md` for the ongoing inventory. The
agent-facing summary lives in `run_registry/best_runs.csv` and is the
source of truth — run `pretrained_registry.py --summary` any time you
need a fresh snapshot.
