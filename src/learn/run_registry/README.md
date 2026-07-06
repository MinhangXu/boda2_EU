# `src/learn/run_registry` Guide

This directory is the lightweight machine-readable registry layer for training
and HPO bookkeeping.

## Files

- `best_runs.csv`
  - curated best-known runs across task families
- `runs.csv`
  - append-only per-run manifest written by `src/learn/train_wandb_log.py`
- `sweep_launches.csv`
  - append-only sweep launch log written by scripts in `src/learn/launch/`
- `wandb_history_exports/`
  - local fallback exports from `src/learn/export_wandb_history.py` when W&B
    cloud summaries exist but chart/history rows are not queryable

## Intended Use

Use `best_runs.csv` when you need to answer:

- what is the current best run for this task?
- which config path produced it?
- where should I look for artifacts or model paths?
- what comparison group did it belong to?

Use `runs.csv` when you need to compare all runs in a sweep or recover the
exact metric values that selected a checkpoint. Legacy `val_r2`, `test_r2`,
and `train_r2` are Pearson correlation squared; prefer explicit
`*_pearson_r2`, `*_cod_r2`, and `*_mse` columns for new analyses.

Use `sweep_launches.csv` when you need to answer:

- which sweep did I just launch?
- which config and launch script created it?
- how many agents / runs were used?
- which GPUs were assigned?

Use `wandb_history_exports/` when W&B Charts or `scan_history` are blank for a
run that has local `src/learn/wandb/run-*/run-*.wandb` files. Export a sweep
with:

```bash
conda run --no-capture-output -n boda_env python src/learn/export_wandb_history.py \
  --project <wandb_project> \
  --sweep-id <sweep_id> \
  --output-dir src/learn/run_registry/wandb_history_exports/<short_name>
```

## Workflow

1. launch a sweep using `src/learn/launch/`
2. let the launcher append a row to `sweep_launches.csv`
3. inspect results in W&B
4. if W&B charts are blank, export local history rows with
   `src/learn/export_wandb_history.py`
5. confirm best checkpoints in notebook or local cache
6. promote the winning run into `best_runs.csv`
