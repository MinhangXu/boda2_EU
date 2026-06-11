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
bash launch/utr3_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh
```

```bash
cd /home/minhang/synBio_AL/boda2_EU/src/learn
NUM_AGENTS=4 NUM_RUNS=10 GPU_LIST="0 1 2 3" \
  bash launch/introns_seelig_a5ss_sd1_basset_branched_sweep.sh
```

```bash
cd /home/minhang/synBio_AL/boda2_EU/src/learn
CREATE_SWEEP_ONLY=1 bash launch/utr5_hani_basset_branched_delta_aux_sweep.sh
```

Lib1 in-house scratch orchestration:

```bash
cd /home/minhang/synBio_AL/boda2_EU
DRY_RUN=1 PREPARE_ONCE=0 GPU_LIST="0 1 2 3 4 5 6 7" RUNS_PER_SWEEP=128 \
  PARTS="promoter intron utr3 utr5 enhancer" MODE=parallel_by_part \
  bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh
```

```bash
cd /home/minhang/synBio_AL/boda2_EU
GPU_LIST="0 1 2 3" RUNS_PER_SWEEP=128 MODE=sequential \
  PARTS="promoter intron utr3 utr5 enhancer" \
  bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh
```

```bash
cd /home/minhang/synBio_AL/boda2_EU
GPU_LIST="0 1 2 3 4 5 6 7" RUNS_PER_SWEEP=128 MODE=parallel_by_part \
  PARTS="promoter intron utr3 utr5 enhancer" \
  bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh
```

```bash
cd /home/minhang/synBio_AL/boda2_EU
DRY_RUN=1 PILOT=1 GPU_LIST="0" PARTS="enhancer" MODE=sequential \
  bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh
```

`MODE=sequential` gives all selected GPUs to one sweep at a time. With
`RUNS_PER_SWEEP=128` and `GPU_LIST="0 1 2 3"`, each sweep launches
`NUM_AGENTS=4`, `NUM_RUNS=32`. `MODE=parallel_by_part` runs the architecture
sweeps for one part concurrently and waits before the next part; with two
architecture launchers and four GPUs, each sweep gets two agents and
`NUM_RUNS=64`. `PILOT=1` forces each launched sweep to one GPU and one run, so
use `MODE=sequential` for a one-GPU smoke test or provide enough GPUs for the
parallel architecture split.

W&B history verification after a real pilot:

```bash
cd /home/minhang/synBio_AL/boda2_EU
PILOT=1 GPU_LIST="0" PARTS="enhancer" MODE=sequential \
  bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh

conda run --no-capture-output -n boda_env python src/learn/verify_wandb_history.py \
  --latest \
  --project enhancer__bashor_in_house__no_flank_hq8__scratch__bassetvl_fp32 \
  --keys val_pearson val_loss trainer/global_step
```

The verifier uses `wandb.Api().run(...).scan_history(keys=[...])` against W&B
cloud history. Standardized `logger_type: wandb` HPO runs define canonical
`{train,val,test}_{loss,mse,pearson,pearson_r2,spearman,cod_r2}` metrics on
`trainer/global_step`, write a `wandb_history_canary` row at run start, log
canonical metrics through both Lightning and explicit `wandb.log`, and fail
loudly instead of falling back to a non-W&B logger when W&B init fails.

If `SWEEP_ID` is already known, set it before launching agents. Use:

- the full sweep path: `entity/project/sweep_id`

Sweep identity note:

- curated sweep YAMLs now carry explicit top-level `entity` and `project`
- the helper validates that the created sweep path matches those values
- `parameters.logger_project` is still useful experiment metadata, but it is not the authoritative W&B project locator under sweep execution
- see `../WANDB_SWEEP_WORKFLOW.md` for the full workflow

## Current Scripts

- `enhancer_malinois_basset_branched_baseline.sh`
- `lib1_enhancer_no_flank_hq8_scratch_bassetvl_sweep.sh`
- `lib1_enhancer_no_flank_hq8_scratch_resnet1d_sweep.sh`
- `lib1_fiveprime_scratch_resnet1d_sweep.sh`
- `lib1_fiveprime_scratch_utr_bassetvl_sweep.sh`
- `lib1_inhouse_scratch_orchestrator.sh` — orchestrates current standardized
  Lib1 in-house scratch launchers. `MODE=sequential` gives all GPUs to one
  sweep at a time; `MODE=parallel_by_part` splits GPUs across architecture
  sweeps within each part and waits before moving to the next part. It includes
  the standardized enhancer no-flank HQ8 ResNet1D/BassetVL launchers.
- `lib1_intron_scratch_resnet1d_sweep.sh`
- `lib1_promoter_scratch_promoter_bassetvl_sweep.sh`
- `lib1_promoter_scratch_resnet1d_sweep.sh`
- `lib1_threeprime_scratch_resnet1d_sweep.sh`
- `lib1_threeprime_scratch_utr_bassetvl_sweep.sh`
- `introns_seelig_a5ss_sd1_basset_branched_sweep.sh`
- `utr3_hani_basset_branched_delta_aux_sweep.sh`
- `utr3_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh`
- `utr5_hani_basset_branched_delta_aux_sweep.sh`
- `utr5_hani_resnet1d_cell_conditioned_delta_aux_sweep.sh`
- `utr_hani_resnet1d_cell_conditioned_delta_aux_sweeps.sh`
- `utr5_polysome_fixed_all.sh`

5'UTR polysome note:

- HPO sweep configs/launchers were retired in the June 2026 Lib1 cleanup.
- Keep `utr5_polysome_fixed_all.sh` for fixed-parameter benchmark reruns.

Hani 5'UTR Phase 3 note:

- The old Lib1+Lib2 phase3 scratch launcher/config was retired after it served
  its comparison role.
- Keep `tutorials/lib1_tasks/pretraining_CRE_public_data/hani_utr5_lib1_lib2_phase3_scratch_hpo_analysis_may2026.ipynb`
  as the decision record.

Enhancer legacy note:

- The non-branched Malinois single-head sweep and old Bashor enhancer
  basic/weighted/FASTQ1-5 launchers were retired in the June 2026 cleanup.
- The remaining Malinois launcher is the BassetBranched transfer baseline.

`deploy_wandb_agent_train.sh` is now a legacy one-off example rather than the
preferred launch surface.
