# In-House CRE Scratch And Transfer Diagnostics

This directory is for notebooks that interpret in-house Lib1 CRE modeling runs:
one-shot scoring with pretrained checkpoints, from-scratch HPO on in-house
tables, and the first decision layer before a part-specific fine-tuning workflow
is promoted.

Use this folder when the biological question is:

- does a promoted public checkpoint transfer at all to an in-house Lib1 assay?
- is one-shot scoring good enough to justify a fine-tuning pipeline?
- can in-house Lib1 data support a useful model trained from scratch?
- which part class should receive the next scratch/fine-tune comparison?

## June 2026 Length Fact Check

The length-mismatch premise is real, but two details need to be stated
precisely:

| CRE part | Current in-house Lib1 table used by scratch runs | Current modeled length | Prior/public comparison | Readout |
|---|---|---:|---:|---|
| Promoter | `src/learn/derived_data/promoter/bashor_in_house/lib1_promoter_allvalid_fastqs1_5__learn_ready.tsv` | Mostly 50 nt, not exact: 7,774 of 7,893 usable rows are 50 nt; valid tail is 41-51 nt. Current scratch configs pad to `padded_seq_len=51` with neutral `N`. | Legacy e7/e30 split-safe table is exactly 84 nt. | Mismatch holds. If the intended experiment is exact 50 nt only, create a modal-50 promoter table; current runs are the `allvalid`/51-padded branch. |
| Intron | `src/learn/derived_data/introns/bashor_in_house/lib1_intron_modal80_fastqs1_5__learn_ready.tsv` | 80 nt exactly after modal-length filtering, 7,848 rows. | Seelig A5SS local processed table/config is 101 nt, not 100 nt. | Mismatch holds, and one-shot Seelig transfer should not be treated as a fair exact-length test. |
| 3 Prime UTR | `src/learn/derived_data/utr3/bashor_in_house/lib1_threeprime_modal100_fastqs1_5__learn_ready.tsv` | 100 nt exactly after modal-length filtering, 6,845 rows. | Hani 3 Prime UTR table is exactly 240 nt. | Mismatch holds; a dedicated 100 nt in-house model is justified before padding/truncating into the 240 nt public checkpoint. |
| 5 Prime UTR | `src/learn/derived_data/utr5/bashor_in_house/lib1_fiveprime_modal50_fastqs1_5__learn_ready.tsv` is prepared by the new scratch launcher. | Exact/modal 50 nt branch: 8,331 rows; 1,797 rows have `n_barcodes >= 8`. | Hani 5 Prime public models also use 50 nt inputs, but predict Hani RNA activity heads rather than in-house `log2_RNA_DNA`. | Length geometry is compatible; target/assay mismatch still makes an in-house-only scratch baseline useful for transfer-vs-scratch comparison. |
| Enhancer | Standardized no-flank HQ8 scratch configs now use `src/learn/derived_data/enhancer/bashor_in_house/lib1_fastqs1_5_0filtered_out__learn_ready.tsv`. | Raw enhancer insert column is mostly 200 nt, with a valid tail up to 211 nt; standardized scratch configs pad neutrally to `padded_seq_len=216`. | BODA2/Malinois transfer remains the stronger prior route. | Old scratch evidence was weak, but it predates the June fp32 / `val_pearson` / HQ8 standard; rerun before using enhancer as a negative prior for all parts. |

## Current In-House Scratch HPO Runs

Registry snapshot checked on 2026-06-09 from
`src/learn/run_registry/runs.csv` and `sweep_launches.csv`.

| Part | Run families | What is being tested | Current validation-first readout |
|---|---|---|---|
| Promoter | `promoter__bashor_in_house__lib1_allvalid__scratch__resnet1d` and `promoter__bashor_in_house__lib1_allvalid__scratch__promoter_bassetvl` | ResNet1D vs promoter-specific BassetVL; RC augmentation on/off; LR, weight decay, batch size, dropout/kernel/channel HPO. | BassetVL has the better validation leader: run `8fa94khq`, val Pearson 0.407, test Pearson 0.338. ResNet1D leader `dj68kcj0` has val Pearson 0.365, test Pearson 0.320; the ResNet test-max run reaches test Pearson 0.393 but should not be selected by test. |
| Intron | `introns__bashor_in_house__lib1_intron_modal80__scratch__resnet1d` | ResNet1D HPO; RC augmentation on/off; LR, weight decay, batch size, dropout/kernel/channel HPO. | Strong first pass. Validation leader `uyjz44qs` has val Pearson 0.603 and test Pearson 0.663. Test-max run `pkc2c2aa` reaches test Pearson 0.718. BassetVL has not yet been tested for this in-house intron branch. |
| 3 Prime UTR | `utr3__bashor_in_house__threeprime_modal100__scratch__resnet1d_fp32`, `utr3__bashor_in_house__threeprime_modal100__scratch__utr_bassetvl_fp32`, and `utr3__bashor_in_house__threeprime_modal100__scratch__utr_bassetvl_focused_rc_factorial_fp32` | ResNet1D vs UTR_BassetVL; fp32 numerical stability; focused BassetVL RC factorial over RC, split seed, model seed, LR, weight decay, and linear dropout. | BassetVL is ahead of ResNet1D. Focused BassetVL leader `fiqdi316` has val Pearson 0.437 and test Pearson 0.321; the test-max focused BassetVL run reaches test Pearson 0.448. ResNet1D fp32 validation leader `z0phesuc` has val Pearson 0.280 and test Pearson 0.254. |
| 3 Prime UTR early attempt | `utr3__bashor_in_house__threeprime_modal100__scratch__resnet1d` | Earlier ResNet1D pass before the fp32 rerun. | Exclude from model selection: multiple logged Pearson/R2 values are numerically impossible, so this run family is a metric/precision health diagnostic, not evidence. |
| 5 Prime UTR | `utr5__bashor_in_house__fiveprime_modal50__scratch__resnet1d_fp32` and `utr5__bashor_in_house__fiveprime_modal50__scratch__utr_bassetvl_fp32` | Exact/modal50 in-house-only scratch baseline; ResNet1D vs UTR_BassetVL; RC augmentation on/off; split/model seed, LR, weight decay, scheduler, batch norm, dropout/kernel/channel HPO. | Initial validation-first result favors BassetVL but not by the one-run `0.0831` contrast. BassetVL leader `lrjup2g1` has val Pearson 0.524 and test Pearson 0.389. ResNet1D leader `0ek1r95l` has val Pearson 0.482 and test Pearson 0.378. The early ResNet run `z7vooab8` / `f64lzjji` has best metric 0.083 and final val Pearson 0.000, but it is a poor trial, not the ResNet architecture result. |
| Enhancer standardized rerun | `enhancer__bashor_in_house__no_flank_hq8__scratch__resnet1d_fp32` and `enhancer__bashor_in_house__no_flank_hq8__scratch__bassetvl_fp32` | No-flank enhancer scratch rerun under the same June policy: HQ8 val/test, fp32, clean `val_pearson`, split seed, model seed, RC, and ResNet1D/BassetVL architecture comparison. | Not run yet at the time of this note. This is the fair comparison to the newer promoter/intron/UTR scratch runs; keep old enhancer scratch results labeled legacy. |

The current length facts support prioritizing in-house from-scratch training for
promoter, intron, and 3 Prime UTR. More precise interpretation:

- Intron is the strongest current in-house scratch signal, meaning it has the
  highest validation-selected held-out correlations so far. This does not mean
  intron is the most important Phase 2 biology; it means the current modal80
  scratch run already demonstrates trainable Lib1 signal.
- 3 Prime UTR is the most decisive recent architecture comparison because both
  ResNet1D and BassetVL were swept cleanly on the same modal100 branch, and
  BassetVL currently changes the model choice. It still needs the RC-factorial
  synthesis before promotion.
- Promoter is promising but not fully decision-ready because the active scratch
  branch is `allvalid` with neutral padding to 51 nt. If Phase 2 wants a pure
  50 nt promoter encoder, run the modal50 table and compare it against the
  current allvalid/51-padded branch before interpreting transfer or scratch
  differences.
- 5 Prime UTR now has the direct exact/modal50 in-house scratch comparison that
  was missing. BassetVL currently leads validation, but ResNet1D also trains
  well, so this is a promotion and transfer-vs-scratch question rather than a
  failed-architecture question.

## What Is Missing By CRE Part

| CRE part | Current coverage | Missing before a clean promotion decision |
|---|---|---|
| Enhancer | BODA2/Malinois transfer, barcode-bin studies, HQ8 analyses, and weak scratch evidence exist. | Finalize the canonical split/barcode policy and decide whether any scratch rerun would change the conclusion that transfer is the primary route. |
| Promoter | Lib1 scratch ResNet1D and PromoterBassetVL HPO exist on allvalid/51-padded data; legacy e7/e30 context exists. | Modal50 versus allvalid/51-padded decision, RC synthesis from current HPO, and legacy e7/e30 -> Lib1 fine-tune on the same split. |
| 5 Prime UTR | Hani public pretraining, Hani Lib2 fine-tune, Phase 3 Lib1+Lib2 scratch, in-house transfer/proxy HPO, and exact/modal50 in-house scratch ResNet1D/BassetVL HPO now exist. | Synthesize the new scratch HPO with Hani-pretrained transfer under the same validation-first policy; then choose whether the production seed is public scratch, transfer, or in-house scratch. |
| Intron | Lib1 modal80 ResNet1D scratch HPO is strong; Seelig one-shot transfer was negative. | Second architecture or confirmation seed, RC synthesis, and a decision on whether Seelig fine-tune is still worth running after the scratch baseline. |
| 3 Prime UTR | Lib1 modal100 ResNet1D and BassetVL scratch HPO exist; BassetVL focused RC factorial exists. | Final RC-factorial synthesis, seed/split stability check, and any deliberately labeled length-context or Hani-240 fine-tune comparison. |

## Run Checklist

Use this checklist for each scratch run family before promoting a checkpoint:

- Data policy: active sequence column, raw length distribution, modeled
  `input_len`/`padded_seq_len`, padding mode, target transform, train/val/test
  split seed, train barcode threshold, and held-out barcode threshold.
- RC augmentation: whether `use_reverse_complements` is an HPO axis or fixed
  value; compare RC within the same split/model-seed block whenever possible.
- Architecture: model module, channel counts, kernel sizes, pooling, linear
  layers, dropout, batch norm, and output heads.
- Optimization: optimizer, LR, weight decay, beta values, scheduler, batch
  size, precision, early-stopping patience, minimum/maximum epochs, and
  checkpoint monitor.
- Metrics: select by validation only, then report train/val/test Pearson,
  Spearman, Pearson R2, coefficient-of-determination R2, MSE, and loss from the
  selected checkpoint.
- Health checks: flag `abs(pearson) > 1`, very large metric values, negative
  COD R2 with high Pearson, train/test gaps, missing test metrics, and mismatch
  between `best_metric_value` and final selected-checkpoint metrics.

## Standardized Lib1 Scratch HPO Policy

The enhancer scratch result should not be used as a blanket prior that all Lib1
scratch models will fail. The current registry separates into two regimes:
enhancer scratch generalizes weakly, while the newer promoter, intron, 3 Prime,
and 5 Prime scratch runs all show learnable in-house signal. A fair all-part
campaign should therefore standardize the setup before making cross-part claims.

Recommended standard:

- Data: one declared learn-ready table per part; report raw length distribution
  and modeled `input_len`/`padded_seq_len`; use `train_min_barcodes=1` and
  HQ heldout rows with `test_min_barcodes=8` when enough HQ rows exist.
- Splits: include `split_seed` as a reported HPO/blocking axis, with values
  such as `[101, 202, 303]`; select by validation within each declared policy,
  then report held-out test once.
- Architectures: run a ResNet1D baseline for every part; run the part-matched
  Basset-family model where available: `PromoterBassetVL` for promoter,
  `UTR_BassetVL` for 3 Prime and 5 Prime, BassetVL for enhancer, and either a
  Basset-family intron branch or a focused ResNet confirmation grid before
  claiming the intron architecture question is done.
- RC: include `use_reverse_complements: [false, true]` unless a part has a
  documented orientation-specific reason to fix it; analyze RC within matched
  split/model-seed blocks rather than across arbitrary HPO trials.
- Seeds: include `model_seed` values such as `[1701, 1702]`; do not search over
  split seeds and then choose by test.
- Optimization: keep the broad June search space unless a focused confirmation
  grid is being run: Adam/AdamW, LR roughly `3e-5` to `3e-3`, weight decay
  roughly `1e-6` to `3e-3`, batch size `[64, 128, 256]`, scheduler
  `CosineAnnealingWarmRestarts` or `None`.
- Metrics/logging: fp32, `checkpoint_monitor: val_pearson`,
  `log_legacy_metric_aliases: false`, `epoch_eval_splits: [train, val, test]`,
  canonical names, and `runs.csv` registry rows.

What fp32 changes: fp32 means model activations, gradients, losses, and metric
reductions are run in 32-bit floating point rather than mixed/16-bit precision.
It costs more memory and may be slower, but it makes small regressions,
correlations, and checkpoint metrics less vulnerable to numerical noise,
underflow/overflow, or precision-specific metric failures. It is probably not
the sole reason the newer runs improved; the bigger policy changes are HQ8
heldout, clean `val_pearson` monitoring, part-specific length handling, enough
HPO, and architecture coverage. Use fp32 here mainly to make the HPO evidence
trustworthy.

Minimum campaign scale: use 64 runs per architecture for a first read. Use 128
runs per architecture for a serious part-level comparison. Use 256 runs per
architecture only after a 64/128-run screen shows the architecture is plausible
and the question is promotion-grade. With two architectures, "256 runs per part"
usually means 128 ResNet1D + 128 Basset-family, not 256 for each architecture.

Current standardization status:

| Part | Standard-ready? | Next action |
|---|---|---|
| Promoter | Mostly yes on allvalid/51-padded branch. | Decide whether to add a modal50 branch before promotion; otherwise synthesize ResNet1D vs PromoterBassetVL and RC. |
| Intron | Partly: ResNet1D is standardized and strong. | Add/run a Basset-family intron branch or a focused ResNet confirmation seed grid. |
| 3 Prime UTR | Yes for modal100 fp32; focused BassetVL RC grid exists. | Finish RC/split-seed synthesis and compare only to deliberately labeled length-context or fine-tune branches. |
| 5 Prime UTR | Yes for exact/modal50 ResNet1D and UTR_BassetVL. | Synthesize scratch versus Hani-pretrained transfer under the same validation-first policy. |
| Enhancer | Yes for a new no-flank HQ8 rerun; not yet complete. | Pilot the new ResNet1D/BassetVL launchers, then run 64/128 runs per architecture before deciding whether the transfer-first conclusion still holds. |

## Orchestrated HPO Campaign

The current repo supports orchestration at the `.sh` launcher layer. Each
curated part launcher owns data prep, W&B sweep creation, agent launch, and
`run_registry/sweep_launches.csv` / `runs.csv` provenance. The meta-launcher
`src/learn/launch/lib1_inhouse_scratch_orchestrator.sh` composes those
launchers without replacing them.

Dry-run the ready parts:

```bash
DRY_RUN=1 PREPARE_ONCE=0 GPU_LIST="0 1 2 3 4 5 6 7" RUNS_PER_SWEEP=128 \
  PARTS="promoter intron utr3 utr5 enhancer" MODE=parallel_by_part \
  bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh
```

Run one sweep after another, giving every selected GPU to each sweep:

```bash
GPU_LIST="0 1 2 3" RUNS_PER_SWEEP=128 MODE=sequential \
  PARTS="promoter intron utr3 utr5 enhancer" \
  bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh
```

Run architecture sweeps in parallel within each part, then wait before moving
to the next part:

```bash
GPU_LIST="0 1 2 3 4 5 6 7" RUNS_PER_SWEEP=128 MODE=parallel_by_part \
  PARTS="promoter intron utr3 utr5 enhancer" \
  bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh
```

The orchestrator requires `RUNS_PER_SWEEP` to divide evenly by the number of
agents assigned to each sweep, so a 4-GPU sequential 128-run sweep becomes
`NUM_AGENTS=4`, `NUM_RUNS=32`, while a 4-GPU parallel-by-part two-architecture
run becomes two concurrent sweeps with `NUM_AGENTS=2`, `NUM_RUNS=64` each.
Use `PILOT=1` first to force one 1-run smoke test per launcher and verify W&B
history/summary capture before a long campaign.

With the same `PARTS` and `RUNS_PER_SWEEP`, `MODE=sequential` and
`MODE=parallel_by_part` request the same number of HPO trials per sweep; they
only differ in scheduling. In sequential mode, every architecture sweep gets all
selected GPUs and the next sweep starts only after the previous one finishes. In
parallel-by-part mode, the orchestrator splits the selected GPUs across the
architecture launchers for one part, runs those sweeps concurrently, waits for
all of them, and then moves to the next part.

Pilot examples:

```bash
DRY_RUN=1 PILOT=1 GPU_LIST="0" PARTS="enhancer" MODE=sequential \
  bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh
```

Real pilot plus W&B cloud-history verification:

```bash
PILOT=1 GPU_LIST="0" PARTS="enhancer" MODE=sequential \
  bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh

conda run --no-capture-output -n boda_env python src/learn/verify_wandb_history.py \
  --latest \
  --project enhancer__bashor_in_house__no_flank_hq8__scratch__bassetvl_fp32 \
  --keys val_pearson val_loss trainer/global_step
```

The verifier uses `wandb.Api().run(...).scan_history(keys=[...])`, so it checks
the same cloud history backing W&B Charts rather than only `runs.csv` or local
`.wandb` files. Future standardized `logger_type: wandb` Lib1 scratch runs must
have canonical history rows for the train/val/test loss, MSE, Pearson, squared
Pearson, Spearman, and coefficient-of-determination metrics on
`trainer/global_step`, plus a startup `wandb_history_canary` row. W&B
init/history failures should be treated as pilot failures, not silently
downgraded logger runs.

```bash
PILOT=1 GPU_LIST="0" PARTS="promoter intron utr3 utr5 enhancer" MODE=sequential \
  bash src/learn/launch/lib1_inhouse_scratch_orchestrator.sh
```

For a pilot with `MODE=parallel_by_part`, provide at least as many GPUs as the
number of architecture sweeps being run concurrently for that part; enhancer,
promoter, 3 Prime, and 5 Prime currently have two architecture launchers.

## Metric Naming Cleanup

The current in-house scratch configs already move in the right direction:
`checkpoint_monitor: val_pearson` and `log_legacy_metric_aliases: false`.
Older and public-data runs still use names such as `epoch_end_val_pearson_r2`,
`epoch_end_val_r2`, and `val_r2_score`, so cross-run analysis must normalize
names before ranking.

Recommended logging contract:

- Canonical scalar names should be
  `{split}_{loss,pearson,spearman,pearson_r2,cod_r2,mse}` for `train`, `val`,
  and `test`.
- `val_pearson` should be the default monitor for one-output in-house Lib1
  scratch runs. Use `val_pearson_r2` only when a legacy/public run truly
  monitors squared Pearson.
- In the registry, keep `val_r2` only as a backwards-compatible alias when
  needed; prefer explicit `val_pearson_r2` and `val_cod_r2`.
- Analysis notebooks should map legacy aliases into the canonical columns and
  display the original `best_metric_name` so metric provenance is visible.
- For future configs, keep `epoch_eval_splits: [train, val, test]` when
  runtime allows, so per-epoch diagnostics and selected-checkpoint summaries
  use the same canonical split names.

Current cleanup status: current in-house Lib1 scratch configs that run through
`src/learn/train_wandb_log.py` with `checkpoint_monitor: val_pearson`,
`log_legacy_metric_aliases: false`, fp32, and
`epoch_eval_splits: [train, val, test]` should produce the cleanest HPO logging.
Older public and early scratch runs are not retroactively renamed; analysis
notebooks should keep normalizing their aliases. A rerun of every region is only
worth it if the existing run lacks clean validation/test metrics, was affected
by the 3 Prime non-fp32 metric issue, or needs a different split/length policy.

### W&B chart-history recovery

Observed on 2026-06-09: some new 5 Prime scratch runs have valid local W&B
history files and valid cloud summaries, but the W&B Charts tab and
`scan_history` API return no history rows. In that case, do not rerun solely
because the panels are blank. First check `src/learn/run_registry/runs.csv` for
final validation/test metrics and export the local history curves:

```bash
conda run --no-capture-output -n boda_env python src/learn/export_wandb_history.py \
  --project utr5__bashor_in_house__fiveprime_modal50__scratch__utr_bassetvl_fp32 \
  --sweep-id ba6t98l8 \
  --output-dir src/learn/run_registry/wandb_history_exports/utr5_fiveprime_bassetvl_ba6t98l8

conda run --no-capture-output -n boda_env python src/learn/export_wandb_history.py \
  --project utr5__bashor_in_house__fiveprime_modal50__scratch__resnet1d_fp32 \
  --sweep-id 0ieyxjkk \
  --output-dir src/learn/run_registry/wandb_history_exports/utr5_fiveprime_resnet1d_0ieyxjkk
```

The exported `manifest.tsv` and per-run `*__history.tsv` files contain
`trainer/global_step`, `val_pearson`, `val_spearman`, `val_loss`, `train_*`,
and `test_*` columns for notebook plots. Checksum warnings usually mean a local
run file is still being written; rerun the exporter after agents finish.
The same-day promoter, intron, and 3 Prime W&B projects displayed metrics
normally, so this should be treated as a 5 Prime project/run-history recovery
issue rather than a global reason to rerun HPO.

For split seeds, randomizing val/test from the `n_barcodes >= 8` pool is useful
as a stability check because it estimates how dependent a result is on one HQ
heldout partition. It is not a substitute for a held-out test set: do not select
final models by repeatedly searching across split seeds. Treat split seed as a
reported blocking factor, then promote by validation-first selection and a final
test readout under the declared policy.

## FivePrime Scratch Baseline

The in-house 5 Prime baseline added for direct comparison with Hani 5 Prime
transfer uses exact/modal 50 nt FivePrime rows, `log2_RNA_DNA`, HQ8 val/test
rows, fp32, clean metric names, and from-scratch ResNet1D/BassetVL sweeps:

- dataset prep: `src/learn/prepare_lib1_fiveprime_inhouse_dataset.py`
- ResNet1D sweep config:
  `src/learn/configs/utr5/bashor_in_house/resnet1d/lib1_fiveprime_modal50__scratch_resnet1d__bayes.yml`
- ResNet1D launcher: `src/learn/launch/lib1_fiveprime_scratch_resnet1d_sweep.sh`
- BassetVL sweep config:
  `src/learn/configs/utr5/bashor_in_house/utr_bassetvl/lib1_fiveprime_modal50__scratch_utr_bassetvl__bayes.yml`
- BassetVL launcher: `src/learn/launch/lib1_fiveprime_scratch_utr_bassetvl_sweep.sh`

Both scratch configs test RC as an HPO factor with
`use_reverse_complements: [false, true]`. In this data module, RC doubles only
the training examples; validation and test are evaluated on the original
orientation.

Use this as the first transfer-vs-scratch comparator for the Hani 5 Prime
pretrained route. If it beats or ties transfer, run a focused confirmation grid
over split seed, model seed, and RC; if it loses, the scratch result still
anchors how much public pretraining is helping the in-house proxy target.

Initial readout from `runs.csv`: BassetVL is the current validation leader
(`lrjup2g1`, val Pearson 0.524, test Pearson 0.389), with ResNet1D close behind
(`0ek1r95l`, val Pearson 0.482, test Pearson 0.378). Compare architectures by
their validation leaders or by a matched seed/RC summary, not by the early poor
ResNet trial `z7vooab8` / `f64lzjji` whose best metric was 0.083 and final
`val_pearson` was 0.000.

## Current Canonical Notebooks

| Notebook | Role | Current status |
|---|---|---|
| `promoter_intron_inhouse_pretrained_eval_may2026.ipynb` | One-shot in-house promoter and intron evaluation using promoted public checkpoints. | canonical diagnostic |
| `lib1_inhouse_scratch_hpo_best_models_june2026.ipynb` | In-house Lib1 scratch HPO review for promoter, intron, and 3 Prime UTR; links sequence-length/data-distribution checks to best-run and metric-naming diagnostics. | canonical scratch-HPO analysis scaffold |

## Boundary With Neighboring Folders

- Public-data HPO and checkpoint-promotion notebooks stay in
  `../pretraining_CRE_public_data/`.
- Dedicated fine-tuning notebooks should move under
  `../fine_tuning/<part_class_or_project>/`.
- Reusable scoring or plotting code should move into `src/analysis/` or
  `src/finetune/` once it is called by more than one notebook.

Generated prediction tables, checkpoints, and per-run folders should stay out of
Git unless they are small decision artifacts.
