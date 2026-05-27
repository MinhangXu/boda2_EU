# Connect Hani Lib1/Lib2 With In-House UTR Data: May 2026 Update

## Current Status

Phase 2 is complete for the BODA-first 5' UTR transfer baseline.

We fine-tuned the canonical Hani/Goodarzi 5' UTR Lib1-pretrained BODA `ResNet1DRegressor` checkpoint `1mmy39ku` on Hani/Goodarzi 5' UTR Lib2. The run used a deterministic sequence-level Lib2 split and evaluated:

- Lib2 validation and test.
- Lib1 test retention.
- In-house exact-length `FivePrime` candidates as a proxy transfer diagnostic.

The strongest Lib2 validation config was:

```text
unfreeze_scope = full
head_lr = 3e-4
backbone_lr = 1e-5
target_scaler_source = pretrained_lib1_train
monitor_metric = val_average_activity_pearson
```

## Phase 2 Results

### Lib2 generalization

Fine-tuning improves held-out Lib2 performance in a consistent but moderate way.

| Readout | Pretrained `1mmy39ku` | Selected fine-tuned config mean | Delta |
|---|---:|---:|---:|
| Lib2 test average-activity Pearson | 0.714 | 0.786 | +0.072 |
| Lib2 test mean per-head Pearson | 0.548 | 0.618 | +0.070 |
| Lib2 test flattened Pearson | 0.538 | 0.624 | +0.086 |
| Lib2 test average-activity RMSE | 0.138 | 0.094 | -0.043 |

Interpretation: Lib2 fine-tuning is doing real transfer adaptation, but this was still a small first sweep. It is worth running a broader transfer HPO before concluding that the observed gain is close to the ceiling.

### Lib1 retention

Lib1 retention drops after Lib2 fine-tuning, which is expected for this kind of adaptation but still important to diagnose.

| Readout | Pretrained `1mmy39ku` | Selected fine-tuned config mean | Delta |
|---|---:|---:|---:|
| Lib1 retention average-activity Pearson | 0.833 | 0.820 | -0.013 |
| Lib1 retention mean per-head Pearson | 0.755 | 0.736 | -0.019 |
| Lib1 retention flattened Pearson | 0.750 | 0.703 | -0.046 |
| Lib1 retention average-activity RMSE | 0.089 | 0.096 | +0.007 |

Interpretation: the average-activity retention loss is modest, but the flattened Pearson drop is large enough that we need sequence-level diagnostics. The next question is not just "did retention drop?", but "which Lib1 sequences got worse, which improved, and where do they sit relative to Lib2 and in-house sequence space?"

### In-house FivePrime proxy

The in-house signal does not favor the Lib2-validation-selected full-unfreeze checkpoint.

| In-house HQ subset readout | Spearman |
|---|---:|
| Pretrained average-head predictor vs `log2_RNA_DNA` | 0.227 |
| Selected full-unfreeze average-head predictor vs `log2_RNA_DNA` | 0.207 |
| Best observed proxy row: `head_only`, `head_lr=1e-4`, seed 11, `pred_activity_c1` | 0.237 |
| Pretrained `pred_activity_c6` | 0.232 |

The high-quality in-house diagnostic subset used `number_of_barcodes >= 8` and contained 1,797 exact-length finite rows. All exact finite `FivePrime` rows total 8,331.

Interpretation: do not use this in-house proxy as an HPO target yet. The Hani public task is fluorescence/FACS-bin-derived RNA activity across specific cell lines, while the in-house target is an RNA/DNA sequencing ratio. That difference may reflect assay biology, preprocessing, barcode/count effects, or sequence distribution shift.

### Early stopping and training dynamics

The first sweep did not show meaningful early stopping. Best checkpoints were selected very late:

```text
best epoch range: 95-100 out of 100
```

This differs from prior enhancer fine-tuning runs where full unfreeze often stopped earlier. For Hani Lib2 transfer, the validation curve may still have been improving near the epoch cap. Follow-up HPO should include longer runs and/or scheduler variants before deciding whether full unfreeze is truly optimal.

## Architecture Note

For this ResNet1D run, `last_stage_plus_head` is not a tiny head-adaptation setting. In `1mmy39ku`, the encoder has six residual blocks from `stage_blocks = [2, 2, 2]`. `last_stage_plus_head` trains:

```text
encoder.4.*
encoder.5.*
head.*
```

The stem and `encoder.0.*` through `encoder.3.*` stay frozen after the head-only warmup. This is roughly 2.15M trainable parameters out of roughly 2.87M total, so it is better thought of as late-backbone adaptation plus head calibration.

## Biological Interpretation Notes

The Hani heads are:

```text
c1  = MDA-MB-231
c2  = HepG2
c4  = Jurkat
c6  = SW480
c17 = NALM6
```

If the in-house context is HEK293, then epithelial/adherent carcinoma-like heads such as c1, c2, and c6 are a more plausible prior than lymphoid heads c4 and c17. However, the current in-house proxy plot does not cleanly identify c2 as best. c1 head-only is the best observed fine-tuned proxy row, and pretrained c6/c17 also carry signal. Treat biology as a weak prior; use sequence-neighborhood analysis to test where each head transfers.

## Phase 3 Results Placeholder

Phase 3 should remain open until the next experiment finishes.

Planned comparison surface:

| Experiment | Selection metric | Lib2 test | Lib1 retention | In-house FivePrime proxy | Notes |
|---|---|---:|---:|---:|---|
| Phase 2: Lib2 fine-tune from `1mmy39ku` | Lib2 val average-activity Pearson | complete | complete | diagnostic only | current baseline |
| Phase 3: Lib1+Lib2 production pretraining or broader transfer HPO | TBD | pending | pending | diagnostic only unless holdout is defined | leave untouched in-house holdout |

Phase 3 should not overwrite the current canonical pretrained row until it beats Phase 2 on the intended promotion criteria.

## Next Steps

1. Build the `sequence_landscape` package as a sibling repo at `/home/minhang/synBio_AL/sequence_landscape`, installed editable into `boda_env`.
2. Add a thin BODA adapter that loads the Phase 2 prediction CSVs and normalizes them into the landscape schema.
3. For Lib2, rank sequences by `abs_err_pretrained - abs_err_finetuned` to find top improved and worsened sequences.
4. For Lib1 retention, rank sequences by the same transfer effect to identify retention failures and retained/improved neighborhoods.
5. Project Lib1 train/test, Lib2, and in-house `FivePrime` into one-hot and k-mer PCA spaces.
6. Quantify nearest-neighbor distance to Lib1 train and ask whether error or fine-tuning improvement depends on distance.
7. Debug in-house preprocessing with barcode/count stratification before treating `log2_RNA_DNA` as a model-selection target.
8. Run a broader transfer HPO: longer max epochs, scheduler variants, `target_scaler_source=lib2_train`, smaller backbone LRs, and explicit comparison of `head_only`, `last_stage_plus_head`, and `full`.

