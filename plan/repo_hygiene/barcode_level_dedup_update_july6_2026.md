# Barcode-Level Dedup Data Update Plan

Generated: 2026-07-06

Status: active planning note for updating the MattLee Lib1 barcode-level and
single-part variant-level data products after confirming repeated same-
construct/same-barcode rows are duplicate records. Multi-part regeneration is
deferred until the pML299 normalization discussion is resolved.

Execution update, 2026-07-06: single-part data-level steps 1-4 were completed
with `src/data_prep/generate_lib1_single_part_dedup_data.py`. New
`.dedup_exact` barcode-level, single-part variant-level, per-library
barcode-level, manifest, validation, and pre-dedup archive-copy artifacts now
exist under the MattLee Lib1 data root. Original CSV paths were left in place
pending the repo code/default update.

`$BODA_WORK_ROOT` denotes the workspace containing this checkout and the
private data roots; set it explicitly when those directories are not siblings.

Primary data root:

```text
$BODA_WORK_ROOT/opt_EU_learn_n_design/MattLee_lib1/
```

Primary repo root:

```text
<path-to-boda2_EU>/
```

## Current Finding

The barcode-level file
`barcode_level/L1_variant_bc_expr_combined_20251107_np_fastq1-5.csv` contains
repeated rows for the same `parts_concatenated` plus same exact
`bba1_ddc1_concat` barcode.

Current variant-level tables are reconstructable from raw barcode-row sums:

- `number_of_barcodes` matches distinct nonblank `bba1_ddc1_concat` values.
- `DNA_bc_counts_sum` matches the sum over all raw barcode rows, including
  duplicate rows.
- `RNA_bc_counts_sum` matches the sum over all raw barcode rows, including
  duplicate rows.
- `RNA/DNA` matches `RNA_bc_counts_sum / DNA_bc_counts_sum`.

After exact duplicate rows are removed, `number_of_barcodes` stays matched but
DNA sums, RNA sums, and many ratio targets change. Therefore the current
variant-level expression targets are incorrect if the repeated rows are data
duplication rather than intended repeated observations.

## What Previous Modeling Used

The previous Lib1 single-part HPO paths did not apply an experimental
library-specific normalization upstream of the variant-level target. The
variant CSVs themselves show that `RNA/DNA` is exactly
`RNA_bc_counts_sum / DNA_bc_counts_sum` for all current single-part and
multi-part files, to floating-point precision.

What they used:

- Promoter, intron, 5 Prime, and 3 Prime learn-ready preparation read the
  variant-level `RNA/DNA` column and wrote `log2_RNA_DNA`.
- Enhancer learn-ready preparation read variant-level `RNA/DNA` and created
  `RNA_DNA_Ratio_log10_scaled = log10(RNA/DNA) + 2`.
- Some enhancer finetuning scripts recomputed the target as
  `log10(RNA_bc_counts_sum / DNA_bc_counts_sum)`.
- Lib1 HPO configs typically set `normalize: true`, which standardizes the
  already-created target using train-set mean and standard deviation. That is
  ML scaling for optimization, not experimental/library normalization.

For multi-part variant files, `log_RNA/DNA_pML299_norm` is also reconstructable
from the stored aggregate counts:

```text
log_RNA/DNA_pML299_norm = log10(RNA_bc_counts_sum / DNA_bc_counts_sum)
                          + 0.837791179471
```

Equivalently:

```text
log_RNA/DNA_pML299_norm = log10((RNA/DNA) / 0.145281)
```

That looks like a pML299 control baseline offset, but it is not a different
per-library offset in the current CSVs: the same constant holds across every
multi-part row and file.

Implication: previous HPOs are affected at the label level if the duplicate
rows are accidental. The old HPOs can still be useful as rough architectural
signals, but they should not be treated as final clean-label results.

## Experimental Normalization Versus ML Z-Score Scaling

Experimental or library-level normalization would change the biological target
before the model sees it. It would answer questions like: "relative to which
assay control, sequencing-depth baseline, library batch, or backbone should
this construct be compared?"

Examples:

```text
raw_log_activity_i = log10(sum_RNA_i / sum_DNA_i)
control_normalized_i = raw_log_activity_i - raw_log_activity_control_library
```

or equivalently:

```text
control_normalized_i = log10((RNA/DNA)_i / (RNA/DNA)_control)
```

This kind of normalization can:

- make values more comparable across libraries or batches;
- remove a known backbone/control baseline;
- change the biological interpretation from "absolute aggregate RNA/DNA" to
  "activity relative to control";
- change ranking across libraries if each library has a different baseline;
- affect active-learning objectives if acquisition compares scores from
  different libraries.

For current single-part files, the data do not show this extra experimental
normalization. The immediate single-part target is:

```text
RNA/DNA = sum_RNA_bc_counts / sum_DNA_bc_counts
```

with sums now needing to be recomputed after exact deduplication. The log
targets used in modeling are simple transforms of that ratio.

Train-set z-score scaling inside the ML pipeline is different. The Lib1 HPO
configs usually set `normalize: true`, which makes the data module compute:

```text
y_train_scaled = (y_train - mean(y_train)) / std(y_train)
y_val_scaled   = (y_val   - mean(y_train)) / std(y_train)
y_test_scaled  = (y_test  - mean(y_train)) / std(y_train)
```

This helps optimization by putting targets on a stable numeric scale. It is
not a biological correction, does not use controls, and does not fix duplicate-
inflated labels. It is an affine transform of the already-created target, fit
from the training split only to avoid validation/test leakage.

## Weekly Meeting Topic: Multi-Part / Biological Normalization

Most target-construction details can be answered from the data. The remaining
questions are now narrower and mostly about confirming the biological meaning
of the pML299 baseline.

1. Does the constant pML299 offset correspond to a pML299 control ratio of
   approximately `0.145281`?

2. Was that pML299 baseline intended to be global across all multi-part
   libraries, as the current files imply?

3. For dedup-derived multi-part files, should we keep using the same pML299
   baseline constant, or recompute it from a deduplicated pML299 control row or
   control set?

4. Were pseudocounts used anywhere upstream before the current CSVs were
   written? The current `RNA/DNA` columns do not show pseudocount use for rows
   with positive DNA, because they exactly equal `RNA_sum / DNA_sum`.

5. Should final modeling targets include both raw log aggregate ratio and
   pML299-baseline log ratio for multi-part work?

Concrete framing to send:

```text
We found repeated identical rows for the same parts_concatenated +
bba1_ddc1_concat barcode. Current variant tables count those repeats in
DNA_bc_counts_sum and RNA_bc_counts_sum, but not in number_of_barcodes. The
current RNA/DNA columns are exactly RNA_sum / DNA_sum. The multi-part
log_RNA/DNA_pML299_norm column is exactly log10(RNA/DNA) + 0.837791179471,
equivalent to normalizing by a pML299 ratio of about 0.145281. For the
deduplicated regeneration, should we keep that same pML299 baseline constant,
or recompute the baseline from deduplicated pML299 control observations?
```

## Data-Level Update: Single-Part First

Goal: create new deduplicated barcode-level and variant-level files without
destroying the current files until validation passes. The immediate scope is
single-part data only. Multi-part files are a TODO after the weekly pML299 /
normalization discussion.

### Proposed Output Names

Barcode level:

```text
opt_EU_learn_n_design/MattLee_lib1/barcode_level/
  L1_variant_bc_expr_combined_20251107_np_fastq1-5.dedup_exact.csv
  L1_variant_bc_expr_combined_20251107_np_fastq1-5.dedup_exact.manifest.json
```

Single-part variant level, flattened into the existing root:

```text
opt_EU_learn_n_design/MattLee_lib1/single_part_variant_level/
  L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.dedup_exact.csv
  L1_final_fastqs1-5_sublibrary_Promoter_subset.dedup_exact.csv
  L1_final_fastqs1-5_sublibrary_FivePrime_subset.dedup_exact.csv
  L1_final_fastqs1-5_sublibrary_Intron_subset.dedup_exact.csv
  L1_final_fastqs1-5_sublibrary_ThreePrime_subset.dedup_exact.csv
```

The non-filtered enhancer superset
`L1_final_fastqs1-5_sublibrary_enhancer_subset.csv` has 5273 rows. The current
canonical modeling file is
`L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.csv`, with 4788
rows. The removed 485 rows are exactly non-positive or nonfinite expression
rows:

```text
RNA/DNA NaN:        151
RNA/DNA == 0:       334
DNA == 0 & RNA == 0: 145
DNA > 0 & RNA == 0:  334
DNA == 0 & RNA > 0:    6
```

The June-standard `src/learn` enhancer launchers and configs use the
`0filtered_out` source and derived files:

```text
src/learn/derived_data/enhancer/bashor_in_house/
  lib1_fastqs1_5_0filtered_out__learn_ready.tsv
  lib1_fastqs1_5_0filtered_out__learn_ready_log2_target.tsv
```

The more recent filtered/raw-ratio enhancer finetuning scripts also default to
`L1_final_fastqs1-5_sublibrary_enhancer_subset_0filtered_out.csv`. Therefore
the dedup update should treat `0filtered_out` as the canonical enhancer
single-part table and archive the unfiltered superset as an audit artifact
unless a future zero-count modeling plan needs it.

Per-library barcode-level modeling tables:

```text
opt_EU_learn_n_design/MattLee_lib1/barcode_level/by_library/
  single_part__enhancer_subset_0filtered_out.dedup_exact.barcode_level.csv
  single_part__Promoter_subset.dedup_exact.barcode_level.csv
  single_part__FivePrime_subset.dedup_exact.barcode_level.csv
  single_part__Intron_subset.dedup_exact.barcode_level.csv
  single_part__ThreePrime_subset.dedup_exact.barcode_level.csv
```

### Step 1: Deduplicate Barcode-Level Data

Policy:

- Drop exact duplicate rows, not merely repeated
  `parts_concatenated + bba1_ddc1_concat` keys.
- Keep one representative of each exact duplicated row.
- Preserve all original columns.
- Add a manifest with:
  - source file path
  - source SHA256
  - output SHA256
  - row counts before and after dedup
  - duplicate rows removed
  - deduplication subset used
  - timestamp
  - script/notebook used

Critical check:

- Repeated same-construct/same-barcode groups should have identical DNA/RNA
  counts and barcode scores before exact dedup is made canonical. If any group
  has count variability, it should be flagged instead of silently deduped.

### Step 2: Recreate Single-Part Variant-Level Data

For each canonical single-part variant-level table:

1. Read original table to get the intended row set and column order.
2. Join to deduplicated barcode rows by `parts_concatenated`.
3. Recompute:
   - `number_of_barcodes` as distinct nonblank `bba1_ddc1_concat`
   - `DNA_bc_counts_sum` as deduped barcode DNA sum
   - `RNA_bc_counts_sum` as deduped barcode RNA sum
   - `RNA/DNA` as `RNA_bc_counts_sum / DNA_bc_counts_sum`
4. Preserve sequence/part columns from the original table.
5. Recompute file-specific log columns from the deduped aggregate counts:
   - single-part modeling target: `log2_RNA_DNA = log2(RNA/DNA)` in
     learn-ready outputs
   - enhancer legacy target: `RNA_DNA_Ratio_log10_scaled = log10(RNA/DNA) + 2`

Acceptance checks:

- `parts_concatenated` row set matches the corresponding old variant file,
  unless we deliberately choose to expand the row set.
- Recomputed `number_of_barcodes` matches the old value unless a barcode ID
  itself was duplicated in a way that changes distinctness. Expected: it should
  match.
- Deduped DNA/RNA sums are less than or equal to old raw-row sums.
- `RNA/DNA` equals recomputed `RNA_bc_counts_sum / DNA_bc_counts_sum`.
- Each output gets a manifest recording the source variant file, barcode-level
  source, and aggregation policy.

### Step 3: Extract Single-Part Per-Library Barcode-Level Tables

For barcode-level modeling, create one deduplicated barcode table per
single-part variant library by filtering the deduped barcode table to the library's
`parts_concatenated` set.

Each per-library barcode file should include:

- all original barcode-level columns
- `library_layer`: `single_part`
- `library_name`
- `variant_file`
- optional `part_pattern`
- optional per-barcode target columns such as
  `log2_RNA_DNA_barcode = log2((RNA_bc_counts + alpha_R) /
  (DNA_bc_counts + alpha_D))`, but only after pseudocount policy is chosen

Important leakage rule:

- Barcode-level modeling splits must group by `parts_concatenated`, not by
  individual barcode row, if the goal is sequence-to-expression
  generalization. Otherwise the same sequence can appear in train and test
  through different barcodes.

### Step 4: Archive Old Data

Do not compress or move the old CSVs until the deduped outputs and manifests
are validated.

After validation:

- Compress the old duplicate-containing barcode file.
- Compress old single-part variant-level CSVs whose targets were derived from
  duplicate-inflated sums.
- Keep multi-part files in place for now, or archive only after the pML299
  discussion and multi-part dedup regeneration.
- Prefer an explicit archive folder over replacing paths with compressed files:

```text
opt_EU_learn_n_design/MattLee_lib1/archive_pre_dedup_20260706/
  barcode_level/
  single_part_variant_level/
  MANIFEST.md
```

Reason: preserving old paths while converting files to `.gz` can break code
that expects `.csv`. If path preservation matters for notebooks, keep a small
readme or symlink strategy instead of silently changing file formats in place.

### Completed Data-Level Artifacts, 2026-07-06

Regeneration script:

```text
boda2_EU/src/data_prep/generate_lib1_single_part_dedup_data.py
```

Run command:

```text
python src/data_prep/generate_lib1_single_part_dedup_data.py --archive-old
```

Barcode-level result:

```text
rows before exact dedup: 927311
rows after exact dedup:  549160
exact duplicate rows removed: 378151
```

The audit found 2 same `parts_concatenated + bba1_ddc1_concat` groups with
non-identical barcode split/score fields. They were zero-count rows and were
not collapsed by exact-row deduplication. Details are recorded in:

```text
opt_EU_learn_n_design/MattLee_lib1/barcode_level/
  L1_variant_bc_expr_combined_20251107_np_fastq1-5.dedup_exact.variable_key_audit.csv
```

Validation summary:

```text
opt_EU_learn_n_design/MattLee_lib1/single_part_variant_level/
  dedup_exact.validation_summary.json
```

All five canonical single-part row sets matched the original variant files,
`number_of_barcodes` matched exactly, deduped DNA/RNA sums never increased,
and `RNA/DNA` exactly matched the recomputed deduped aggregate ratio.

Archive policy used:

```text
opt_EU_learn_n_design/MattLee_lib1/archive_pre_dedup_20260706/
```

The archive contains gzip copies of the old barcode, canonical single-part
variant files, and unfiltered enhancer superset. Original CSV paths were left
in place so current notebooks and scripts continue to work until the repo code
defaults are updated to the `.dedup_exact` products.

## Repo Code Updates After Data Products Exist

Required updates in `boda2_EU`:

- Update `src/learn/prepare_lib1_*` defaults to point to the deduped
  variant-level CSVs.
- Update finetuning scripts that read single-part variant-level files.
- Update tutorials to prefer deduped files by default and keep the raw duplicate
  file only as an audit artifact.
- Add a small data manifest or README pointing from repo code to the external
  data root.
- Add an explicit `dedup_policy` or source filename field to learn-ready
  metadata JSONs.

Optional but useful:

- Add a standalone script under `src/analysis` or `src/data_prep` to regenerate
  barcode-level and variant-level deduped tables. Notebook-only regeneration is
  too easy to drift.

## Subsequent Data Update Git Plan

This plan should be executed after the current dirty working tree is committed
as a checkpoint and pushed by the user.

Recommended version-control sequence:

1. Start from a clean working tree on
   `checkpoint/learn-finetune-docs-may2026`.
2. Confirm the branch tip matches remote before beginning the dedup work.
3. Generate external data products under:

```text
$BODA_WORK_ROOT/opt_EU_learn_n_design/MattLee_lib1/
```

4. Keep large CSV data outside Git unless the project explicitly decides to
   track them elsewhere.
5. Commit only repo-side changes in `boda2_EU`, such as:
   - reproducible data-prep scripts
   - validation notebooks or compact validation outputs
   - updated source-data defaults
   - updated learn-ready metadata
   - documentation and manifests pointing to the external data root
6. Make the data-update commit separate from the current checkpoint commit.
7. Use a commit message like:

```text
Update Lib1 data pipeline for barcode deduplication
```

8. Before pushing, run:

```text
git status --short --branch
git diff --stat HEAD
```

Acceptance for the data-update commit:

- The commit should not mix old run-history cleanup with dedup implementation.
- The commit should identify deduped data paths and aggregation policy.
- The commit should make it clear which single-part CSVs are canonical.
- The commit should leave multi-part regeneration deferred until the pML299
  normalization discussion is resolved.

## Deferred Multi-Part TODO

Do not regenerate multi-part canonical files until after the weekly discussion
on pML299 normalization.

Known from the current CSVs:

```text
RNA/DNA = RNA_bc_counts_sum / DNA_bc_counts_sum
log_RNA/DNA_pML299_norm = log10(RNA/DNA) + 0.837791179471
```

Open decision:

- Keep the same pML299 offset after dedup, or recompute it from deduplicated
  pML299 control observations?
- Generate both raw `log10(RNA/DNA)` and pML299-normalized targets for
  downstream multi-part modeling?
- Decide whether multi-part barcode-level modeling should use per-library
  baselines before pooling different multi-part libraries.

## Follow-Up 2: Redo Focused HPOs

Purpose: check whether deduped targets change generalization performance and
whether barcode-weighted loss now behaves more sensibly.

Initial scope:

- Promoter from scratch
- 5 Prime UTR from scratch
- Existing ResNet1D and/or BassetVL configs
- Same or near-same hyperparameter ranges as previous HPOs
- No barcode weighting first, then paired barcode-weighted runs

Why barcode-weighted loss may work better after dedup:

- Before dedup, `number_of_barcodes` counted distinct barcode identities, but
  DNA/RNA sums counted duplicate raw rows. That means target construction and
  barcode reliability weights were not aligned.
- If duplicates inflated some barcode observations more than others, the
  aggregate target became an arbitrary duplicate-weighted barcode average.
- After dedup, `number_of_barcodes` is a cleaner proxy for independent barcode
  support.
- Weighted loss can then more honestly ask whether variants with more
  independent barcode measurements have lower label noise.

Critical notes:

- Do not compare old and new HPOs as pure architecture comparisons. The label
  changed.
- Use the same train/val/test split policy where possible, but regenerate split
  manifests from deduped learn-ready tables.
- Keep metrics unweighted on validation/test so weighted and unweighted
  training remain comparable.
- Track both raw target distribution changes and performance changes. A better
  target can lower apparent correlation if the old target had duplicate-driven
  artifacts that were easy to fit.

Minimal paired experiment grid:

| Part | Architecture | Target | Loss | Purpose |
|---|---|---|---|---|
| Promoter | ResNet1D | deduped `log2_RNA_DNA` | unweighted MSE | baseline rerun |
| Promoter | ResNet1D | deduped `log2_RNA_DNA` | barcode-weighted MSE | reliability ablation |
| 5 Prime | ResNet1D | deduped `log2_RNA_DNA` | unweighted MSE | baseline rerun |
| 5 Prime | ResNet1D | deduped `log2_RNA_DNA` | barcode-weighted MSE | reliability ablation |
| Promoter/5 Prime | BassetVL | deduped `log2_RNA_DNA` | optional | architecture check |

## Follow-Up 3: Barcode-Level Uncertainty HPO Design

Priority: higher than broad reruns of all variant-level HPOs.

Discussion handoff/context brief:

```text
boda2_EU/plan/phase1_lib1/learn/barcode_level_uncertainty_discussion_context_july7_2026.md
```

Goal: model barcode-level observations and uncertainty directly, rather than
collapsing immediately to one construct-level target.

Candidate modeling target:

```text
y_ij = log2((RNA_bc_counts_ij + alpha_R) / (DNA_bc_counts_ij + alpha_D))
```

where `i` is construct and `j` is barcode. The pseudocounts `alpha_R` and
`alpha_D` should be decided before this becomes canonical.

Split policy:

- Split by `parts_concatenated`, not barcode row.
- Never allow the same DNA construct in train and validation/test through
  different barcodes.

Model families to test:

1. Mean-only barcode-level model
   - Treat each deduped barcode row as one observation.
   - Evaluate construct-level prediction by aggregating barcode-level
     predictions or by comparing sequence-level prediction to deduped
     construct target.

2. Heteroscedastic model
   - Predict mean and variance.
   - Optimize Gaussian negative log likelihood.
   - Evaluate calibration by barcode count and library.

3. Hierarchical or two-stage model
   - Sequence model predicts construct mean.
   - Barcode-level residual model estimates observation noise.
   - Use construct-level aggregate target for final sequence ranking.

Uncertainty outputs that matter:

- predictive mean per construct
- predictive variance per construct
- empirical barcode variance per construct
- number of distinct barcodes
- DNA/RNA coverage status
- calibration curves by library and barcode count bin

Critical risks:

- Per-barcode rows are not guaranteed independent if barcode construction,
  sequencing depth, or merge artifacts introduce shared noise.
- Zero counts need a principled target policy. Dropping all zeros may bias
  toward expressed constructs; pseudocounts may compress low-count biology.
- Barcode-level rows from the same construct must not leak across splits.
- Library-specific baseline normalization, if real, should be applied before
  comparing uncertainty across libraries.

## Immediate Next Checklist

- [ ] Bring pML299 / biological normalization topic to weekly meeting for
      multi-part planning. This is not a blocker for single-part dedup.
- [x] Write a reproducible data-prep script for exact barcode dedup and variant
      reconstruction.
- [x] Generate deduped barcode-level CSV and manifest.
- [x] Generate deduped single-part variant-level CSVs in the flat
      `single_part_variant_level/` root.
- [x] Generate deduped single-part per-library barcode-level modeling CSVs.
- [x] Validate reconstruction against old raw-row policy and new exact-dedup
      policy.
- [x] Archive/compress old duplicate-containing barcode and single-part data
      only after validation.
- [x] Archive the unfiltered enhancer superset as audit/provenance, keeping
      `enhancer_subset_0filtered_out` as the canonical enhancer modeling table.
- [ ] Update repo defaults and tutorials to use deduped data.
- [ ] Regenerate learn-ready TSVs and metadata JSONs from deduped variant data.
- [ ] Run focused promoter and 5 Prime deduped HPO smoke/baseline jobs.
- [ ] Design and launch barcode-level uncertainty HPO.

## Stop Conditions

Pause before making deduped files canonical if any of these happen:

- Same construct plus exact barcode groups have non-identical DNA/RNA counts or
  scores after a full audit.
- Teammate confirms repeated rows were intentional independent observations.
- Deduped variant row sets no longer match the intended library membership.
- Any downstream code silently mixes old raw-row targets and new deduped
  targets in the same experiment.

For multi-part work, add another stop condition until resolved:

- The pML299/library normalization recipe should not be guessed if the weekly
  discussion contradicts the constant-offset reconstruction observed in the
  current CSVs.
