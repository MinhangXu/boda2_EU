# Sequence Landscape Diagnostics: Fast Implementation Plan for CRE MPRA Model Transfer

**Working name:** `sequence_landscape`  
**Recommended repo target:** a sibling repo at `/home/minhang/synBio_AL/sequence_landscape`, installed editable into the same `boda_env` environment. Keep BODA-specific notebooks and tiny adapters in `boda2_EU`, but keep the reusable feature/projector/metric code outside `boda2_EU/boda`.  
**Primary goal:** quickly implement quantitative and visual diagnostics for fixed-length synthetic CRE sequence landscapes, then empirically test whether these diagnostics explain model transfer, fine-tuning gains, and failures across Hani/Goodarzi Lib1, Hani/Goodarzi Lib2, and in-house 5′ UTR data.

---

## 0. Why this file exists

We have a concrete model-transfer observation from the Phase 2 BODA-first 5′ UTR fine-tuning run:

```text
Fine-tuning the BODA ResNet1DRegressor Lib1-pretrained checkpoint 1mmy39ku on Lib2 improves Lib2 test average-activity Pearson:
  pretrained: 0.714
  selected fine-tuned config mean: 0.786

But Lib1 retention takes a modest hit:
  average-activity Pearson: 0.833 -> 0.820
  flattened Pearson drop is larger.

The validation-selected config in the first sweep is:
  unfreeze_scope = full
  head_lr = 3e-4
  backbone_lr = 1e-5
  target_scaler_source = pretrained_lib1_train
```

We want to answer three immediate questions:

1. **Which Lib2 sequences get better predictions after Lib2 fine-tuning?**
2. **Which Lib1 sequences get worse predictions after Lib2 fine-tuning?**
3. **Where do Lib1, Lib2, and in-house 5′ UTR sequences sit in raw sequence space and, later, model embedding space?**
4. **Are Lib1 retention losses concentrated in recognizable sequence neighborhoods, such as low-density Lib1 regions, Lib2-like regions, or GC/composition extremes?**
5. **Is the weak in-house 5′ UTR proxy signal an assay/preprocessing issue, a cell-type mismatch issue, or a true public-to-in-house sequence-distribution shift?**

This module should be implementable quickly. It should not become a general-purpose genomics visualization framework at first. It should be a lightweight diagnostic package that produces figures and tables we can immediately use in BODA notebooks and presentations.

Current BODA notebook that should consume these diagnostics:

```text
boda2_EU/tutorials/lib1_tasks/fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/
  hani_utr5_lib2_phase2_finetune_analysis_may2026.ipynb
```

Current fine-tuning outputs to support first:

```text
boda2_EU/src/finetune/learning_curve/hani_utr5_lib2_resnet1d_1mmy39ku_phase2_may2026/
```

---

## 1. Relationship to the deep research report

The attached deep-research report is useful for framing the broader literature: MPRA sequence-function maps, alignment-free sequence comparison, PCA/UMAP/MDS, distribution-shift testing, and model-embedding diagnostics.

However, for implementation, the report is broader than we need. The first version should focus on our exact setting:

```text
fixed-length or near-fixed-length synthetic CRE regions
+ ACTG sequence strings
+ MPRA / FACS / RNA-DNA activity labels
+ model predictions before vs after fine-tuning
+ raw-sequence 2D landscapes
+ quantitative train-vs-design-vs-in-house shift metrics
```

The broader literature review should be treated as a separate background task. The implementation should not wait for a perfect literature map.

---

## 2. Scope of v1

### In scope

- One CRE region at a time, especially 5′ UTR first.
- Fixed-length sequence handling as the default.
- Exact-length filtering for position-aware analyses.
- Raw-sequence landscape construction from:
  - flattened one-hot encoding,
  - k-mer frequency vectors,
  - optional Hamming distance / MDS for small subsets.
- PCA as the default reducer.
- UMAP as optional if already installed or easy to add.
- Overlaying measured expression/activity, prediction error, and fine-tuning improvement/regression.
- Comparing Lib1, Lib2, and in-house data in the same coordinate system.
- Quantifying distribution shift and coverage using simple metrics.
- Producing CSV tables plus publication-safe PNG/PDF figures.

### Out of scope for v1

- Full model-embedding extraction from every architecture.
- Foundation model embeddings.
- Large-scale interactive web viewers.
- Active-learning acquisition optimization.
- Full automated literature/tool benchmarking.
- Handling arbitrary variable-length genomics datasets.

### v1 philosophy

Start with a small, interpretable, ACTG-close diagnostic system:

```text
sequence string -> raw feature vector -> 2D map -> expression/error/shift overlays
```

Then add model embeddings only after raw-sequence diagnostics are working.

---

## 3. Data assumptions

### Sequence assumptions

For each module run:

- Analyze one CRE region at a time.
- Most sequences should have approximately the same expected length.
- For position-aware views, require exact length.
- For near-fixed-length or deviating sequences, either:
  - filter them out for one-hot/Hamming analysis, or
  - analyze them with k-mer features only.

Example expected lengths:

```text
BODA/PARADE 5′ UTR: 50 nt
BODA/PARADE 3′ UTR: 240 nt for the current public-checkpoint comparison surface
in-house 5′ UTR: likely comparable fixed length, depending on preprocessing
in-house 3′ UTR: often 100 nt and should not be directly compared to the 240 nt public-checkpoint surface using one-hot geometry
```

### Activity assumptions

Activity can be any scalar target, including:

- expected FACS bin,
- log2 RNA/DNA,
- average activity across cell-type heads,
- specific cell-type activity,
- cell-type deviation / delta,
- prediction residual or absolute error.

For the first implementation, use:

```text
average_activity = mean activity across available cell-type heads
absolute_error = abs(predicted_average_activity - true_average_activity)
error_improvement = abs_error_pretrained - abs_error_finetuned
```

Positive `error_improvement` means fine-tuning improved the prediction.

---

## 4. Immediate implementation questions

### Question A: Which Lib2 sequences improved after fine-tuning?

For each Lib2 sequence with truth, pretrained prediction, and fine-tuned prediction:

```text
true_avg = mean observed activity across heads
pred_avg_pre = mean pretrained predicted activity across heads
pred_avg_ft = mean fine-tuned predicted activity across heads
abs_err_pre = abs(pred_avg_pre - true_avg)
abs_err_ft = abs(pred_avg_ft - true_avg)
error_improvement = abs_err_pre - abs_err_ft
```

Then rank:

```text
top improved Lib2 sequences: largest positive error_improvement
top worsened Lib2 sequences: most negative error_improvement
```

Useful columns:

```text
seq
library
region
true_avg
pred_avg_pre
pred_avg_ft
abs_err_pre
abs_err_ft
error_improvement
true_activity_by_head...
pred_pre_by_head...
pred_ft_by_head...
GC
length
nearest_lib1_distance
raw_PC1
raw_PC2
```

### Question B: Which Lib1 sequences got worse after Lib2 fine-tuning?

Same calculation, but on Lib1 test or Lib1 retained validation/test sequences.

Rank:

```text
top Lib1 retention failures: most negative error_improvement
```

This table is biologically important because it asks whether Lib2 fine-tuning overwrote or shifted the model away from specific Lib1 sequence neighborhoods.

For the current run, start with the validation-selected checkpoint:

```text
finetuned_1mmy39ku__seed7__full__hlr3p0e04__blr1p0e05__scaler_pretrained_lib1_train
```

Then repeat the same tables for gentler in-house-favorable configs, especially:

```text
head_only, head_lr=1e-4
```

Reason: the full unfreeze config is best on Lib2 validation/test, but the in-house proxy plot suggests head-only fine-tuning preserves or improves some output-head transfer signals better.

### Question C: Are improved/worsened sequences localized in sequence space?

Project sequences into raw-sequence PCA space. Overlay:

- Lib1 train/test,
- Lib2,
- in-house 5′ UTR,
- error improvement,
- true activity,
- pretrained error,
- fine-tuned error,
- distance to Lib1 train manifold.

Key plot:

```text
PCA raw-sequence map:
  points = sequences
  x/y = raw sequence PCs
  color = error_improvement
  shape = dataset/language: Lib1, Lib2, in-house
```

Interpretation:

```text
Lib2 improved far from Lib1 train:
  fine-tuning helps extrapolated regions.

Lib1 worsened near Lib2 cluster:
  possible local interference / task shift.

Lib1 worsened in isolated Lib1-specific neighborhoods:
  fine-tuning may lose rare natural sequence modes.

In-house overlaps Lib1/Lib2:
  public model may be relevant.

In-house outside both:
  stronger need for in-house fine-tuning / active learning.
```

### Question D: Is the in-house 5′ UTR proxy target appropriate?

The current in-house `FivePrime` target is `log2_RNA_DNA`, apparently from an RNA-seq-style count ratio, while Hani/Goodarzi public labels are RNA activity values inferred from fluorescence/FACS bin measurements across cell lines. Do not assume these are equivalent target surfaces.

Use the landscape package to separate three issues:

```text
1. sequence shift:
   Are in-house 5′ UTRs in the same raw sequence neighborhoods as Hani Lib1/Lib2?

2. assay/preprocessing shift:
   Does RNA/DNA behave smoothly over the in-house sequence landscape, and does it correlate with barcode depth or count totals?

3. output-head transfer:
   Are particular Hani heads locally predictive of in-house RNA/DNA only in certain sequence neighborhoods?
```

For in-house rows, always carry:

```text
number_of_barcodes
DNA_bc_counts_sum
RNA_bc_counts_sum
RNA/DNA
log2_RNA_DNA
exact_length_flag
valid_dna_flag
```

The first implementation should produce barcode/count-stratified versions of the in-house placement and predictor-correlation plots.

---

## 5. Recommended package structure

Create a small sibling repo rather than adding this to `boda2_EU/boda`:

```text
synBio_AL/
  sequence_landscape/
    pyproject.toml
    README.md
    src/sequence_landscape/
      __init__.py
      schema.py
      features.py
      projectors.py
      metrics.py
      transfer_effects.py
      plots.py
      io.py
      cli.py
    examples/
    tests/
```

Use `pip install -e /home/minhang/synBio_AL/sequence_landscape` inside `boda_env` so notebooks in `boda2_EU` can import it.

Rationale:

- `boda2_EU/boda` should stay focused on model/data modules used by training and inference.
- Landscape diagnostics should also apply to PARADE outputs, future in-house models, active-learning candidate pools, and possibly non-BODA repositories.
- A sibling package keeps the implementation reusable while keeping BODA-specific paths and experiment semantics in notebooks or adapter scripts.

Recommended BODA-specific adapter location:

```text
boda2_EU/src/analysis/sequence_landscape_adapters/
  hani_utr5_phase2.py
```

This adapter can load the concrete Phase 2 files and return normalized `sequence_landscape` tables, but it should not contain core feature extraction or plotting logic.

---

## 6. Input schema

The module should work with a simple long or wide table.

### Required columns

```text
seq_id       unique sequence identifier; fallback to sequence string if missing
seq          ACTG sequence string
dataset      e.g. lib1_train, lib1_test, lib2_test, inhouse_5p
region       e.g. 5UTR, 3UTR, enhancer, promoter, intron
split        optional: train/val/test/overlap/candidate
```

### Optional truth columns

Either wide cell-type heads:

```text
c1, c2, c4, c6, c13, c17, ...
```

or explicit columns:

```text
true_average_activity
true_activity_c1
true_activity_c2
...
```

### Optional prediction columns

For pretrained and fine-tuned models:

```text
pred_pre_c1
pred_pre_c2
...
pred_ft_c1
pred_ft_c2
...
pred_pre_average_activity
pred_ft_average_activity
```

The module should be flexible: if per-head predictions exist, compute averages; if average predictions already exist, use them.

### Optional assay confidence columns

```text
number_of_barcodes
DNA_bc_counts_sum
RNA_bc_counts_sum
total_counts
activity_se
replicate_count
```

These are not required for v1 but should be preserved in output.

### Current Phase 2 file wiring

Inputs for the first integration:

```text
Sweep root:
  boda2_EU/src/finetune/learning_curve/hani_utr5_lib2_resnet1d_1mmy39ku_phase2_may2026/

Lib2 split manifest:
  combined/lib2_sequence_split_manifest.csv

Combined metrics:
  combined/model_comparison_summary.csv
  combined/per_head_metrics.csv
  combined/inhouse_fiveprime_metrics.csv

Pretrained prediction tables:
  per_seed/seed_7/predictions/pretrained_1mmy39ku__lib2_test_predictions.csv
  per_seed/seed_7/predictions/pretrained_1mmy39ku__lib1_test_retention_predictions.csv
  per_seed/seed_7/predictions/pretrained_1mmy39ku__inhouse_fiveprime_predictions.csv

Validation-selected fine-tuned prediction tables:
  per_seed/seed_7/runs/seed7__full__hlr3p0e04__blr1p0e05__scaler_pretrained_lib1_train/predictions/
```

For Lib1 train reference geometry, read the original Lib1 wide table and use its existing `fold` column:

```text
/home/minhang/synBio_AL/opt_EU_learn_n_design/utr_hani_2025/processed_utr_data/5UTR_lib1_branched_observed_heads.csv
```

The fine-tuning script reports:

```text
Lib1 train rows: 17,288
Lib1 test retention rows: 2,160
Lib2 sequences after all-head filter: 10,174
Lib2 split: 8,140 train / 1,017 val / 1,017 test
FivePrime exact finite rows: 8,331
FivePrime barcode_min_8 rows used in diagnostic metrics: 1,797
```

---

## 7. Feature extractors

### 7.1 One-hot flattened features

Use this when all sequences are exact same length.

```text
Input:  N sequences of length L
Output: N x (4L) matrix
```

Encoding order:

```text
A,C,G,T
```

Unknown bases should either be all-zero or raise a warning. Default: all-zero with a count of invalid characters reported.

Pros:

- Position-aware.
- Directly meaningful for fixed-length synthetic constructs.
- Good baseline for PCA.

Cons:

- Sensitive to small indels or length differences.
- May emphasize positional base composition more than motif grammar.

### 7.2 k-mer count / frequency features

Use this as the alignment-free baseline and for sequences with small length deviations.

Parameters:

```text
k = 3, 4, 5, or 6
normalize = frequency by total k-mers
canonical_rc = optional; probably False for UTRs unless strand-invariance is intended
```

Pros:

- Alignment-free.
- Motif-composition-like.
- Handles modest length variation.

Cons:

- Loses position.
- Different sequences with different motif order can collapse together.

### 7.3 GC / simple composition features

Always compute as metadata:

```text
length
GC_fraction
A_fraction
C_fraction
G_fraction
T_fraction
contains_N
```

These help interpret PCA axes.

### 7.4 Hamming distance / MDS

Optional for exact-length sequences and moderate N only.

Use case:

```text
small diagnostic subset where actual edit distance is the scientific object
```

Default: do not run on huge datasets.

---

## 8. Projectors

### 8.1 PCA

Default v1 reducer.

Two modes:

#### Mode A: reference-fit PCA

```text
fit PCA on Lib1 train
transform Lib1 test, Lib2, in-house
```

Best for distribution-shift / train-manifold analysis.

#### Mode B: combined-fit PCA

```text
fit PCA on Lib1 + Lib2 + in-house
```

Best for presentation / descriptive map of all datasets.

The CLI should support both, with reference-fit as default for diagnostics.

### 8.2 UMAP

Optional.

Use if `umap-learn` is installed. Fit on reference set and transform new sets if using UMAP’s `transform` method.

### 8.3 Standardization

For one-hot features:

- Default: no standard scaling before PCA.
- Optional: center features for PCA as sklearn does internally.

For k-mer frequencies:

- Default: standardize features before PCA.
- Alternative: use CLR-like transform later if compositional effects become important.

---

## 9. Distribution and coverage metrics

Compute metrics for pairs such as:

```text
Lib1 train vs Lib1 test
Lib1 train vs Lib2 test
Lib1 train vs in-house 5′ UTR
Lib2 test vs in-house 5′ UTR
```

For each feature space:

```text
onehot_pca_input
kmer4
kmer5
model_embedding_later
```

### 9.1 Nearest-neighbor distance to reference

For each query sequence, compute distance to nearest Lib1 train sequence in feature space.

Outputs:

```text
nn_dist_to_lib1_train
knn_mean_dist_to_lib1_train
```

Dataset-level summaries:

```text
median_nn_distance
p90_nn_distance
p95_nn_distance
outlier_fraction_above_lib1_train_p95
```

### 9.2 kNN density proxy

For each query point:

```text
density_proxy = 1 / (epsilon + mean distance to k nearest Lib1 train points)
```

Use this for coloring or stratifying points.

### 9.3 Classifier two-sample test

Train a simple logistic regression classifier to distinguish:

```text
reference = Lib1 train
query = Lib2 or in-house
```

Report cross-validated AUC.

Interpretation:

```text
AUC ~ 0.5: hard to distinguish; distributions overlap
AUC high: strong distribution shift
```

### 9.4 MMD / energy distance

Optional v1 if easy. Do not block initial implementation.

Suggested priorities:

```text
1. nearest-neighbor distance
2. classifier two-sample AUC
3. MMD if available
4. Sinkhorn/OT later
```

---

## 10. Activity landscape metrics

These metrics ask whether expression is smooth over raw sequence space.

### 10.1 kNN activity smoothness

For each sequence with true activity:

```text
local_activity_mean = mean activity among k nearest neighbors
local_activity_std = std activity among k nearest neighbors
activity_residual_to_local_mean = activity - local_activity_mean
```

This helps identify isolated high-activity or low-activity points.

### 10.2 Distance vs activity difference

Sample sequence pairs and compute:

```text
sequence_distance
absolute_activity_difference
```

Plot or summarize whether nearby sequences have similar activity.

### 10.3 Error vs distance to train

For Lib2 or in-house sequences with labels:

```text
x = nn_dist_to_lib1_train
y = abs prediction error
```

This is one of the most important plots for active learning. If error increases with distance from train, novelty/distance is a useful acquisition signal.

### 10.4 Fine-tuning improvement vs distance to train

For Lib2:

```text
x = nn_dist_to_lib1_train
y = error_improvement after fine-tuning
```

If improvement is largest far from Lib1, that supports the idea that Lib2 fine-tuning improves extrapolated sequence neighborhoods.

For Lib1:

```text
x = nn_dist_to_lib2
y = error_improvement after fine-tuning
```

This can test whether Lib1 degradation is concentrated near/far from Lib2-like regions.

---

## 11. Improvement/regression analysis

Implement:

```python
def compute_prediction_transfer_effects(
    df,
    true_cols,
    pred_pre_cols,
    pred_ft_cols,
    id_col="seq_id",
    seq_col="seq",
):
    ...
```

Return columns:

```text
true_avg
pred_avg_pre
pred_avg_ft
err_pre
err_ft
abs_err_pre
abs_err_ft
signed_error_pre
signed_error_ft
error_improvement
squared_error_improvement
pre_to_ft_prediction_shift
```

Definitions:

```text
error_improvement = abs_err_pre - abs_err_ft
squared_error_improvement = err_pre^2 - err_ft^2
pre_to_ft_prediction_shift = pred_avg_ft - pred_avg_pre
```

Also compute per-head values when heads exist:

```text
error_improvement_c1
error_improvement_c2
...
```

### Output tables

```text
lib2_top_improved_sequences.csv
lib2_top_worsened_sequences.csv
lib1_top_retention_failures.csv
lib1_top_retained_or_improved_sequences.csv
all_transfer_effects.csv
```

Each top table should include the sequence string and enough metadata to inspect motifs manually.

---

## 12. Figures to generate in v1

### Figure 1: raw sequence PCA, colored by dataset

```text
x = PC1
y = PC2
color = dataset: Lib1 train/test, Lib2, in-house
alpha = 0.4
```

Make one version for one-hot PCA and one version for k-mer PCA.

### Figure 2: raw sequence PCA, colored by true average activity

For labeled public data:

```text
color = true_avg
```

For in-house 5′ data:

```text
color = log2_RNA_DNA or relevant in-house activity
```

### Figure 3: Lib2 fine-tuning improvement map

```text
Lib2 only
x/y = PCA coordinates
color = error_improvement
```

Positive values indicate improved predictions after Lib2 fine-tuning.

### Figure 4: Lib1 retention loss map

```text
Lib1 test only
x/y = PCA coordinates
color = error_improvement
```

Negative values indicate worse predictions after Lib2 fine-tuning.

### Figure 5: error vs distance to Lib1 train

```text
x = nearest-neighbor distance to Lib1 train
 y = abs prediction error
panel/line = pretrained vs fine-tuned
```

### Figure 6: improvement vs distance to Lib1 train

```text
x = nearest-neighbor distance to Lib1 train
 y = error_improvement
```

### Figure 7: in-house placement map

```text
Lib1 + Lib2 + in-house 5′ UTR
x/y = PCA coordinates
color = dataset
optional second panel color = in-house measured log2_RNA_DNA
```

### Figure 8: nearest-neighbor example panels

For top improved and top worsened sequences:

```text
query sequence
nearest Lib1 sequences
nearest Lib2 sequences
true/pred values
simple alignment-like display for exact-length sequences
```

This does not require formal alignment because lengths are fixed; display positional differences as mismatches.

---

## 13. CLI design

Implement a small command-line entry point, for example:

```bash
python -m analysis.sequence_landscape.cli \
  --region 5UTR \
  --reference-csv path/to/lib1_train.csv \
  --lib1-test-csv path/to/lib1_test_with_predictions.csv \
  --lib2-csv path/to/lib2_test_with_predictions.csv \
  --inhouse-csv path/to/inhouse_5p.csv \
  --seq-col seq \
  --true-heads c1 c2 c4 c6 c17 \
  --pred-pre-prefix pred_pre_ \
  --pred-ft-prefix pred_ft_ \
  --feature onehot kmer4 kmer5 \
  --reducer pca \
  --fit-mode reference \
  --out-dir outputs/sequence_landscape/5UTR_lib2_finetune
```

If exact file wiring is inconvenient at first, implement as a notebook-friendly Python API first, then wrap CLI later.

---

## 14. Notebook-first implementation plan

Because the existing work is already in a notebook, implement in this order:

### Step 1: Create package skeleton

Create the sibling repo and install it editable:

```bash
cd /home/minhang/synBio_AL
mkdir -p sequence_landscape/src/sequence_landscape sequence_landscape/tests
conda run -n boda_env pip install -e sequence_landscape
```

Do not modify BODA model-training code for v1.

### Step 2: Create `sequence_landscape.features`

Functions:

```python
one_hot_flatten(seqs, alphabet="ACGT", expected_length=None)
kmer_frequency_matrix(seqs, k=4, canonical_rc=False)
sequence_metadata(seqs)
```

### Step 3: Create `sequence_landscape.projectors`

Functions/classes:

```python
fit_projector(X_ref, method="pca", n_components=2, standardize=False)
transform_projector(projector, X)
project_datasets(feature_mats, reference_key="lib1_train", method="pca")
```

### Step 4: Create `sequence_landscape.transfer_effects`

Functions:

```python
compute_average_activity(df, head_cols)
compute_transfer_effects(df, true_cols, pred_pre_cols, pred_ft_cols)
rank_transfer_sequences(effects_df, top_n=100)
```

### Step 5: Create `sequence_landscape.metrics`

Functions:

```python
nearest_reference_distances(X_ref, X_query, k=5)
classifier_two_sample_auc(X_ref, X_query)
summarize_shift_metrics(X_ref, X_query, label_ref="lib1_train", label_query="lib2")
```

### Step 6: Create `sequence_landscape.plots`

Functions:

```python
plot_landscape(points_df, x="PC1", y="PC2", color="dataset", ...)
plot_error_vs_distance(df, distance_col, error_cols)
plot_improvement_vs_distance(df)
plot_top_sequence_neighbors(...)
```

### Step 7: Add one BODA adapter

Suggested adapter:

```text
boda2_EU/src/analysis/sequence_landscape_adapters/hani_utr5_phase2.py
```

Responsibilities:

```text
1. Load the Phase 2 prediction CSVs.
2. Normalize pretrained/fine-tuned prediction columns into package schema.
3. Attach Lib1 train/test, Lib2, and in-house metadata.
4. Return ready-to-analyze DataFrames.
```

### Step 8: Add one integration notebook

Suggested notebook:

```text
boda2_EU/tutorials/lib1_tasks/fine_tuning/fivePrime_finetune_w_lib1_hani_pretrain/
  hani_utr5_sequence_landscape_phase2_may2026.ipynb
```

Notebook sections:

```text
1. Load Lib1/Lib2 truth and prediction tables
2. Build pretrained vs fine-tuned transfer effects
3. Build raw one-hot/k-mer landscapes
4. Plot Lib2 improved sequences
5. Plot Lib1 retention failures
6. Place in-house 5′ data in the same raw sequence space
7. Export tables and figures
```

---

## 15. Expected output directory

Use a timestamped or run-labeled output directory:

```text
outputs/sequence_landscape/5UTR_lib2_finetune_YYYYMMDD/
  metadata.json
  all_points_onehot_pca.csv
  all_points_kmer4_pca.csv
  transfer_effects_lib2.csv
  transfer_effects_lib1.csv
  top_lib2_improved.csv
  top_lib2_worsened.csv
  top_lib1_retention_failures.csv
  shift_metrics.csv
  figures/
    onehot_pca_dataset_overlay.png
    kmer4_pca_dataset_overlay.png
    lib2_improvement_onehot_pca.png
    lib1_retention_loss_onehot_pca.png
    inhouse_placement_onehot_pca.png
    error_vs_nn_distance.png
    improvement_vs_nn_distance.png
```

---

## 16. Minimal acceptance criteria for v1

v1 is successful if it can:

1. Load Lib1 test, Lib2 test/overlap, and in-house 5′ sequences.
2. Compute average activity and prediction errors for pretrained and fine-tuned models.
3. Export top Lib2 improved and Lib1 worsened sequence tables.
4. Produce one-hot PCA and k-mer PCA maps with Lib1/Lib2/in-house overlays.
5. Quantify nearest-neighbor distance of Lib2 and in-house sequences to Lib1 train.
6. Plot prediction error / fine-tuning improvement as a function of distance to Lib1 train.
7. Repeat the transfer-effect tables for both the Lib2-validation-selected full-unfreeze checkpoint and at least one head-only checkpoint.
8. Include barcode/count-stratified in-house diagnostics so RNA/DNA proxy behavior is not conflated with sequence-space transfer.
9. Run in under a few minutes on the current dataset sizes.
10. Avoid changing existing model training code.

---

## 17. Important design guardrails

### Guardrail 1: Do not over-interpret PCA axes

PCA coordinates are diagnostic, not mechanistic. Always pair maps with quantitative metrics.

### Guardrail 2: Do not mix different CRE lengths in one-hot space

One-hot PCA is only valid for exact same length and alignment.

### Guardrail 3: Preserve raw sequence IDs and strings

The top improved/worsened tables must include sequence strings so we can inspect them manually.

### Guardrail 4: Separate direct comparison and descriptive comparison

Reference-fit PCA answers:

```text
Where do new sequences fall relative to Lib1 train?
```

Combined-fit PCA answers:

```text
What is the joint descriptive geometry of all sequences?
```

Both are useful but should not be confused.

### Guardrail 5: Treat model embeddings as v2

Model embeddings are valuable, but v1 should first establish raw sequence-space diagnostics. Add model embeddings after the raw pipeline is stable.

---

## 18. Possible v2 extensions

### 18.1 Model embedding landscape

Extract embeddings from:

- PARADE released checkpoints,
- fine-tuned PARADE checkpoints,
- BODA/HPO models,
- possibly foundation models later.

Then repeat the same analyses:

```text
embedding PCA/UMAP
Lib1/Lib2/in-house distribution metrics
error vs embedding-distance-to-train
improvement vs embedding-distance-to-train
```

Important comparison:

```text
raw sequence distance predicts error?
model embedding distance predicts error?
which is more useful for active learning?
```

### 18.2 Active-learning diagnostics

For a candidate pool, compute:

```text
predicted activity
uncertainty if available
nearest-neighbor distance to measured train
local density
diversity among selected candidates
coverage gain
```

Plot candidate pool and selected sequences on raw/model landscapes.

### 18.3 Motif enrichment of improved/worsened regions

For top improved/worsened sets:

```text
k-mer enrichment
motif enrichment
GC/length/composition comparison
position-wise base frequency logos
```

This can reveal whether fine-tuning helps or hurts specific motif/composition regimes.

---

## 19. Separate updated deep-research prompt

Use this only if we want a more comprehensive literature/tool scan. This is separate from the implementation plan above.

### Deep research prompt

```markdown
I am working on synthetic biology / MPRA sequence-to-function modeling for fixed-length or near-fixed-length cis-regulatory element regions, especially 5′ UTRs and other CRE parts in synthetic DNA constructs. I want a literature and tool review focused on methods that quantify and visualize sequence landscapes and connect those landscapes to measured activity, model prediction error, model transfer, and active learning.

Please do a comprehensive, source-grounded review that is less biased toward any pre-specified package list. Search broadly across genomics, MPRA/MAVE, regulatory DNA design, protein/DNA sequence-function landscapes, active learning, out-of-distribution detection, and embedding visualization.

Key problem setting:
- Each sequence is an ACTG string for one CRE region.
- Within a run, sequences are usually fixed length and design-aligned, though experimental deviations can occur.
- Each sequence has measured activity, such as expected FACS bin, RNA/DNA, or cell-type-specific expression.
- We train predictive models and want to understand where train/test/designed/in-house sequences lie in raw sequence space and learned model-embedding space.
- We especially want to diagnose cases where fine-tuning on a designed library improves predictions on that library but slightly worsens retention on the original library.

Please answer:

1. What existing methods/tools visualize DNA/RNA sequence landscapes from raw sequences?
   - Include one-hot/position-aware approaches, k-mer/alignment-free approaches, edit/Hamming-distance approaches, manifold learning, and sequence-function maps.
   - Distinguish methods designed for fixed-length engineered libraries from those designed for genomes, taxonomy, or arbitrary variable-length sequences.

2. What existing methods/tools connect sequence landscape coordinates to measured quantitative phenotype?
   - MPRA/MAVE genotype-phenotype maps
   - activity surfaces
   - local smoothness metrics
   - sequence-function ruggedness
   - fitness landscape or expression landscape analogies

3. What literature exists on using model embeddings to visualize regulatory sequence space?
   - CNN/Transformer embeddings
   - supervised vs self-supervised embeddings
   - foundation-model embeddings
   - methods for validating whether embedding neighborhoods are biologically meaningful

4. What distribution-shift / OOD / domain adaptation methods are appropriate for comparing train, designed, and in-house sequence sets?
   - MMD, energy distance, Wasserstein/Sinkhorn, classifier two-sample tests, nearest-neighbor coverage, density/outlier metrics
   - Discuss strengths, weaknesses, and sample-size considerations.

5. What active-learning literature uses sequence-space or embedding-space novelty/diversity to select new biological sequences?
   - Include MPRA/regulatory DNA design, protein engineering, MAVE, Bayesian optimization, and pool-based active learning.
   - Focus on exploration/exploitation diagnostics.

6. What existing software packages could we reuse rather than reinvent?
   - Include genomics-specific frameworks, MAVE/MPRA packages, sequence analysis packages, drift/OOD libraries, embedding visualization tools, and active-learning libraries.
   - For each package: language, maintenance status, installation burden, relevance to fixed-length synthetic CREs, and whether it supports our use case directly or only partially.

7. What are the gaps?
   - Is there an existing package that already does: fixed-length CRE raw-sequence landscape + model embedding landscape + expression/error overlays + train/design/in-house distribution shift + active-learning diagnostics?
   - If not, what minimal repo-local wrapper would be scientifically justified?

Please produce:
- A structured literature review with citations.
- A table of methods/tools and whether they directly solve our use case.
- A recommended minimal implementation plan.
- A list of 10–20 most important references.
- A clear statement of what would be novel or gap-filling about applying these methods to synthetic CRE MPRA active-learning workflows.
```

---

## 20. First Codex task prompt

Use this prompt to start implementation.

```markdown
You are working in the boda2_EU repository. Implement a lightweight sequence landscape diagnostics module for fixed-length synthetic CRE/UTR MPRA datasets.

Goal:
Create a repo-local module that can compare Lib1, Lib2, and in-house 5′ UTR sequences in raw sequence space and connect the landscape to model prediction changes after fine-tuning.

Please implement v1 only. Do not modify model training code.

Required functionality:
1. Feature extraction:
   - one-hot flattened encoding for exact-length ACTG sequences
   - k-mer frequency encoding for k=4 and k=5
   - sequence metadata: length, GC, base fractions, invalid base count

2. Projection:
   - PCA with reference-fit mode: fit on Lib1 train/reference, transform Lib1 test, Lib2, in-house
   - combined-fit mode: fit on all provided datasets for descriptive plots

3. Prediction transfer effects:
   - given truth heads, pretrained prediction heads, and fine-tuned prediction heads, compute:
     true_avg, pred_avg_pre, pred_avg_ft, abs_err_pre, abs_err_ft, error_improvement
   - positive error_improvement means fine-tuning improved prediction
   - output top Lib2 improved sequences and top Lib1 worsened sequences

4. Distribution metrics:
   - nearest-neighbor distance from each query dataset to Lib1 reference
   - p50/p90/p95 nearest-neighbor distance summaries
   - outlier fraction above Lib1 reference p95
   - optional logistic-regression classifier two-sample AUC

5. Plots:
   - PCA map colored by dataset
   - PCA map colored by true average activity
   - Lib2 PCA map colored by error_improvement
   - Lib1 PCA map colored by error_improvement / retention loss
   - in-house 5′ placement map
   - prediction error vs nearest-neighbor distance
   - error improvement vs nearest-neighbor distance

6. Outputs:
   - CSV files for projected points, shift metrics, transfer effects, and top sequences
   - PNG figures under an output directory
   - metadata JSON describing inputs, feature type, reducer, and parameters

Suggested module structure:
src/analysis/sequence_landscape/
  __init__.py
  features.py
  projectors.py
  metrics.py
  improvement.py
  plots.py
  io.py
  cli.py

Also create a notebook or script demonstrating use with the existing PARADE released checkpoint evaluation outputs.

Acceptance criteria:
- The module runs on 5′ UTR data with exact-length filtering.
- It produces top Lib2 improved and Lib1 worsened sequence tables.
- It produces raw one-hot PCA and k-mer PCA overlays for Lib1, Lib2, and in-house 5′ sequences.
- It computes distance-to-Lib1-reference metrics and plots error/improvement vs distance.
- It does not require UMAP, model embedding extraction, or foundation models in v1.
```

---

## 21. Recommended first empirical analyses

After implementation, run these analyses first:

1. **Lib2 improvement map**
   - Are the improved Lib2 sequences far from Lib1 train in raw sequence space?

2. **Lib1 retention loss map**
   - Are the worsened Lib1 sequences concentrated in specific raw-sequence neighborhoods?

3. **In-house 5′ placement**
   - Does in-house 5′ data overlap Lib1/Lib2 or occupy a distinct region?

4. **Distance vs error**
   - Does distance to Lib1 train predict pretrained model error?
   - Does fine-tuning reduce that distance-error relationship?

5. **Raw one-hot vs k-mer comparison**
   - Does position-aware space or k-mer space better explain prediction improvement/failure?

These results will tell us whether this landscape module is merely pretty or genuinely useful for active learning and model-transfer diagnostics.
