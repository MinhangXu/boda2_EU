# Barcode-Level Uncertainty Modeling Discussion Brief

Generated: 2026-07-07

Purpose: portable context for a longer modeling discussion in web ChatGPT or
with collaborators. This is a handoff brief, not a final implementation plan.

## Recommended Use

Use this file as a context package for a web ChatGPT conversation. "Handoff" or
"discussion brief" is probably more precise than "escalation", although
"escalate to web ChatGPT for literature-backed modeling discussion" is a fine
informal phrase.

Suggested flow:

1. Discuss the high-level modeling math and target estimand first.
2. Decide whether to use log-ratio targets with pseudocounts, count likelihoods,
   or a two-stage MPRA statistics tool.
3. Translate the chosen model family into a small experiment grid.
4. Only then implement the dataset/split/loss code.

This ordering should prevent us from prematurely coding around an arbitrary
pseudocount policy.

## Local Project Context

`$BODA_WORK_ROOT` denotes the workspace containing this checkout and the
private data roots; set it explicitly when those directories are not siblings.

Primary dedup planning note:

```text
boda2_EU/plan/repo_hygiene/barcode_level_dedup_update_july6_2026.md
```

External Lib1 data root:

```text
$BODA_WORK_ROOT/opt_EU_learn_n_design/MattLee_lib1/
```

Deduped single-part barcode-level files now exist:

```text
$BODA_WORK_ROOT/opt_EU_learn_n_design/MattLee_lib1/barcode_level/by_library/
  single_part__enhancer_subset_0filtered_out.dedup_exact.barcode_level.csv
  single_part__Promoter_subset.dedup_exact.barcode_level.csv
  single_part__FivePrime_subset.dedup_exact.barcode_level.csv
  single_part__Intron_subset.dedup_exact.barcode_level.csv
  single_part__ThreePrime_subset.dedup_exact.barcode_level.csv
```

The key leakage rule is non-negotiable:

```text
Split train/validation/test by parts_concatenated, not by individual barcode.
```

The intended predictive input for Lib1 barcode-level sequence models should be
the construct sequence or construct parts, not the barcode sequence. If one DNA
construct has many barcodes, the model can see multiple noisy observations of
the same sequence during training, but validation/test constructs must be
entirely held out.

## Current Follow-Up 3 Question

Candidate barcode-level target:

```text
y_ij = log2((RNA_bc_counts_ij + alpha_R) / (DNA_bc_counts_ij + alpha_D))
```

where `i` is construct and `j` is barcode. The unresolved policy is how to pick
`alpha_R` and `alpha_D`, or whether to avoid pseudocount log-ratios by modeling
counts directly.

Important distinction:

- A mean-only model trained on repeated barcode rows with sequence-only input
  mostly learns a construct-level average, with constructs weighted by their
  number of barcode observations unless we explicitly rebalance the loss.
- It does not automatically learn uncertainty unless the model has an
  uncertainty head, a count likelihood, an ensemble, or another calibrated
  uncertainty mechanism.
- The deduped construct-level target remains:

```text
log2(sum RNA_bc_counts / sum DNA_bc_counts)
```

This is still the clean baseline target for sequence-to-expression ranking.

## Prior Local Work: Lib0 Distribution Modeling

There is relevant preliminary work under:

```text
raw_data_bashor/lib0_distribution_learn/
```

The README says this directory is an active workspace for replicate-level
expression distributions and hierarchical models for Library 0 MPRA data.
Useful prior-art notebooks include:

```text
raw_data_bashor/lib0_distribution_learn/lib0_notebooks/
  lib0_rep_model.ipynb
  lib0_hierarchical_model_jan14_2026.ipynb
  coverage_aware_targets_jan19.ipynb
  histogram_learn_jan21.ipynb
  non_filter_histogram_learn_jan25.ipynb
  lib0_replicate_expression_distribution_descriptor_jan11_2026.ipynb
  phase2B_knobs_parts_to_params.ipynb
```

Relevant Lib0 artifacts already include:

```text
artifacts/coverage_aware_targets_jan19/
  level1_poisson_offset_targets_xmin23.csv
  level2_nb_offset_targets_xmin23.csv

artifacts/lib0_replicate_expression_distribution_descriptor_jan11_2026/
  summary_metrics.csv
  per_construct_metrics.csv
  example_construct_fits.json

artifacts/phase2B_knobs_parts_to_params/
  phase2B_construct_holdout_summary.csv
  phase2B_replicate_holdout_summary.csv
```

This matters because Lib0 already explored count-aware ideas:

- Poisson offset targets with `log(DNA)` as an exposure/offset.
- Negative-binomial offset targets and overdispersion.
- Hurdle/zero behavior.
- Mapping categorical parts to distribution parameters.

Lib0 is categorical and has fewer construct variants than Lib1 sequence
modeling, but it is still valuable as a sandbox for observation models,
calibration plots, and count-likelihood thinking.

## What To Ask Web ChatGPT

Paste this file and ask:

```text
I am modeling MPRA barcode-level RNA/DNA data for sequence-to-expression
prediction. I have deduplicated barcode rows. Each row has construct sequence
identity parts_concatenated, barcode ID, DNA_bc_counts, RNA_bc_counts, and
library_name. I must split train/val/test by parts_concatenated, never by
barcode row. The model input should be construct sequence only, not barcode.

Please help choose a statistically principled model family and small experiment
grid for barcode-level uncertainty modeling. Compare:
1. construct-level aggregate log2(sum RNA / sum DNA);
2. mean-only barcode-row log-ratio training with pseudocounts;
3. heteroscedastic Gaussian regression on barcode log-ratios;
4. Poisson or negative-binomial count likelihood with log DNA as offset;
5. two-stage use of MPRA-specific tools such as mpralm, MPRAnalyze, or BCalm to
   estimate construct activity and uncertainty, followed by sequence modeling.

Please search the web for current MPRA barcode-level analysis tools and
recommend which methods are suitable as out-of-the-box baselines versus which
should be reimplemented as PyTorch losses. Please also advise how to handle
zero RNA and zero DNA barcode rows, pseudocount policy if log-ratios are used,
construct-balanced vs barcode-weighted training, and validation metrics for
uncertainty calibration.
```

Useful follow-up prompts:

```text
Given my active-learning goal, distinguish aleatoric uncertainty from barcode
noise, epistemic uncertainty from model extrapolation, and acquisition
uncertainty for choosing new construct sequences.
```

```text
Please produce a minimal first experiment grid with exact targets, losses,
filtering, split policy, and construct-level evaluation metrics.
```

```text
Please search specifically for whether BCalm, MPRAnalyze, or mpralm can export
per-element activity estimates and uncertainty/precision weights that can be
used as supervised targets for a DNA sequence model.
```

## Web Search Starting Points

These are good sources to ask web ChatGPT to read deeply and update with any
newer work:

- `mpra` / `mpralm` Bioconductor guide:
  https://www.bioconductor.org/packages/release/bioc/vignettes/mpra/inst/doc/mpra.html
  - The guide says `MPRASet` can take barcode-level DNA/RNA count matrices.
  - It discusses `mpralm`, log2 RNA/DNA activity, barcode aggregation by
    average barcode log-ratio versus summed-count aggregate ratio, and precision
    weights based on DNA count/variance behavior.

- MPRAnalyze paper:
  https://link.springer.com/article/10.1186/s13059-019-1787-z
  - Models DNA and RNA count uncertainty with nested GLMs.
  - Treats transcription rate `alpha` as the activity parameter.
  - Relevant as an out-of-the-box MPRA count-model baseline or as inspiration
    for a deep count-likelihood loss.

- MPRAnalyze vignette/source:
  https://rdrr.io/bioc/MPRAnalyze/f/vignettes/vignette.Rmd
  - Important practical note: barcode effects are generally modeled in the DNA
    model, but not necessarily in the RNA model, because per-barcode RNA effects
    would imply different transcription rates for barcodes of the same enhancer.

- BCalm paper:
  https://link.springer.com/article/10.1186/s12859-025-06065-9
  - Barcode-level extension/adaptation of `mpralm`.
  - Claims individual barcode modeling can improve power and robustness to
    outliers while remaining more scalable than MPRAnalyze.
  - Especially relevant as an out-of-the-box barcode-level baseline to compare
    with our own sequence model outputs.

- QuASAR-MPRA:
  https://academic.oup.com/bioinformatics/article/34/5/787/4209990
  - Uses beta-binomial ideas for allele-specific MPRA.
  - Probably less directly suitable for our single-sequence activity prediction
    target, but useful context for DNA/RNA count uncertainty and overdispersion.

## Candidate Model Families For Lib1 Follow-Up 3

### A. Construct-Level Aggregate Baseline

Target:

```text
log2(sum_j RNA_ij / sum_j DNA_ij)
```

Pros:

- Closest to current deduped variant-level target.
- No per-barcode pseudocount needed when aggregate DNA/RNA are positive.
- Best baseline for sequence-to-expression ranking.

Cons:

- Does not directly model barcode variability.
- Uncertainty must come from barcode count, empirical variance, ensembles, or
  another secondary model.

### B. Mean-Only Barcode-Row Regression

Target:

```text
log2((RNA_ij + alpha_R) / (DNA_ij + alpha_D))
```

Two loss policies should be separated:

- Barcode-weighted: every barcode row has equal weight, so highly barcoded
  constructs contribute more.
- Construct-balanced: barcode row weight is `1 / n_barcodes_for_construct`, so
  every construct contributes about equally.

Pros:

- Easy implementation.
- Tests whether barcode-level training behaves differently from construct
  aggregate training.

Cons:

- With sequence-only input, the model predicts the same value for all barcodes
  of the same construct.
- This is not true uncertainty modeling unless paired with a variance head or
  external uncertainty estimate.
- Results may be sensitive to pseudocounts and low-count rows.

### C. Heteroscedastic Gaussian On Barcode Log-Ratios

Model predicts:

```text
mu_i, log_sigma2_i = f(sequence_i)
```

Loss:

```text
0.5 * (log_sigma2_i + (y_ij - mu_i)^2 / exp(log_sigma2_i))
```

Pros:

- Directly produces an aleatoric variance estimate.
- Easy to implement in PyTorch.
- Can be evaluated against empirical barcode variance and calibration curves.

Cons:

- Still depends on pseudocount log-ratio targets.
- Gaussian assumptions may be poor for low counts and zeros.
- Predicted variance can become a catch-all for unmodeled biology, count depth,
  and technical artifacts.

### D. Poisson Or Negative-Binomial Count Likelihood With DNA Offset

A simple observation model:

```text
RNA_ij ~ Poisson(DNA_ij * exp(eta_i))
```

or:

```text
RNA_ij ~ NegativeBinomial(mean = DNA_ij * exp(eta_i), dispersion = phi_i)
```

where the sequence model predicts `eta_i`, and possibly dispersion.

Pros:

- Avoids arbitrary RNA/DNA pseudocounts.
- DNA count naturally acts as exposure/coverage.
- Aligns with the Lib0 Poisson/NB offset work.
- Produces likelihood-based uncertainty metrics.

Cons:

- Needs careful handling of `DNA_ij = 0`.
- Needs decisions about dispersion sharing: global, library-specific,
  construct-specific, or predicted by sequence.
- Slightly more work than log-ratio MSE/NLL.

This is probably the most principled custom PyTorch direction if the goal is
barcode-level uncertainty and active-learning calibration.

### E. Two-Stage MPRA Tool Target Extraction

Use an MPRA-specific tool to estimate per-construct activity and uncertainty,
then train the sequence model on those estimates.

Possible tools:

- `mpralm` / `mpra`: aggregate or barcode-aware activity/log-ratio modeling with
  precision weights.
- `MPRAnalyze`: nested GLM estimating transcription rate `alpha`.
- `BCalm`: barcode-level mpralm-style modeling.

Pros:

- Lets established MPRA statistical tooling handle count quirks first.
- May produce target estimates and uncertainty/precision weights that are
  easier for BODA-style models to consume.
- Good external baseline for our own count-likelihood implementation.

Cons:

- These tools solve MPRA statistical analysis, not DNA sequence prediction.
- We still need to map outputs into supervised sequence-model targets.
- Tool assumptions may not match our one-library, single-condition,
  active-learning setting.

### F. Epistemic Uncertainty For Active Learning

Barcode/count models mostly address observation noise. For active learning over
new construct sequences, we likely also need epistemic uncertainty from:

- ensembles across seeds;
- MC dropout or stochastic heads;
- deep ensembles with different train splits;
- uncertainty from model disagreement in predicted construct activity.

These should be discussed separately from barcode-level aleatoric noise.

## Zero Count And Pseudocount Policy Ideas

For log-ratio experiments only, do not make one pseudocount canonical before
checking sensitivity. Reasonable first-pass policies:

```text
alpha_R = alpha_D = 0.5
alpha_R = alpha_D = 1.0
```

A more interpretable shrinkage target is:

```text
y_ij = log2((RNA_ij + k * r0) / (DNA_ij + k))
```

where `r0` is a train-set or library baseline RNA/DNA ratio and `k` is prior DNA
count strength. This shrinks low-count barcode ratios toward the assay/library
baseline rather than toward `RNA/DNA = 1`.

Initial zero policy to discuss:

- Keep `DNA > 0, RNA = 0` rows for count-likelihood models.
- For log-ratio models, include `DNA > 0, RNA = 0` only through an explicit
  pseudocount/shrinkage policy.
- Drop or separately flag `DNA = 0` barcode rows for target modeling because the
  denominator/exposure is undefined.
- Track zero/low-coverage rows as QC features for diagnostics, not necessarily
  model inputs.

## Recommended Minimal First Experiment Grid

The first implementation should be small and diagnostic:

| ID | Input | Target/Likelihood | Loss Weighting | Main Question |
|---|---|---|---|---|
| A1 | sequence | dedup construct aggregate log2 ratio | one row per construct | clean baseline |
| B1 | sequence | barcode log-ratio, alpha=0.5 | barcode-weighted | does row expansion help? |
| B2 | sequence | barcode log-ratio, alpha=0.5 | construct-balanced | is any gain just barcode-count weighting? |
| C1 | sequence | barcode log-ratio, alpha=0.5 | heteroscedastic Gaussian NLL | can variance head calibrate barcode noise? |
| D1 | sequence | RNA counts with DNA offset | Poisson or NB NLL | can count likelihood avoid pseudocounts? |
| E1 | sequence | MPRA-tool activity estimate | precision-weighted MSE if available | can off-the-shelf statistics produce better targets? |

Evaluation should always be construct-level:

- Spearman/Pearson/RMSE/MAE of predicted mean versus dedup aggregate target.
- Same metrics stratified by barcode count bins and DNA coverage bins.
- NLL on held-out barcode rows, grouped by held-out constructs.
- Calibration: predicted variance versus empirical barcode variance.
- Reliability curves by library and barcode count bin.
- For active learning: compare acquisition ranking stability under mean,
  aleatoric uncertainty, epistemic uncertainty, and diversity penalties.

## Concrete Decision Needed Before Coding

The main decision is not "which pseudocount is standard?" There is no perfectly
neutral standard pseudocount for this problem.

The better decision tree is:

1. Is the first Follow-Up 3 experiment only a quick baseline?
   - Use barcode log-ratio with `alpha_R = alpha_D = 0.5`, and run
     construct-balanced plus barcode-weighted losses.
2. Is the goal genuine barcode-level uncertainty?
   - Prefer Poisson/NB offset likelihood or heteroscedastic Gaussian as the
     first uncertainty model.
3. Is the goal to use established MPRA statistics out of the box?
   - Try `mpra`/`mpralm`, MPRAnalyze, and/or BCalm on deduped barcode counts to
     extract activity estimates, standard errors, or precision weights; then
     train sequence models on those construct-level estimates.

## Implementation After Discussion

After the web discussion, reduce the outcome to a short implementation spec:

```text
Data file(s):
Split key:
Rows included/excluded:
Target or likelihood:
Pseudocount policy, if any:
Loss weighting:
Model heads:
Validation metrics:
First two libraries to run:
Stop conditions:
```

Then implement the smallest dataset/loss changes needed for that spec.
