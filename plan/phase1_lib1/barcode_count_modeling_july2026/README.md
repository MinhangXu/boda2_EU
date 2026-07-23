# Lib1 Barcode-Count EDA, Target Estimands, And Probabilistic Modeling Roadmap

**Date:** 2026-07-14

**Status:** active planning and EDA-definition work

**Immediate scope:** deduplicated single-part Promoter, 5'UTR, Intron, and 3'UTR

**Later scope:** repaired two-head training, Poisson/NB sequence models, and a
hierarchical observation model

**Explicitly out of scope now:** re-running the deduplicated mean-expression
campaign, multi-part modeling, enhancer modeling, and active-learning library
selection

## Executive Decision

The next implementation should not begin with another neural-network sweep.
It should first make the barcode-count estimands and observation models
inspectable in the `mpra_eda_tool`, then use that evidence to freeze a small
model comparison.

The completed deduplicated variant-level campaign remains the mean-expression
baseline. This plan does not replace or rerun it. It connects to that campaign
by requiring the same data provenance, construct-level split identities,
target units, and held-out evaluation policy whenever a new target or
likelihood is compared with the baseline.

The immediate work has two products:

1. A Lib1-capable interactive EDA tool that exposes the raw counts, support,
   zero behavior, candidate activity estimators, and distribution diagnostics.
2. A frozen mathematical data dictionary that says exactly what each symbol,
   target, mask, weight, uncertainty quantity, and likelihood means.

Only after those products pass their acceptance gates should an agent modify
`boda2_EU/src/learn/` for the next modeling run.

## Start Here For A New Agent

In paths below, `$BODA_WORK_ROOT` means the workspace containing this checkout
and the private data repositories. The repo-side scripts default it to the
parent of the checkout and allow an explicit environment-variable override.

Use this file as the primary handoff. Read in this order:

1. This roadmap.
2. [`probabilistic_ml_learning_and_stage5_gate_july2026.md`](probabilistic_ml_learning_and_stage5_gate_july2026.md)
   for the July 17 learning prerequisite, library decision, exact-NLL reporting
   contract, depth-offset identifiability correction, and Stage 5 stop rules.
3. [`../dedup_phase1_rerun_july2026/README.md`](../dedup_phase1_rerun_july2026/README.md)
   for the fixed deduplicated mean-expression baseline and its current status.
4. [`../learn/barcode_level_uncertainty_discussion_context_july7_2026.md`](../learn/barcode_level_uncertainty_discussion_context_july7_2026.md)
   for earlier alternatives and MPRA-tool references.
5. [`../../../tutorials/lib1_tasks/barcode_level/variant_level_redo_jul7_2026/build_lib1_probabilistic_ml_tutorial.py`](../../../tutorials/lib1_tasks/barcode_level/variant_level_redo_jul7_2026/build_lib1_probabilistic_ml_tutorial.py)
   for the reviewable source that regenerates the local
   `lib1_probabilistic_ml_from_mse_to_count_nll_july2026.ipynb` prerequisite
   and synthetic five-part model comparison.
6. The local, generated
   `tutorials/lib1_tasks/barcode_level/data_prep_eda/lib1_dedup_barcode_level_pre_modeling_eda_july2026.ipynb`
   for the primary barcode-level evidence and presentation plots.
7. The local, generated
   `tutorials/lib1_tasks/barcode_level/variant_level_redo_jul7_2026/lib1_two_head_mean_spread_quick_analysis_jul8_2026.ipynb`
   for the diagnostic two-head run.
8. `$BODA_WORK_ROOT/raw_data_bashor/mpra_eda_tool/plan/lib1_barcode_count_eda_revamp_july2026.md`
   for the separate EDA-tool implementation contract.

The first agent assignment should be:

```text
Work only on Stages 0-3 of the mpra_eda_tool Lib1 revamp plan. Do not launch
models and do not change the deduplicated data products. Implement and test
the schema adapter and pure estimand/QC functions before changing the Dash UI.
Every displayed statistic must link to a symbol/formula in this roadmap. Keep
Lib0 behavior available through its own schema adapter. Produce a small toy
fixture with hand-verifiable values, a Lib1 smoke report for all four current
single-part libraries, and an updated plan status. Stop at the decision gate;
do not choose Poisson versus NB from visual fit alone.
```

## What Is Already Complete

### Deduplicated data products

The canonical lineage is:

```text
legacy combined barcode table
  -> exact-row-deduplicated combined barcode table
  -> per-library deduplicated barcode tables
  -> deduplicated single-part variant aggregates
```

The relevant roots are:

```text
$BODA_WORK_ROOT/opt_EU_learn_n_design/MattLee_lib1/
  barcode_level/
    L1_variant_bc_expr_combined_20251107_np_fastq1-5.dedup_exact.csv
    by_library/
      single_part__Promoter_subset.dedup_exact.barcode_level.csv
      single_part__FivePrime_subset.dedup_exact.barcode_level.csv
      single_part__Intron_subset.dedup_exact.barcode_level.csv
      single_part__ThreePrime_subset.dedup_exact.barcode_level.csv
  single_part_variant_level/
    L1_final_fastqs1-5_sublibrary_Promoter_subset.dedup_exact.csv
    L1_final_fastqs1-5_sublibrary_FivePrime_subset.dedup_exact.csv
    L1_final_fastqs1-5_sublibrary_Intron_subset.dedup_exact.csv
    L1_final_fastqs1-5_sublibrary_ThreePrime_subset.dedup_exact.csv
```

The manifests record exact paths, hashes, row counts, aggregation policy, and
validation. The per-library barcode files are filtered splits of the full
deduplicated barcode table, and the single-part variant tables are aggregates
of those matching barcode rows.

### Deduplicated mean-expression baseline

Stages 1--4, including the bounded targeted 3'UTR HPO, fixed-budget final
refits, and the one-time locked final-test evaluation, are complete. The five
selected Stage 3 policies and final-test products are frozen. This plan treats
that completed campaign as the mean-expression baseline and does not reopen
its development or final-test boundaries.

Important comparison rule: a new barcode/count model is not fairly compared
with the baseline merely because both use deduplicated data. It must also use
the same construct identities in each split, the same eligible sequence set,
the same target scale for the mean readout, and the same model-selection
boundary. If any of these differ, label the result exploratory rather than a
paired baseline comparison.

### Current two-head diagnostic

The quick run trained one model for each of Promoter, 5'UTR, Intron, and 3'UTR.
It established that the current mean head is learnable and the current spread
head is not yet useful.

| Library | Mean test Pearson | Spread test Pearson |
|---|---:|---:|
| Promoter | 0.4790 | -0.0073 |
| 5'UTR | 0.4146 | -0.0069 |
| Intron | 0.6306 | 0.0979 |
| 3'UTR | 0.3453 | -0.1089 |

These are diagnostic single-run results, not a model-class verdict. The run
used different architectures by part, a single seed, an unmasked singleton
spread target, equal MSE treatment after target scaling, and aggregate
validation Pearson for checkpoint selection. Its mean target also differs from
the deduplicated variant baseline: the baseline sums all exact-deduplicated
barcode rows, whereas the quick target first drops `DNA < 1` rows and then adds
one 0.5 pseudocount to each retained construct total.

## Evidence From The Barcode EDA

The primary notebook auto-discovers five single-part files, including enhancer.
Its original headline therefore reports a broader population than the immediate
four-part scope:

- 197,295 barcode rows, 36,487 globally unique construct strings, and 36,490
  library-qualified construct groups across the five files;
- 75.24% of the five-file rows retained at `DNA >= 1`;
- 163,355 barcode rows, 31,702 library-qualified construct groups, and 122,231
  retained rows in the immediate Promoter/5'UTR/Intron/3'UTR scope;
- 24.76% of rows with `DNA == 0`;
- zero-RNA row fractions of 21.72% to 24.88% for the four current non-enhancer
  libraries;
- 4,034 `RNA > 0, DNA == 0` rows across those four libraries, which contribute
  RNA to the canonical all-row variant aggregate but are dropped from the
  quick retained-row target; and
- no aggregate-zero-RNA or aggregate-zero-DNA constructs before the retained
  target calculation, although 14 retained target groups become aggregate
  zero RNA after the DNA filter.

The support distribution is more consequential for the spread target:

| Library | Constructs | Retained singleton constructs | Singleton fraction | Median retained barcodes | Median `n_eff` |
|---|---:|---:|---:|---:|---:|
| Promoter | 7,894 | 1,570 | 19.89% | 3 | 2.498 |
| 5'UTR | 8,461 | 1,836 | 21.70% | 3 | 2.365 |
| Intron | 8,089 | 1,917 | 23.70% | 3 | 2.180 |
| 3'UTR | 7,258 | 2,412 | 33.23% | 2 | 1.862 |

For the immediate four parts, 7,735 of 31,702 target groups, or 24.40%, are
singletons. The five-file total including enhancer is 8,588 of 36,490, or
23.54%. The current
variance is identically zero for these groups, so almost all of the spike at
`log_barcode_var = log(1e-4)` is deterministic from the estimator. The
mathematical cause of the spike is one observation. The experimental reason a
construct has only one retained barcode may involve synthesis, library
construction, amplification, sequencing, or filtering, but that causal claim
has not yet been established.

An additional audit of the current target-preview table shows that the
aggregate log ratio and the DNA-weighted mean of per-barcode log ratios are
highly ranked but not numerically interchangeable:

| Library | Mean absolute difference | 95th percentile absolute difference | Spearman |
|---|---:|---:|---:|
| Promoter | 0.1087 | 0.3249 | 0.9776 |
| 5'UTR | 0.1181 | 0.3463 | 0.9725 |
| Intron | 0.1486 | 0.4497 | 0.9793 |
| 3'UTR | 0.1496 | 0.5199 | 0.9630 |

That difference is large enough to treat the mean estimator as a first-class
experimental factor, as the team discussion requested.

The retained quick-run mean also differs from the exact deduplicated variant
target because of DNA-zero filtering and the aggregate pseudocount:

| Library | Mean absolute quick-minus-baseline difference | 95th percentile absolute difference | Spearman |
|---|---:|---:|---:|
| Promoter | 0.0252 | 0.0902 | 0.9966 |
| 5'UTR | 0.0404 | 0.1460 | 0.9932 |
| Intron | 0.0647 | 0.2634 | 0.9897 |
| 3'UTR | 0.0813 | 0.3021 | 0.9857 |

The overall ranking remains close, but the low-support tail can move by several
log2 units. A future paired comparison must decide whether it predicts the
canonical all-row baseline activity, the retained-row activity used by the
count model, or both as separately named readouts.

## Presentation Story For The Five Attached Figures

Use the figures as one argument rather than five independent results:

1. **RNA/DNA detection categories:** about one quarter of rows have zero DNA,
   while a smaller but meaningful group has zero RNA with positive DNA. This
   motivates explicit denominator filtering and keeping RNA zeros in a count
   likelihood.
2. **Target distributions:** the mean target has a broad continuous
   distribution, while observed spread and mean-uncertainty diagnostics have a
   sharp floor. The floor must be decomposed before calling the target
   multimodal biology.
3. **Support scatter plots:** spread and estimated mean uncertainty depend
   visibly on `n_eff`; therefore support is a confounder and a required
   baseline, not merely a helpful covariate to mention later.
4. **3A/3B pipeline illustration:** the current workflow deliberately converts
   barcode rows into two construct-level labels. Its 3A mean and 3B spread use
   different centers, so the next version of the illustration should name both
   estimands rather than call them a mean/variance pair.
5. **Hierarchical observation model:** this is the destination model in which
   sequence predicts a prior mean for construct activity and barcode counts
   update a latent construct activity under an NB observation model. It is not
   the next coding step; the EDA, normalization, likelihood, and identifiability
   gates come first.

The concise team message is: deduplication gives us a trustworthy observation
table; the quick mean head recovers sequence signal; the current spread label
is dominated by measurement support; and a count likelihood is the path to
separating activity from observation noise without discarding useful
singletons.

## Meeting Agreements, With Precise Interpretations

The meeting established these modeling tiers:

1. Variant-level mean-expression baseline on deduplicated aggregates.
2. Summary-target model with a mean/activity output and an observed
   barcode-spread output.
3. Count-likelihood model operating on barcode-level RNA and DNA counts.
4. Later hierarchical model with sequence-predicted activity and explicit
   observation noise or partial pooling.

Several phrases from the meeting need tighter interpretation:

- "Weighted mean" is ambiguous. The current `mean_expr` is a log ratio of
  summed counts. It is not the DNA-weighted mean of pseudocounted barcode log
  ratios.
- The current mean and spread heads are not the mean and variance of one random
  variable. The mean head uses a ratio of construct-level sums. The spread
  head uses the variance of per-barcode pseudocounted log ratios around their
  own DNA-weighted mean.
- Poisson and NB are distributions for observed counts, not distributions to
  apply independently to a real-valued mean label and a real-valued variance
  label. A count model may have neural outputs for activity and dispersion,
  but those are likelihood parameters rather than the existing two summary
  targets.
- MSE corresponds to a Gaussian observation model with fixed variance up to a
  constant. Two independent MSE terms imply two conditionally independent
  Gaussian label models after any target standardization. That is a useful
  baseline, not a complete probabilistic account of barcode counts.
- Excluding singleton constructs globally would throw away useful activity
  information. The default repair is to keep them for the mean/activity loss
  and mask them from a spread loss that cannot be defined from one value.
- Better spread prediction after removing the zero spike would not by itself
  prove biological predictability. It may only mean the support artifact was
  removed. We must measure spread repeatability before asking a sequence model
  to learn it.

### Review Amendments, 2026-07-15

- In a one-construct Poisson offset fit with positive retained RNA and a fixed
  depth convention, `eta_hat = ln(R_i / D_i) - c_l`. M5 is therefore the same
  point activity as unpseudocounted M1 after log-base conversion; its added
  value is likelihood uncertainty, diagnostics, and pooling. If `R_i = 0`, the
  unconstrained fixed-effect MLE is negative infinity.
- `log d_ij` is a true fixed offset. `c_l` is a depth offset only if supplied
  from a known assay-depth ratio; if learned, it is an intercept and is
  shift-confounded with construct effects or a neural output bias unless a
  centering/control constraint anchors it.
- Here `l` labels a CRE-part subset, not an observed sequencing sample. A
  part-specific intercept must not be called a technical depth factor without
  sample-level provenance.
- Split-half spread agreement measures barcode-resampling stability, not
  biological-repeat reproducibility. Spearman-Brown correction can be shown as
  an assumption-dependent approximation, not an unconditional ceiling.
- Candidate-estimand sensitivity, reliability, likelihood selection, and
  calibration use only `train_only` and `development` constructs from the
  canonical split manifests. Protected `audit_test` and sequence-ineligible
  constructs remain available only for clearly labeled assay QC.

## Canonical Data Contract

### Current observation unit

One Lib1 barcode-table row is one deduplicated construct-barcode observation
with paired DNA and RNA counts. The current files do not expose a separate
biological-replicate, technical-replicate, or FASTQ index. The `fastqs1-5`
name describes the upstream combined product; it must not be turned into a
replicate index in downstream notation without recovering and validating
unaggregated files.

Use "barcode observation" for index `j`, not "replicate", in new code and
figures. Introduce a replicate index `k` only if a later source table actually
contains independent per-replicate counts.

### Required columns

| Concept | Lib1 column |
|---|---|
| Construct identity and split key | `parts_concatenated` |
| Barcode identity | `bba1_ddc1_concat` |
| DNA count | `DNA_bc_counts` |
| RNA count | `RNA_bc_counts` |
| Library membership | `library_name` |
| Variable sequence | `Promoter`, `FivePrime`, `Intron`, or `ThreePrime` |

The model input is the variable construct sequence. Barcode identity and
barcode sequence are not predictive inputs in the first model family.

### Non-negotiable split rule

All barcode observations for one `parts_concatenated` value must remain in the
same outer split. For the immediate four separately trained part models, the
stable identity is `(library_name, parts_concatenated)`; the library component
prevents the rare cross-library construct collision from becoming ambiguous.

For observation-model diagnostics, an inner barcode holdout may be used only
inside outer-training constructs. It never replaces the outer construct
holdout used to evaluate sequence generalization.

## Symbol Table

| Symbol | Meaning | Current column or source |
|---|---|---|
| `l` | library/CRE-part index | `library_name` |
| `i` | construct index within library | `parts_concatenated` |
| `j` | barcode observation index within construct | `bba1_ddc1_concat` |
| `k` | biological/technical replicate index | unavailable in current combined table |
| `x_i` | variable construct sequence used by the neural model | part-specific sequence column |
| `d_ij` | observed DNA barcode count | `DNA_bc_counts` |
| `r_ij` | observed RNA barcode count | `RNA_bc_counts` |
| `B_i` | all exact-deduplicated barcode rows for construct `i` | before DNA filtering |
| `t_D` | minimum DNA count for a retained barcode | currently 1 in the quick run |
| `J_i(t_D)` | retained barcode set for construct `i` | finite RNA/DNA and `d_ij >= t_D` |
| `n_i` | number of retained barcodes | `|J_i|` |
| `D_i` | total retained DNA | `sum_j d_ij` |
| `R_i` | total retained RNA | `sum_j r_ij` |
| `D_i^all`, `R_i^all` | totals over all rows in `B_i` | canonical dedup variant aggregation |
| `w_ij` | normalized DNA weight | `d_ij / D_i` |
| `q_ij` | raw barcode activity ratio | `r_ij / d_ij`, only for `d_ij > 0` |
| `y_ij^(a)` | pseudocounted barcode log ratio | defined below |
| `n_eff,i` | Kish-style effective barcode count | defined below |
| `eta_i` | latent or predicted construct log activity in a count model | model parameter |
| `alpha_l` | NB2 overdispersion, where variance is `mu + alpha * mu^2` | model parameter |
| `z_i` | spread-loss eligibility mask | support-dependent |

Use one log base per quantity and write it in the name. The current activity
targets use `log2`; the current variance transform uses the natural logarithm.

## Retention And Zero Policy

Define the retained set as:

```math
J_i(t_D) = \{j: d_{ij}, r_{ij}\text{ are finite and }d_{ij} \ge t_D\}.
```

The default exploratory threshold is `t_D = 1`, with sensitivity views for at
least `1, 2, 5, 10` if support permits.

- Keep `r_ij = 0, d_ij > 0`. These are informative low-expression outcomes
  and are handled naturally by Poisson/NB likelihoods.
- Exclude `d_ij = 0` from ordinary ratio and simple DNA-offset models. A
  positive RNA count with zero DNA is impossible under `mu_ij = d_ij * rate`,
  so silently adding an offset pseudocount changes the observation model.
- Exclude both-zero rows from the simple target/likelihood path; retain their
  frequency in QC reporting.
- A later joint DNA/RNA latent-abundance model may revisit zero-DNA rows, but
  that is a different model and must be labeled as such.

## 3A: Candidate Construct Activity Estimands

The word "mean" should not appear alone in code or slides. Use one of the
names below.

### Exact deduplicated variant baseline

The current baseline target is computed before the barcode-level DNA filter:

```math
m_i^{baseline} = \log_2\left(\frac{R_i^{all}}{D_i^{all}}\right).
```

The canonical variant files aggregate every exact-deduplicated matching row,
including rows with `d_ij = 0`. Those rows add no DNA denominator but can add
RNA numerator. All current single-part construct totals have positive DNA and
RNA, so this baseline requires no pseudocount.

This target remains the primary connection to the completed deduplicated
campaign. It must not be silently replaced by a retained-row target.

### Barcode-level pseudocounted log ratio

```math
y_{ij}^{(a)} = \log_2\left(\frac{r_{ij} + a_R}{d_{ij} + a_D}\right).
```

The quick run used `a_R = a_D = 0.5`. This quantity is finite for RNA zeros
after DNA filtering, but its value for low counts depends strongly on the
pseudocount and sequencing scale.

### A. Retained-row log ratio of summed counts: current quick `mean_expr`

```math
m_i^{sum}(a_R,a_D) =
\log_2\left(\frac{R_i + a_R}{D_i + a_D}\right).
```

This is the exact current quick-run mean target. A single pseudocount is added
after summing the construct's retained counts. It differs from
`m_i^baseline` through both the retained population and the pseudocount.

Without pseudocounts and with positive counts:

```math
\frac{R_i}{D_i} = \sum_j w_{ij} q_{ij}.
```

Therefore `m_i^sum` is the logarithm of a DNA-weighted arithmetic mean of raw
ratios. It is not a weighted arithmetic mean of log ratios.

### B. Equal-weight mean of barcode log ratios

```math
m_i^{log,eq} = \frac{1}{n_i}\sum_{j \in J_i} y_{ij}^{(a)}.
```

This gives every retained barcode equal influence and corresponds to the
center of the unweighted barcode-log distribution.

### C. DNA-weighted mean of barcode log ratios

```math
m_i^{log,DNA} = \sum_{j \in J_i} w_{ij}y_{ij}^{(a)},
\qquad
w_{ij}=\frac{d_{ij}}{D_i}.
```

This gives high-DNA barcodes greater influence. It is the center used by the
current weighted spread target, but it is not the current mean head.

DNA weighting is an exposure-weighting heuristic, not automatically an
inverse-variance weight. Under an idealized independent Poisson approximation
with positive counts, a delta-method variance for `ln(RNA/DNA)` is roughly
`1/RNA + 1/DNA`, so a precision weight would depend on both counts.
Overdispersion and shared normalization change it again. Therefore compare DNA
weighting empirically, as the meeting proposed, and call it "DNA weighted"
rather than "precision weighted" unless a variance model supplies the weight.

The non-equivalence is structural, not a coding error. Even with no
pseudocounts, the log of a weighted arithmetic mean generally differs from a
weighted mean of logs. The EDA tool must show this difference and any induced
rank changes directly.

### D. Count-regression activity estimate

The simplest Poisson offset estimator for one construct is:

```math
r_{ij} \sim \operatorname{Poisson}(\mu_{ij}),
\qquad
\log \mu_{ij} = \log d_{ij} + c_l + \eta_i.
```

Here `log d_ij` is a fixed offset with coefficient one, `c_l` is a library or
RNA/DNA depth offset, and `eta_i` is construct activity. Replacing Poisson with
NB2 allows extra-Poisson variation.

The GLM link uses the natural logarithm, so `eta_i` is in natural-log units.
After accounting for `c_l`, compare it with a log2 activity target using
`eta_i / ln(2)`, with the exact conversion recorded in the export manifest.

The existing `mpra_eda_tool` dropdown does not currently implement this
estimand. Its formula is `rna_counts ~ dna_counts`, which treats DNA as an
ordinary linear predictor under the GLM link. The revamp must add an explicit
`exposure=dna_counts` or `offset=log(dna_counts)` implementation and test it on
hand-calculated data.

For positive retained totals and a fixed `c_l`, the per-construct Poisson MLE
obeys `eta_hat_i = ln(R_i / D_i) - c_l`. M5 should therefore not be promoted as
a numerically new summary target relative to M1; it supplies a probability
model, uncertainty, residual diagnostics, and a bridge to pooled/sequence
models. An all-RNA-zero construct lies on the `eta -> -infinity` boundary
unless pooling or a prior regularizes it.

### Mean-estimator experiment

Treat 3A as a controlled estimand bake-off, not a broad neural HPO:

| ID | Activity estimate | Main question |
|---|---|---|
| M0 | all-row `m_baseline`, no pseudocount | exact dedup campaign target |
| M1 | retained-row `m_sum`, no pseudocount when retained totals are positive | effect of DNA-zero filtering |
| M2 | retained-row `m_sum(0.5, 0.5)` | compatibility with quick two-head run |
| M3 | equal mean of `y_ij(0.5, 0.5)` | equal barcode influence |
| M4 | DNA-weighted mean of `y_ij(0.5, 0.5)` | high-DNA influence |
| M5 | Poisson offset `eta_hat_i` | regression activity without ratio pseudocount |
| M6 | NB2 offset `eta_hat_i` | regression activity with overdispersion |

For every pair, report numerical difference, rank change, support dependence,
zero-RNA dependence, barcode-downsampling stability, and downstream sequence
prediction under an otherwise paired setup.

The meeting's reference to a "CLASSIC" paper and an approximately `0.5 R^2`
effect is not yet bibliographically resolved. Do not repeat that effect size as
established evidence until the exact paper, estimator, evaluation split, and
definition of `R^2` are recorded.

## 3B: Candidate Observed Barcode-Spread Estimands

These are summaries of the observed `y_ij^(a)` values. They combine biological
activity, count sampling, DNA support, barcode-specific technical effects, and
pseudocount behavior. Call them "observed barcode spread", not intrinsic
biological variance.

### A. Equal-weight population variance

```math
v_i^{eq,pop} = \frac{1}{n_i}\sum_j
\left(y_{ij}^{(a)} - m_i^{log,eq}\right)^2.
```

### B. DNA-weighted population variance: current `barcode_var`

```math
v_i^{DNA,pop} = \sum_j w_{ij}
\left(y_{ij}^{(a)} - m_i^{log,DNA}\right)^2.
```

The quick run used:

```math
\texttt{log_barcode_var}_i = \ln(v_i^{DNA,pop} + 10^{-4}).
```

This is a natural-log transform of a variance measured in squared log2-ratio
units. It is centered on `m_i^(log,DNA)`, not on `m_i^sum`.

The two spread definitions answer different questions. Equal weighting
describes variation across retained barcode identities. DNA weighting
describes variation under a distribution that samples barcodes in proportion
to their DNA abundance. If the intended purpose is measurement precision,
neither should be called optimal until an observation model derives the
appropriate variance weights.

### C. Finite-sample-corrected weighted variance

For normalized weights, a candidate correction is:

```math
v_i^{DNA,corr} =
\frac{\sum_j w_{ij}(y_{ij}^{(a)}-m_i^{log,DNA})^2}
{1-\sum_j w_{ij}^2}.
```

This is undefined for one effective observation and unstable when one barcode
dominates. It should be presented as a candidate sample-variance correction,
not automatically declared the true variance. The interpretation of DNA
weights as reliability weights versus frequency weights must be stated.

### Effective barcode support

```math
n_{eff,i} = \frac{(\sum_j d_{ij})^2}{\sum_j d_{ij}^2}
= \frac{1}{\sum_j w_{ij}^2}.
```

`n_eff` is at most the retained barcode count and becomes one when a single
barcode carries all DNA weight. It measures weight concentration, not a new
independent barcode count.

Under an iid equal-variance approximation, the variance of a weighted mean is
proportional to `1 / n_eff`. Therefore:

```math
\widehat{Var}(m_i^{log,DNA}) \approx
\frac{v_i^{DNA}}{n_{eff,i}}.
```

The notebook's `mean_se_var = barcode_var / n_eff` is a useful diagnostic
heuristic under that approximation. It is not an exact standard error under
arbitrary heteroskedastic counts and estimated weights, and it should not be a
third neural output in the first experiment.

### Spread eligibility and masking

Define:

```math
z_i^{(2)} = 1[n_i >= 2],
\qquad
z_i^{(3)} = 1[n_i >= 3].
```

The default repaired two-head loss should be:

```math
L = \frac{1}{N}\sum_i L_{mean,i}
+ \lambda_{spread}
\frac{\sum_i z_i L_{spread,i}}{\max(1,\sum_i z_i)}.
```

Do not encode a missing or undefined spread target as zero. Keep singleton
constructs in `L_mean`, set their spread mask to zero, and log the eligible
count per split and epoch.

Run `n_i >= 2`, `n_i >= 3`, and an `n_eff`-based sensitivity analysis. Do not
pick the threshold from test performance.

### Spike-and-slab decision

A literal spike-and-slab output is not the default recommendation for the next
run. Most of the spike is determined by the observed support rule `n_i = 1`,
not by an unknown sequence property. Asking a sequence-only head to predict
that spike risks learning synthesis/coverage artifacts. The simpler estimand is
conditional spread among support-eligible constructs, implemented with a mask.

If exact-zero spread remains common when `n_i >= 2`, quantify whether it comes
from identical rounded count ratios or a meaningful second process. Only then
consider a hurdle/censored/spike-and-slab label model. If barcode recovery
itself is scientifically interesting, model it as a separately named QC or
assay-yield outcome rather than folding it into biological spread.

### Spread reliability ceiling

Before another spread-head HPO, estimate whether the target is repeatable:

1. Restrict to constructs with enough barcodes to form two nontrivial groups,
   initially `n_i >= 4` or `n_i >= 6`.
2. Repeatedly split barcodes within each construct into two halves, stratifying
   or balancing by DNA support when possible.
3. Recompute each spread estimand in both halves.
4. Report half-to-half Pearson, Spearman, RMSE, and rank stability by library,
   `n_i`, `n_eff`, total DNA, and zero-RNA burden.
5. Bootstrap the barcode set to show estimator uncertainty.

If a spread target has weak split-half repeatability, poor sequence prediction
is expected and more architecture HPO is not the next remedy. The target may
need shrinkage, a count likelihood, more barcodes, or a narrower estimand.

## Probabilistic Count Models

### Poisson offset baseline

```math
r_{ij} \mid d_{ij},x_i \sim \operatorname{Poisson}(\mu_{ij}),
\qquad
\log \mu_{ij} = \log d_{ij} + c_l + f_\theta(x_i).
```

This removes the per-barcode log-ratio pseudocount and naturally includes RNA
zeros. It conditions on observed DNA as fixed exposure.

### NB2 offset baseline

```math
r_{ij} \mid d_{ij},x_i \sim \operatorname{NB2}(\mu_{ij},\alpha_l),
```

with:

```math
E[r_{ij}] = \mu_{ij},
\qquad
Var(r_{ij}) = \mu_{ij} + \alpha_l\mu_{ij}^2.
```

Start with one global or one library-specific dispersion. A sequence-predicted
dispersion head is a later ablation because it adds a flexible path that can
absorb mean-model misspecification and support artifacts.

A singleton contributes valid count likelihood information about activity, but
it cannot identify a construct-specific dispersion from one observation. Its
uncertainty is handled through a pooled/global dispersion or hierarchical
prior, not by pretending that its observed variance is known to be zero.

In this count model, a two-output neural head would naturally produce an
activity parameter and a constrained dispersion parameter, for example:

```math
(\widehat{\eta}_i, h_i) = f_\theta(x_i),
\qquad
\widehat{\alpha}_i = \operatorname{softplus}(h_i) + \epsilon.
```

These are not the old `mean_expr` and `log_barcode_var` labels. The model is
trained by count negative log likelihood and evaluated through both activity
prediction and posterior predictive count behavior.

### Hierarchical partial-pooling model

The fifth meeting illustration can be written as:

```math
r_{ij} \sim \operatorname{NB2}(\mu_{ij},\alpha_l),
\qquad
\log \mu_{ij} = \log d_{ij} + c_l + \eta_i,
```

```math
\eta_i \sim \mathcal{N}(f_\theta(x_i),\tau_l^2).
```

`f_theta(x_i)` is the sequence-predicted activity mean. `eta_i` is a latent
construct activity informed by both the sequence prior and its barcode counts.
`tau_l` describes construct-level residual variation not explained by the
sequence model.

This distinction matters at evaluation time:

- Reconstructing a training construct with its fitted `eta_i` measures
  posterior fit.
- Predicting a new construct must use the prior predictive distribution from
  `f_theta(x_new)` and `tau_l`, without a fitted construct-specific latent.

The primary held-out metric must test the second case.

### Optional joint DNA/RNA model

The simple offset model treats DNA counts as known exposure. A richer model can
represent barcode abundance explicitly:

```math
d_{ij} \sim \operatorname{NB2}(s_D a_{ij},\alpha_D),
```

```math
r_{ij} \sim \operatorname{NB2}(s_R a_{ij}\exp(\eta_i),\alpha_R).
```

Here `a_ij` is latent barcode/construct abundance and `s_D`, `s_R` are assay
depth factors. This is closer in spirit to MPRAnalyze, which models uncertainty
in the DNA and RNA libraries. It is a later stage because it adds latent
variables and normalization decisions.

### Identifiability warning

With one aggregated DNA/RNA count pair per barcode, a barcode-specific random
effect, NB overdispersion, and generic count noise can be difficult to separate.
Do not add all of these at once. A term such as:

```math
u_{ij} \sim \mathcal{N}(0,\sigma_{bc}^2),
\qquad
\log \mu_{ij}=\log d_{ij}+\eta_i+u_{ij}
```

is not automatically identifiable from the current table. Revisit it only if
FASTQ-, batch-, or biological-replicate counts can be recovered, or if a
simulation/recovery study shows the parameters can be distinguished.

## Normalization Questions That Must Be Closed

Before fitting count likelihoods, record:

1. Whether `DNA_bc_counts` and `RNA_bc_counts` are raw summed reads, normalized
   counts, or counts after any upstream clipping/filtering.
2. Whether FASTQs 1-5 are independent samples, lanes, or pooled technical
   pieces, and whether unaggregated paired counts still exist.
3. Whether library-specific RNA and DNA depth factors are known.
4. Whether each CRE part will be fit separately. If so, a learned intercept can
   absorb one global RNA/DNA scale ratio, but the provenance still needs to be
   explicit.
5. Whether depth factors are estimated from training data only and then frozen
   for validation/audit.

No agent should infer independent replicates from a filename.

## Immediate EDA Tool Questions

The tool should help answer these in order:

1. How do zero categories, total DNA/RNA, retained `n_i`, and `n_eff` differ by
   part and support stratum?
2. How much do M0-M6 activity estimators differ, and where do rankings change?
3. How much of each spread estimator is explained mechanically by `n_i`,
   `n_eff`, total DNA, zero-RNA fraction, and pseudocount choice?
4. Is the singleton spike fully accounted for by support, and what remains in
   the slab after masking?
5. Is observed variance larger than Poisson mean-variance behavior predicts?
6. Does NB2 improve held-out count log likelihood and zero calibration over
   Poisson within each part/support stratum?
7. Is a hurdle or zero-inflated model still needed after DNA support is
   controlled, or does NB2 account for the zeros?
8. Which spread estimand has enough split-half repeatability to justify a
   sequence-prediction head?

The tool-specific views, code layout, tests, and acceptance criteria are in
the sibling-repo plan.

## Phased Experiment Roadmap

### Phase 0: freeze provenance and comparison contract

Deliverables:

- machine-readable registry of the four input barcode files and manifests;
- hashes, row/construct counts, required columns, and library labels;
- confirmed current baseline split artifacts and construct IDs;
- explicit statement that current rows are barcode observations, not
  independent assay replicates; and
- one target-definition JSON/YAML record for every candidate estimand.

Gate: all formulas can be mapped to columns and reproduced on a toy fixture.

### Phase 1: pure estimand and QC library in `mpra_eda_tool`

Implement schema adapters, zero categories, support metrics, M0-M4 summary
estimators, spread estimators, eligibility masks, and provenance exports as
pure functions with tests. Define stable result interfaces for M5-M6, which
are fit and evaluated in Phase 3. Preserve Lib0 through a separate adapter.

Gate: tests cover singleton, equal counts, unequal DNA weights, RNA zero, DNA
zero, both zero, repeated barcode IDs, and hand-calculated offset examples.

### Phase 2: interactive Lib1 EDA

Add dataset/part selectors, threshold and pseudocount controls, estimator
comparison, a selected-construct formula microscope, singleton decomposition,
and count-distribution diagnostics. Generate a static all-four-library report
from the same core functions.

Gate: every plot names the estimand and support population; changing a control
updates both formula and eligible-row count; no UI callback contains duplicate
math.

### Phase 3: non-neural observation-model bake-off

Compare Poisson offset, NB2 offset, and only if diagnostics justify them,
hurdle/zero-inflated variants. Use held-out barcode observations within
outer-training constructs for observation-model selection. Report NLL,
deviance/Pearson residuals, empirical-versus-predicted zero rates, and
posterior predictive mean/variance/tails.

Gate: the preferred likelihood improves held-out NLL and calibration in more
than one support stratum without relying on test constructs.

### Phase 4: repaired summary-target neural baseline

Use one promoted architecture per part and the frozen baseline split. Compare:

| ID | Mean head | Spread head | Spread eligibility |
|---|---|---|---|
| S0 | baseline mean only | none | n/a |
| S1 | current `m_sum(0.5)` | current weighted population variance | all, diagnostic replay only |
| S2 | current `m_sum(0.5)` | current weighted population variance | `n_i >= 2` |
| S3 | selected 3A mean | selected weighted/unweighted spread | `n_i >= 3` or frozen support gate |
| S4 | selected 3A mean | finite-sample-corrected or shrinkage spread | frozen support gate |

Vary `lambda_spread` only after target scaling and mask behavior are verified.
Compare S0 and each two-head model on the same mean metric to detect negative
transfer.

The current `Lib1MeanSpreadDataModule` requires every target to be finite,
standardizes each target from the training split, and returns a dense target
tensor. `CNNBasicTraining` then applies one criterion to the full tensor. A
repaired run therefore needs explicit per-output masks and a masked loss/metric
path; placing zero in the singleton spread column would recreate the current
problem. Unit tests must verify each head's denominator and logged eligible
count.

Gate: spread performance exceeds a support-only baseline and is consistent
with the empirical reliability ceiling; mean performance is not materially
degraded.

### Phase 5: neural Poisson/NB offset models

Run sequentially:

1. Poisson, sequence predicts activity only.
2. NB2, one global dispersion.
3. NB2, one dispersion per CRE part.
4. NB2, sequence-predicted dispersion with regularization, only if 2-3 show
   stable gain and dispersion signal.

For the simple models, use `torch.distributions` likelihoods inside the current
Lightning training structure. This avoids a second inference framework when
there are no latent variables. Freeze and unit-test the NB mean/dispersion to
library-parameter conversion because software parameterizations differ.

Gate: count NLL and calibration improve over Poisson, and sequence-derived
activity remains competitive with the paired mean-expression baseline.

### Phase 6: hierarchical latent-activity model

Use Pyro only when latent `eta_i`, partial pooling, or the joint DNA/RNA model
is actually required. `pyro-ppl` is already a BODA dependency, but the current
`boda_env` is Python 3.7 with Pyro 1.8.6 and Torch 1.13.1, so implementation
must target the installed environment or deliberately create a versioned new
environment.

Gate: simulation-based parameter recovery succeeds; held-out constructs are
evaluated prior-predictively; posterior predictive checks beat the simpler NB
model enough to justify the extra inference complexity.

### Phase 7: active-learning uncertainty

Keep these uncertainty sources separate:

- observation/aleatoric uncertainty from the count likelihood;
- construct residual uncertainty from hierarchical partial pooling;
- epistemic uncertainty from seed/model ensembles or posterior uncertainty;
- acquisition score uncertainty used to select new sequences; and
- diversity/novelty constraints in sequence space.

This phase begins only after activity calibration and out-of-distribution
behavior are understood.

## Evaluation Contract

### Mean/activity metrics

- Pearson and Spearman;
- RMSE and MAE in a named log unit;
- coefficient of determination `1 - SSE/SST`, labeled `COD R2` rather than
  squared Pearson;
- calibration intercept and slope;
- metrics by retained barcode count, `n_eff`, total DNA, zero-RNA fraction,
  sequence length/mask stratum, and CRE part; and
- paired difference versus the deduplicated baseline on identical constructs.

### Spread metrics

- report only on the frozen eligible support population;
- Pearson, Spearman, RMSE, and MAE;
- support-only baselines using library, `n_i`, `n_eff`, total DNA, and zero-RNA
  burden;
- split-half reliability and bootstrap uncertainty of the observed target;
- calibration of predicted versus observed spread by quantile; and
- mean-head performance beside spread performance to show multi-task tradeoff.

### Count-likelihood metrics

- mean NLL per barcode and per construct;
- Poisson/NB deviance or randomized quantile residual diagnostics;
- observed versus predicted zero rate;
- posterior predictive count mean, variance, median, and upper tail;
- calibration by DNA exposure and CRE part; and
- activity ranking against the fixed dedup aggregate target.

### Selection boundary

Training and validation metrics may be logged every epoch. The frozen audit
set must not be scored every epoch or used for model selection. If an
exploratory split is called `test` in the quick-run tooling, label it
development-test and do not confuse it with the protected campaign audit.

## Off-The-Shelf Software Strategy

Use the least complex established layer that matches the model:

- `statsmodels` in the EDA tool for transparent Poisson/NB offset baselines and
  diagnostics, with explicit exposure and dispersion handling;
- `torch.distributions.Poisson` and `NegativeBinomial` for differentiable
  likelihoods in the existing sequence-model training loop;
- Pyro SVI and plates for latent hierarchical models after a simulation gate;
- MPRAnalyze, `mpralm`, and BCalm as external MPRA-specific estimators or
  benchmarks, not as substitutes for sequence prediction.

Do not import the whole BODA training stack into `mpra_eda_tool`. Export a
versioned target/diagnostic table plus manifest from the EDA/statistics layer,
then let `boda2_EU` consume that product.

## Decision Gates And Stop Rules

### Gate A: data semantics

Stop if the source of FASTQ aggregation, normalization, or count pairing
cannot be established. Document uncertainty before fitting a count model.

### Gate B: estimand stability

Do not promote a 3A mean estimator solely because it correlates with the
current target. Require barcode-downsampling stability and paired sequence
generalization evidence.

### Gate C: spread learnability

Do not launch spread HPO if split-half spread repeatability is weak or if a
support-only baseline explains the apparent signal.

### Gate D: distribution choice

Do not choose NB because the marginal count histogram looks overdispersed.
Require held-out conditional likelihood and zero/tail calibration after DNA
exposure is modeled.

### Gate E: hierarchical complexity

Do not add construct and barcode random effects without simulation-based
recovery and a clear new-construct prediction path.

### Gate F: fair model comparison

Do not compare raw scores across different split identities, support filters,
target definitions, or audit access policies as if they were paired.

## Missing Or Underemphasized Meeting Considerations

The following should be added to future team presentations:

1. Barcode rows are not currently independent biological replicates. The
   combined table has no replicate index.
2. The observed spread target mixes count noise, support, pseudocount effects,
   barcode technology, and any true construct-dependent variability.
3. The singleton spike is mathematically forced; its upstream abundance may be
   technical, but that causal explanation remains a hypothesis.
4. DNA is noisy. A DNA-offset model conditions on it; a joint DNA/RNA model is
   needed to propagate DNA-count uncertainty.
5. Library-size/depth normalization must be explicit for count likelihoods.
6. A sequence-predicted dispersion head is not automatically identifiable or
   biologically meaningful.
7. Spread target reliability places an empirical ceiling on model performance.
8. Mean and spread losses need separate eligibility masks and denominators.
9. A support-only model is a required baseline for the spread head.
10. Test/audit metrics should not drive epoch selection even if the logging
    stack can compute them.
11. Poisson versus NB versus hurdle is a conditional model comparison, not a
    choice made from a marginal histogram.
12. A count model's activity and dispersion outputs are not the same objects as
    the current summary-target mean and variance.

## Learning Plan For Minhang

The goal is not to become a mathematical statistician before moving forward.
It is to build a reliable habit for translating a biological question into an
estimand, then into a likelihood, then into a prediction experiment.

### Use one estimand card per quantity

Before an agent implements a new target, fill in this template together:

```text
Name:
Biological question in one sentence:
Observed rows/columns used:
Formula:
Log base and units:
Rows excluded and why:
What happens for RNA=0, DNA=0, and n=1:
What assumptions make this meaningful:
What technical effects can change it:
How we will test repeatability:
How a model will be evaluated on it:
```

If an equation cannot be completed in this card, it is not ready for HPO.

### Learn each concept through one construct

Use a selected construct with three barcode rows and calculate by hand or in a
small table:

1. `D_i`, `R_i`, and `m_sum`.
2. Each `y_ij` with the current pseudocount.
3. Equal and DNA-weighted means.
4. Equal and weighted variances.
5. `n_eff` and the finite-sample correction.
6. Poisson expected RNA counts from a fixed activity.
7. NB variance at the same mean for two dispersion values.

Then repeat with a singleton and an RNA-zero barcode. This will make the spike,
mask, and count likelihood concrete much faster than reading equations alone.

The EDA tool should include this as a "formula microscope": selected raw rows
on the left, formula substitutions in the middle, and resulting estimands on
the right. Changing threshold, pseudocount, or weighting should update all
three.

### Five short learning modules

1. **Ratios and logs:** understand why `log(sum ratio)` differs from `mean(log
   ratio)` and how pseudocount placement changes low counts.
2. **Weights and effective sample size:** understand what DNA weighting values,
   what it sacrifices, and why unequal weights reduce `n_eff`.
3. **Likelihoods:** for Poisson, learn `Var=mean`; for NB2, learn
   `Var=mean+alpha*mean^2`; connect each to visible mean-variance plots.
4. **Hierarchy and partial pooling:** use a three-construct toy example to see
   low-support activities shrink more strongly toward the sequence-predicted
   mean.
5. **Calibration and generalization:** distinguish fitting observed constructs,
   predicting held-out constructs, and forecasting counts for a new construct.

### Questions to ask an agent at every milestone

- Which exact data column is each symbol?
- Is this a target summary or a probability-model parameter?
- What is random and what is conditioned on?
- What happens at zero and at one barcode?
- Which assumption makes this weight valid?
- Can the parameter be identified from one DNA/RNA pair per barcode?
- What is the simplest baseline that could explain this plot?
- Is this metric computed on the same construct population as the baseline?
- Could this apparent improvement be caused only by filtering easier cases?
- How would we predict a completely new sequence with this model?

### A practical collaboration rhythm

For each stage, ask the agent for four artifacts before accepting code:

1. a toy calculation with expected numbers;
2. one presentation plot with a one-sentence question and takeaway;
3. a failure-case plot or test;
4. a decision note saying what evidence would change the next step.

This keeps you in control of the scientific meaning while the agent handles
implementation detail. Your most valuable role is not deriving every loss
from memory; it is catching when the code's target no longer answers the
question the team thinks it is answering.

## Deliverables Checklist

- [ ] Lib1 and Lib0 schema contracts in `mpra_eda_tool`.
- [ ] Pure, tested 3A and 3B estimand functions.
- [ ] Toy fixture and hand-calculated expected values.
- [ ] Provenance-aware four-library static QC report.
- [ ] Interactive formula microscope.
- [ ] Estimator comparison and rank-change views.
- [ ] Singleton/spread decomposition and target reliability analysis.
- [ ] Correct Poisson offset and explicitly parameterized NB2 diagnostics.
- [ ] Held-out conditional likelihood and zero calibration.
- [ ] Frozen target-definition manifest for BODA consumption.
- [ ] Paired repaired two-head experiment spec.
- [ ] Paired Poisson/NB sequence-model experiment spec.
- [ ] Simulation/recovery notebook before hierarchical implementation.
- [ ] Updated team diagram with barcode observations, not unlabeled
  "replicates".
- [ ] Exact citation for the meeting's "CLASSIC" result or removal of the
  unverified effect-size claim.

## References And Local Anchors

Local:

- [Dedup data-product plan](../../repo_hygiene/barcode_level_dedup_update_july6_2026.md)
- [Dedup campaign status](../dedup_phase1_rerun_july2026/README.md)
- [Earlier barcode uncertainty discussion](../learn/barcode_level_uncertainty_discussion_context_july7_2026.md)
- [Two-head data preparation code](../../../src/learn/prepare_lib1_two_head_mean_spread_dataset.py)
- [Two-head quick-run launcher](../../../src/learn/launch/lib1_two_head_mean_spread_quick_july2026_runs.sh)

External/statistical:

- [PyTorch probability distributions](https://docs.pytorch.org/docs/stable/distributions.html)
- [Pyro SVI introduction](https://pyro.ai/examples/svi_part_i.html)
- [statsmodels GLM exposure/offset API](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html)
- [MPRAnalyze paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6717970/)
- [`mpralm` aggregation options](https://rdrr.io/bioc/mpra/man/mpralm.html)
- [BCalm paper](https://doi.org/10.1186/s12859-025-06065-9)

## Final Handoff Rule

This file governs the scientific question and cross-repo boundary. The
`mpra_eda_tool` plan governs implementation inside that repository. The
deduplicated campaign documents govern the fixed baseline and audit boundary.
When they conflict, stop and write a dated amendment rather than silently
changing a target, split, or likelihood in code.
