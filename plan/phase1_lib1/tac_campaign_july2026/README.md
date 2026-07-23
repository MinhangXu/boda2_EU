# The TAC Campaign: Lib1 Mean-Expression Baseline

**Campaign opened:** 2026-07-20, after the semi-annual TAC meeting

**Current status:** the deduplicated Lib1 Stages 1--4 baseline and one-time
locked final test are complete; the items below are post-TAC follow-up work

**Primary target:** one construct-level value,
`log2(total RNA barcode counts / total DNA barcode counts)`

**Canonical completed-campaign index:**
[`../dedup_phase1_rerun_july2026/README.md`](../dedup_phase1_rerun_july2026/README.md)

This page is the durable context and decision queue for work that follows the
July 20 TAC. It intentionally links to the completed deduplicated campaign
instead of copying its protocol and results into another family of status
files. Add a new document only when a new experiment has a frozen protocol or
a completed result that cannot be represented clearly here.

## What The TAC Baseline Established

Lib1 contains five single-part sequence-to-expression problems. Exact-
deduplicated barcode rows are grouped by construct, DNA and RNA counts are
summed, and one aggregate expression target is formed:

\[
y_i = \log_2\!\left(\frac{\sum_j R_{ij}}{\sum_j D_{ij}}\right).
\]

Lower-support constructs remain eligible for training (`n_barcodes >= 1`),
while development and the already-opened final test used constructs with at
least eight distinct barcode identities. Barcode count is measurement support,
not a count of confirmed independent biological replicates.

| Part | Modeled constructs | Constructs with at least 8 barcodes | Selected route | RC | Loss | Development OOF r | Locked final-test r |
|---|---:|---:|---|---|---|---:|---:|
| Enhancer | 4,787 | 1,229 | transferred BassetBranched, K562/full | on | unweighted | 0.565 | 0.365 |
| Promoter | 7,893 | 1,931 | PromoterBassetVL, scratch | off | barcode weighted | 0.478 | 0.444 |
| Intron | 7,848 | 1,326 | ResNet1D, scratch | off | barcode weighted | 0.690 | 0.681 |
| 3'UTR | 6,845 | 775 | UTRBassetVL, scratch | off | barcode weighted | 0.493 | 0.452 |
| 5'UTR | 8,331 | 1,797 | UTRBassetVL, scratch | off | barcode weighted | 0.542 | 0.512 |

The selected policies are closed historical results. New work may compare new
methods on development folds, but it must not use the opened final test for
selection or calibration.

## Post-TAC Priority Queue

1. **High priority: disentangle the BODA2 augmentation recipe.** Reproduce the
   author recipe correctly, then test whether high-activity or high-support
   exposure changes the Enhancer result under controlled optimizer budgets.
2. **High priority: change the Intron estimand and run a residual-target
   correction.** Within-sequence-group performance, not pooled Pearson alone,
   is the model-improvement target.
3. **Medium priority: add deliberately simple sequence baselines.** Start with
   GC/length and k-mer ridge; treat clustering and motif discovery as distinct
   questions rather than one vague baseline.
4. **Medium/low priority: adapt model interpretation.** Start with a small,
   validated contribution-score and mutagenesis pilot before TF-MoDISco-scale
   analysis.
5. **Continuous: preserve campaign context and reduce redundancy by indexing,
   not deleting.** Keep canonical artifacts and provenance; do not copy figures
   or results into multiple new folders.

No new model was launched and no locked-final-test product was reopened while
creating this first campaign record.

---

## 1. What BODA2 Actually Did With Reverse Complements

The Nature paper's Methods text is unambiguous: Malinois used two separate training
augmentations:

1. include the reverse complement of **every** padded 600-bp training
   sequence; and
2. duplicate an oligo when its `log2[FC]` exceeded 0.5 in any of the three cell
   types.

At inference, the paper averaged predictions for the forward and
reverse-complement padded sequences. Therefore the author recipe was **not
conditional RC**. It was global RC plus conditional high-activity
oversampling, followed by strand-averaged prediction.

Primary sources:

- [Nature article, Methods: Malinois data preprocessing](https://www.nature.com/articles/s41586-024-08070-z#Sec12)
- [Official BODA2 repository and exact training command](https://github.com/sjgosai/boda2#model-training)
- local implementation:
  [`../../../boda/data/mpra_datamodule.py`](../../../boda/data/mpra_datamodule.py)

The local `DNAActivityDataset` implements the same composition: a high-
activity prefix is repeated through `duplication_cutoff`, after which
`use_reverse_complements` doubles the entire effective dataset. High-activity
oligos therefore receive four effective presentations under the full recipe;
other oligos receive two.

### Why This Changes The Follow-Up Question

Our completed Stage-2 intervention compared global RC off versus global RC on.
It did **not** separately test:

- high-expression oversampling;
- high-barcode-support oversampling;
- forward/RC prediction averaging at inference; or
- the interaction of any of those choices with the RC-trained source
  checkpoint.

Calling the next test only “conditional RC” would conflate orientation with
sample weighting. A selected row that receives an extra RC copy has both a new
orientation and twice the optimizer exposure. The experiment needs a matched
forward-copy control.

### Zero-Training Diagnostic First

Before changing the training loader, use the saved five-fold Enhancer
development checkpoints to compute, for each OOF construct:

- the forward prediction;
- the full-padded-sequence RC prediction;
- their arithmetic mean; and
- absolute forward--RC disagreement.

Compare forward-only with RC-averaged OOF Pearson, RMSE, COD R2, calibration,
and bias. This directly tests the inference component of the Malinois recipe
without retraining. It must remain development-only; do not reopen the locked
final test.

### Recommended First Training Experiment: Enhancer Only

Use the selected transferred K562/full Enhancer configuration, the existing
five development folds, original-orientation validation, model seed 1701, and
the same unweighted objective. Do not inspect the opened final test. Start with
the route on which global RC was consistently helpful; extend to other parts
only if this experiment yields a clear interaction worth pursuing.

Freeze two selection rules inside each training fold:

- **high expression:** top quartile of the training-fold target only;
- **high support:** `n_barcodes >= 8` in the training rows.

The expression threshold must be recomputed from training rows only. The
support threshold uses only pre-existing measurement metadata. A top quartile
roughly matches the fraction of Enhancer constructs with at least eight
barcodes, making the two policies closer in exposure.

| Arm | Extra presentation for selected rows | Question |
|---|---|---|
| A | none | uniform, RC-off anchor; reuse only if the implementation contract is identical |
| B | global RC for every row | completed Stage-2-style global-RC anchor |
| C | global forward duplicate for every row | same global multiplicity as B; isolates orientation from exposure |
| D | global RC plus duplication of both orientations for high-expression rows | closest transportable Malinois author recipe |
| E | the same pair duplication for a stable random subset of equal size | distinguishes targeted high-expression exposure from more rows/steps |
| F | forward copy for high-expression rows | effect of high-expression exposure without a new orientation |
| G | RC copy for the same high-expression rows | conditional orientation effect, paired to F |
| H | forward copy for high-support rows | effect of high-support exposure without a new orientation |
| I | RC copy for the same high-support rows | conditional orientation effect, paired to H |

This design adds seven five-fold arms, or 35 new cells, if the two existing
anchors can be reused under an identical implementation contract. The author-
style arm uses a training-fold quantile because transplanting the paper's
absolute `log2[FC] > 0.5` cutoff is inappropriate: in current Lib1 fold-0
training rows that cutoff would select only about 4% of Enhancers but more than
95% of each of the other four part libraries.

### Required Controls And Readouts

- Hold total optimizer steps, scheduler steps, checkpoint opportunity, and
  batch count fixed across arms. A larger dataset must not silently receive a
  larger training budget.
- Use a fixed sampler/manifest so the selected row identities and presentation
  counts are auditable.
- Keep development evaluation unweighted and original-orientation for the
  primary comparison.
- Secondary inference diagnostic: forward-only prediction versus the average
  of forward and full-padded-sequence RC predictions. Do this on development
  OOF checkpoints; it is a different intervention from training augmentation.
- Primary metric: pooled five-fold OOF Pearson. Co-report the five fold deltas,
  RMSE, COD R2, raw calibration slope/bias, and performance inside and outside
  the selected exposure stratum.
- Use paired construct bootstrap intervals on prediction differences. Include
  a matched random-row exposure control if either conditional arm appears to
  help, so “high expression/support” is distinguished from “more repeated
  rows.”
- Record `prediction(x)` versus `prediction(RC(x))` disagreement. This measures
  learned orientation sensitivity; it is not itself a performance metric.

### Interpretation Boundary

Even a positive result would not establish a universal biological strand
invariance. Enhancers are assayed in fixed vector context, the transferred
source checkpoint was itself trained with RC augmentation, and the current
scratch comparator changes architecture and input framing. State the result as
an augmentation/exposure effect within a specific route.

---

## 2. Resolving The Intron Sequence-Group Problem

### Diagnosis

The pooled Intron score is valid for ranking the observed natural mixture, but
it is composition assisted. Three nested, sequence-defined mask groups have
different expression means, and the CNN receives substantial credit for
separating those means. This is an estimand mismatch, not evidence of leakage.

The groups must be called **sequence-defined groups** or **inferred mask
strata**. They are not verified synthesis pools or measured splicing classes.
True synthesis-pool membership cannot be reconstructed uniquely because the
masks are nested.

| Evidence set | Natural pooled r | Within-group-centered r | Group-specific r |
|---|---:|---:|---|
| Stage-2 five-fold OOF leader | 0.682 | 0.451 | 0.599, 0.276, 0.176 |
| Selected policy, locked final test | 0.681 | 0.473 | 0.579, 0.338, 0.206 |
| Stage-4 full-N development OOF | 0.656 | 0.394 | 0.500, 0.333, 0.103 |

The Stage-2 decomposition attributed 70.6% of target--prediction covariance
to differences between inferred group means. At the same time, the nonzero
within-group-centered result shows that the model learns real variation beyond
group recognition. Do not subtract two correlations and call the difference
an exact “inflation”; they use different covariance and variance denominators.

Canonical evidence:

- [Intron estimand and challenge-set protocol](../dedup_phase1_rerun_july2026/lib1_dedup_intron_estimand_and_challenge_set_protocol_july2026.md)
- [post-presentation interpretation addendum](../dedup_phase1_rerun_july2026/lib1_dedup_post_presentation_interpretation_addendum_july17_2026.md)
- [Stage-2 estimand table](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_intron_leader_estimand_summary.csv)
- [Stage-2 covariance decomposition](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/stage2_intron_leader_signal_decomposition.csv)
- [locked-final-test Intron metrics](../../../src/learn/outputs/audit/lib1_dedup_final_audit_july2026/frozen_products/audit_intron_estimand_metrics.csv)

### Immediate Evaluation Change

Every future Intron development result should report, on the same OOF rows:

1. natural-mixture pooled Pearson;
2. within-group-centered Pearson;
3. macro-average and minimum group Pearson;
4. all three group-specific correlations, RMSE, and calibration; and
5. a fold-training-fitted group-mean baseline.

Use within-group-centered Pearson as the primary **model-improvement** estimand
and natural pooled performance as a non-degradation/deployment guardrail. Also
audit close sequence/design-family similarity across folds; exact
deduplication alone does not isolate near-neighbour design families.

### First Model Correction: Fold-Safe Residual Target

For each outer fold, estimate each group mean using that fold's training rows:

\[
\mu_{g,f}=\operatorname{mean}\{y_i:i\in\text{training}(f),g_i=g\},
\qquad
y_i^{\mathrm{resid}}=y_i-\mu_{g_i,f}.
\]

Train the same selected barcode-weighted ResNet1D to predict
`y_resid`, then reconstruct raw expression as

\[
\widehat y_i=\mu_{g_i,f}+\widehat y_i^{\mathrm{resid}}.
\]

Compare it with the unchanged selected model using identical construct IDs,
folds, architecture, weight rule, optimizer budget, and model seed. The paired
primary comparison is within-group-centered OOF performance. Pooled Pearson,
RMSE, calibration, and minimum-group performance are guardrails.

This is the simplest direct test of whether forcing the network to model
within-group residuals improves the scientific estimand. No broad HPO is
needed.

### Bounded Second Step

Only if residual-target training helps, compare a small hierarchy:

1. current shared ResNet1D;
2. training-mean offset plus one shared residual head;
3. shared encoder with group-specific residual heads; and
4. separate per-group models as a diagnostic comparator.

Group-count reweighting is low priority because the inferred groups are
already similar in size, and equal-group weighting barely changes pooled
correlation. An adversarial objective that erases all group information is
also not the first choice: splice-boundary/group signal may be real biology.

A boundary-neutralization retraining can be a useful mechanistic ablation, but
it must be labeled as such because it removes biologically meaningful splice
boundary information as well as easy group identity.

### Metadata And Prospective Data

Request the highest-value missing metadata before making biological sublibrary
claims:

- true construct-to-synthesis-pool membership;
- original design-parent or motif-family identifiers; and
- measured splicing/junction evidence, if available.

For a genuinely new challenge library, preserve fixed splice boundaries and
balance only mutable positions within each verified design family's allowed
alphabet. Jointly cover GC, motifs, k-mers, and design parents. Isolate exact
sequences and close siblings across roles, keep barcode-support QC separate
from sequence-composition design, and freeze an untouched one-time evaluation
partition before reading its targets.

Do not force 25% A/C/G/T at every position: the current masks make that
infeasible, and the previously opened final test cannot be reweighted into a
new untouched test.

---

## 3. Existing Interpretation Machinery And A Safe Adaptation Path

The BODA2 paper used Sampled Integrated Gradients (SIG) to obtain observed and
hypothetical nucleotide contribution scores, TF-MoDISco Lite to aggregate
seqlets into patterns, TOMTOM for motif matching, and in-silico block/motif
ablation for validation. The paper's exact method is described in the
[Nature article's Methods](https://www.nature.com/articles/s41586-024-08070-z).

This fork contains useful pieces:

- [`../../../src/analysis/contrib_score.py`](../../../src/analysis/contrib_score.py):
  SIG-style sampled contribution scoring;
- [`../../../src/analysis/sat_mut.py`](../../../src/analysis/sat_mut.py):
  saturation-mutagenesis machinery;
- [`../../../boda/common/pymeme.py`](../../../boda/common/pymeme.py): STREME
  wrapper and parser;
- [`../../../boda/generator/energy.py`](../../../boda/generator/energy.py):
  motif filters and STREME-based penalties; and
- [`../../../boda/common/utils.py`](../../../boda/common/utils.py): k-mer
  generation/filter utilities.

These are not yet a turnkey Lib1 interpretation pipeline. The two analysis
scripts assume the Malinois setting in several places (200-nt inserts,
600-nt model input, three outputs, GPU execution, and artifact layout) and
need basic correctness/smoke-test work before being trusted on the five
single-output Lib1 models. In particular, both currently use `h5py` without
importing it; the contribution CLI leaves `start_idx/stop_idx` undefined in
single-job mode; and the saturation-mutagenesis script always averages forward
and RC predictions, which is not the selected policy for the direction-sensitive
parts.

Recommended order:

1. write one model-agnostic predictor adapter that accepts a checkpoint,
   variable sequence length, padding/flank policy, output index, target inverse
   transform, and optional strand averaging;
2. unit-test attribution shapes and the completeness/sanity behavior on a
   tiny deterministic model;
3. pilot forward and hypothetical SIG on 100--500 high-confidence development
   constructs from one part, plus shuffled controls;
4. validate top contribution blocks by in-silico mutagenesis/ablation;
5. only then run TF-MoDISco Lite and motif annotation at scale; and
6. compare motifs across OOF models/seeds and, for Intron, within each inferred
   group so easy boundary/group features are not mistaken for fine-grained
   regulatory grammar.

Enhancer is the most natural first biological pilot because the transferred
model has the strongest source-method continuity. Intron is the most useful
diagnostic pilot only after the analysis is explicitly stratified.

---

## 4. Simple Predictive Baselines And Motif/Cluster Analyses

The committee suggestion should be split into three products with different
claims.

### A. Predictive k-mer Baseline

On the exact five development folds, fit:

1. intercept/mean and GC-plus-length controls;
2. normalized k-mer counts with ridge regression for `k = 3, 4, 5, 6`;
3. an optional elastic-net sensitivity; and
4. a concatenated multi-k ridge only if it wins nested development selection.

Choose `k` and regularization inside the outer training data. Report one OOF
prediction per development construct, on the raw target scale, with the same
Pearson/RMSE/COD/calibration suite as the neural models. Use directional k-mer
counts for promoter, UTR, and Intron models. For Enhancer, compare directional
and RC-canonicalized counts as a prespecified sensitivity instead of assuming
strand invariance for every part.

Existing feature code can be adapted from
[`../../../src/analysis/sequence_landscape_adapters/hani_utr5_phase2.py`](../../../src/analysis/sequence_landscape_adapters/hani_utr5_phase2.py),
which imports the reusable normalized/RC-canonical feature implementation in
an optional workspace-level `sequence_landscape` package. That package is not
part of this repository, so a self-contained five-part k-mer regression runner
does not currently exist here.

### B. Cluster-Mean Baseline

Fit a k-means model on training-fold k-mer features only, assign held-out rows
to the nearest training centroid, and predict the training-fold mean expression
of that cluster. Use a small nested grid such as `K = 4, 8, 16, 32`. This asks
how much coarse sequence-family membership predicts expression. For Intron,
compare discovered clusters with the deterministic mask groups and report
within-cluster residual performance.

Do not fit clusters on all rows before OOF evaluation; that would leak held-out
sequence geometry into a nominally predictive baseline.

### C. Motif Discovery And Motif-Feature Model

Run STREME or a comparable method on training-fold high-versus-low expression
contrasts, scan the learned motifs into training and held-out sequences, and
fit a regularized linear model on motif scores/counts. Discovery, motif width,
and regularization must remain inside the training fold. This is more expensive
than k-mer ridge and belongs after the k-mer baseline.

Unsupervised UMAP/k-mer plots are valuable exploratory views, but they are not
predictive baselines and should not be compared numerically with OOF Pearson
unless a fold-safe prediction rule is defined.

---

## 5. Context Engineering, Git, And Redundancy Policy

### Repository State At Campaign Creation

- repository: `MinhangXu/boda2_EU`;
- branch: `checkpoint/learn-finetune-docs-may2026`;
- local `HEAD`: `f5176c6479291f4bd617e054d19ef05a158f36f0`;
- connected GitHub branch head: the same commit;
- working tree before this folder: 24 modified tracked paths and 83 untracked
  paths.

The normal SSH remote could not be fetched on this machine because no usable
SSH key was available. The connected GitHub repository view independently
confirmed the branch head. No pull, merge, cleanup, commit, or push was
performed.

### File Policy

A first-pass content audit found no exact duplicate Markdown or notebook files.
Three meeting-specific documents substantially overlap---the July 18
presentation brief and the two July 19 slide/plot plans---but they capture the
evolution of the talk and should be marked historical behind this index rather
than deleted during an already-dirty worktree. The short compatibility files
in `plan/phase1_lib1/` are relocation stubs with inbound links and should also
remain. Large embedded notebook outputs are the clearest later repository-size
target; strip or pair them with Jupytext only through a separately reviewed
cleanup.

1. Keep the completed campaign's
   [`../dedup_phase1_rerun_july2026/README.md`](../dedup_phase1_rerun_july2026/README.md)
   as the source of truth for Stages 1--4.
2. Keep this page as the source of truth for post-TAC priorities and decisions.
3. Store executable code under `src/`, tests under `tests/`, and generated
   outputs under the existing `src/learn/outputs/analysis/` convention.
4. Keep generated analysis figures/tables in their canonical output
   directories and link to them. Accepted source illustrations supplied for the
   TAC report are stored locally under [`figures/`](figures/).
5. Do not delete dated protocol amendments. They preserve what was frozen
   before results were visible.
6. Before removing anything that appears redundant, generate a manifest with
   path, size, SHA-256, git-tracked state, inbound references, and semantic
   role. Exact duplicates can be proposed for deletion; near-duplicate reports
   should usually be consolidated through indexes and status labels.
7. Keep one short `decision + evidence + next action` update here after each
   follow-up, and link to its machine-readable result rather than adding
   another campaign-wide recap.

### Next Safe Git Step

After reviewing the TAC source files selected for version control, make a
**scoped local commit** containing only this campaign's public-safe report,
source figures, and explicitly selected companion files. Keep the identifiable
meeting transcript, generated PDF preview, and local notebooks untracked. Do
not stage the whole dirty tree, and do not push this work to the current public
origin. Creating and authenticating a standalone private remote is deferred
until the repository-visibility workflow has been reviewed.

---

## Report And Figure Map

The first LaTeX report draft is
[`tac_report_july20_2026_draft.tex`](tac_report_july20_2026_draft.tex).
It follows the actual 37-slide meeting order and uses source-asset pointers
rather than embedding copies.

### Local PDF Preview

From this directory, run:

```bash
make preview
```

This uses the user-local Tectonic installation to rebuild
`tac_report_july20_2026_draft.pdf` beside the LaTeX source. In Codex, the PDF
can be opened directly from the task; for a persistent split-pane editor, use
VS Code Remote-SSH with LaTeX Workshop and select Tectonic as the build engine.

High-value figure assets:

| Report role | TAC slide | Canonical asset |
|---|---:|---|
| project scope and composition-to-function search | 3 | [`project_scope_composition_to_function_search.png`](figures/project_scope_composition_to_function_search.png) |
| barcode observations to construct target | 8 | [`lib1_barcode_observations_to_construct_target.png`](figures/lib1_barcode_observations_to_construct_target.png) |
| expression and barcode-support overview | 9--10 | [`lib1_dedup_expression_target_distributions.svg`](../../../src/learn/outputs/analysis/lib1_dedup_data_summary_july2026/reporting/lib1_dedup_expression_target_distributions.svg), [`lib1_dedup_barcode_support_distributions.svg`](../../../src/learn/outputs/analysis/lib1_dedup_data_summary_july2026/reporting/lib1_dedup_barcode_support_distributions.svg) |
| common Stage 1--4 design | 13--14 | [`lib1_single_part_modeling_workflow.png`](figures/lib1_single_part_modeling_workflow.png) |
| selected from-scratch model architectures | future/supplement | [`lib1_single_part_model_architectures.png`](figures/lib1_single_part_model_architectures.png) |
| RC effect | 15 | [`main_rc_augmentation_effect.svg`](../../../src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_rc_augmentation_effect.svg) |
| weighted-loss effect | 16 | [`main_barcode_weighted_loss_effect.svg`](../../../src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_barcode_weighted_loss_effect.svg) |
| broad HPO landscape | 17 | [`supplement_hpo_configuration_landscape.svg`](../../../src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/supplement_hpo_configuration_landscape.svg) |
| one-time final-test behavior | 18 | [`main_locked_final_test_scatter_hexbin.svg`](../../../src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_locked_final_test_scatter_hexbin.svg) |
| Intron signal decomposition | 19 | [`main_intron_composition_triptych.svg`](../../../src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_intron_composition_triptych.svg) |
| sample-efficiency curves | 20 | [`01_primary_pearson_learning_curves.pdf`](../../../src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/01_primary_pearson_learning_curves.pdf) |
| observed 400-to-4,000 gains | 21 | [`02_observed_10x_pearson_forest.pdf`](../../../src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/02_observed_10x_pearson_forest.pdf) |
| Intron pooled versus conditional learning | supplement | [`03_intron_scoped_learning_curves.pdf`](../../../src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/03_intron_scoped_learning_curves.pdf) |
| frozen Evo 2 embeddings with a trainable prediction head | future direction | [`evo2_embedding_prediction_plan.png`](figures/evo2_embedding_prediction_plan.png) |

## Immediate Next Actions

- [ ] Review and edit the LaTeX narrative for voice and committee-specific
      discussion notes.
- [ ] Freeze a short Enhancer augmentation/exposure amendment before writing
      code or launching cells.
- [ ] Add fold-safe residual-target support and tests for the selected Intron
      DataModule/training path.
- [ ] Request verified Intron design-family and splicing metadata from the
      collaborator.
- [ ] Implement a five-part k-mer ridge OOF baseline before motif discovery.
- [ ] Adapt and smoke-test one Lib1 contribution-score path; do not launch a
      full TF-MoDISco campaign yet.
- [ ] Review the dirty-tree manifest and select a narrow commit boundary.
