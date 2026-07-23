# Lib1 deduplicated baseline: TAC plot and slide plan

Date: 2026-07-19

## Executive recommendation

The main deck should be organized around decisions and biological interpretation, not around notebook stages or audit terminology. After the data introduction and the high-level workflow illustration, use one slide for each of these claims:

1. Reverse-complement augmentation is useful only for a specific model route.
2. Barcode-weighted loss provides modest but reproducible gains for four selected configurations.
3. With policies fixed, additional training constructs still improve development performance.
4. The frozen models generalize to the one-time locked test, but their calibration differs by CRE part.
5. The strong pooled Intron result needs a sequence-group qualification.

Keep the broad HPO landscape and formal decision-rule details in the supplement. This preserves the important methodological evidence without making the audience decode `gate_pass`, internal route names, or configuration identifiers in the main narrative.

## Main-deck slide order

### Slide 1 — Where the project stood at the previous TAC

Purpose: reconnect to the prior meeting in one minute.

- Lib0 and public data were used to establish sequence-to-expression modeling for individual CRE parts.
- The new work uses collaborator-generated Lib1 measurements.
- Lib1 currently provides an RNA/DNA expression target; the baseline therefore predicts mean construct expression only.

Do not introduce RC, weighting, or downsampling here.

### Slide 2 — What is in Lib1?

Use the five-single-part-library illustration and the high-level barcode/construct counts.

Recommended headline: `Lib1 varies one CRE part at a time in a shared construct context`.

Define the terms once:

- A construct variant is identified by its variable-part sequence.
- Multiple distinct barcode observations may support one construct.
- Barcode observations are technical measurement support, not confirmed independent biological replicates.

### Slide 3 — From barcode observations to one construct-level target

Use the finalized barcode-to-construct illustration, limited to data preprocessing and target generation.

Recommended headline: `Repeated barcode measurements are aggregated into one expression target per construct`.

Say explicitly that the model target is raw construct-level `log₂(total RNA / total DNA)`. Do not describe this as z-scored, standardized, or calibrated.

### Slide 4 — What do the post-dedup data look like?

Use the post-dedup expression-distribution figure and the post-dedup barcode-support figure already produced in this project.

The slide should answer two questions:

- What expression range is available for each variable CRE part?
- How much barcode support does a typical construct have, and how many constructs meet the ≥8-barcode evaluation criterion?

Use raw target values for the violin distributions. Standardization would hide real between-library location and scale differences and is unnecessary for a descriptive data slide.

### Slide 5 — Modeling and evaluation strategy

Use the finalized workflow illustration.

Recommended transition: `We then held the target and evaluation design fixed while testing architecture, sequence symmetry, label-support weighting, and sample efficiency.`

Keep this slide high level. The HPO details belong in the supplement.

### Slide 6 — RC is a route-specific intervention

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_rc_augmentation_effect.svg`

Headline: `Reverse-complement augmentation helped the transferred Enhancer route, not every CRE part`.

Speaker points:

- Every point is an exact RC-on versus RC-off comparison using the same configuration, folds, seed, and loss.
- The common y-axis makes effect sizes directly comparable.
- RC was retained only for the transferred Enhancer policy.
- Say `RC helped the transferred Enhancer route`, not `pretraining caused the RC benefit`: the transfer and scratch routes differ in more than initialization.

Do not use the compound `gate_pass` encoding on this slide.

### Slide 7 — Weighting by barcode support gives small, targeted gains

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_barcode_weighted_loss_effect.svg`

Headline: `Barcode-weighted loss improved four selected configurations; Enhancer remained unweighted`.

Speaker points:

- Each CRE part shows the exact comparison for the configuration that ultimately supplied its frozen policy.
- Open circles are the five held-out-fold changes; the diamond is the pooled five-fold OOF change.
- Selected pooled changes are Enhancer −0.009, Promoter +0.009, Intron +0.016, 3′UTR +0.010, and 5′UTR +0.023.
- The intervention decision also required consistency and raw-error guardrails; Pearson improvement alone was not the entire rule.

The negative 3′UTR fold is worth acknowledging. The pooled gain was positive, but the fold-level evidence was heterogeneous.

### Slide 8 — More constructs still improve development performance

Use the existing Stage-4 presentation figure unchanged:

`src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/01_primary_pearson_learning_curves.png`

Headline: `The value of additional constructs differs substantially across CRE parts`.

Why this figure works:

- All five panels share one y-axis.
- The x-axis is training constructs on a log scale.
- The policies are held fixed, so the slide answers a sample-efficiency question rather than reopening HPO.

Speaker emphasis:

- Enhancer and Intron start higher and show shallower gains over the observed range.
- Promoter, 3′UTR, and 5′UTR show larger gains from more constructs.
- These are development-only learning curves; they do not repeatedly access the final test.

Put the 400→4,000 contrast forest plot in the supplement unless the committee wants a quantitative follow-up.

### Slide 9 — One-time locked-test performance

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_locked_final_test_calibration_shared_axes.svg`

Headline: `Frozen models generalize, but prediction range and calibration differ by CRE part`.

Speaker points:

- These are the raw predictions from the frozen three-seed arithmetic ensemble.
- No standardization, normalization, or post-test recalibration was applied.
- All five panels use identical x and y limits and equal aspect ratios. The identity line therefore has the same meaning everywhere.
- Final-test Pearson r is 0.365, 0.444, 0.681, 0.452, and 0.512 for Enhancer, Promoter, Intron, 3′UTR, and 5′UTR, respectively.
- Enhancer shows the clearest prediction-range compression: its calibration slope is 0.50 despite small mean bias.

Define the displayed slope as observed expression regressed on prediction. The ideal is slope 1 with intercept 0.

### Slide 10 — Why the pooled Intron score needs qualification

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_intron_sequence_group_audit.svg`

Headline: `Intron performance contains both between-group and within-group signal`.

Speaker points:

- The pooled final-test correlation is 0.681.
- After removing frozen group means, the within-group centered correlation is still 0.473, so the result is not only a group-label classifier.
- Performance is heterogeneous: the three final-test group correlations are 0.579, 0.338, and 0.206.
- The inferred groups have substantially different target means, which helps the pooled score.

Use `sequence-defined groups` or `inferred sequence-mask groups`. Do not call them confirmed sublibraries or measured splicing classes; those identities have not been verified.

### Slide 11 — Take-home model

Suggested summary:

- A shared construct-level expression target supports five single-part sequence models.
- Model choices are CRE-part specific: sequence symmetry and barcode weighting should not be assumed universally.
- Locked-test performance is real but calibration and effective signal differ by part.
- Intron is the strongest pooled predictor, with an important group-composition qualification.
- Additional constructs are likely most valuable for Promoter and the UTR models over the sampled range.

## Supplement order

### Supplement 1 — Broad HPO landscape

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/supplement_hpo_configuration_landscape.svg`

The figure shows all 885 settings on one shared performance scale and highlights the ten configurations per part advanced to paired five-fold testing. It intentionally avoids the words `replay` and `source_kind` because neither helps the committee understand the model search.

### Supplement 2 — What varied in HPO

Use a compact methods table rather than a parameter-effect plot:

| Component | Tested variation |
|---|---|
| Convolutional body | Channel widths, kernel sizes, number/type of layers |
| Prediction head | Depth, width, dropout, batch normalization |
| Optimizer | Adam and AdamW |
| Learning rate | Approximately 1.1×10⁻⁵ to 2.8×10⁻³ |
| Weight decay | Approximately 1.0×10⁻⁶ to 7.9×10⁻³ |
| Batch size | 64, 128, 256 |
| Scheduler | None or cosine restarts where supported |

Caption: `Settings were tested as joint configurations, not as a complete factorial grid.`

Do not show the univariate hyperparameter-association table as a causal effect analysis. The search is correlated and nonuniform, and its p-values are unadjusted.

### Supplement 3 — Formal RC decision audit

If the existing RC gate figure is retained, replace internal language:

- `gate_pass=True/False` → `predeclared RC evidence rule satisfied / not satisfied`
- `core` → `scratch CNN`
- `transfer` → `transferred Enhancer model`
- `challenger` → the specific model family, such as `3′UTR UTRBasset`

Footnote: the evidence rule required mean fold Δr ≥ 0.005, positive Δr in at least four of five folds, and RMSE/COD R² guardrails; Intron also required centered-performance consistency.

### Supplement 4 — Formal weighted-loss decision audit

Use the existing one-standard-error selection figure or a compact policy table. Define:

- Blue/eligible: within one standard error of the best admissible arm.
- Star: selected arm.
- Gray: admissible but outside the one-standard-error set.

Do not call this an uncertainty interval for the final model; it is a development-stage selection device.

### Supplement 5 — Stage-4 quantitative contrasts

Use:

`src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/02_observed_10x_pearson_forest.png`

This quantifies the observed 400→4,000 gain. Keep the primary learning curves in the main deck and use this only when someone asks how large the decade-scale gain is.

### Supplement 6 — Intron learning curves by inferred group

Use:

`src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/03_intron_scoped_learning_curves.png`

This is a useful deeper diagnostic after the main Intron qualification slide. Its two panels already share a y-axis.

## Treatment of the attached `gate_pass` plot

The current plot should not be the primary weighted-loss slide. Its y-axis displays only the mean of five Pearson deltas, while its color encodes a compound rule containing fold consistency, RMSE, COD R², and an additional centered Intron check. A viewer cannot infer the color from the plotted y-value alone.

The replacement weighted-loss figure solves that communication problem by showing the selected exact paired effects and stating the additional decision checks in a footnote. Preserve the full gate audit in the supplement for methodological transparency.

## Final illustration check

One numerical inconsistency remains in the current workflow illustration: the Stage-3 header correctly says the weight at one barcode is approximately 0.32, but the lower icon still says `wᵢ ≈ 0.1`. Under the displayed formula, `log(2) / log(9) ≈ 0.315`, so the clipping floor is not active. Change the lower icon to `wᵢ ≈ 0.32` or simply `lower relative weight` before presenting.

In speech, describe barcodes as distinct technical observations/support. If the parenthetical `(replicates)` remains in the illustration, clarify that it does not mean independently collected biological replicates.

## Reproducibility

The new presentation figures are generated by:

`src/analysis/lib1_dedup_tac_presentation_figures.py`

Outputs and source tables are indexed in:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figure_manifest.json`

The locked-test calibration plot is a reporting-only replot from frozen ensemble predictions. It does not refit a model, alter predictions, or recalibrate against the test set.
