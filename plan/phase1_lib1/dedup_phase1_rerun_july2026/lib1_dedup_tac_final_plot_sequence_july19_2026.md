# Lib1 deduplicated baseline TAC: final plot sequence

Date: 2026-07-19

## Narrative spine

The cleanest story is:

> define the new Lib1 data and target → explain the protected evaluation design → show how the model route and two training interventions were chosen → ask whether more constructs help → report the one-time final test → qualify what the strong Intron result means.

This is more coherent than narrating notebook Stages 1–4 literally. The stage labels can remain in methods notes, but each main slide should make one scientific or model-development claim.

## Recommended main-deck order

### 1. Catch up from the previous TAC

Headline: `From public/Lib0 component models to collaborator-generated Lib1 single-part libraries`.

- Previous meeting: public data and Lib0 were used to establish the component-wise modeling approach.
- This meeting: Lib1 varies one CRE part at a time in a shared construct context.
- Current baseline predicts one construct-level mean-expression target. Expression-spread/two-head models are a separate later question.

### 2. What is in Lib1?

Use the finalized five-single-part-library illustration and the high-level construct counts.

Accurate length language:

- Enhancer: observed valid lengths 76–211 nt, neutral-padded to 216 for the scratch input; the selected transferred Enhancer route uses a 600-nt assay-framed input.
- Promoter: observed valid lengths 41–51 nt, neutral-padded to 51.
- 5′UTR: modal 50 nt.
- Intron: modal 80 nt.
- 3′UTR: modal 100 nt.

Do not describe Enhancer as a fixed 200-nt library or Promoter as 80 nt.

### 3. From barcode observations to one construct target

Use the finalized barcode-to-construct illustration.

Key language:

- `distinct barcode observation` or `barcode support`, not independent biological replicate;
- group exact-deduplicated rows by construct sequence;
- sum aggregate RNA and DNA counts;
- target: `log₂(total RNA / total DNA)`;
- one row and one target per construct.

The descriptive target is shown raw. Models z-score the target from training rows only and predictions are inverse-transformed before reporting.

### 4. What do the post-dedup measurements look like?

Pair the post-dedup expression violin with the barcode-support composition plot.

Use raw `log₂(total RNA / total DNA)` for the violin. Do not standardize this descriptive plot; the between-library location and range differences are part of the data summary.

Explain the ≥8 cutoff as a frozen quality-versus-quantity compromise:

- higher distinct-barcode count is a measurement-support proxy;
- lower-support constructs are retained for training rather than discarded;
- the threshold was carried into the common evaluation contract before the locked-test results;
- ≥8 still leaves enough 3′UTR rows for 250 locked-test constructs and five development folds of 105 constructs each, whereas ≥10 would leave only 149 development rows after preserving the same test size.

Do not call eight a biological threshold, a noise-free target, or an optimized change point.

### 5. Are the five development folds comparable?

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_development_fold_expression_barcode_balance.svg`

Headline: `Five development folds cover similar expression and barcode-support ranges`.

Speaker points:

- every point is one ≥8-barcode development construct;
- color is the fold in which that construct was held out;
- axes are shared across all five CRE parts;
- the barcode axis is log₂-scaled so the two 2,466-barcode control constructs remain visible;
- fold medians are close, with 9–11 barcodes and only small expression-median shifts.

Important qualification: this is a visual balance check, not evidence that the ≥8 cutoff is optimal. The folds were assigned by stable-ID hash ranking and round-robin, not stratified on expression or barcode count.

### 6. Modeling and evaluation strategy

Use the finalized broad workflow illustration.

Transition:

> We held the target and protected evaluation design fixed while choosing a model route, testing sequence symmetry, testing label-support weighting, and estimating sample efficiency.

Keep the full HPO landscape in the supplement. The workflow slide is the map; the following slides are the evidence.

### 7. Enhancer required a different model route

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_enhancer_transfer_vs_scratch_oof.svg`

Headline: `A pretrained Enhancer route outperformed all tested scratch configurations`.

Panel A is the clean route contrast with RC off on both sides:

- scratch ResNet1D: 10 configurations, median pooled five-fold OOF `r=0.113`, maximum `r=0.185`;
- pretrained BassetBranched: 6 policies, median `r=0.513`, maximum `r=0.535`.

Panel B then isolates the smaller RC effect within the transferred policies; the frozen K562/full/RC-on policy reached `r=0.565`.

Do not add a transfer violin to the Stage-1 HPO figure. Stage-1 HPO uses one screening fold, whereas the transfer evidence uses pooled five-fold OOF; putting them on one violin plot would mix estimands. Six structured transfer policies are also too few for a violin to be especially meaningful. Use `route comparison`, not `pretraining caused the difference`, because architecture, input framing, and initialization also differ.

### 8. RC is a route-specific intervention

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_rc_augmentation_effect.svg`

Headline: `Reverse-complement augmentation helped the transferred Enhancer route, not every CRE part`.

Every point is an exact RC-on versus RC-off comparison using the same configuration, folds, seed, and loss. Keep the compound formal decision rule out of the main visual; it belongs in the supplement.

### 9. Barcode-weighted loss gives small, targeted gains

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_barcode_weighted_loss_effect.svg`

Headline: `Barcode-weighted loss improved four selected configurations; Enhancer remained unweighted`.

- open circles: the five held-out-fold changes;
- diamond: pooled five-fold OOF change;
- selected pooled Δr: Enhancer −0.009, Promoter +0.009, Intron +0.016, 3′UTR +0.010, 5′UTR +0.023.

Say that selection also included fold-consistency and raw-error guardrails. Do not ask the audience to infer a compound `gate_pass` variable from a Pearson-only y-axis.

### 10. More constructs still help, but by different amounts

Use:

`src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/01_primary_pearson_learning_curves.png`

Headline: `The value of additional training constructs differs across CRE parts`.

These are development-only learning curves with frozen policies. They estimate sample efficiency without repeatedly inspecting the locked final test.

The separate Enhancer transfer-versus-scratch learning curve can be a backup to Slide 7 or 10:

`src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/04_enhancer_transfer_vs_scratch.png`

It shows that the operational route gap persists across sample sizes, but it is still not a causal pretraining-only comparison.

### 11. One-time locked final test

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_locked_final_test_scatter_hexbin.svg`

Headline: `Frozen models generalize, but raw-scale calibration differs by CRE part`.

- top row: one dot per construct;
- bottom row: the exact same constructs summarized as counts per hexagon;
- dashed line: identity;
- orange line in every panel: observed expression regressed on prediction;
- axes are identical between the two rows within each CRE part;
- raw inverse-transformed `log₂(total RNA / total DNA)` values; no normalization or post-test recalibration.

Final-test Pearson r is 0.365, 0.444, 0.681, 0.452, and 0.512 for Enhancer, Promoter, Intron, 3′UTR, and 5′UTR. The displayed calibration slope is `observed ~ prediction`; slope 1 and intercept 0 are ideal.

### 12. Why the pooled Intron result needs qualification

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/main_intron_composition_triptych.svg`

Headline: `Much of the pooled Intron score is group separation, but within-group ranking remains`.

This is the intuitive Stage-1 fold-0 diagnostic from the Notion analysis:

- CNN pooled correlation: `r=0.778`;
- three training-fitted sequence-group means alone: `r=0.703`;
- CNN performance after within-group centering: `r=0.472`.

The middle panel is leakage-safe: its three means were fitted on the fold-0 training rows, then applied to the held-out fold. Describe the groups as `sequence-inferred mask groups`, not confirmed synthesis sublibraries or measured splicing classes. Also say clearly that this figure diagnoses the signal structure; it is not the frozen final-policy score.

### 13. Take-home

- Lib1 supports five sequence-to-mean-expression baselines on one shared construct target.
- Architecture and augmentation choices are part-specific rather than universal.
- Barcode support can improve training when used selectively, while evaluation remains unweighted.
- More constructs are still valuable, especially for the weaker routes.
- Locked-test association is reproducible, but calibration and the source of predictive signal differ across CRE parts.
- The strong pooled Intron result contains both group-composition and genuine within-group ranking signal.

## Supplement order

### S1. Model-family architecture

Generate the ResNet1D/BassetVL architecture figure from:

`plan/phase1_lib1/dedup_phase1_rerun_july2026/resnet1d_bassetvl_architecture_illustration_prompt.md`

Keep the optional BassetBranched transfer architecture as a separate third panel or inset so it is not confused with the short-input BassetVL family.

### S2. Broad HPO landscape

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/supplement_hpo_configuration_landscape.svg`

Label it `Broad hyperparameter screening across five Lib1 single-part models`. Do not say `repeating pre-dedup configurations`; the presentational point is that 885 joint settings spanning architecture and optimization choices were tested. The displayed score is one fixed screening-fold Pearson r, not five-fold OOF.

### S3. What varied in HPO

Use a compact table: convolutional widths/kernels/layer count, prediction-head depth/width/dropout/normalization, optimizer, learning rate, weight decay, batch size, and scheduler. State that these were joint settings, not a full factorial design. Avoid interpreting marginal hyperparameter associations causally.

### S4. Enhancer fine-tuning dynamics

Use:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/figures/supplement_enhancer_unfreeze_training_dynamics.svg`

This is the requested transposed layout:

- rows: Pearson r, standardized-target MSE, COD R²;
- columns: branch + output, top convolution block + dense head, full network;
- thin lines: folds;
- thick lines: fold median while at least three folds remain;
- train and development-validation only.

Do not show a test trajectory; that would conflict with the locked-final-test design.

### S5. Training behavior across all five parts

Use the all-part training-dynamics figure generated with the follow-up bundle. Its rows are the same three metrics and its columns are CRE parts. The five-part figure cannot honestly use `unfreeze scope` as its columns: Promoter, Intron, 3′UTR, and the selected 5′UTR routes are scratch models and have no fine-tuning scope.

For barcode-weighted selected policies, label the logged canonical MSE as an **unweighted diagnostic**; the optimized training loss itself is barcode-weighted.

### S6. Formal RC decision audit

If the older gate figure is retained, relabel:

- `gate_pass=True/False` → `predeclared RC evidence rule satisfied / not satisfied`;
- `core` → `scratch CNN`;
- `transfer` → `transferred Enhancer model`;
- `challenger` → the named model family.

### S7. Formal weighted-loss decision audit

Use the one-standard-error selection figure or a compact policy table. Define eligibility and the selected arm in words. Do not call the one-standard-error set a final-model confidence interval.

### S8. Quantitative learning-curve contrasts

- 400→4,000 contrast: `src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/02_observed_10x_pearson_forest.png`
- Intron group-scoped curves: `src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/03_intron_scoped_learning_curves.png`
- Enhancer route curves: `src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/04_enhancer_transfer_vs_scratch.png`

## Final visual and language checks

- In the workflow illustration, the one-barcode weight is approximately 0.32 under the displayed formula. Do not leave a lower icon labeled `wᵢ ≈ 0.1`.
- `barcode` means a distinct technical identity/observation supporting a construct, not an independently collected biological replicate.
- Use `training constructs ≥1 barcode; evaluation constructs ≥8 barcodes`.
- Use raw targets for descriptive distributions and inverse-transformed raw units for reported predictions.
- Use common y-axes within comparison rows. The final-test scatter/hexbin exception follows the Notion design: axes are shared within each CRE-part column so the top and bottom representations are directly comparable.
- Never compare Stage-1 single-fold HPO scores directly with Stage-2 five-fold OOF scores on one visual scale.

## Reproducibility

Follow-up figures are generated by:

`src/analysis/lib1_dedup_tac_followup_figures.py`

Manifest:

`src/learn/outputs/analysis/lib1_dedup_tac_presentation_july2026/followup_figure_manifest.json`

The final-test figure is a reporting-only replot from frozen ensemble predictions. No models were refit, predictions changed, or calibration applied.
