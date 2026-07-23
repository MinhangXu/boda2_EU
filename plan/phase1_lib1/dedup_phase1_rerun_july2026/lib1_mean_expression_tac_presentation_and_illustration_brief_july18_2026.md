# Lib1 Mean-Expression TAC Presentation And Illustration Brief

**Prepared:** 2026-07-18
**Scope:** the completed five-part Lib1 construct-mean expression baseline,
Stages 1-4, the bounded 3'UTR branch, and the one-time locked final test
**Audience:** semi-annual thesis advisory committee
**Primary recommendation:** organize the talk around scientific questions and
decisions, not around run chronology. Use one data-lineage figure and one
modeling-roadmap figure, then let the analysis plots carry the detailed results.

This is a presentation and illustration handoff, not a replacement for the
dated protocol amendments or frozen machine-readable manifests.

## Accepted Illustration Assets

The finalized raster illustrations supplied for the July TAC report are stored
under the adjacent campaign folder:

| Illustration | Accepted asset | Current use |
|---|---|---|
| Project scope: individual CREs to composition-to-function search | [`project_scope_composition_to_function_search.png`](../tac_campaign_july2026/figures/project_scope_composition_to_function_search.png) | Included in the report; schematic lengths should not be read as the modeled input contracts. |
| Barcode observations to one construct target | [`lib1_barcode_observations_to_construct_target.png`](../tac_campaign_july2026/figures/lib1_barcode_observations_to_construct_target.png) | Included in the report. |
| Shared Stage 1--4 modeling workflow | [`lib1_single_part_modeling_workflow.png`](../tac_campaign_july2026/figures/lib1_single_part_modeling_workflow.png) | Included in the report; the one-barcode weight is 0.315, so the small `w_i ~ 0.1` label remains a known illustration correction. |
| ResNet1D and short-input BassetVL architectures | [`lib1_single_part_model_architectures.png`](../tac_campaign_july2026/figures/lib1_single_part_model_architectures.png) | Saved for a future methods/supplement figure; it does not show the transferred Enhancer route. |
| Evo 2 embeddings with a trainable prediction head | [`evo2_embedding_prediction_plan.png`](../tac_campaign_july2026/figures/evo2_embedding_prediction_plan.png) | Included as a conceptual future direction; exact input length, pooling, tokenizer details, and output transforms remain to be finalized. |

## The Story In One Sentence

We asked whether the sequence of each of five individually varied Lib1
regulatory parts predicts a single construct-level RNA-expression readout,
then tested—in controlled paired experiments—which modeling choices improve
robustness and how performance changes with more training constructs.

## Recommended Framing From The Previous TAC

Use a short bridge rather than a long recap:

> At the last TAC, the work centered on Library 0 and public datasets for the
> individual CRE parts. Those studies established part-specific sequence-model
> starting points. The new question is whether those ideas support a rigorous
> baseline on Library 1, a collaborator-generated library with one measured
> expression phenotype per construct.

Then state the scope of this talk:

> Today I am focusing on the simplest common baseline: one sequence input and
> one construct-level expression target. Modeling barcode spread or predictive
> uncertainty is a separate follow-up and is not mixed into this baseline.

The assay phenotype is RNA expression/activity, operationalized as an aggregate
RNA/DNA ratio. DNA counts are the abundance denominator, not a second modeled
phenotype.

## Language To Freeze Before Making Figures

| Term | Presentation-safe definition |
|---|---|
| **Barcode observation** | One exact-deduplicated construct-barcode row with paired DNA and RNA counts. |
| **Barcode** | A molecular tag associated with a construct. It provides repeated measurement support, but is not an independently indexed biological or technical replicate in the current combined file. |
| **Construct variant** | One unique full construct identity, keyed by `parts_concatenated`. Multiple barcode observations can map to it. |
| **Single-part library** | A Lib1 subset in which the model uses the sequence of one varied part: Enhancer, Promoter, 5'UTR, Intron, or 3'UTR. |
| **Model input** | The part-specific variable sequence, not the barcode sequence and not the entire `parts_concatenated` string. |
| **Expression target** | `log2(total RNA barcode counts / total DNA barcode counts)` for one construct after exact-row deduplication. |
| **Development fold** | One of five disjoint high-support construct sets used to create out-of-fold predictions. Folds are not biological replicates. |
| **Locked final test** | A high-support construct set frozen before model development and evaluated once after policy and checkpoint freezing. Historical internal artifacts call this the `audit` set. |

Do **not** write `barcode = replicate` on a slide. A precise short label is:

```text
barcode = repeated construct-associated observation
```

The current table has no recoverable replicate index. If the word “replicate”
is useful conversationally, say “barcode-level repeated measurements” and add
that they are not confirmed independent biological replicates.

## The Exact Mean-Target Contract

For construct `i` and its exact-deduplicated barcode observations `j`:

```math
D_i = \sum_j d_{ij}, \qquad R_i = \sum_j r_{ij}, \qquad
y_i = \log_2\left(\frac{R_i}{D_i}\right).
```

The target is a **log ratio of summed counts**. It is not an arithmetic mean of
per-barcode log ratios.

For this baseline:

- exact duplicate records are removed once upstream;
- barcode rows are grouped by `parts_concatenated`;
- `n_barcodes` is the number of distinct nonblank barcode identities;
- all matching exact-deduplicated barcode rows contribute to the aggregate;
- an individual barcode row is not removed merely because its DNA count is
  zero;
- no pseudocount is added; and
- every modeled construct has positive aggregate DNA and RNA totals.

The DNA-row filter and `+0.5` pseudocount shown in the earlier two-head
mean/spread illustration belong to a separate exploratory target. They must not
appear in the baseline mean-target figure.

During model fitting, `y_i` is z-scored using the current training rows only.
Predictions are inverse-transformed before reporting, so plotted predictions,
RMSE, COD R2, slopes, and biases are on the raw `log2(RNA/DNA)` scale.

## Lib1 Single-Part Modeling Sets

| Part modeled | Model sequence input | Length policy | Construct rows | Constructs with `n_barcodes >= 8` |
|---|---|---|---:|---:|
| Enhancer | `Enhancer` | valid 76-211 nt, neutral-pad to 216 | 4,787 | 1,229 |
| Promoter | `Promoter` | valid 41-51 nt, neutral-pad to 51 | 7,893 | 1,931 |
| 5'UTR | `FivePrime` | exact modal 50 nt | 8,331 | 1,797 |
| Intron | `Intron` | exact modal 80 nt | 7,848 | 1,326 |
| 3'UTR | `ThreePrime` | exact modal 100 nt | 6,845 | 775 |

The figure should show the conceptual construct order
`Enhancer -> Promoter -> 5'UTR -> Intron -> 3'UTR`, with one part highlighted
at a time. Do not imply that the neural model consumes all five sequences at
once in this baseline.

---

# Figure 1: Barcode Observations To One Construct Target

**Accepted asset:**
[`lib1_barcode_observations_to_construct_target.png`](../tac_campaign_july2026/figures/lib1_barcode_observations_to_construct_target.png)

## Recommended Visual Design

Use a crop-friendly 16:9 landscape graphic with four panels:

1. **Five Lib1 single-part libraries.** Show a five-slot construct strip and
   five miniature copies in which one slot is colored and the other slots are
   muted. Label the colored slots Enhancer, Promoter, 5'UTR, Intron, and 3'UTR.
2. **Barcode observations.** Show several rows for the same construct sequence,
   each with a distinct barcode ID and paired DNA/RNA counts. Bracket the rows
   with “many barcode observations support one construct.”
3. **Group and aggregate.** Show a small QC chip for exact-record deduplication,
   then group by `parts_concatenated`, count distinct barcodes, and sum DNA and
   RNA. Do not show per-barcode filtering, spread, variance, or pseudocounts.
4. **Construct-level target table.** Show one row per construct with the
   variable-part sequence, `n_barcodes`, `total_DNA`, `total_RNA`, and the one
   target `log2_RNA_DNA`. End at the target table—no neural-network icon.

Use this real, compact Promoter example if a numeric example is desired:

| Construct | Promoter sequence | Barcode | DNA | RNA |
|---|---|---|---:|---:|
| A | `CTGTGCGT...ATTGTG` | bc01 | 9 | 37 |
| A | `CTGTGCGT...ATTGTG` | bc02 | 20 | 50 |
| A | `CTGTGCGT...ATTGTG` | bc03 | 12 | 70 |

The correct aggregation is:

```text
n_barcodes = 3
total_DNA = 9 + 20 + 12 = 41
total_RNA = 37 + 50 + 70 = 157
log2_RNA_DNA = log2(157 / 41) = 1.937
```

The numeric example is for explaining the operation, not for representing the
full distribution.

## Self-Contained Web-ChatGPT Prompt For Figure 1

Copy the following block into web ChatGPT. Asking for editable SVG is strongly
preferred because equations and table text are more reliable as real text than
inside a generated raster image.

```text
Create an editable 1920 x 1080 SVG scientific infographic titled:
“From Lib1 barcode observations to one expression target per construct”.

Use a clean white background, restrained accessible colors, dark navy text,
rounded rectangular panels, thin arrows, and large presentation-readable type.
All labels must be actual SVG text elements, not rasterized text. Return the SVG
code and a rendered preview. Keep the composition modular so the left quarter
or right three quarters can be cropped into separate slides.

The figure must contain exactly four numbered panels and must stop before any
modeling.

Panel 1 — “Five Lib1 single-part libraries”
- Draw a horizontal five-slot construct schematic in this order:
  Enhancer | Promoter | 5'UTR | Intron | 3'UTR.
- Below it, draw five small copies. In each copy, highlight exactly one part in
  color and mute the other four, showing that one part sequence is modeled at a
  time.
- Add the short caption: “one variable part sequence per model”.

Panel 2 — “Many barcode observations per construct”
- Draw a small table with columns:
  construct, variable-part sequence, barcode observation, DNA count, RNA count.
- Use these three rows:
  A | CTGTGCGT...ATTGTG | bc01 | 9  | 37
  A | CTGTGCGT...ATTGTG | bc02 | 20 | 50
  A | CTGTGCGT...ATTGTG | bc03 | 12 | 70
- Use a bracket to group the three rows and label it:
  “distinct barcodes supporting the same construct”.
- Add a small footnote: “barcode observation ≠ confirmed biological replicate”.

Panel 3 — “QC, group, and aggregate”
- Show a small shield/check icon with the label:
  “remove exact duplicate records once”.
- Then show: “group by parts_concatenated”.
- Show the exact arithmetic:
  n_barcodes = 3
  total_DNA = 9 + 20 + 12 = 41
  total_RNA = 37 + 50 + 70 = 157
- Do not show a DNA-row filter. Do not add any pseudocount. Do not calculate
  per-barcode log ratios, variance, spread, uncertainty, or effective barcode
  count.

Panel 4 — “One row and one target per construct”
- Draw a compact target table with columns:
  construct_id | variable-part sequence | n_barcodes | total_DNA | total_RNA |
  log2_RNA_DNA.
- Show one row:
  A | CTGTGCGT...ATTGTG | 3 | 41 | 157 | 1.937
- Above the last column show the exact formula:
  y_i = log2(total_RNA_i / total_DNA_i)
- Highlight only log2_RNA_DNA as the modeling target.

Add a bottom goal banner:
“Goal: convert repeated barcode observations into one aggregate RNA/DNA
expression target for each construct sequence.”

Hard exclusions:
- no neural network or model;
- no two-headed output;
- no expression-spread or variance target;
- no +0.5 pseudocount;
- no DNA <= 0 barcode-row removal;
- do not call barcode observations biological or technical replicates;
- no deduplication run counts or implementation filenames.

Use exact spelling and punctuation. If any text would be too small, simplify
decorative elements rather than shrinking the text.
```

If raster image generation is used instead, ask it to omit the equation and
table text, then overlay those items in PowerPoint. The attached example is too
text-dense for reliable raster generation.

---

# Why The Held-Out Evaluation Pool Uses `n_barcodes >= 8`

## The Defensible Reason

The threshold is a predeclared **quality-versus-sample-size compromise**, not a
biological constant and not a claim that eight barcodes guarantee an accurate
label.

1. After exact deduplication, the number of distinct barcode identities is a
   practical proxy for measurement support behind a construct-level target.
2. Earlier Enhancer quality-bin experiments showed that higher-support held-out
   sets produced stronger and cleaner evaluation signal. Low-barcode constructs
   could still be useful for training, so they were not discarded.
3. `>=8` was carried into the common five-part evaluation contract before the
   present selections. It was not tuned on the locked final-test outcomes.
4. It leaves enough constructs for both five disjoint development folds and a
   meaningful locked final test, including the limiting 3'UTR library.

The quantity tradeoff is visible directly in the current tables:

| Part | All modeled | `>=8` | `>=10` | `>=12` |
|---|---:|---:|---:|---:|
| Enhancer | 4,787 | 1,229 | 809 | 539 |
| Promoter | 7,893 | 1,931 | 1,226 | 756 |
| Intron | 7,848 | 1,326 | 717 | 392 |
| 3'UTR | 6,845 | 775 | 399 | 204 |
| 5'UTR | 8,331 | 1,797 | 1,070 | 648 |

At `>=8`, 3'UTR can reserve 250 constructs for the locked final test and retain
525 development constructs, or 105 per fold. At `>=10`, only 399 high-support
3'UTR constructs exist in total; retaining a 250-row final test would leave
only 149 development rows, roughly 30 per fold. A stricter common threshold
would therefore make model selection much noisier for the limiting part.

## Suggested Slide And Spoken Wording

Slide:

```text
Training: all eligible constructs (>=1 barcode)
Evaluation: higher-support constructs (>=8 barcodes)
Reason: cleaner target estimates while preserving enough rows for five-fold OOF
```

Spoken:

> Eight is not a magic reliability cutoff. We froze it as a pragmatic common
> threshold: earlier quality-resolved experiments supported higher-barcode
> held-out evaluation, and eight still leaves enough constructs—especially for
> 3'UTR—to support five out-of-fold evaluations plus a locked final test. We
> keep the lower-support constructs in training because they still add sequence
> diversity and can carry useful signal.

Do not say that the validation target is noise-free, that barcodes are
independent replicates, or that the cutoff was estimated as an optimal change
point.

One additional technical caveat belongs in the speaker notes: `n_barcodes`
counts distinct nonblank barcode identities whether or not an individual
barcode row has a nonzero DNA or RNA count. The aggregate construct totals are
positive, but `n_barcodes >= 8` does **not** mean “eight independent nonzero
measurements.” In a read-only join back to the barcode table, roughly 38-56%
of HQ8 constructs (depending on the CRE part) had fewer than eight
DNA-positive barcode rows. This does not invalidate the prespecified support
threshold; it limits what should be claimed about it.

---

# Figure 2: Broad Stage 1-4 Modeling Workflow

**Accepted asset:**
[`lib1_single_part_modeling_workflow.png`](../tac_campaign_july2026/figures/lib1_single_part_modeling_workflow.png)

## Recommended Visual Hierarchy

The main workflow should answer four questions:

1. **Which already-observed model configurations are promising on Lib1?**
2. **Does reverse-complement augmentation help the same configuration?**
3. **Does downweighting low-barcode training targets help the same arm?**
4. **How much does performance improve as the number of training constructs
   increases?**

Use four large numbered boxes. Put the common target and validation design in a
thin band above them. Put the one-time locked final test in a separate locked
branch after Stage 3, because it is essential to rigor but is not a fifth model
development stage. The bounded 3'UTR search should be a small branch from Stage
2 into Stage 3; omitting it would make the final 3'UTR portfolio appear
unmotivated.

## Exact Completed Stage Specification

### Shared contract

```text
one target: construct-level log2(total RNA / total DNA)
training eligibility: n_barcodes >= 1
development and final-test eligibility: n_barcodes >= 8
five disjoint development folds
one locked final-test partition per part
primary development metric: pooled five-fold raw-scale OOF Pearson r
guardrails: fold stability, RMSE, COD R2, raw calibration
```

### Stage 1 — Broad configuration screen

Presentation label:

```text
Broad HPO/configuration replay
```

Precise meaning: this was not a new adaptive Bayesian sweep. It replayed 885
exact configurations that had already completed in the earlier HPO/outer-run
families, now under one target and fold-0 contract.

```text
885 exact configurations
development fold 0
model seed 1701
RC off
unweighted MSE
32-bit precision
locked final test unavailable
promote 10 configurations per part
```

The exact configuration inventory was:

| Part / route | Exact configurations |
|---|---:|
| Enhancer ResNet1D | 128 |
| Promoter PromoterBassetVL | 158 |
| Intron ResNet1D | 156 |
| 3'UTR ResNet1D | 157 |
| 5'UTR ResNet1D | 158 |
| 5'UTR UTRBassetVL | 128 |
| **Total** | **885** |

For the TAC main slide, the total and “exact prior configurations” are enough;
put the per-route counts in backup.

### Stage 2 — Five-fold paired RC test

```text
same base configuration, split, seed, loss, and training policy
paired change: RC augmentation off versus on
validation is always original orientation
five held-out-fold cells per arm
primary score: pooled OOF Pearson across all five held-out folds
```

Completed accounting:

```text
500 core cells = 5 parts x 10 configs x 5 folds x 2 RC modes
 60 bounded Enhancer-transfer challenger cells
100 bounded 3'UTR UTRBassetVL challenger cells
660 total cells = 50 Stage-1 cells reused + 610 new launches
```

The key result was that RC was not a universal invariance. It was supported for
the Enhancer transfer route, was configuration-specific for Intron, and was not
supported as the default for Promoter, 3'UTR, or 5'UTR.

### Bounded 3'UTR branch between Stages 2 and 3

Stage 2 showed UTRBassetVL headroom but large configuration/fold variability,
so a bounded, fully crossed search was added:

```text
UTRBassetVL architecture fixed
learning rate: 0.001, 0.002, 0.004, 0.006
weight decay: 0.0001, 0.0007, 0.003
linear dropout: 0.35, 0.50
24 configurations x 5 folds x 2 RC modes = 240 cells
```

It identified a useful neighborhood around learning rate 0.002 and reinforced
RC off for 3'UTR. The frozen Stage 3 3'UTR portfolio contained seven
UTRBassetVL configurations and three ResNet1D anchors.

### Stage 3 — Paired barcode-weighted loss

Stage 3 was a paired intervention study, not another open HPO:

```math
w_i = \operatorname{clip}\left(
\frac{\log(1+n_i)}{\log(1+8)},\ 0.1,\ 1.0
\right)
```

```math
L = \frac{\sum_i w_i(\hat y_i-y_i)^2}{\sum_i w_i}.
```

Training loss was weighted; development and final-test metrics stayed
unweighted. Every weighted arm was compared with its exact unweighted mate.

```text
10 configurations per part
450 new weighted cells + 450 immutable unweighted mates
900 analysis cells; 180 complete five-fold OOF arms
RC off/on for Enhancer, Promoter, Intron, and 5'UTR
RC off only for 3'UTR
```

An intervention had to achieve a mean five-fold Pearson gain of at least 0.005,
be positive in at least four of five folds, and stay within frozen RMSE/COD R2
guardrails. Intron also had to avoid degradation after within-inferred-stratum
centering. Only admissible arms entered the one-standard-error selection.

Selected development policies:

| Part | Selected route | RC | Loss | Development OOF r | Locked final-test r |
|---|---|---|---|---:|---:|
| Enhancer | BassetBranched, K562 full transfer | on | unweighted | 0.565 | 0.365 |
| Promoter | PromoterBassetVL scratch | off | barcode weighted | 0.478 | 0.444 |
| Intron | ResNet1D scratch | off | barcode weighted | 0.690 | 0.681 |
| 3'UTR | UTRBassetVL scratch | off | barcode weighted | 0.493 | 0.452 |
| 5'UTR | UTRBassetVL scratch | off | barcode weighted | 0.542 | 0.512 |

The main take-home is more memorable than the run count: weighting was selected
for four parts, whereas RC was selected only for the Enhancer transfer policy.

### Locked final-test branch after Stage 3

```text
one development-selected policy per part
fixed epoch budget from the five development folds
refit on all eligible non-final-test rows
three prespecified seeds: 1701, 1702, 1703
freeze a 15-checkpoint allowlist
score the locked final test once
primary prediction = arithmetic mean of the three raw seed predictions
no return to model selection
```

The locked final test is internal to the same Lib1 library distribution. It is
an honest held-out generalization test, not an external-library validation.

### Stage 4 — Development-only downsampling

Stage 4 held the selected policies and evaluation structure fixed and changed
the number of training constructs.

```text
primary N = 40, 250, 400, 2,500, 4,000, full
three frozen nested subset tracks at each finite N
five outer OOF folds
distinct inner checkpoint fold: inner = (outer + 1) mod 5
five primary configs + nine sparse alternatives + one Enhancer scratch diagnostic
660 development-only cells
locked final-test products never read
```

The clean common comparison was the observed `400 -> 4,000` increase:

| Part | Full-N development OOF r | Observed `400 -> 4,000` delta r, 95% CI |
|---|---:|---:|
| Enhancer | 0.516 | +0.065 [0.034, 0.096] |
| Promoter | 0.455 | +0.194 [0.148, 0.242] |
| Intron, pooled | 0.656 | +0.078 [0.043, 0.117] |
| Intron, within-stratum centered | 0.394 | +0.231 [0.153, 0.313] |
| 3'UTR | 0.309 | +0.242 [0.070, 0.389] |
| 5'UTR | 0.513 | +0.254 [0.196, 0.308] |

The acquisition conclusion is:

- highest generic-volume priority: 5'UTR and 3'UTR, without claiming a precise
  ordering between them;
- next generic-volume priority: Promoter;
- Intron: targeted, position-balanced and within-stratum acquisition rather
  than optimizing pooled Pearson alone; and
- lowest generic-volume priority under the tested route: Enhancer.

Do not put a projected “10x beyond full” number in the talk. The fitted curve
families were unstable and often hit their allowed boundaries.

## Self-Contained Web-ChatGPT Prompt For Figure 2

```text
Create an editable 1920 x 1080 SVG scientific workflow titled:
“Building robust Lib1 single-part sequence-to-expression baselines”.

Use a clean white background, large presentation-readable text, accessible
colors, rounded boxes, and clear arrows. All labels must be actual SVG text.
The figure should communicate the scientific logic, not resemble a software
pipeline. Use four large numbered stage boxes from left to right, one narrow
shared-design band across the top, one small 3'UTR branch, and one locked final-
test branch below Stage 3. Do not include deduplication implementation details.

Top shared-design band — blue
Label: “Shared target and evaluation design”.
Include four compact chips:
1. one target: construct log2(total RNA / total DNA)
2. training: constructs with >=1 distinct barcode identity
3. evaluation: constructs with >=8 distinct barcode identities
4. five disjoint development folds + one locked final test
Add: “primary development score = pooled five-fold OOF Pearson r”.

Stage 1 — blue-green
Title: “1. Broad HPO / exact configuration screen”.
Body:
- 885 previously observed exact configurations
- common fold-0, seed-1701 baseline
- RC off; unweighted MSE
- promote 10 configurations per CRE part
Use a small grid-of-configurations icon. Do not call this a new adaptive
Bayesian sweep.

Stage 2 — green
Title: “2. Paired RC test with five-fold OOF”.
Show a five-segment fold ring and a paired DNA-sequence icon labeled RC off / RC
on. Body:
- same configuration, rows, seed, and loss
- change only reverse-complement augmentation
- validation remains original orientation
- 660 total development cells
Result chip: “RC benefit was route/part specific”.

Small purple branch above the arrow from Stage 2 to Stage 3
Title: “Bounded 3'UTR HPO”.
Body:
- 24 UTRBassetVL configs
- 5 folds x RC off/on = 240 cells
- fixed grid over LR, weight decay, dropout
- rejoins Stage 3 portfolio
Do not draw it as a fifth main stage.

Stage 3 — purple
Title: “3. Paired barcode-weighted loss”.
Show two matched scales labeled unweighted and weighted. Include the compact
formula:
w_i = clip[log(1+n_i) / log(9), 0.1, 1]
Body:
- exact weighted vs unweighted mates
- validation metrics remain unweighted
- 10 configurations per part; 900 analysis cells
Result chips:
- weighted loss selected for Promoter, Intron, 3'UTR, 5'UTR
- RC selected only for Enhancer transfer

Locked branch below Stage 3 — white box with a red outline and lock icon
Title: “One-time locked final test”.
Body:
- freeze one policy per part
- 3 all-development refit seeds
- score once; no feedback to selection
Draw arrows into this box but no arrow returning to any earlier stage.

Stage 4 — amber
Title: “4. Development-only downsampling”.
Show a small rising learning curve. Body:
- N = 40, 250, 400, 2,500, 4,000, full
- 3 nested subset tracks; 5 outer folds
- separate inner checkpoint fold
- 660 cells; final test never read
Result chip:
“largest generic data gains: both UTRs, then Promoter”.

At the bottom, add a one-line scientific progression:
“Choose a model family -> test sequence symmetry -> test label-support weighting
-> estimate the value of more constructs”.

Legend:
- solid arrow = candidate or frozen-policy flow
- dashed purple arrow = bounded 3'UTR follow-up
- red lock = one-time final-test boundary
- amber = sample-efficiency study, not model reselection

Hard exclusions:
- no two-headed model;
- no barcode-spread or variance target;
- no pseudocount;
- no arrows from the final test back to HPO, policy selection, or Stage 4;
- no claim that five folds are biological replicates;
- no claim that >=8 barcodes is a noise-free or optimal cutoff;
- no beyond-full learning-curve projection;
- do not fill the figure with full configuration hashes or file paths.

Use exact spelling. If the content is crowded, prioritize the stage question,
paired factor, and output; remove decorative icons before shrinking text.
```

---

# Recommended TAC Slide Flow

The proposed content is strong, but a purely chronological “Stage 1, then 2,
then 3, then 4” talk risks sounding like an engineering status report. Use the
stages as the methods spine while making each slide headline a scientific
answer.

## Main deck

1. **From public/Lib0 part models to a collaborator-built Lib1 test bed**
   - One-slide recap of the previous TAC.
   - End with the new thesis question, not a list of old runs.

2. **Lib1 gives five single-part sequence problems and one expression readout**
   - Show the five-part construct strip.
   - Explain which sequence varies and which sequence each model consumes.
   - State that RNA/DNA is the only phenotype modeled here.

3. **Barcode observations become one aggregate expression target per construct**
   - Use Figure 1.
   - Explicitly distinguish barcode observation, construct variant, and model
     input.
   - One spoken sentence on exact-record QC is sufficient.

4. **The evaluation design separates target support, model selection, and final testing**
   - Show `train >=1`, `evaluate >=8`, five disjoint OOF folds, and the locked
     final test.
   - Explain the `>=8` quality/quantity compromise.

5. **Four controlled questions build the baseline**
   - Use Figure 2 as the roadmap.
   - Tell the audience that detailed plots will follow in the same order.

6. **Architecture and initialization route dominate the first comparison**
   - Use the Stage 2 route-landscape plot.
   - Main points: Enhancer transfer clearly beats the tested scratch route;
     3'UTR has a promising but variable UTRBassetVL neighborhood.

7. **Reverse-complement augmentation is not a universal biological invariance**
   - Use the paired-RC plot.
   - State the route-specific Enhancer result and RC-off defaults elsewhere.

8. **A bounded 3'UTR search resolves a local optimizer neighborhood**
   - Use the targeted-search surface and/or RC-delta plot.
   - Keep this short unless 3'UTR is a major thesis focus.

9. **Barcode-weighted training gives modest, repeatable gains for four parts**
   - Show the weight curve or formula, paired fold deltas, and the selected
     policy table.
   - Emphasize that evaluation stayed unweighted.

10. **Frozen policies retain within-Lib1 signal, with Enhancer as the main limitation**
    - Use the development-versus-final-test plot with Pearson, RMSE, and COD R2.
    - Avoid a correlation-only success claim; discuss calibration.

11. **Intron pooled performance contains substantial between-design-group signal**
    - Use the natural versus within-inferred-mask plot.
    - State that the groups are sequence-inferred masks, not verified
      sublibraries or measured splicing classes.

12. **More constructs still help most for the UTRs and Promoter**
    - Use the primary learning curves and the observed `400 -> 4,000` forest
      plot.
    - Pair the pooled Intron curve with the centered Intron curve.

13. **What this baseline establishes—and what it does not**
    - Establishes: sequence signal exists for all five parts; optimal RC/loss
      policy is part specific; data value is part specific.
    - Does not establish: barcode-level uncertainty, causal value of pretraining,
      external-library generalization, or uniform Intron performance.

14. **Next experimental decision**
    - More generic UTR/Promoter variants.
    - Targeted position-balanced and within-design Intron variants.
    - A matched Enhancer experiment only if the causal value of pretraining is
      itself important.

## Backup slides

- exact Stage 1 route/configuration counts;
- fold sizes and split construction;
- full RC and weighted-loss gates;
- selected full `base_config_id` values and fixed epoch budgets;
- target normalization and raw-unit inverse transform;
- final-test calibration scatter/hexbin plots;
- 3'UTR one-SE stability tradeoff;
- Intron mask definitions and positional composition;
- Stage 4 portfolio sensitivity and curve-family failure;
- exact-record deduplication counts and zero-count policy; and
- why the separate two-head spread model is not part of this baseline.

---

# Critical Interpretation Guardrails

1. **Stage 1 terminology.** “Broad HPO” is audience-friendly, but “broad exact
   configuration screen” is scientifically precise. The 885 combinations were
   replayed from already completed HPO/outer-run configurations; they were not
   proposed by a new adaptive sweep.

2. **Barcode terminology.** Barcode observations are not confirmed biological
   or technical replicates. Barcode count is a support proxy, not a direct
   measurement of label variance.

3. **Target terminology.** The baseline target is a log ratio of aggregate
   counts, not a mean of barcode log ratios. Use “construct-level aggregate
   expression” at least once before shortening it to “mean expression.”

4. **Threshold claim.** `>=8` is a frozen compromise supported by earlier
   quality-resolved evidence and current sample-size needs. It is not an
   optimized universal cutoff.

5. **Fold claim.** The five folds quantify split/training stability and create
   one OOF prediction per development construct. They are not five independent
   experimental replicates.

6. **Metric claim.** Pearson measures association, not calibration. Always pair
   it with RMSE, COD R2, raw slope/bias, or a calibration plot when making a
   performance claim.

7. **Enhancer claim.** The tested transfer route decisively outperformed the
   tested scratch route operationally, but the comparison also changes
   architecture, input framing, and initialization. It is not a causal
   pretraining-only experiment. The selected checkpoint was itself trained in
   an RC-compatible context, so the RC result is route specific.

8. **3'UTR claim.** The selected Stage 3 policy is not the largest pooled
   Pearson point estimate. It lies inside the predeclared one-SE band and was
   chosen for better worst-fold stability. Present this as an intentional
   robustness tradeoff.

9. **Intron claim.** Natural pooled Pearson is composition assisted. The
   within-inferred-stratum and per-stratum results must accompany it. The
   current final test is a natural-mixture internal test, not the proposed
   position-balanced external challenge set.

10. **Stage 4 claim.** The observed `400 -> 4,000` gains support acquisition
    priorities. Stage 4 full-N values are not direct replications of Stage 3:
    nested validation changes the training pool and checkpoint-selection
    design. The unstable fitted curves do not support numerical 10x- or
    100x-beyond-full forecasts.

11. **Final-test claim.** The locked final test supports within-Lib1
    generalization for frozen policies. It does not establish transfer to a new
    library, batch, backbone, or assay.

12. **Scope claim.** This presentation establishes the mean-expression
    baseline. It does not need to defend or resolve the two-headed spread model;
    that is a later uncertainty/observation-modeling question.

---

# Existing Plots That Best Support The Main Deck

- post-dedup expression-target distributions, all modeled constructs and HQ8:
  [`lib1_dedup_expression_target_distributions.svg`](../../../src/learn/outputs/analysis/lib1_dedup_data_summary_july2026/reporting/lib1_dedup_expression_target_distributions.svg)
- post-dedup barcode-support distribution and HQ8 composition:
  [`lib1_dedup_barcode_support_distributions.svg`](../../../src/learn/outputs/analysis/lib1_dedup_data_summary_july2026/reporting/lib1_dedup_barcode_support_distributions.svg)
- Stage 2 route comparison:
  [`stage2_oof_route_landscape.png`](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/figures/stage2_oof_route_landscape.png)
- paired RC effects:
  [`stage2_paired_rc_effects.png`](../../../src/learn/outputs/analysis/lib1_dedup_stage2_july2026/reporting/figures/stage2_paired_rc_effects.png)
- 3'UTR targeted search surface:
  [`utr3_targeted_search_surface.png`](../../../src/learn/outputs/analysis/lib1_dedup_utr3_targeted_hpo_july2026/reporting/figures/utr3_targeted_search_surface.png)
- 3'UTR targeted RC effect:
  [`utr3_targeted_rc_delta.png`](../../../src/learn/outputs/analysis/lib1_dedup_utr3_targeted_hpo_july2026/reporting/figures/utr3_targeted_rc_delta.png)
- Stage 3 admissible one-SE selection:
  [`stage3_admissible_one_se_selection.png`](../../../src/learn/outputs/analysis/lib1_dedup_stage3_weighted_loss_july2026/reporting/figures/stage3_admissible_one_se_selection.png)
- development versus locked final test:
  [`stage3_selected_policies_development_vs_audit.png`](../../../src/learn/outputs/audit/lib1_dedup_final_audit_july2026/reporting/final_audit_figures/stage3_selected_policies_development_vs_audit.png)
- final-test raw-scale calibration:
  [`stage3_audit_raw_calibration.png`](../../../src/learn/outputs/audit/lib1_dedup_final_audit_july2026/reporting/final_audit_figures/stage3_audit_raw_calibration.png)
- final-test Intron robustness:
  [`stage3_intron_inferred_mask_audit.png`](../../../src/learn/outputs/audit/lib1_dedup_final_audit_july2026/reporting/final_audit_figures/stage3_intron_inferred_mask_audit.png)
- Stage 4 primary learning curves:
  [`01_primary_pearson_learning_curves.png`](../../../src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/01_primary_pearson_learning_curves.png)
- observed common-decade gains:
  [`02_observed_10x_pearson_forest.png`](../../../src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/02_observed_10x_pearson_forest.png)
- Intron pooled versus centered learning:
  [`03_intron_scoped_learning_curves.png`](../../../src/learn/outputs/analysis/lib1_dedup_stage4_downsampling_july2026/presentation/figures/03_intron_scoped_learning_curves.png)

Prefer the SVG versions where available when assembling slides.

## Canonical Evidence Sources

- [`README.md`](README.md)
- [`lib1_dedup_phase1_hpo_rerun_plan_july2026.md`](lib1_dedup_phase1_hpo_rerun_plan_july2026.md)
- [`lib1_dedup_stage2_analysis_report_july2026.md`](lib1_dedup_stage2_analysis_report_july2026.md)
- [`lib1_dedup_utr3_targeted_hpo_analysis_report_july14_2026.md`](lib1_dedup_utr3_targeted_hpo_analysis_report_july14_2026.md)
- [`lib1_dedup_stage3_analysis_and_next_stage_handoff_july15_2026.md`](lib1_dedup_stage3_analysis_and_next_stage_handoff_july15_2026.md)
- [`lib1_dedup_post_presentation_interpretation_addendum_july17_2026.md`](lib1_dedup_post_presentation_interpretation_addendum_july17_2026.md)
- [`lib1_dedup_stage4_downsampling_analysis_and_handoff_july18_2026.md`](lib1_dedup_stage4_downsampling_analysis_and_handoff_july18_2026.md)
- [`../barcode_count_modeling_july2026/README.md`](../barcode_count_modeling_july2026/README.md)
- [`../../repo_hygiene/barcode_level_dedup_update_july6_2026.md`](../../repo_hygiene/barcode_level_dedup_update_july6_2026.md)
