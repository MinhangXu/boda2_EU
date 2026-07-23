# Prompt brief: Lib1 model-architecture illustration

Date: 2026-07-19

**Rendered result:**
[`lib1_single_part_model_architectures.png`](../tac_campaign_july2026/figures/lib1_single_part_model_architectures.png)

The rendered figure is retained for future methods/supplement use because it
shows the selected from-scratch architectures but not the transferred
BassetBranched Enhancer route.

## What this figure needs to communicate

Create a professional, publication-quality architecture figure for a thesis advisory committee presentation. The figure should explain the two scratch-model families used for the Lib1 single-part expression baseline:

1. **ResNet1D**
2. **Short-input BassetVL family** (`PromoterBassetVL` / `UTR_BassetVL`)

Use the visual grammar of the supplied BODA `BassetBranched` reference—clean rounded layer boxes, a clear flow direction, and enclosing outlines for larger modules—but do not copy its topology where the current implementation differs.

The figure is architectural, not an HPO result. It should show how one-hot DNA becomes one scalar expression prediction. Do not include folds, loss weighting, barcode aggregation, performance values, or a second prediction head.

## Copy/paste prompt for web ChatGPT

> Create a clean, editable-looking scientific architecture diagram on a white 16:9 canvas. Use a restrained navy/teal palette, thin dark-blue connectors, rounded rectangles, consistent typography, and generous whitespace. The style should resemble a polished vector figure from a computational-biology paper. Avoid gradients, 3-D blocks, heavy shadows, tiny text, decorative DNA imagery, and photorealism.
>
> Title: **Sequence models used for the Lib1 single-part expression baseline**
>
> Subtitle: **One variable-part DNA sequence → one construct-level log₂(total RNA / total DNA) prediction**
>
> Make two equally sized architecture cards, labeled **A** and **B**, arranged left to right. Use the same input and output visual language in both cards so that the architectural difference is obvious.
>
> **Shared input icon in both cards:** a compact one-hot DNA matrix labeled `one-hot sequence (4 × L)`, with A, C, G, and T rows. An arrow points into the model. Do not show barcode data entering the network; barcodes have already been aggregated into the construct-level target.
>
> **Panel A — ResNet1D**
>
> Show this left-to-right flow:
>
> `one-hot sequence (4 × L)` → `stem Conv1D` → `residual stage 1 (64 channels; 2 blocks)` → `residual stage 2 (128 channels; 2 blocks; stride 2)` → `residual stage 3 (256 channels; 2 blocks; stride 2)` → `adaptive global-average pooling` → `dense + ReLU + dropout` → `one expression scalar`.
>
> Include a small residual-block inset below Panel A. It should show a main branch `Conv1D → normalization/activation → Conv1D` and a skip branch that bypasses the two convolutions before an addition symbol. Label the skip `identity` and add `1 × 1 projection when dimensions change`. Keep the inset schematic; do not invent tensor sizes.
>
> Add a small badge at the bottom of Panel A: `Scratch CNN; selected for Intron`. A second, lighter note may say `also evaluated as a scratch candidate for other parts`.
>
> **Panel B — short-input BassetVL family**
>
> Show this left-to-right flow:
>
> `one-hot sequence (4 × L)` → `same-padded Conv1D block 1` → `same-padded Conv1D block 2` → `same-padded Conv1D block 3` → `optional adaptive-average pooling` → `flatten` → `1–2 dense blocks` → `one expression scalar`.
>
> Under the three convolution blocks, use one bracket labeled `each block: Conv1D + activation + dropout`. Do **not** draw MaxPool after each convolution. Do **not** draw multiple tissue or cell-type output branches. The current `PromoterBassetVL` class inherits from the short-input `UTR_BassetVL` implementation; it is not the classic pooled Basset topology.
>
> Add a small badge at the bottom of Panel B: `Scratch CNN; selected for Promoter, 3′UTR, and 5′UTR`.
>
> **Shared output icon:** a single rounded output box in a distinct teal color labeled on two lines: `predicted construct expression` and `ŷ = log₂(total RNA / total DNA)`. Make it visually unambiguous that there is one scalar output and one training target.
>
> Use a consistent layer color key at the bottom: convolution = muted green; residual module = blue; activation/dropout = pale orange; pooling/flatten = muted gold; dense layer = periwinkle; scalar output = teal. Use dashed outlines only to group modules, not to imply freezing in Panels A or B.
>
> Add a compact, readable configuration strip below the two cards, titled `Selected examples (topology is shared; widths were tuned)`, with these exact entries:
>
> - `Intron ResNet1D: L=80; stem 53, k=5; residual channels 64/128/256; 2 blocks per stage; block k=7; head 100`
> - `Promoter BassetVL: L=51; channels 115/126/59; kernels 5/5/5; adaptive pool 12; 1 dense layer × 191`
> - `3′UTR BassetVL: L=100; channels 184/142/50; kernels 5/9/5; adaptive pool 5; 1 dense layer × 68`
> - `5′UTR BassetVL: L=50; channels 158/146/77; kernels 7/5/5; adaptive pool 8; 2 dense layers × 134`
>
> Use true typographic primes in `3′UTR` and `5′UTR`, and a subscript 2 in `log₂`. Check every number and label for transcription errors. Keep the main blocks large enough to read from the back of a seminar room. Deliver both an SVG-style vector version and a high-resolution PNG with a white background.

## Optional companion inset for the selected Enhancer transfer route

The selected Enhancer model is **BassetBranched**, not the short-input BassetVL model in Panel B. If the architecture slide should also show the actual final Enhancer route, ask for a separate narrow companion panel rather than merging it into Panel B.

Copy/paste add-on prompt:

> Add a separate Panel C titled **Transferred BassetBranched Enhancer model**. Show `600-nt assay-framed sequence` → three classic shared convolution blocks with `Conv 300, k=19, pool 3`, `Conv 200, k=11, pool 4`, and `Conv 200, k=7, pool 4` → `shared dense representation (1000)` → three pretrained cell-type branches. Visually de-emphasize the unused HepG2 and SK-N-SH branches and highlight `K562 branch (3 layers × 140)` → `one Lib1 Enhancer expression scalar`. Use a dashed enclosure labeled `Malinois-pretrained BassetBranched weights`, and a clear color/shading key for pretrained versus fine-tuned parameters. Add a note: `Selected route: K562 initialization, full-network fine-tuning, RC augmentation`. Do not imply that the transfer-versus-scratch performance difference isolates pretraining alone; the routes also differ in architecture and input framing.

## Accuracy checks before accepting the generated image

- The network input is sequence only; barcode count is not a model input.
- Every model ends in one scalar mean-expression output.
- ResNet1D visibly contains residual skip connections.
- Short-input BassetVL has three same-padded convolution blocks and no invented MaxPool stack.
- Classic MaxPool factors 3/4/4 appear only in the optional BassetBranched transfer panel.
- `PromoterBassetVL` and `UTR_BassetVL` are presented as one short-input family, not as the classic BassetBranched network.
- Enhancer transfer uses the selected K562 branch and a 600-nt assay-framed input. The modeled Lib1 enhancer sequences span 76–211 nt before assay framing; do not label them all as exactly 200 nt.
- The target text is exactly `log₂(total RNA / total DNA)`.
- No “two-headed model,” expression-spread target, fold diagram, HPO result, or barcode-weighted loss appears in this architecture figure.

## Local implementation references

- `boda/model/resnet.py`
- `boda/model/basset.py`
- `boda/graph/cnn_prediction.py`
- `src/learn/outputs/analysis/lib1_dedup_stage3_weighted_loss_july2026/stage3_selected_part_policies.csv`
