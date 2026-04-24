# Model Architecture Notes

This note gives a high-level, practical description of the main sequence
regression architectures currently used in `boda2_EU`.

It is meant to answer questions like:

- What is the difference between `BassetVL` and `UTR_BassetVL`?
- Why might `ResNet1DRegressor` behave differently?
- What is `BassetBranched`, and when is it relevant?

## Quick Summary

- `UTR_BassetVL`
  - short-sequence CNN
  - same-padded convolutions
  - no built-in max-pooling stack
  - a natural fit for short inputs like 5'UTR / 3'UTR libraries
  - can also be tried on promoter if you want a "short-local-pattern" CNN

- `BassetVL`
  - classic Basset-style CNN
  - conv stack plus max-pooling
  - designed more in the style of longer CRE sequence modeling
  - a natural fit for enhancer or promoter settings with longer inputs

- `ResNet1DRegressor`
  - residual 1D CNN
  - uses residual blocks and adaptive average pooling
  - often a good alternative when a plain stacked CNN is too rigid
  - worth treating as a real architecture comparison, not just a minor variant

- `BassetBranched`
  - classic Basset-style trunk with per-output branched heads
  - mainly relevant for multi-output settings
  - in this repo it is mostly used as a baseline / transfer-learning style path

## `UTR_BassetVL`

`UTR_BassetVL` is the short-sequence variant.

Practical properties:

- uses three Conv1D layers with `"same"` padding
- sequence length is preserved through the conv stack unless optional adaptive
  pooling is turned on
- does not use the classic Basset max-pooling cascade
- flattens the representation and passes it through one or more linear layers

How to think about it:

- It is biased toward preserving position-by-position local information across
  short sequences.
- It is a good default for short fixed-length libraries such as 5'UTR and 3'UTR
  tasks.
- When used on promoter, it is not because promoter is "a UTR task"; it is
  because the architecture itself is a compact short-sequence CNN that may still
  work well on promoter-length inputs.

In other words, `UTR_BassetVL` is better understood as a *short-input Basset
variant* than as something biologically restricted to UTRs.

## `BassetVL`

`BassetVL` is the more classic Basset-style architecture.

Practical properties:

- three Conv1D layers with explicit padding
- interleaved max-pooling behavior that progressively reduces sequence length
- larger "compressed feature hierarchy" than the UTR-specific variant
- one or more dense layers after flattening

How to think about it:

- It is a stronger "standard CRE CNN" baseline.
- It tends to make more sense for longer inputs where hierarchical compression
  and pooling are useful.
- For promoter tasks, comparing `BassetVL` against `UTR_BassetVL` asks a real
  question:
  - should we use a classic pooled CRE CNN?
  - or a shorter-sequence, less aggressively pooled CNN?

So in promoter sweeps:

- `utr_bassetvl` means "short-sequence Basset-style variant"
- `bassetvl` means "classic pooled Basset-style variant"

## `ResNet1DRegressor`

`ResNet1DRegressor` is a residual 1D CNN rather than a plain stacked-conv
network.

Practical properties:

- stem conv followed by residual blocks
- stage-wise channel changes
- stride-based downsampling inside the residual stack
- adaptive average pooling to length 1 before the regression head

How to think about it:

- It is usually a more flexible architecture family than a plain 3-layer CNN.
- Residual connections can make optimization easier and help deeper feature
  extraction behave more stably.
- Adaptive average pooling makes the head less tied to a particular flattened
  spatial layout.

This is why `ResNet1DRegressor` should be treated as a genuine architecture
comparison against `BassetVL` / `UTR_BassetVL`, not just a small implementation
variation.

## `BassetBranched`

`BassetBranched` is the classic Basset-style trunk plus additional branched
layers before the final outputs.

Practical properties:

- shared convolutional trunk
- shared dense trunk
- additional branched dense layers
- naturally suited to multi-output prediction

How to think about it:

- It is most relevant when outputs share a common representation but still need
  some output-specific specialization.
- In this repo, the main visible use is the Malinois enhancer branched baseline,
  which is closer to an inherited / reproduction-style baseline than to a new
  top-priority HPO surface.

## Which One To Reach For

If the task is:

- short UTR library regression:
  - start with `UTR_BassetVL`

- promoter regression:
  - compare `UTR_BassetVL`, `BassetVL`, and `ResNet1DRegressor`
  - this is a meaningful architecture sweep, not redundant bookkeeping

- enhancer regression on longer CRE inputs:
  - `BassetVL` and `ResNet1DRegressor` are usually the most natural primary
    baselines

- multi-output enhancer transfer baseline:
  - `BassetBranched` can still be useful as a reference point

## Important Naming Caution

The names can be misleading if read too literally:

- `UTR_BassetVL` does **not** mean "only valid for UTR biology"
- it means "the Basset variant designed for shorter inputs"

That distinction matters for promoter experiments, where `UTR_BassetVL` is
really an architecture choice rather than a dataset-family choice.
