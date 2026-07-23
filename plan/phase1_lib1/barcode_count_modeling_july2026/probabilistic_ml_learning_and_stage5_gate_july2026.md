# Probabilistic-ML learning gate and Stage 5 decision contract

**Date:** 2026-07-17
**Status:** learning tutorial implemented and verified; real Lib1 C1/C2 fitting not started

## Why this gate exists

The next model change should be interpretable before it is computationally
expensive. The committed, reviewable prerequisite source is:

[`build_lib1_probabilistic_ml_tutorial.py`](../../../tutorials/lib1_tasks/barcode_level/variant_level_redo_jul7_2026/build_lib1_probabilistic_ml_tutorial.py)

It regenerates the executed notebook and its rendered HTML companion locally.
Those generated artifacts remain untracked under the repository data/artifact
policy.

It establishes, with executable assertions, that:

- fixed-variance Gaussian NLL is scaled MSE plus a constant;
- a learned scale/dispersion is trained through one joint likelihood and does
  not require a separate observed variance label;
- Adam is a general differentiable-objective optimizer, not an NLL-specific
  algorithm;
- the Poisson exposure model uses
  `log_mu = log(dna) + f_theta(sequence)` and exact count log probability;
- the PyTorch NB adapter has mean `mu` and variance
  `mu + alpha * mu**2`;
- model selection requires held-out full NLL on identical rows; and
- cross-part reporting needs a declared part-matched reference, not a
  supposedly universal bounded NLL.

## Frozen implementation stack for C1/C2

1. Use native `torch.distributions.Poisson` and
   `torch.distributions.NegativeBinomial` for the neural training and exact
   per-row `log_prob` path.
2. Use `statsmodels.discrete.Poisson` and
   `statsmodels.discrete.NegativeBinomial(loglike_method="nb2",
   exposure=dna)` as independent CPU reference fits and simulation-recovery
   checks.
3. Reserve Pyro for a later hierarchical model with latent construct activity,
   priors, partial pooling, or posterior uncertainty.
4. Do not add NumPyro, TensorFlow Probability, or scvi-tools for C1/C2.

Do not use `torch.nn.NLLLoss`; it is a classification loss. Do not use
`PoissonNLLLoss(full=False)` for reported scores because it omits
`log(r!)`; even `full=True` uses a Stirling approximation. Use the exact
distribution `log_prob` for both fitting and evaluation unless a separately
tested optimization shortcut is explicitly documented.

## Parameterization contract

For every eligible row with `dna > 0`:

```text
eta_i  = f_theta(sequence_i)
log_mu = log(dna_ij) + eta_i                  # c_s = 0 convention
mu_ij  = exp(log_mu)
```

For NB2:

```text
alpha = softplus(raw_alpha) + epsilon
size  = 1 / alpha
NegativeBinomial(total_count=size, logits=log_mu - log(size))
```

Automated tests must prove that the resulting distribution has
`E[R] = mu` and `Var[R] = mu + alpha * mu**2` over a parameter grid.

## Depth-offset decision

With the currently aggregated counts, one free global `c_s` and the neural
intercept are non-identifiable. Opposite shifts leave every `mu` and NLL
unchanged. Therefore:

- set `c_s = 0` now as an explicit scale convention;
- do not add one learnable global `c_s` and interpret an NLL comparison as a
  depth test;
- later use a fixed external offset only when paired RNA/DNA usable-depth
  totals and row-to-sample mappings exist; or
- fit sample effects only with a reference/sum-to-zero constraint and call
  them fitted sample effects, not known depth factors.

Experimental metadata request:

> For every DNA and RNA sequencing library/sample/lane contributing to
> `DNA_bc` and `RNA_bc`, provide its sample ID and pairing, usable read totals
> after the same demultiplexing/barcode-validity/clipping/filtering pipeline,
> the lane/replicate pooling rule, and whether exported counts were rescaled.

## Frozen comparison ladder

Use the same construct-grouped train/development/test identities and identical
eligible barcode rows at every rung:

1. **P0:** part-only Poisson reference with DNA offset.
2. **P1:** sequence Poisson with the same offset.
3. **N1:** the same sequence mean with one global NB2 `alpha`.
4. **N2:** the same sequence mean with part-specific `alpha_p`.
5. **N3, deferred:** sequence-dependent dispersion only after N1/N2 pass the
   held-out and calibration gates.

The roadmap currently declares four immediate parts and excludes enhancer,
whereas the July 17 meeting description of N2 assumes five parts. This changes
the parameter count: four part dispersions add three parameters relative to
one global dispersion; five add four. Freeze the real-data scope before N2 and
state it in every table. The tutorial uses all five parts only to demonstrate
the five-versus-one comparison; it is not a silent change to the current EDA
scope.

## Evaluation contract

Primary predictive score:

```text
heldout_mean_nll = -mean(exact_full_log_prob)  # nats per barcode row
```

Incremental information gains:

```text
G_P1_P0 = (mean_nll_P0 - mean_nll_P1) / log(2)
G_N1_P1 = (mean_nll_P1 - mean_nll_N1) / log(2)
G_N2_N1 = (mean_nll_N1 - mean_nll_N2) / log(2)
```

These are bits per held-out barcode; positive is better. Report every part,
pooled barcode-micro, construct-macro, and equal-part macro results. Estimate
paired uncertainty by resampling constructs, not barcode rows, and repeat the
comparison over fixed training seeds. Do not rank low-support individual
constructs without showing barcode support and total DNA.

Complement NLL with:

- observed-versus-predicted mean calibration;
- observed-versus-predicted zero probability;
- randomized quantile/PIT and Pearson residuals versus `mu`, DNA, and part;
- predictive interval coverage and upper-tail/rootogram checks; and
- fit/convergence and excluded-row reports.

Within-family deviance and `D^2` are secondary diagnostics. Do not compare
Poisson and NB using their family-specific saturated deviances. Pearson,
Spearman, COD `R2`, and MSE may describe inferred activity relative to old
summary targets, but they do not select the count distribution.

## Stage 5 acceptance and stop rules

Before real data:

- analytic PMFs and Torch/statsmodels log probabilities agree in float64;
- simulation recovers Poisson activity, global NB dispersion, and correct
  part indexing;
- `alpha -> 0` approaches Poisson;
- gradients for activity and raw dispersion are finite;
- `dna = 0, rna > 0` is explicitly rejected from the conditional likelihood
  and remains visible in QC; and
- all compared candidates score exactly the same held-out rows.

For real data, do not adopt N2 after an arbitrary tiny NLL improvement. Require
a positive held-out N2:N1 gain that is stable under construct resampling and
training seeds, plus improvement in the calibration pattern N2 was intended
to fix. If NB2 still has structured zero failure, diagnose it before proposing
a complete hurdle or zero-inflated PMF.

## Regeneration

```bash
conda run --no-capture-output -n boda_evo2_env python \
  tutorials/lib1_tasks/barcode_level/variant_level_redo_jul7_2026/build_lib1_probabilistic_ml_tutorial.py

conda run --no-capture-output -n boda_evo2_env python -m jupyter nbconvert \
  --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=boda_evo2_env \
  tutorials/lib1_tasks/barcode_level/variant_level_redo_jul7_2026/lib1_probabilistic_ml_from_mse_to_count_nll_july2026.ipynb
```

The real-data Stage 5 implementation remains a separate reviewed change.
