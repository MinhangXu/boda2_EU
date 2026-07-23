"""Build the local executable Lib1 MSE-to-count-NLL learning notebook.

The generator is committed so the long mathematical narrative remains
reviewable as text while ignored .ipynb/.html artifacts can be regenerated
deterministically.
"""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "lib1_probabilistic_ml_from_mse_to_count_nll_july2026.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "boda_evo2_env",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3"},
}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb["cells"] = [
    md(r"""
# Probabilistic ML from MSE to count NLL — a Lib1 prerequisite

This notebook builds one continuous chain:

\[
\text{MSE}
\longrightarrow
\text{Gaussian NLL}
\longrightarrow
\text{gradient optimization}
\longrightarrow
\text{Poisson count NLL}
\longrightarrow
\text{NB2 count NLL}
\longrightarrow
\text{held-out model selection}.
\]

It is deliberately **toy-first**. It does not fit the real Lib1 files and it does not choose the final count model. Its job is to make every later implementation choice inspectable.

Use it in four passes for each section: **predict the result → calculate one row by hand → run the cell → explain the output aloud**. The intended next references are:

- [typeset formula reference](lib1_barcode_math_reference.html#part-iv-probabilistic-count-models), especially C1–C5;
- [short five-module recap](lib1_barcode_count_learning_modules_july2026.html#3.-Likelihoods:-Poisson-versus-NB2); and
- [Stage 5 implementation plan](../../../../../raw_data_bashor/mpra_eda_tool/plan/lib1_barcode_count_eda_revamp_july2026.md#stage-5-conditional-count-model-diagnostics).

**Current project boundary:** set the assay-depth term \(c_s=0\) as a scale convention until real sample identifiers or externally computed depth factors exist. A single free global \(c\) is confounded with a neural-network intercept and cannot be validated by asking whether training or validation NLL improves.
"""),
    code(r"""
from pathlib import Path
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from IPython.display import display, Markdown

SEED = 20260717
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_default_dtype(torch.float64)

plt.rcParams.update({
    "figure.figsize": (8, 4.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
})

print(f"Python {sys.version.split()[0]}")
print(f"PyTorch {torch.__version__}; NumPy {np.__version__}; SciPy {scipy.__version__}")
print("CPU-only deterministic toy tutorial; no Lib1 data loaded.")
"""),
    md(r"""
## 0. The objects in the model

| Symbol | Role | Observed or learned? | Lib1 meaning |
|---|---|---|---|
| \(x_i\) | input | observed | sequence/features of construct \(i\) |
| \(y_i\) | old continuous response | observed summary | construct activity/spread target in the old regression |
| \(d_{ij}\) | exposure | observed count | DNA count for barcode \(j\) of construct \(i\) |
| \(r_{ij}\) | response | observed count | RNA count for the same barcode |
| \(\eta_i=f_\theta(x_i)\) | likelihood parameter on log scale | learned from sequence | log RNA-per-DNA activity |
| \(\mu_{ij}\) | likelihood mean | computed | expected RNA count, not the observed count |
| \(\alpha\) or \(\alpha_p\) | NB2 dispersion | learned and pooled | extra-Poisson variation |

The network does **not** output a random observed RNA count. It outputs parameters of a predictive distribution. The observed count is then scored under that distribution:

```text
observed sequence x_i ──network──> eta_i
observed DNA d_ij ───────────────> mu_ij = d_ij exp(eta_i)
observed RNA r_ij + distribution(mu_ij, alpha) ──> row log probability ──> NLL
```

This tutorial performs maximum-likelihood estimation: \(\theta\) and \(\alpha\) are point estimates. A model can be *probabilistic about observations* without yet being Bayesian about its weights. Pyro becomes useful later if \(\eta_i\), \(\theta\), or dispersion receive priors and posterior uncertainty.
"""),
    md(r"""
## 1. Probability, likelihood, and NLL

After observations \(z_1,\ldots,z_N\) are collected, they are fixed. The model parameters \(\theta\) move during fitting. Assuming conditional independence,

\[
p(z_1,\ldots,z_N\mid\theta)=\prod_{n=1}^N p(z_n\mid\theta).
\]

Taking logs turns a numerically fragile product into a sum:

\[
\operatorname{NLL}(\theta)
=-\sum_{n=1}^N\log p(z_n\mid\theta).
\]

Lower NLL means the model assigned more probability (for discrete data) or density (for continuous data) to what occurred. The logarithmic score is a **proper scoring rule**: in expectation, it rewards reporting the full predictive distribution honestly, not merely getting its mean close.

Two distinctions matter:

1. A discrete PMF assigns probability mass to exact counts. A continuous PDF is a density; its value can exceed one, so continuous NLL can be negative.
2. For counts, the PMF is a dimensionless probability, but “dimensionless” still does not mean “bounded” or “normalized.” Count NLL is nonnegative because a PMF is at most one, yet it is unbounded above. A continuous density also depends on the response's measurement units and may exceed one, so Gaussian NLL can even be negative. We will define every averaging and reference convention explicitly.
"""),
    code(r"""
# One observation, viewed as a function of a candidate parameter.
observed_count = torch.tensor(3.0)
candidate_mu = torch.tensor([0.5, 1.0, 3.0, 8.0])
poisson_prob = torch.exp(torch.distributions.Poisson(candidate_mu).log_prob(observed_count))

observed_continuous = torch.tensor(0.0)
candidate_sigma = torch.tensor([0.1, 0.5, 1.0, 2.0])
normal_density = torch.exp(torch.distributions.Normal(0.0, candidate_sigma).log_prob(observed_continuous))

display(pd.DataFrame({"candidate Poisson mean": candidate_mu.numpy(), "P(R=3)": poisson_prob.numpy()}))
display(pd.DataFrame({"candidate Gaussian sigma": candidate_sigma.numpy(), "density at y=0": normal_density.numpy()}))
print("The Gaussian density exceeds 1 for sigma=0.1; that is legal because density is not point probability.")
"""),
    md(r"""
## 2. Fixed-variance Gaussian NLL is scaled MSE

Suppose the old continuous target follows

\[
y_n\sim\mathcal N(\mu_n,\sigma^2),\qquad \mu_n=f_\theta(x_n).
\]

The mean NLL is

\[
\overline{\operatorname{NLL}}
=\frac12\log(2\pi\sigma^2)
+\frac{1}{2\sigma^2}\underbrace{\frac1N\sum_n(y_n-\mu_n)^2}_{\operatorname{MSE}}.
\]

If one common \(\sigma\) is fixed, the first term is constant and the second is a positive rescaling of MSE. They therefore choose the same \(\theta\). This is the precise sense in which MSE corresponds to a Gaussian likelihood—not a universal identity for every regression setting.
"""),
    code(r"""
y_hand = torch.tensor([1.0, 2.0, 4.0])
mu_hand = torch.tensor([1.5, 1.5, 3.0])
sigma_hand = torch.tensor(1.0)
sq_error = (y_hand - mu_hand).square()
row_nll = -torch.distributions.Normal(mu_hand, sigma_hand).log_prob(y_hand)
mse = sq_error.mean()
formula_nll = 0.5 * torch.log(2 * torch.pi * sigma_hand.square()) + mse / (2 * sigma_hand.square())

hand_table = pd.DataFrame({
    "observed y": y_hand.numpy(),
    "predicted mu": mu_hand.numpy(),
    "residual": (y_hand - mu_hand).numpy(),
    "squared residual": sq_error.numpy(),
    "Gaussian row NLL": row_nll.numpy(),
})
display(hand_table.round(5))
print(f"MSE = {mse.item():.5f}; mean Gaussian NLL = {row_nll.mean().item():.5f}")
assert torch.allclose(row_nll.mean(), formula_nll)
"""),
    code(r"""
# The two objectives have the same minimum when sigma is fixed.
candidate_mean = torch.linspace(0.0, 5.0, 401)
grid_mse = torch.stack([((y_hand - m) ** 2).mean() for m in candidate_mean])
grid_nll = torch.stack([-torch.distributions.Normal(m, sigma_hand).log_prob(y_hand).mean() for m in candidate_mean])

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(candidate_mean, grid_mse, label="MSE")
ax.plot(candidate_mean, grid_nll, label="Gaussian mean NLL (sigma=1)")
ax.axvline(y_hand.mean(), color="black", linestyle="--", label=f"shared optimum = {y_hand.mean():.3f}")
ax.set(xlabel="candidate common mean", ylabel="objective", title="Different scales, same minimizer")
ax.legend()
plt.show()

assert candidate_mean[grid_mse.argmin()] == candidate_mean[grid_nll.argmin()]
"""),
    md(r"""
## 3. What changes when variance is learned?

For one Gaussian observation,

\[
\operatorname{NLL}_n
=\underbrace{\frac{(y_n-\mu_n)^2}{2\sigma_n^2}}_{\text{standardized error}}
+\underbrace{\log\sigma_n}_{\text{width penalty}}
+\underbrace{\frac12\log(2\pi)}_{\text{normalizer}}.
\]

The standardized-error term alone would reward \(\sigma\to\infty\). The \(\log\sigma\) term charges for a needlessly wide distribution. A two-headed network can therefore emit

\[
(\widehat\mu_i,h_i)=f_\theta(x_i),\qquad
\widehat\sigma_i=\operatorname{softplus}(h_i)+\epsilon.
\]

Here \(h_i\) is an unconstrained real-valued pre-link score. It is **not** an observed variance target. The likelihood teaches the scale head from how surprising each observed \(y_i\) is under the joint \((\widehat\mu_i,\widehat\sigma_i)\).
"""),
    code(r"""
sigma_grid = torch.logspace(-1.5, 1.0, 300)
fixed_mse = mse.detach()
mean_nll_grid = 0.5 * torch.log(2 * torch.pi * sigma_grid.square()) + fixed_mse / (2 * sigma_grid.square())
sigma_mle = torch.sqrt(fixed_mse)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogx(sigma_grid, mean_nll_grid)
ax.axvline(sigma_mle, color="black", linestyle="--", label=f"sqrt(MSE) = {sigma_mle:.4f}")
ax.set(xlabel="candidate sigma", ylabel="mean Gaussian NLL", title="The width penalty prevents infinite uncertainty")
ax.legend()
plt.show()

for s in [0.25, sigma_mle.item(), 1.0, 2.0]:
    value = -torch.distributions.Normal(mu_hand, torch.tensor(s)).log_prob(y_hand).mean()
    print(f"sigma={s:0.4f} -> mean NLL={value.item():0.4f}")
"""),
    code(r"""
# Fit a linear Gaussian mean and one pooled scale by maximum likelihood.
torch.manual_seed(SEED)
x_gauss = torch.linspace(-2, 2, 160).unsqueeze(1)
true_sigma = 0.65
y_gauss = 1.25 + 1.8 * x_gauss.squeeze() + true_sigma * torch.randn(len(x_gauss))

mean_model = nn.Linear(1, 1)
raw_sigma = nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.Adam(list(mean_model.parameters()) + [raw_sigma], lr=0.04)
trace = []

for step in range(700):
    optimizer.zero_grad()
    predicted_mean = mean_model(x_gauss).squeeze()
    sigma = F.softplus(raw_sigma) + 1e-6
    loss = -torch.distributions.Normal(predicted_mean, sigma).log_prob(y_gauss).mean()
    loss.backward()
    optimizer.step()
    if step % 20 == 0 or step == 699:
        trace.append((step, loss.item(), sigma.item()))

with torch.no_grad():
    predicted_mean = mean_model(x_gauss).squeeze()
    fitted_sigma = (F.softplus(raw_sigma) + 1e-6).item()
    residual_mse = (y_gauss - predicted_mean).square().mean().item()

print(f"true sigma={true_sigma:.3f}; fitted sigma={fitted_sigma:.3f}; sqrt(residual MSE)={math.sqrt(residual_mse):.3f}")
assert abs(fitted_sigma**2 - residual_mse) < 0.02

trace_df = pd.DataFrame(trace, columns=["step", "mean NLL", "sigma"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
axes[0].plot(trace_df["step"], trace_df["mean NLL"])
axes[0].set(xlabel="Adam step", ylabel="mean NLL", title="Optimization trace")
axes[1].scatter(x_gauss, y_gauss, s=10, alpha=0.45, label="observed y")
axes[1].plot(x_gauss, predicted_mean.detach(), color="black", label="fitted mean")
axes[1].fill_between(
    x_gauss.squeeze(),
    predicted_mean.detach() - 1.96 * fitted_sigma,
    predicted_mean.detach() + 1.96 * fitted_sigma,
    alpha=0.2,
    label="approx. 95% predictive interval",
)
axes[1].set(xlabel="x", ylabel="continuous y", title="Mean and observation uncertainty")
axes[1].legend()
plt.tight_layout()
plt.show()
"""),
    code(r"""
# A literal two-headed example: the model learns a mean and an input-dependent scale.
torch.manual_seed(SEED + 10)
x_hetero = 4 * torch.rand(360, 1) - 2
true_mean_hetero = 0.8 + 1.4 * x_hetero.squeeze()
true_sigma_hetero = 0.18 + 0.75 * torch.sigmoid(2.2 * x_hetero.squeeze())
y_hetero = true_mean_hetero + true_sigma_hetero * torch.randn(len(x_hetero))

class TwoHeadGaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 16), nn.Tanh())
        self.mean_head = nn.Linear(16, 1)
        self.raw_scale_head = nn.Linear(16, 1)

    def forward(self, x):
        hidden = self.trunk(x)
        mean = self.mean_head(hidden).squeeze(-1)
        scale = F.softplus(self.raw_scale_head(hidden).squeeze(-1)) + 1e-4
        return mean, scale

two_head = TwoHeadGaussian()
optimizer = torch.optim.Adam(two_head.parameters(), lr=0.018)
for step in range(1_200):
    optimizer.zero_grad()
    predicted_mean, predicted_scale = two_head(x_hetero)
    loss = -torch.distributions.Normal(predicted_mean, predicted_scale).log_prob(y_hetero).mean()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    fitted_mean, fitted_scale = two_head(x_hetero)
    coverage_80 = ((y_hetero >= fitted_mean - 1.2816 * fitted_scale) &
                   (y_hetero <= fitted_mean + 1.2816 * fitted_scale)).double().mean().item()
    coverage_95 = ((y_hetero >= fitted_mean - 1.9600 * fitted_scale) &
                   (y_hetero <= fitted_mean + 1.9600 * fitted_scale)).double().mean().item()

order = torch.argsort(x_hetero.squeeze())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
axes[0].scatter(x_hetero, y_hetero, s=9, alpha=0.35, label="observations")
axes[0].plot(x_hetero[order], fitted_mean[order], color="black", label="learned mean")
axes[0].fill_between(
    x_hetero[order].squeeze(),
    (fitted_mean - 1.96 * fitted_scale)[order],
    (fitted_mean + 1.96 * fitted_scale)[order],
    alpha=0.2, label="learned 95% interval",
)
axes[0].set(xlabel="x", ylabel="continuous y", title="Two heads, one likelihood")
axes[0].legend()
axes[1].plot(x_hetero[order], true_sigma_hetero[order], label="true sigma")
axes[1].plot(x_hetero[order], fitted_scale[order], label="learned sigma")
axes[1].set(xlabel="x", ylabel="conditional sigma", title="No sigma labels were supplied")
axes[1].legend()
plt.tight_layout()
plt.show()

print(f"in-sample 80% coverage={coverage_80:.3f}; 95% coverage={coverage_95:.3f}")
assert 0.70 < coverage_80 < 0.90 and 0.89 < coverage_95 <= 1.0
"""),
    md(r"""
### Checkpoint

<details><summary>Why can a scale head be trained without a variance label?</summary>

Because each observed response contributes one joint likelihood value. Too-small scale makes residuals extremely surprising; too-large scale pays the width penalty. Many rows jointly identify a pooled or covariate-dependent scale. This is different from regressing independently on an observed mean label and an observed variance label.
</details>

<details><summary>Is learned observation variance the same as uncertainty in neural-network weights?</summary>

No. The scale above describes the conditional variation of observations given model parameters. Parameter or epistemic uncertainty asks how uncertain we are about the fitted weights; that needs resampling, ensembles, or a Bayesian treatment.
</details>
"""),
    md(r"""
## 4. How an optimizer minimizes NLL

For a Gaussian mean,

\[
\frac{\partial\operatorname{NLL}}{\partial\mu}=\frac{\mu-y}{\sigma^2}.
\]

Backpropagation applies the chain rule from this loss through \(\mu=f_\theta(x)\) to every weight. In PyTorch the essential loop is:

```python
optimizer.zero_grad()  # remove gradients from the prior step
loss.backward()        # autograd computes derivatives
optimizer.step()       # update parameters
```

Adam is not an “NLL optimizer.” It can minimize any differentiable scalar objective. It uses running first and second moments of gradients to choose an adaptive step for each parameter. Mean NLL and summed NLL have the same optimum for a fixed, equally weighted dataset; the mean keeps gradient scale more stable across batch sizes.
"""),
    code(r"""
# Verify an analytic Gaussian gradient against autograd.
y_one = torch.tensor(4.0)
mu_one = torch.tensor(1.5, requires_grad=True)
sigma_one = torch.tensor(0.8)
one_nll = -torch.distributions.Normal(mu_one, sigma_one).log_prob(y_one)
one_nll.backward()
analytic_gradient = (mu_one.detach() - y_one) / sigma_one.square()
print(f"autograd={mu_one.grad.item():.6f}; analytic={analytic_gradient.item():.6f}")
assert torch.allclose(mu_one.grad, analytic_gradient)

def optimize_one_mean(optimizer_name, steps=90):
    parameter = nn.Parameter(torch.tensor(-2.0))
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD([parameter], lr=0.05)
    else:
        optimizer = torch.optim.Adam([parameter], lr=0.12)
    rows = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = -torch.distributions.Normal(parameter, sigma_one).log_prob(y_one)
        loss.backward()
        rows.append((step, parameter.item(), loss.item(), parameter.grad.item()))
        optimizer.step()
    return pd.DataFrame(rows, columns=["step", "mu", "NLL", "gradient"])

sgd_trace = optimize_one_mean("SGD")
adam_trace = optimize_one_mean("Adam")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for label, frame in [("SGD", sgd_trace), ("Adam", adam_trace)]:
    axes[0].plot(frame["step"], frame["mu"], label=label)
    axes[1].plot(frame["step"], frame["NLL"], label=label)
axes[0].axhline(y_one, color="black", linestyle="--", label="optimum y")
axes[0].set(xlabel="step", ylabel="candidate mu", title="Parameter path")
axes[1].set(xlabel="step", ylabel="NLL", title="Objective path")
for ax in axes: ax.legend()
plt.tight_layout()
plt.show()
"""),
    md(r"""
## 5. Why counts need a discrete observation model

RNA and DNA barcode values are nonnegative integers. A count model respects that support and assigns probability to zero. It also declares a mean–variance relationship:

| Model | Support | Conditional variance |
|---|---|---|
| Gaussian | all real values | chosen \(\sigma^2\) |
| Poisson | \(0,1,2,\ldots\) | \(\mu\) |
| NB2 | \(0,1,2,\ldots\) | \(\mu+\alpha\mu^2\) |

Poisson and NB2 model **one observed RNA count** using one jointly specified distribution. They are not two independent losses applied to a real-valued “mean target” and a real-valued “variance target.”
"""),
    code(r"""
rng = np.random.default_rng(SEED)
fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=False)
for ax, mu in zip(axes, [0.5, 2.0, 20.0]):
    draws = rng.poisson(mu, size=20_000)
    values, frequencies = np.unique(draws, return_counts=True)
    ax.bar(values, frequencies / frequencies.sum(), width=0.8)
    ax.set(title=f"Poisson mu={mu}\nzero rate={np.mean(draws == 0):.3f}", xlabel="count", ylabel="probability")
plt.tight_layout()
plt.show()
"""),
    md(r"""
## 6. Poisson NLL and the Lib1 exposure offset

The C1 baseline is

\[
r_{ij}\sim\operatorname{Poisson}(\mu_{ij}),\qquad
\log\mu_{ij}=\underbrace{\log d_{ij}}_{\text{fixed coefficient-one offset}}+\underbrace{\eta_i}_{f_\theta(x_i)},
\]

so, with \(c_s=0\),

\[
\mu_{ij}=d_{ij}\exp(f_\theta(x_i)).
\]

This is not the deterministic equation `count_model(d_ij) = r_ij`. Rather, DNA exposure and sequence determine the **expected** RNA count, and the Poisson PMF describes which observed counts are plausible around it.

The exact row NLL is

\[
\ell_{ij}=\mu_{ij}-r_{ij}\log\mu_{ij}+\log\Gamma(r_{ij}+1).
\]

The final term does not depend on model parameters, so omitting it preserves gradients on the same observations. It must be retained for exact reported NLL, bits, and Poisson-versus-NB comparison. `torch.distributions.Poisson.log_prob` includes it.
"""),
    code(r"""
d_hand = torch.tensor([2.0, 5.0, 10.0])
eta_hand = torch.log(torch.tensor(2.0))
mu_count_hand = d_hand * torch.exp(eta_hand)
r_hand = torch.tensor([3.0, 12.0, 18.0])
manual_poisson_nll = mu_count_hand - r_hand * torch.log(mu_count_hand) + torch.lgamma(r_hand + 1)
torch_poisson_nll = -torch.distributions.Poisson(mu_count_hand).log_prob(r_hand)

display(pd.DataFrame({
    "DNA d": d_hand.numpy(),
    "eta": np.repeat(eta_hand.item(), 3),
    "expected RNA mu": mu_count_hand.numpy(),
    "observed RNA r": r_hand.numpy(),
    "exact row NLL": torch_poisson_nll.numpy(),
}).round(5))
print(f"mean exact NLL = {torch_poisson_nll.mean().item():.5f}")
assert torch.allclose(manual_poisson_nll, torch_poisson_nll)
"""),
    code(r"""
# The Poisson gradient has an intuitive sign: d(NLL)/d(eta) = mu - r.
eta_probe = torch.tensor(0.3, requires_grad=True)
mu_probe = d_hand[0] * torch.exp(eta_probe)
probe_loss = -torch.distributions.Poisson(mu_probe).log_prob(r_hand[0])
probe_loss.backward()
print(f"mu={mu_probe.item():.4f}, r={r_hand[0].item():.0f}")
print(f"autograd dNLL/deta={eta_probe.grad.item():.6f}; mu-r={(mu_probe-r_hand[0]).item():.6f}")
assert torch.allclose(eta_probe.grad, mu_probe.detach() - r_hand[0])

# For a freely fitted construct, eta_hat = log(sum RNA / sum DNA).
closed_form_eta = torch.log(r_hand.sum() / d_hand.sum())
eta_fitted = nn.Parameter(torch.tensor(-1.0))
optimizer = torch.optim.Adam([eta_fitted], lr=0.08)
fit_trace = []
for step in range(500):
    optimizer.zero_grad()
    mu = d_hand * torch.exp(eta_fitted)
    loss = -torch.distributions.Poisson(mu).log_prob(r_hand).mean()
    loss.backward()
    optimizer.step()
    if step % 10 == 0: fit_trace.append((step, eta_fitted.item(), loss.item()))

print(f"closed form eta={closed_form_eta.item():.6f}; Adam eta={eta_fitted.item():.6f}")
assert abs(eta_fitted.item() - closed_form_eta.item()) < 1e-5

eta_grid = torch.linspace(-1.0, 1.6, 300)
nll_grid = torch.stack([
    -torch.distributions.Poisson(d_hand * torch.exp(e)).log_prob(r_hand).mean()
    for e in eta_grid
])
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(eta_grid, nll_grid)
axes[0].axvline(closed_form_eta, color="black", linestyle="--", label="closed-form optimum")
axes[0].set(xlabel="eta", ylabel="mean exact NLL", title="Poisson likelihood surface")
axes[0].legend()
trace_frame = pd.DataFrame(fit_trace, columns=["step", "eta", "NLL"])
axes[1].plot(trace_frame["step"], trace_frame["eta"])
axes[1].axhline(closed_form_eta, color="black", linestyle="--")
axes[1].set(xlabel="Adam step", ylabel="eta", title="Numerical fit reaches analytic MLE")
plt.tight_layout()
plt.show()
"""),
    code(r"""
# Independent classical reference: statsmodels uses exposure=d, hence coefficient 1 on log(d).
from statsmodels.discrete.discrete_model import Poisson as StatsmodelsPoisson

reference_fit = StatsmodelsPoisson(
    endog=r_hand.numpy(),
    exog=np.ones((len(r_hand), 1)),
    exposure=d_hand.numpy(),
).fit(disp=0)
reference_eta = reference_fit.params[0]
torch_log_likelihood = torch.distributions.Poisson(
    d_hand * torch.exp(closed_form_eta)
).log_prob(r_hand).sum().item()

print(f"statsmodels exposure fit eta={reference_eta:.6f}")
print(f"statsmodels log likelihood={reference_fit.llf:.6f}; Torch exact log likelihood={torch_log_likelihood:.6f}")
assert np.isclose(reference_eta, closed_form_eta.item(), atol=1e-7)
assert np.isclose(reference_fit.llf, torch_log_likelihood, atol=1e-7)
"""),
    md(r"""
### What the offset commits us to

- The coefficient of \(\log d_{ij}\) is fixed at one. DNA is exposure, not a covariate with a learned slope.
- Natural log is convenient because positive multiplicative effects become additive. A base-2 activity model is equivalent after \(\eta=\ln(2)g_\theta(x)\).
- \(d_{ij}=0,r_{ij}>0\) has \(\mu_{ij}=0\) and zero probability. Do not hide this with an evaluation epsilon. Keep such rows as QC and exclude them from this conditional-offset likelihood, or later define a joint DNA/RNA observation model.
- The baseline treats measured DNA as fixed exposure even though DNA itself has experimental uncertainty. That is an explicit approximation, not a claim that DNA is error-free.
"""),
    md(r"""
## 7. Where the sequence model enters—and why one global \(c\) is not identifiable

Every barcode from construct \(i\) shares \(x_i\) and \(\eta_i=f_\theta(x_i)\), but each row has its own \(d_{ij}\):

\[
\eta_i=f_\theta(x_i),\qquad
\mu_{ij}=d_{ij}e^{\eta_i}.
\]

For a real CNN, \(f_\theta\) reads sequence. The toy below uses one feature so the likelihood remains visible.

Now suppose

\[
\log\mu=\log d+c+(\beta_0+g_\theta(x)).
\]

For any \(\delta\), replacing \(c\) by \(c+\delta\) and \(\beta_0\) by \(\beta_0-\delta\) leaves every \(\mu\) unchanged. Therefore adding one free global \(c\) cannot test for a depth effect. A meaningful depth offset needs paired assay/sample IDs and either externally measured RNA:DNA depth factors or constrained sample effects (for example, a reference level or sum-to-zero constraint). Fitted sample effects should be called sample effects, not known depth offsets.
"""),
    code(r"""
# A tiny construct/barcode table makes the i and j indices concrete.
rng = np.random.default_rng(SEED + 1)
constructs = pd.DataFrame({
    "construct_id": ["A", "B", "C", "D", "E", "F"],
    "sequence_feature x_i": np.linspace(0.05, 0.95, 6),
})
constructs["true eta_i"] = -0.8 + 1.7 * constructs["sequence_feature x_i"]
rows = []
for row in constructs.itertuples(index=False):
    for barcode_j, dna in enumerate([4, 10, 25, 50]):
        expected = dna * np.exp(row[2])
        rows.append({
            "construct_id": row[0], "barcode_j": barcode_j,
            "x_i": row[1], "eta_i": row[2], "DNA d_ij": dna,
            "expected RNA mu_ij": expected, "observed RNA r_ij": rng.poisson(expected),
        })
toy_offset_rows = pd.DataFrame(rows)
display(toy_offset_rows.head(12).round(3))

# Direct numerical invariance check for c versus network intercept.
d_demo = torch.tensor([3.0, 9.0, 15.0])
x_demo = torch.tensor([-1.0, 0.0, 1.0])
c, beta0, beta1, delta = 0.4, -0.7, 1.1, 2.3
log_mu_a = torch.log(d_demo) + c + beta0 + beta1 * x_demo
log_mu_b = torch.log(d_demo) + (c + delta) + (beta0 - delta) + beta1 * x_demo
assert torch.allclose(log_mu_a, log_mu_b)
print("Opposite shifts in c and the model intercept leave every prediction exactly unchanged.")
"""),
    md(r"""
## 8. NB2: the same conditional mean with extra dispersion

C2 keeps the C1 mean but relaxes the Poisson variance restriction:

\[
r_{ij}\sim\operatorname{NB2}(\mu_{ij},\alpha),\qquad
\operatorname{Var}(r_{ij}\mid\mu_{ij})=\mu_{ij}+\alpha\mu_{ij}^2.
\]

Let \(k=1/\alpha\). In PyTorch the correct adapter is

```python
alpha = softplus(raw_alpha) + eps
k = 1 / alpha
dist = NegativeBinomial(total_count=k, logits=log_mu - log(k))
```

`raw_alpha` is called *raw* because the optimizer may move it anywhere on the real line; it is *unconstrained* for the same reason. `softplus` maps it to the valid positive dispersion \(\widehat\alpha\). The hat means “estimated.” This is an implementation parameterization, not another observed target.

A global \(\alpha\) pools all rows. Five part-specific \(\alpha_p\) replace one number with five, adding exactly four free dispersion parameters. Sequence-dependent \(\alpha_i\) remains deferred because it is much easier to overfit.
"""),
    code(r"""
def nb2_distribution(log_mu, alpha):
    # PyTorch NB parameterized by project mean mu and NB2 dispersion alpha.
    alpha = torch.as_tensor(alpha, dtype=log_mu.dtype, device=log_mu.device)
    size = alpha.reciprocal()
    return torch.distributions.NegativeBinomial(
        total_count=size,
        logits=log_mu - torch.log(size),
    )

requested_mu = torch.tensor([0.5, 2.0, 10.0, 50.0])
requested_alpha = torch.tensor([0.4, 0.4, 0.4, 0.4])
nb_check = nb2_distribution(torch.log(requested_mu), requested_alpha)
expected_variance = requested_mu + requested_alpha * requested_mu.square()

display(pd.DataFrame({
    "requested mu": requested_mu.numpy(),
    "Torch mean": nb_check.mean.numpy(),
    "requested NB2 variance": expected_variance.numpy(),
    "Torch variance": nb_check.variance.numpy(),
}))
assert torch.allclose(nb_check.mean, requested_mu)
assert torch.allclose(nb_check.variance, expected_variance)
"""),
    code(r"""
# Simulation check: NB2 has the requested mean, larger variance, more zeros, and a heavier upper tail.
torch.manual_seed(SEED + 2)
mu_sim = torch.tensor(20.0)
alpha_sim = torch.tensor(0.35)
n_draws = 120_000
pois_draws = torch.distributions.Poisson(mu_sim).sample((n_draws,))
nb_draws = nb2_distribution(torch.log(mu_sim), alpha_sim).sample((n_draws,))

sim_summary = pd.DataFrame([
    {"model": "Poisson", "mean": pois_draws.mean().item(), "variance": pois_draws.var(unbiased=False).item(),
     "zero rate": (pois_draws == 0).double().mean().item(), "99th percentile": torch.quantile(pois_draws, 0.99).item()},
    {"model": "NB2", "mean": nb_draws.mean().item(), "variance": nb_draws.var(unbiased=False).item(),
     "zero rate": (nb_draws == 0).double().mean().item(), "99th percentile": torch.quantile(nb_draws, 0.99).item()},
])
display(sim_summary.round(4))
print(f"NB2 theoretical variance = {(mu_sim + alpha_sim * mu_sim**2).item():.3f}")
assert abs(nb_draws.var(unbiased=False).item() - (mu_sim + alpha_sim * mu_sim**2).item()) < 3.0

# As alpha approaches zero, NB2 log probabilities approach Poisson log probabilities.
count_grid = torch.arange(0.0, 45.0)
poisson_lp = torch.distributions.Poisson(mu_sim).log_prob(count_grid)
small_alpha_lp = nb2_distribution(torch.log(mu_sim), torch.tensor(1e-5)).log_prob(count_grid)
print(f"max |NB(alpha=1e-5) log_prob - Poisson log_prob| = {(small_alpha_lp-poisson_lp).abs().max().item():.5f}")
assert (small_alpha_lp - poisson_lp).abs().max() < 0.01
"""),
    md(r"""
## 9. Synthetic five-part bake-off

This capstone creates five CRE parts with different true dispersions and a simple sequence feature. It fits the planned ladder on identical construct-grouped splits:

1. **P0:** part-only Poisson reference;
2. **P1:** sequence Poisson;
3. **N1:** sequence NB2 with one global \(\alpha\);
4. **N2:** sequence NB2 with five part-specific \(\alpha_p\).

The mean architecture is intentionally a linear stand-in for the future CNN. All barcodes from a construct stay together, because a barcode-row split would leak the same sequence into training and validation.
"""),
    code(r"""
PARTS = ["Enhancer", "Promoter", "5'UTR", "Intron", "3'UTR"]
TRUE_ALPHA = np.array([0.05, 0.15, 0.35, 0.65, 1.00])
rng = np.random.default_rng(SEED + 3)
records = []

for p, part in enumerate(PARTS):
    for construct_number in range(80):
        x = rng.normal()
        # Part baseline plus a shared sequence effect.
        eta = -0.25 + 0.55 * x + 0.18 * p
        for barcode in range(5):
            dna = int(rng.poisson(24) + 1)
            mu = dna * np.exp(eta)
            size = 1.0 / TRUE_ALPHA[p]
            success_probability = size / (size + mu)  # NumPy NB convention
            rna = int(rng.negative_binomial(size, success_probability))
            records.append({
                "part": part, "part_index": p,
                "construct_id": f"{part}:{construct_number:03d}",
                "construct_number": construct_number, "barcode": barcode,
                "sequence_feature": x, "dna": dna, "rna": rna,
            })

synthetic = pd.DataFrame(records)
synthetic["split"] = np.where(synthetic["construct_number"] % 4 == 0, "validation", "train")

# Design matrices use one reference part to avoid a redundant intercept.
part_onehot = np.eye(len(PARTS))[synthetic["part_index"].to_numpy()][:, 1:]
X_part = np.column_stack([np.ones(len(synthetic)), part_onehot])
X_sequence = np.column_stack([np.ones(len(synthetic)), synthetic["sequence_feature"].to_numpy(), part_onehot])

tensors = {
    "part": torch.tensor(synthetic["part_index"].to_numpy(), dtype=torch.long),
    "dna": torch.tensor(synthetic["dna"].to_numpy()),
    "rna": torch.tensor(synthetic["rna"].to_numpy()),
    "X_part": torch.tensor(X_part),
    "X_sequence": torch.tensor(X_sequence),
    "train": torch.tensor((synthetic["split"] == "train").to_numpy()),
    "validation": torch.tensor((synthetic["split"] == "validation").to_numpy()),
}

split_a = set(synthetic.loc[synthetic.split == "train", "construct_id"])
split_b = set(synthetic.loc[synthetic.split == "validation", "construct_id"])
assert split_a.isdisjoint(split_b)
display(synthetic.head())
display(synthetic.groupby(["part", "split"]).agg(rows=("rna", "size"), constructs=("construct_id", "nunique")))
"""),
    code(r"""
class CountRegressor(nn.Module):
    def __init__(self, n_features, family, n_parts=5):
        super().__init__()
        self.family = family
        self.mean = nn.Linear(n_features, 1, bias=False)
        nn.init.zeros_(self.mean.weight)
        if family == "nb_global":
            self.raw_alpha = nn.Parameter(torch.tensor(-1.0))
        elif family == "nb_part":
            self.raw_alpha = nn.Parameter(torch.full((n_parts,), -1.0))

    def distribution(self, X, dna, part):
        eta = self.mean(X).squeeze(-1)
        log_mu = torch.log(dna) + eta
        mu = torch.exp(torch.clamp(log_mu, min=-20, max=20))
        if self.family == "poisson":
            return torch.distributions.Poisson(mu)
        alpha_all = F.softplus(self.raw_alpha) + 1e-6
        alpha = alpha_all if self.family == "nb_global" else alpha_all[part]
        return nb2_distribution(torch.log(mu), alpha)

    def alpha(self):
        if self.family == "poisson": return None
        return F.softplus(self.raw_alpha) + 1e-6


def fit_model(model, X, steps=900, learning_rate=0.035):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train = tensors["train"]
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        distribution = model.distribution(X[train], tensors["dna"][train], tensors["part"][train])
        loss = -distribution.log_prob(tensors["rna"][train]).mean()
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == steps - 1:
            losses.append((step, loss.item()))
    return pd.DataFrame(losses, columns=["step", "train mean NLL"])


torch.manual_seed(SEED + 4)
models = {
    "P0 part-only Poisson": (CountRegressor(X_part.shape[1], "poisson"), tensors["X_part"]),
    "P1 sequence Poisson": (CountRegressor(X_sequence.shape[1], "poisson"), tensors["X_sequence"]),
    "N1 sequence NB global": (CountRegressor(X_sequence.shape[1], "nb_global"), tensors["X_sequence"]),
    "N2 sequence NB by part": (CountRegressor(X_sequence.shape[1], "nb_part"), tensors["X_sequence"]),
}
training_traces = {}
for name, (model, X) in models.items():
    training_traces[name] = fit_model(model, X)
    print(name, "finished")
"""),
    code(r"""
def per_row_evaluation(name, model, X, mask):
    with torch.no_grad():
        distribution = model.distribution(X[mask], tensors["dna"][mask], tensors["part"][mask])
        log_prob = distribution.log_prob(tensors["rna"][mask])
        mu = distribution.mean
        zero_probability = torch.exp(distribution.log_prob(torch.zeros_like(tensors["rna"][mask])))
    out = synthetic.loc[mask.numpy(), ["part", "part_index", "construct_id", "barcode", "dna", "rna"]].copy()
    out["model"] = name
    out["log_prob"] = log_prob.numpy()
    out["mu"] = mu.numpy()
    out["predicted_p0"] = zero_probability.numpy()
    return out

validation_rows = pd.concat([
    per_row_evaluation(name, model, X, tensors["validation"])
    for name, (model, X) in models.items()
], ignore_index=True)

ROW_KEY = ["construct_id", "barcode"]
reference = (
    validation_rows.query("model == 'P0 part-only Poisson'")
    .set_index(ROW_KEY)["log_prob"]
)
log_prob_by_model = {
    name: group.set_index(ROW_KEY)["log_prob"]
    for name, group in validation_rows.groupby("model", sort=False)
}
prior_rung = {
    "P0 part-only Poisson": None,
    "P1 sequence Poisson": "P0 part-only Poisson",
    "N1 sequence NB global": "P1 sequence Poisson",
    "N2 sequence NB by part": "N1 sequence NB global",
}

summary_rows = []
for name, group in validation_rows.groupby("model", sort=False):
    key = group.set_index(ROW_KEY)
    gain_bits = (key["log_prob"] - reference).mean() / np.log(2)
    previous = prior_rung[name]
    incremental_gain = np.nan if previous is None else (
        key["log_prob"] - log_prob_by_model[previous]
    ).mean() / np.log(2)
    construct_mean = group.groupby("construct_id")["log_prob"].mean()
    part_mean = group.groupby("part")["log_prob"].mean()
    zero_error = group.assign(observed_zero=(group.rna == 0).astype(float)).groupby("part").apply(
        lambda g: abs(g["observed_zero"].mean() - g["predicted_p0"].mean())
    ).mean()
    summary_rows.append({
        "model": name,
        "validation micro NLL (nats/barcode)": -group["log_prob"].mean(),
        "validation construct-macro NLL": -construct_mean.mean(),
        "validation part-macro NLL": -part_mean.mean(),
        "gain over P0 (bits/barcode)": gain_bits,
        "incremental gain vs prior rung (bits/barcode)": incremental_gain,
        "mean absolute part zero-rate error": zero_error,
    })

model_summary = pd.DataFrame(summary_rows).sort_values("validation micro NLL (nats/barcode)")
display(model_summary.round(4))

dispersion_table = []
for name, (model, _) in models.items():
    if model.alpha() is None:
        values = "Poisson: no alpha"
    else:
        values = np.atleast_1d(model.alpha().detach().numpy()).round(3).tolist()
    dispersion_table.append({"model": name, "fitted alpha": values})
display(pd.DataFrame(dispersion_table))
print("True part alphas:", dict(zip(PARTS, TRUE_ALPHA)))

best_model_name = model_summary.iloc[0]["model"]
assert best_model_name == "N2 sequence NB by part"
"""),
    code(r"""
# Paired uncertainty: resample constructs, keeping all of each construct's barcodes together.
wide_log_prob = validation_rows.pivot(
    index=["part", "construct_id", "barcode"], columns="model", values="log_prob"
)
comparisons = [
    ("P1:P0 sequence gain", "P1 sequence Poisson", "P0 part-only Poisson"),
    ("N1:P1 global-dispersion gain", "N1 sequence NB global", "P1 sequence Poisson"),
    ("N2:N1 part-dispersion gain", "N2 sequence NB by part", "N1 sequence NB global"),
]
bootstrap_rng = np.random.default_rng(SEED + 5)
bootstrap_rows = []
for label, new, old in comparisons:
    per_construct_bits = (
        (wide_log_prob[new] - wide_log_prob[old]) / np.log(2)
    ).groupby(["part", "construct_id"]).mean()
    values = per_construct_bits.to_numpy()
    draws = bootstrap_rng.choice(values, size=(4_000, len(values)), replace=True).mean(axis=1)
    bootstrap_rows.append({
        "increment": label,
        "mean paired gain (bits/barcode)": values.mean(),
        "construct-bootstrap 2.5%": np.quantile(draws, 0.025),
        "construct-bootstrap 97.5%": np.quantile(draws, 0.975),
        "held-out constructs": len(values),
    })

bootstrap_summary = pd.DataFrame(bootstrap_rows)
display(bootstrap_summary.round(4))
print("This toy interval demonstrates the unit of resampling; it is not evidence about Lib1.")
"""),
    code(r"""
# Partwise raw NLL and reference-relative gain show why cross-part raw NLL needs context.
part_rows = []
reference_by_row = validation_rows.query("model == 'P0 part-only Poisson'").set_index(ROW_KEY)["log_prob"]
for (name, part), group in validation_rows.groupby(["model", "part"], sort=False):
    indexed = group.set_index(ROW_KEY)
    part_rows.append({
        "model": name, "part": part,
        "mean NLL": -indexed["log_prob"].mean(),
        "gain vs part-matched P0 (bits/barcode)": (indexed["log_prob"] - reference_by_row.loc[indexed.index]).mean() / np.log(2),
    })
part_scores = pd.DataFrame(part_rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for name, group in part_scores.groupby("model", sort=False):
    ordered = group.set_index("part").loc[PARTS]
    axes[0].plot(PARTS, ordered["mean NLL"], marker="o", label=name)
    axes[1].plot(PARTS, ordered["gain vs part-matched P0 (bits/barcode)"], marker="o", label=name)
axes[0].set(ylabel="held-out mean NLL (nats/barcode)", title="Absolute difficulty differs by part")
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set(ylabel="information gain (bits/barcode)", title="Part-matched reference asks a sharper question")
for ax in axes:
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.show()
"""),
    code(r"""
# Calibration for the selected N2 model: mean counts and zeros by predicted-mean bin.
selected = validation_rows.query("model == 'N2 sequence NB by part'").copy()
selected["mu_bin"] = pd.qcut(selected["mu"], q=10, duplicates="drop")
calibration = selected.groupby("mu_bin", observed=True).agg(
    predicted_mean=("mu", "mean"),
    observed_mean=("rna", "mean"),
    predicted_zero_rate=("predicted_p0", "mean"),
    observed_zero_rate=("rna", lambda x: np.mean(x == 0)),
    rows=("rna", "size"),
).reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
axes[0].plot(calibration["predicted_mean"], calibration["observed_mean"], marker="o")
lims = [0, max(calibration["predicted_mean"].max(), calibration["observed_mean"].max())]
axes[0].plot(lims, lims, color="black", linestyle="--")
axes[0].set(xlabel="mean predicted mu", ylabel="mean observed RNA", title="Mean calibration")
axes[1].plot(calibration["predicted_zero_rate"], calibration["observed_zero_rate"], marker="o")
axes[1].plot([0, 1], [0, 1], color="black", linestyle="--")
axes[1].set(xlabel="predicted zero probability", ylabel="observed zero fraction", title="Zero calibration")
plt.tight_layout()
plt.show()

# Randomized quantile residuals: calibrated discrete forecasts should be near N(0,1).
selected_model, selected_X = models["N2 sequence NB by part"]
mask = tensors["validation"]
with torch.no_grad():
    selected_dist = selected_model.distribution(selected_X[mask], tensors["dna"][mask], tensors["part"][mask])
    mu_np = selected_dist.mean.numpy()
    alpha_np = selected_model.alpha()[tensors["part"][mask]].numpy()
y_np = tensors["rna"][mask].numpy()
size_np = 1.0 / alpha_np
prob_np = size_np / (size_np + mu_np)
lower = stats.nbinom.cdf(y_np - 1, size_np, prob_np)
upper = stats.nbinom.cdf(y_np, size_np, prob_np)
u = lower + np.random.default_rng(SEED).uniform(size=len(y_np)) * (upper - lower)
rq_residual = stats.norm.ppf(np.clip(u, 1e-9, 1 - 1e-9))

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
stats.probplot(rq_residual, dist="norm", plot=axes[0])
axes[0].set_title("Randomized-quantile residual Q-Q")
axes[1].scatter(mu_np, rq_residual, s=9, alpha=0.35)
axes[1].axhline(0, color="black", linestyle="--")
axes[1].set(xscale="log", xlabel="predicted mu", ylabel="randomized-quantile residual", title="Residuals versus fitted mean")
plt.tight_layout()
plt.show()
"""),
    md(r"""
## 10. What “normalized NLL” should mean here

There is no universal normalized NLL. Name each quantity explicitly.

### Exact held-out mean NLL

\[
\overline{\operatorname{NLL}}
=-\frac1N\sum_{(i,j)\in\mathrm{test}}\log p_\theta(r_{ij}\mid d_{ij},x_i,p_i)
\]

This is **nats per barcode row**. It removes dependence on the number of rows, but not intrinsic differences in count depth, zero burden, or entropy.

### Reference-relative information gain

For a fixed baseline \(B\) fitted only on training data,

\[
G_{M:B}
=\frac{\overline{\operatorname{NLL}}_B-\overline{\operatorname{NLL}}_M}{\log 2}
\quad\text{bits per barcode}.
\]

Positive gain means \(M\) assigned more held-out probability than \(B\). Use a part-matched reference and show every part. An equal-part macro-average prevents the largest library from dominating.

### Micro versus macro weighting

- **Barcode-micro:** average rows; constructs with more barcodes receive more weight.
- **Construct-macro:** average within each construct, then across constructs; constructs receive equal weight.
- **Part-macro:** average part-level scores; CRE parts receive equal weight.

Per-construct NLL is valid but extremely noisy at low support. Always show barcode count and total DNA; do not rank singletons as if their scores were precise.

### Deviance \(D^2\)

Within one declared likelihood family and dispersion convention,

\[
D^2=1-\frac{D_{\mathrm{model}}}{D_{\mathrm{null}}}.
\]

This is fraction of null deviance removed, not variance explained, and it can be negative on held-out data. Do **not** select Poisson versus NB by comparing family-specific saturated deviances: their saturated reference likelihoods differ, and NB’s changes with \(\alpha\). Use exact held-out full NLL for cross-family selection.
"""),
    code(r"""
# The bits identity used above, checked on the synthetic validation rows.
n2_nll = model_summary.set_index("model").loc["N2 sequence NB by part", "validation micro NLL (nats/barcode)"]
p0_nll = model_summary.set_index("model").loc["P0 part-only Poisson", "validation micro NLL (nats/barcode)"]
stated_gain = model_summary.set_index("model").loc["N2 sequence NB by part", "gain over P0 (bits/barcode)"]
assert np.isclose(stated_gain, (p0_nll - n2_nll) / np.log(2))
print(f"N2 gain over P0 = {stated_gain:.4f} bits per held-out barcode row")
"""),
    md(r"""
## 11. The goodness-of-fit panel

NLL is primary for distribution selection because it scores the entire predictive PMF. It does not show *how* a model fails. Pair it with:

1. **Mean calibration:** observed versus predicted mean and \(\sum r/\sum\mu\), stratified by part, predicted \(\mu\), and DNA exposure.
2. **Zero calibration:** observed zero fraction versus predicted \(P(R=0)\). For NB2, \(P(R=0)=(1+\alpha\mu)^{-1/\alpha}\).
3. **Tail and rootogram checks:** observed versus model-expected frequencies for 0, 1, 2, … and upper-tail exceedances.
4. **Randomized PIT/quantile residuals:** \(u=F(r-1)+vP(R=r)\), then \(z=\Phi^{-1}(u)\). A calibrated model gives approximately uniform \(u\) and standard-normal \(z\).
5. **Pearson residuals:** \((r-\mu)/\sqrt{\mu+\alpha\mu^2}\), viewed against \(\mu\), DNA, part, and zero status.
6. **Predictive interval coverage:** 50%, 80%, and 95% coverage plus interval width.

Pearson \(r\), Spearman \(\rho\), COD \(R^2\), and MSE remain useful descriptive comparisons of predicted activity with old summary targets. Pearson measures linear association; Spearman is the rank-association metric. Neither measures calibration. They do not score dispersion, zero probability, or tails, so they are not primary selectors of Poisson versus NB.
"""),
    md(r"""
## 12. Library decision

| Tool | Use now | Why | Main trap |
|---|---|---|---|
| `torch.distributions` | **Primary C1/C2 neural implementation** | exact PMFs, `log_prob`, sampling, autograd, batching, GPU, already in BODA | verify NB parameterization with moment tests |
| `statsmodels.discrete.Poisson/NegativeBinomial` | **Independent classical reference** | correct coefficient-one `exposure`, NB2 MLE, useful for intercept/part-only recovery checks | GLM-family `NegativeBinomial(alpha=1)` fixes alpha; it does not estimate it |
| Pyro | **Later hierarchy** | priors, latent \(\eta_i\), partial pooling, posterior uncertainty; already PyTorch-compatible | model/guide/SVI complexity adds nothing to simple point-estimate C1/C2 |
| NumPyro / TensorFlow Probability / scvi-tools | not for this stage | capable libraries, but introduce another stack or domain-specific abstractions | duplicated model code and parameterization risk |

For exact reporting, prefer

```python
-torch.distributions.Poisson(mu).log_prob(r)
```

over `torch.nn.PoissonNLLLoss`: its default omits \(\log(r!)\), and its `full=True` uses a Stirling approximation. `torch.nn.NLLLoss` is a classification API expecting class log-probabilities; it is not a generic likelihood wrapper.
"""),
    md(r"""
## 13. Project implementation gate after this tutorial

Do not jump directly from the toy result to a CNN. Implement the real-data ladder as auditable rungs:

| Rung | Mean model | Dispersion | Question |
|---|---|---|---|
| P0 | part-only training intercept + DNA offset | Poisson | What does a no-sequence reference predict? |
| P1 | sequence \(f_\theta(x_i)\) + DNA offset | Poisson | Does sequence add held-out information? |
| N1 | same sequence mean | one global \(\alpha\) | Does pooled overdispersion improve held-out prediction? |
| N2 | same sequence mean | five \(\alpha_p\) | Is dispersion reliably different by CRE part? |
| N3, deferred | same mean | sequence-dependent \(\alpha_i\) | Only consider after N1/N2 pass calibration and stability gates |

Required contract:

- same integer, nonnegative response rows and same \(d>0\) eligibility set for all candidates;
- construct-grouped train/validation/test split; untouched test only after selection;
- exact per-row full log probabilities retained;
- report barcode-micro, construct-macro, every part, and equal-part macro scores;
- report incremental gains \(G_{P1:P0}\), \(G_{N1:P1}\), and \(G_{N2:N1}\);
- paired uncertainty by resampling **constructs**, not barcode rows, plus multiple fixed seeds;
- simulation recovery and Torch-versus-statsmodels reference tests before real fitting;
- calibration for means, zeros, residuals, and tails.

Training NLL alone cannot justify N2: it nests N1 and has four more dispersion parameters. Adopt it only if held-out gain is repeatable across construct resamples/seeds and the relevant calibration improves. “Any numerical improvement” is not a decision rule.

### Exact question for the experimental team about depth

> For every DNA and RNA sequencing library/sample/lane contributing to `DNA_bc` and `RNA_bc`, can you provide (1) its sample ID and DNA/RNA pairing, (2) total usable reads after the same demultiplexing, barcode-validity, clipping, and filtering steps used for these counts, (3) how lanes or replicates were pooled into the exported columns, and (4) whether any counts were rescaled or normalized before export?

If only one already-aggregated DNA and RNA count survives per barcode and the mapping is lost, a sample-specific \(c_s\) cannot be reconstructed. Keep \(c_s=0\), document that scale convention, and do not label a learned intercept as measured sequencing depth.
"""),
    md(r"""
## 14. Self-check before real data

Try answering without looking back.

1. Why do fixed-\(\sigma\) Gaussian NLL and MSE choose the same mean model?
2. What two terms stop a learned Gaussian scale from collapsing to zero or exploding to infinity?
3. What does `loss.backward()` compute, and what does Adam do with it?
4. In \(\mu_{ij}=d_{ij}e^{f_\theta(x_i)}\), which terms carry \(i\) and \(j\)?
5. Why is \(\mu_{ij}\) not equal to the observed \(r_{ij}\)?
6. What does the coefficient-one DNA offset assert?
7. Why is a free global \(c\) confounded with a network intercept?
8. What does NB2 \(\alpha\) change, and what does it leave unchanged?
9. Why does part-specific dispersion add four—not five—parameters relative to one global dispersion?
10. Why must full likelihood constants be retained for Poisson-versus-NB NLL comparison?
11. Why are held-out mean NLL and bits/barcode both useful?
12. Why should uncertainty in \(\Delta\)NLL resample constructs rather than individual barcodes?

<details><summary>Compact answers</summary>

1. Their objectives differ only by a constant and positive scale when \(\sigma\) is fixed.
2. Standardized residual error penalizes too-small scale; the log-scale normalizer penalizes too-wide predictions.
3. Backprop computes derivatives through the computation graph; Adam turns their running moments into parameter updates.
4. \(x_i,\eta_i\) carry construct \(i\); \(d_{ij},r_{ij},\mu_{ij}\) also carry barcode \(j\).
5. \(\mu\) is a distributional expectation; \(r\) is one random realization.
6. Doubling DNA exposure doubles expected RNA at fixed sequence activity.
7. Opposite constant shifts leave every log mean unchanged.
8. It changes variance, zero probability, and tails while preserving \(E[R]=\mu\).
9. Five free values replace one free value: \(5-1=4\).
10. Dropped constants differ with the observed count and between families, corrupting absolute cross-family scores.
11. Mean NLL is the exact absolute log score; reference-relative bits state how much predictive information was added over a declared baseline.
12. Barcodes from one construct share the sequence and scientific unit, so row resampling falsely treats dependent evidence as independent.
</details>
"""),
    md(r"""
## Sources and implementation references

- PyTorch [`torch.distributions`](https://docs.pytorch.org/docs/stable/distributions.html), including exact `Poisson.log_prob` and `NegativeBinomial.log_prob`.
- PyTorch [`PoissonNLLLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html) and [`GaussianNLLLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html) API notes.
- statsmodels [`Poisson`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.Poisson.html) and [`NegativeBinomial`](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.NegativeBinomial.html) exposure/offset APIs; the separate [GLM NB family](https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.NegativeBinomial.html) fixes `alpha` supplied by the caller.
- [Gneiting & Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and Estimation*](https://doi.org/10.1198/016214506000001437).
- [Dunn & Smyth (1996), *Randomized Quantile Residuals*](https://gksmyth.github.io/pubs/residual.pdf).
- [Huang (2015), *Goodness-of-Fit Tests and Model Diagnostics for Negative Binomial Regression of RNA Sequencing Data*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4365073/).
- Pyro [Bayesian regression tutorial](https://pyro.ai/examples/bayesian_regression.html), for the later transition from maximum-likelihood point estimates to priors and posterior inference.

**Reproducibility note:** this notebook was built and executed in the project `boda_evo2_env` snapshot (PyTorch 1.13.1, Pyro 1.8.6). Pin and test the actual training environment before Stage 5 implementation rather than assuming current online API versions match this snapshot.
"""),
]

nbf.write(nb, OUTPUT)
print(f"Wrote {OUTPUT}")
