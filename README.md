# Learning from Convergence

Reproducible companion experiments for Vignesh Gopakumar's blog post
**Learning from Convergence: Learning deterministic expectations from deliberately noisy Monte Carlo estimates**.

[Website repository](https://github.com/gitvicky/website) ·
[PTNO paper](https://arxiv.org/abs/2609.40090)

The accompanying post is maintained in the website checkout at `Blog/learning_from_convergence.qmd`.

The examples generalize the noisy-supervision argument in PTNO. They do not reproduce its particle-transport benchmarks.

## Reproduce

Use Python 3.11 or 3.12. No GPU, dataset downloads, or training service is required.

```bash
git clone https://github.com/gitvicky/learning-from-convergence.git
cd learning-from-convergence
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments.py
python verify.py
```

The reference run used Python 3.12.7, NumPy 1.26.4, PyTorch 2.5.1, and Matplotlib 3.10.8.
The experiments took about 13 seconds on the development machine, excluding import/font-cache time.
Fixed seeds and deterministic CPU operations make repeat runs reproducible within that environment;
floating-point results may differ across platforms or dependency versions.

## One problem, two figures

For `X | a ~ Normal(a, 1)`, the exact expectation is `F(a) = a² + 1`.
The label at a configuration is the mean of `N` simulated values of `X²`.
Configurations are drawn uniformly from `[-3, 3]`.

1. **Learning through noise.** Figure 1 shows every training label for `M=4096, N=1`
   and the network trained with seed 0. It receives no exact targets during training.
2. **Fixed simulation budget.** Figure 2 compares `(M,N)=(4096,1),(512,8),(64,64)`,
   with `K=1`. Each fit uses exactly `M*N=4096` normal draws.
   All eight seeds (0–7) are shown; diamonds mark their means.

![Learning the expectation from single-sample labels](results/noisy_expectation.png)

![Equal simulation budget, different allocations](results/budget_allocation.png)

Every fit uses the same `1 → 32 → 32 → 1` tanh MLP, a linear final layer,
full-batch Adam with learning rate `0.01`, and 800 updates. Input division by 3
and target division by 10 are fixed linear scalings, not data-derived or nonlinear transforms.
No architecture comparison, hyperparameter sweep, early stopping, or exact-label selection is used.
The test MSE is the average squared error on a fixed 1,001-point grid in `[-3,3]`.
The exact expectation is used only for evaluation and visualization.

Across allocations, each seed shares model initialization, the configuration RNG seed,
and a stream of 4,096 Gaussian noise draws. The configurations are nested prefixes,
and the noise draws are regrouped as `M × N`. Within each fit all draws are independent.
This coupling makes allocations partially paired without pretending that their training datasets are identical.
All allocations use equal optimizer update counts; they do **not** use equal training wall time.
The budget measures simulation generation, not training cost, diagnostics, or repeated reuse of stored labels.

| M | N | Test MSE, mean ± sample SD (8 seeds) |
|---:|---:|---:|
| 4096 | 1 | 0.0607 ± 0.0856 |
| 512 | 8 | 0.0922 ± 0.1182 |
| 64 | 64 | 0.0414 ± 0.0282 |

These runs show that one-sample labels can learn an accurate map. They do not show that
`N=1` is optimal. Here the lowest mean error occurs at `(64,64)` and seed distributions overlap.
Do not interpret this small illustrative study as a statistically established ranking.

## Jensen bias without another learned model

At `a=0, N=1`, the estimator is `Z²` with `Z ~ Normal(0,1)`.
The exact values are

```text
log E[Z²] = 0
E[log Z²] = -EulerGamma - log(2) = -1.2703628454614782
Var(log Z²) = π² / 2
```

The exact logarithmic mean follows by differentiating
`E[(Z²)^t] = 2^t Γ(t+1/2)/Γ(1/2)` at `t=0`.
The identity `ψ(1/2) = -EulerGamma - 2 log(2)` is given in
[NIST DLMF §5.4](https://dlmf.nist.gov/5.4#E13).

An independent set of 200,000 draws gives a mean log of `-1.27193`
(standard error `0.00497`) and a log sample mean of `0.00003`.
The code takes the natural logarithm directly; it does not clip or add epsilon.
These diagnostic draws are separate from each model's simulation budget.

## Outputs and checks

`experiments.py` writes SVG and 240-dpi PNG versions of both figures, the 24 individual
fit metrics in `metrics.csv`, the exact and predicted grid values in `predictions.npz`,
a generated Markdown results table, and settings/versions/Jensen diagnostics in `summary.json`.
The committed results are from the documented reference run.

`verify.py` checks the budget ledger, recomputes all test errors and seed summaries from
the saved curves, independently checks the Gaussian moments, and compares the numerical
Jensen gap with its exact value. It does not demand a particular allocation ranking.

To copy regenerated figures and the table into a local website checkout:

```bash
python export_site.py /path/to/website
```

This copies only the two figures (SVG/PNG) and generated table to
`Blog/assets/learning_from_convergence/`. Article text is maintained in the website repository.

## Statistical scope

Conditionally unbiased targets with finite second moments leave the squared-loss
**population optimum** unchanged. This does not guarantee finite-data recovery,
optimizer convergence, out-of-distribution generalization, or an unbiased learned predictor.
Nonlinear label transforms generally shift the conditional-mean target.
Composing learned primitive expectations avoids that label bias but can amplify approximation error.
For covariance, distinguish unbiased sample covariance from biased nonlinear transformations
such as Cholesky factors; Bessel-corrected variance labels remain valid direct targets.

## References

The full blog bibliography is provided in [`references.bib`](references.bib).
Its core sources are [PTNO](https://arxiv.org/abs/2609.40090),
[Noise2Noise](https://proceedings.mlr.press/v80/lehtinen18a.html),
[stochastic kriging](https://doi.org/10.1287/opre.1090.0754),
[replication versus exploration](https://arxiv.org/abs/1710.03206),
[WoS-NO](https://arxiv.org/abs/2603.01193),
[CTMC moment maps](https://arxiv.org/abs/2606.18409),
[Nonlinear Noise2Noise](https://arxiv.org/abs/2512.24794),
[nested Monte Carlo](https://proceedings.mlr.press/v80/rainforth18a.html),
and [unbiased coupled MCMC](https://arxiv.org/abs/1708.03625).
