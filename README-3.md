# LogSfBM_Model

This repository contains the Python implementation and numerical experiments associated with the Log S-fBM model, it is mainly based on the paper:

**"From Rough to Multifractal volatility: the Log S-fBM model"**
Peng Wu, Jean-François Muzy, Emmanuel Bacry (2022)
arXiv: https://arxiv.org/abs/2201.09516

The code allows the simulation and statistical analysis of the **Log-Stationary fractional Brownian Motion (Log-SfBM) model**, a rough stochastic volatility model bridging the gap between rough volatility models and multifractal random measures, in both the univariate and multivariate settings.

---

## Overview

The log-volatility field $\omega_t$ is a centred, stationary Gaussian process with covariance

$$
C_{\omega}(t,s) = \frac{\lambda^2}{2H(1-2H)}\left(1 - \frac{|t-s|^{2H}}{T^{2H}}\right) .
$$

driving two associated processes:

$$
X_t = \int_0^t e^{\omega_s/2}\,dW_s \quad \text{(MRW, price path)}, \qquad
M_t = \sigma^2\int_0^t e^{\omega_s}\,ds \quad \text{(MRM, integrated variance)}.
$$

- **Rough volatility regime:** $0 < H < \frac12$
- **Multifractal regime:** $H \rightarrow 0$, recovering the log-normal multifractal random measure (MRM)

Main parameters: $H$ (Hurst/roughness), $T$ (correlation scale), $\lambda^2$ (intermittency), $\sigma^2$ (variance).


---

## Overview
- **`LogSfBM_Class.py`** — simulates the log-vol field $\omega_t$ and derives the MRW/MRM paths $(X_t,M_t)$, in both univariate ($H,\lambda^2$) and multivariate (co-Hurst/cointermittency matrices) form.
- **`LogSfBM_StatisticalProperties.py`** — checks the model's multifractal scaling by estimating the moments $M_q(\tau)=\mathbb E[|\delta_\tau X_t|^q]$ and their exponent $\zeta(q)$, empirically and against theory.
---

## Documentation

Full mathematical documentation — every function with its exact formula, parameter tables, and numerical caveats — is in [`LogSfBM_Model_documentation.pdf`](LogSfBM_Model_documentation.pdf).

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## References

1. Wu, P., Muzy, J.-F., Bacry, E. (2022). "From rough to multifractal volatility: The log S-fBM model." *Physica A: Statistical Mechanics and its Applications*, 604. arXiv: https://arxiv.org/abs/2201.09516
2. Zarhali, O., Bacry, E., Muzy, J.-F. (2026). "From rough to multifractal multidimensional volatility: A multidimensional Log S-fBM model." arXiv: https://arxiv.org/abs/2601.10517
3. Wood, A. T. A., Chan, G. (1994). "Simulation of Stationary Gaussian Processes in $[0,1]^d$." *J. Comput. Graph. Statist.*, 3(4), 409–432.

