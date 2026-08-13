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
C_\omega(t,s) = \frac{\lambda^2}{2H(1-2H)}\left[1 - \frac{|t-s|^{2H}}{T^{2H}}\right] \mathbf{1}_{|t-s|<T},
$$

driving two associated processes:

$$
X_t = \int_0^t e^{\omega_s/2}\,dW_s \quad \text{(MRW, price path)}, \qquad
M_t = \sigma^2\int_0^t e^{\omega_s}\,ds \quad \text{(MRM, integrated variance)}.
$$

- **Rough volatility regime:** $0 < H < \frac12$
- **Multifractal regime:** $H \rightarrow 0$, recovering the log-normal multifractal random measure (MRM)

Main parameters: $H$ (Hurst/roughness), $T$ (correlation scale), $\lambda^2$ (intermittency), $\sigma^2$ (variance).

The $d$-dimensional extension replaces $(H,\lambda^2)$ with a symmetric **co-Hurst matrix** $H_{ij}$ and **cointermittency matrix** $\Lambda_{ij}$; for $d=2$, the cross-covariance is

$$
C_{12}(\tau) = \lambda_{12}\,\xi_{12}\left[\frac{1}{2H_{12}}-\frac{1}{2\bar H-1}+\frac{|\tau|}{T}\left(\frac{-1}{2H_{12}-1}+\frac{1}{2\bar H-1}\right)+\left(\frac{|\tau|}{T}\right)^{2H_{12}}\left(\frac{1}{2H_{12}-1}-\frac{1}{2H_{12}}\right)\right]\mathbf 1_{|\tau|\le T},
$$

with $\bar H=(H_{11}+H_{22})/2$, $\lambda_{12}=\sqrt{\Lambda_{11}\Lambda_{22}}$, $\xi_{12}=\Lambda_{12}$.

---

## Repository structure

```
.
├── LogSfBM_Class.py                  # Core simulation class (univariate & multivariate)
├── LogSfBM_StatisticalProperties.py  # Moment-scaling, autocovariance & distributional diagnostics
├── docs/
│   └── LogSfBM_Model_documentation.pdf  # Full technical reference
├── requirements.txt
└── README.md
```

---

## `LogSfBM_Class.py` — Core simulation

- **`GaussianProcessSimulation(covariance, size)`** — exact stationary Gaussian process simulation via Wood–Chan circulant embedding.
- **`LogS_fBM`** class — univariate (scalar $H,\lambda^2$) or multivariate (co-Hurst/cointermittency matrices) construction.
  - `CovarianceFunction_SfBM`, `CrossAutocovariance_mSfBM` — evaluate $C_\omega(t,s)$ / $C_{12}(\tau)$ above.
  - `_sfbmomcorr` — discretised autocovariance of $\omega_t$, with the $H=0$ (MRW/logarithmic) branch $c_k=\lambda^2\log(T/k\,dt)$.
  - `GenerateSfBM_btwtimebounds`, `Generate_IntegratedSfBM` — simulate $\omega_t$ on a bounded grid, and the integrated field $\Omega_t$.
  - `LogSfBM_Simulation` — main simulator: builds $\omega_t$, then the MRW/MRM paths $(X_t,M_t)$ above. Multivariate backend is `"Cholesky"` (exact, $O((d\cdot\text{size})^3)$) or `"fft"` (default, spectral factorisation).
  - `LogSfBM_Nested_Simulation` — nested variant with externally supplied $\omega$, using the full exponential $e^{\omega_s}$ (hierarchical multifractal construction).
  - `genlogVol`, `genlogVol_New_perscale` — log-realised-volatility proxy via quadratic-variation aggregation, $\mathrm{QV}_k=\sum_j(X_{(j+1)/M}-X_{j/M})^2$.

---

## `LogSfBM_StatisticalProperties.py` — Statistical diagnostics

Operates on increments $\delta_\tau X_t = X_{t+\tau}-X_t$ of the MRW price, the MRM variance, the raw field $\omega_t$, or the log-realised-variance proxy.

- **`Increments`** — shared primitive; simulates or reads increments from empirical data.
- **`LogIncrementsDistributionDensityAcrossScale`** — KDE of standardised increments vs. the Gaussian reference, across scales $\tau$.
- **`MomentIncrementsRepresentation`** — empirical $q$-th moment $M_q(\tau)=\mathbb E[|\delta_\tau X_t|^q]$ and its log-log scaling exponent $\zeta(q)$.
- **`MomentIncrementsRepresentationSimulatedVSTheoretical`** — compares $M_q(\tau)$ against closed-form predictions (separate formulas for MRW, MRM, and S-fbm increments).
- **`MomentIncrementsEvolutionWrtParameter`** — sensitivity of $M_q(\tau)$, $\zeta(q)$ to $H$ and $\lambda^2$.

---

## Documentation

Full mathematical documentation — every function with its exact formula, parameter tables, and numerical caveats — is in [`docs/LogSfBM_Model_documentation.pdf`](docs/LogSfBM_Model_documentation.pdf).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## References

1. Wu, P., Muzy, J.-F., Bacry, E. (2022). "From rough to multifractal volatility: The log S-fBM model." *Physica A: Statistical Mechanics and its Applications*, 604. arXiv: https://arxiv.org/abs/2201.09516
2. Zarhali, O., Bacry, E., Muzy, J.-F. (2026). "From rough to multifractal multidimensional volatility: A multidimensional Log S-fBM model." arXiv: https://arxiv.org/abs/2601.10517
3. Wood, A. T. A., Chan, G. (1994). "Simulation of Stationary Gaussian Processes in $[0,1]^d$." *J. Comput. Graph. Statist.*, 3(4), 409–432.

## License

MIT
