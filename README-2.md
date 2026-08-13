# LogSfBM_Model

This repository contains the Python implementation and numerical experiments associated with the Log S-fBM model, it is mainly based on the paper:

**"From Rough to Multifractal volatility: the Log S-fBM model"**
Peng Wu, Jean-François Muzy, Emmanuel Bacry (2022)
arXiv: https://arxiv.org/abs/2201.09516

The code allows the simulation and statistical analysis of the **Log-Stationary fractional Brownian Motion (Log-SfBM) model**, a rough stochastic volatility model bridging the gap between rough volatility models and multifractal random measures, in both the univariate and multivariate settings.

---

## Overview

The log-volatility field $\omega_t$ driving the model is a centred, stationary Gaussian process with covariance

$$
C_\omega(t,s) = \frac{\lambda^2}{2H(1-2H)}\left[1 - \frac{|t-s|^{2H}}{T^{2H}}\right] \mathbf{1}_{|t-s|<T}.
$$

This is a function of the lag $\tau=|t-s|$ only (stationarity), has compact support $|\tau|<T$, and gives the process variance $C_\omega(0)=\lambda^2/[2H(1-2H)]$.

From $\omega_t$, two associated processes are constructed:

$$
X_t = \int_0^t e^{\omega_s/2}\,dW_s \qquad \text{(MRW — Multifractal Random Walk, the price path)},
$$

$$
M_t = \sigma^2\int_0^t e^{\omega_s}\,ds \qquad \text{(MRM — Multifractal Random Measure, the integrated variance)}.
$$

A mean correction $m=-\lambda^2/[4H(1-2H)]$ is added to every simulated $\omega_t$ path so that $\mathbb E[e^{\omega_t}]=1$, ensuring $M_t$ is a properly normalised measure.

The model provides a unified framework:
- **Rough volatility regime:** $0 < H < \frac12$
- **Multifractal regime:** $H \rightarrow 0$, recovering the log-normal multifractal random measure (MRM)

The main parameters are:
- $H$: Hurst exponent controlling roughness
- $T$: integral (decorrelation) scale
- $\lambda^2$: intermittency coefficient
- $\sigma^2$: variance parameter

---

## Multivariate extension

The $d$-dimensional generalisation replaces the scalar pair $(H,\lambda^2)$ with two symmetric $d\times d$ matrices:
- the **co-Hurst matrix** $H_{ij}$, giving the cross-scaling exponent between components $i$ and $j$ (diagonal entries $H_{ii}$ recover the marginal Hurst exponents);
- the **cointermittency matrix** $\Lambda_{ij}$, giving the cross-intermittency between components $i$ and $j$.

For the bivariate case ($d=2$), the cross-covariance between component 1 and component 2 at lag $\tau$ is

$$
C_{12}(\tau) = \lambda_{12}\,\xi_{12}\left[\frac{1}{2H_{12}}-\frac{1}{2\bar H-1}+\frac{|\tau|}{T}\left(\frac{-1}{2H_{12}-1}+\frac{1}{2\bar H-1}\right)+\left(\frac{|\tau|}{T}\right)^{2H_{12}}\left(\frac{1}{2H_{12}-1}-\frac{1}{2H_{12}}\right)\right]\mathbf 1_{|\tau|\le T},
$$

where $\bar H=(H_{11}+H_{22})/2$, $H_{12}$ is the off-diagonal co-Hurst exponent, $\lambda_{12}=\sqrt{\Lambda_{11}\Lambda_{22}}$ the geometric mean of the marginal intermittency coefficients, and $\xi_{12}=\Lambda_{12}$ the off-diagonal cointermittency coefficient.

---

## Repository structure

```
.
├── LogSfBM_Class.py                  # Core simulation class (univariate & multivariate)
├── LogSfBM_StatisticalProperties.py  # Moment-scaling, autocovariance & distributional diagnostics
├── docs/
│   └── LogSfBM_Model_documentation.pdf  # Full technical reference (this README's source)
├── requirements.txt
└── README.md
```

---

## Module: `LogSfBM_Class.py` — Core Simulation

The core simulation class, together with the module-level helper
`GaussianProcessSimulation`, which performs exact stationary Gaussian
process simulation via the Wood–Chan circulant-embedding algorithm.

### `GaussianProcessSimulation(covariance, size)`
Simulates a stationary Gaussian process with a prescribed autocovariance sequence $c_0,\dots,c_{K-1}$ via circulant embedding — *exact* (matches the prescribed covariance to machine precision) whenever the embedding circulant has non-negative eigenvalues. Small negative eigenvalues from floating-point rounding are absorbed harmlessly by the complex square root and discarded on the final real part.

### Class `LogS_fBM`

- **`__init__(H, lambda2, T, coHurst_matrix, cointermittency_matrix)`** — two mutually exclusive construction paths: *univariate* (both matrices `None`, scalar `H`/`lambda2`) or *multivariate* (at least one matrix supplied; dimension read from the matrix shape). Raises `TypeError`/`ValueError` on invalid combinations.
- **`CovarianceFunction_SfBM(t, s, marginal=None)`** — the univariate covariance $C_\omega(t,s)$ above; with `marginal=k`, returns the marginal covariance of component $k$ in the multivariate case.
- **`CrossAutocovariance_mSfBM(tau, marginal=None)`** — the bivariate cross-covariance $C_{12}(\tau)$ above ($d=2$ only).
- **`_sfbmomcorr(size, dt, T, lambda2, H)`** — builds the discretised autocovariance sequence of $\omega_t$ at step $dt$, with a dedicated $H=0$ (MRW/logarithmic) branch, and returns the mean-correction term $m=-c_0$.
- **`GenerateSfBM_btwtimebounds(t_min, t_max, size, Msubsample)`** — simulates $\omega_t$ on an equally-spaced grid of $[t_{\min},t_{\max}]$, anchored at the left boundary.
- **`Generate_IntegratedSfBM(size, dt)`** — simulates the integrated log-volatility process $\Omega_t$, with its own closed-form autocovariance $C_\Omega(\tau)$.
- **`LogSfBM_Simulation(size, subsample, sigma, flagm, returnomega, returnOmega, simulation_method)`** — the principal simulation routine.
  - *Univariate*: builds the fine-grid covariance via `_sfbmomcorr`, simulates $\omega_t$ via `GaussianProcessSimulation`, then constructs the MRW path $X_t$ and MRM path $M_t$, sub-sampled back to the coarse grid. `returnomega`/`returnOmega` expose the raw or block-integrated field instead; `flagm` returns only $M_t$.
  - *Multivariate*: two interchangeable backends — `"Cholesky"` (exact, $O((d\cdot\text{size})^3)$, builds the full joint covariance matrix) and `"fft"` (default; spectral factorisation via the Wiener–Khinchin theorem, with Hermitian symmetry enforcement and central-quarter cropping to avoid circular boundary artefacts).
- **`LogSfBM_Nested_Simulation(size, subsample, sigma, flagm, modevol)`** — a "nested" variant where $\omega$ may be supplied externally (e.g. from an outer-layer Log-SfBM), enabling hierarchical multifractal constructions; uses the full exponential $e^{\omega_s}$ (not the half-exponential of the standard MRW) in both the price and variance formulas.
- **`genlogVol(size, subsample, M)`** and **`genlogVol_New_perscale(size, subsample, M, scale, sigma, flagm)`** — generate a log-realised-volatility *proxy* via quadratic-variation (QV) aggregation of the fine-grid price path; numerical $\pm\infty$ values (from $\mathrm{QV}=0$) are replaced by linear interpolation before mean-centring.

---

## Module: `LogSfBM_StatisticalProperties.py` — Statistical Diagnostics

Thirteen functions organised into four families — increment generation, distributional diagnostics, second-order (autocovariance) diagnostics, and multifractal moment-scaling diagnostics — all operating on increments $\delta_\tau X_t = X_{t+\tau}-X_t$ of one of four underlying processes: the MRW price path, the MRM integrated variance, the raw field $\omega_t$ ("S-fbm"), or the log-realised-variance QV proxy ("log MRM").

### Increment generation

- **`Increments(model, tau, size, subsample, logincr_flag, type_inc, marketlogvol)`** — the shared primitive. If `marketlogvol` is supplied, increments are computed directly from that empirical series, bypassing simulation; otherwise `type_inc` selects the simulated process (`"increments MRW"`, `"increments MRM"`, `"S-fbm"`, or `"log MRM"`). Returns $\delta_\tau X_t$, or $\log|\delta_\tau X_t|$ if `logincr_flag=True`.

### Distributional diagnostics

- **`LogIncrementsDistributionDensityAcrossScale(model, taulist, size, subsample, logincr_flag, savepath)`** — for each $\tau$, standardises the increments and estimates their density via a Gaussian KDE, plotting $\log \hat p_\tau(x)$ against the standard normal reference curve. Purely visual (returns `None`).

### Multifractal moment-scaling diagnostics

- **`MomentIncrementsRepresentation(logsfbm, q_list, tau_list, size, subsample, type_inc)`** — estimates the $q$-th absolute moment $M_q(\tau)=\mathbb E[|\delta_\tau X_t|^q]$ for each $(q,\tau)$ pair, and fits the log-log power-law scaling $\log M_q(\tau)=\zeta(q)\log\tau + c_q$ by OLS, where $\zeta(q)$ is the empirical multifractal structure-function exponent.
- **`MomentIncrementsRepresentationSimulatedVSTheoretical(logsfbm, q_list, tau_list, size, subsample, type_inc)`** — compares the empirical $M_q(\tau)$ against a closed-form theoretical prediction depending on `type_inc` (separate formulas for MRW, MRM, and S-fbm increments; for `"log MRM"`, an OLS log-log regression of the empirical curve itself is plotted as reference).
- **`MomentIncrementsEvolutionWrtParameter(logsfbm, q, tau_list, size, subsample, type_inc, parameter)`** — studies the sensitivity of $M_q(\tau)$ and $\zeta(q)$ to $H$ and $\lambda^2$, sweeping one while holding the other fixed via a `parameter` dict (`{"H": [...]}` or `{"lambdasquare": [...]}`).

---

## Documentation

Full mathematical documentation — covariance model, MRW/MRM construction, the multivariate co-Hurst/cointermittency extension, and every function documented with its exact formula, parameter conventions, and known numerical caveats — is in [`docs/LogSfBM_Model_documentation.pdf`](docs/LogSfBM_Model_documentation.pdf).

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## References

1. Wu, P., Muzy, J.-F., Bacry, E. (2022). "From rough to multifractal volatility: The log S-fBM model." *Physica A: Statistical Mechanics and its Applications*, 604. arXiv: https://arxiv.org/abs/2201.09516
2. Zarhali, O., Bacry, E., Muzy, J.-F. (2026). "From rough to multifractal multidimensional volatility: A multidimensional Log S-fBM model." arXiv: https://arxiv.org/abs/2601.10517
3. Wood, A. T. A., Chan, G. (1994). "Simulation of Stationary Gaussian Processes in $[0,1]^d$." *J. Comput. Graph. Statist.*, 3(4), 409–432.

---

## License

MIT
