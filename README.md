# The Log S-fBM Model

This repository contains the Python implementation and numerical experiments associated with the Log S-fBM model, it is mainly based on the paper:

**"From Rough to Multifractal volatility: the Log S-fBM model"**  
Peng Wu, Jean-François Muzy, Emmanuel Bacry (2022)

arXiv: https://arxiv.org/abs/2201.09516

The code allows the simulation, calibration, and empirical analysis of the **Log S-fBM model**, a stochastic volatility model bridging the gap between rough volatility models and multifractal random measures. :contentReference[oaicite:1]{index=1}

---

## Overview

The Log S-fBM model is defined through a log-normal random measure

\[
M_{H,T}(dt)=e^{\omega_{H,T}(t)}dt,
\]

where \(\omega_{H,T}(t)\) is a stationary Gaussian process related to a fractional Brownian motion with Hurst parameter \(H\).

The model provides a unified framework:

- **Rough volatility regime:** \(0 < H < 1/2\)
- **Multifractal regime:** \(H \rightarrow 0\), recovering the log-normal multifractal random measure (MRM)

The main parameters are:

- $H$: Hurst exponent controlling roughness
- $T$: correlation scale
- $\lambda^2$: intermittency coefficient
- $\sigma^2$: variance parameter

---


