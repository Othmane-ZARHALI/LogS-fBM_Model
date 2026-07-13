"""
===========================================================
File        : test_genlogVol_examples.py
Project     : LogSfBMCore
Authors     : Othmane Zarhali
Created     : 2026
Description :
    Example script for testing and illustrating the usage of
    the log-realised-volatility estimators implemented in the
    Log-Stationary Fractional Brownian Motion (Log-SfBM) model.

    This script does NOT perform unit testing. Instead, it
    provides executable examples that:

      - Instantiate the LogSfBM model in the univariate setting.
      - Generate log-realised volatility proxies using:
            * genlogVol (baseline quadratic-variation estimator)
            * genlogVol_New_perscale (multi-scale estimator)
      - Compare different aggregation scales (M and M × scale).
      - Output shapes, means, and basic diagnostics for sanity
        checking of numerical stability.

Mathematical context
--------------------
The log-realised volatility proxy is constructed from the
quadratic variation of the simulated price process X_t:

    QV_k = sum_{j in block k} (X_{j+1} - X_j)^2

The log-volatility estimator is then:

    ℓ_k = log(QV_k) - mean(log(QV_k))

A second proxy is constructed from the integrated variance:

    M_t = ∫ σ² exp(ω_t) dt

These estimators are used to study scaling properties of
rough volatility models and to validate the multifractal
structure of the Log-SfBM framework across aggregation scales.

===========================================================
"""

from LogSfBM_Class import *

# ============================================================
# Base configuration (your requested parameters)
# ============================================================
params = dict(
    size=2**14,
    subsample=4,
    T=200,
    lambda2=0.05,
    H=0.01,
    M=32
)

# Instantiate model (univariate case)
model = LogS_fBM(
    H=params["H"],
    lambda2=params["lambda2"],
    T=params["T"]
)

# ============================================================
# SCENARIO 1 — Standard log-vol generation (baseline)
# ============================================================
print("\n--- Scenario 1: genlogVol (baseline) ---")
log_qv, log_mm = model.genlogVol(
    size=params["size"],
    subsample=params["subsample"],
    M=params["M"]
)

print("log_qv shape:", log_qv.shape)
print("log_mm shape:", log_mm.shape)
print("log_qv mean:", np.mean(log_qv))
print("log_mm mean:", np.mean(log_mm))


# ============================================================
# SCENARIO 2 — Larger aggregation scale (coarser volatility proxy)
# ============================================================
print("\n--- Scenario 2: genlogVol_New_perscale (scale=2) ---")
log_qv_s2, log_mm_s2 = model.genlogVol_New_perscale(
    size=params["size"],
    subsample=params["subsample"],
    M=params["M"],
    scale=2
)

print("log_qv shape:", log_qv_s2.shape)
print("log_mm shape:", log_mm_s2.shape)
print("log_qv mean:", np.mean(log_qv_s2))
print("log_mm mean:", np.mean(log_mm_s2))


# ============================================================
# SCENARIO 3 — Even finer aggregation (scale=1, sanity check)
# ============================================================
print("\n--- Scenario 3: genlogVol_New_perscale (scale=1) ---")
log_qv_s1, log_mm_s1 = model.genlogVol_New_perscale(
    size=params["size"],
    subsample=params["subsample"],
    M=params["M"],
    scale=1
)

print("log_qv shape:", log_qv_s1.shape)
print("log_mm shape:", log_mm_s1.shape)
print("log_qv mean:", np.mean(log_qv_s1))
print("log_mm mean:", np.mean(log_mm_s1))


# ============================================================
# Quick diagnostic comparison
# ============================================================
print("\n--- Diagnostics ---")
print("Difference (scale1 vs baseline QV mean):",
      np.mean(log_qv_s1) - np.mean(log_qv))