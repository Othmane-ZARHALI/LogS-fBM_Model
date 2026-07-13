"""
===========================================================
File        : LogSfBM_StatisticalProperties.py
Project     : LogSfBMCore
Authors     : Othmane Zarhali
Created     : 2024
Description :
    Statistical diagnostics and empirical-vs-theoretical comparison
    tools for the Log-Stationary fractional Brownian Motion (Log-SfBM)
    model and its derived processes (MRW price path, MRM integrated
    variance, log-volatility ω, log-MRM).

    The functions in this module fall into four families:

    1.  **Increment generation**
        :func:`Increments`
            Builds raw increments Δ_τX_t = X_{t+τ} − X_t (or their
            log-absolute transform) for any of the model's processes,
            either from a fresh simulation or from supplied market data.

    2.  **Distributional diagnostics**
        :func:`LogIncrementsDistributionDensityAcrossScale`
            Visualises how the shape of the increment distribution
            evolves with the scale τ (the multifractal "fattening"
            of the tails as τ shrinks), benchmarked against the
            standard Gaussian density.

    3.  **Multifractal moment-scaling diagnostics**
        :func:`MomentIncrementsRepresentation`
        :func:`MomentIncrementsRepresentationSimulatedVSTheoretical`
        :func:`MomentIncrementsEvolutionWrtParameter`
            Estimate the q-th order structure function
            M_q(τ) = E[|δ_τX_t|^q] and its power-law scaling
            exponent ζ(q), comparing simulated moments against
            theoretical predictions and studying their sensitivity
            to the model parameters H and λ².

Mathematical background
------------------------
Throughout this module, X_t denotes one of four processes derived
from the Log-SfBM driving field ω_t:

    "increments MRW" :  X_t = price path,        dX_t = e^{ω_t/2} dW_t
    "increments MRM" :  X_t = integrated variance, dM_t = σ² e^{ω_t} dt
    "S-fbm"          :  X_t = ω_t itself (the stationary log-vol field)
    "log MRM"        :  X_t = log-realised-variance QV proxy

For a lag τ, the increment is

    δ_τX_t = X_{t+τ} − X_t

and most diagnostics in this module study either:

  - Density scaling of δ_τX_t,
  - or the q-th moment scaling of the form M_q(τ) = E[|δ_τX_t|^q] ~ τ^{ζ(q)}.

All theoretical formulas below are second-order (small-intermittency)
approximations derived from the Log-SfBM covariance kernel; they become
exact in the limit λ² → 0 and remain accurate for the typically small
λ² (≈ 0.01–0.1) observed in financial log-volatility data.

Dependencies
------------
    numpy, matplotlib, scipy.stats (gaussian_kde, norm), scipy.special
    (gamma), sklearn.linear_model.LinearRegression, math.exp

Note on `LogS_fBM.Increments`
------------------------------
Several functions below (`MomentIncrementsRepresentation`,
`MomentIncrementsRepresentationSimulatedVSTheoretical`,
`MomentIncrementsEvolutionWrtParameter`) call
``logsfbm.Increments(...)`` as a *method* of the model object. This
mirrors the module-level :func:`Increments` function defined here;
in a unified codebase the model class should expose this function as
a bound method (or the calls below should be changed to
``Increments(model=logsfbm, ...)``). The original call signatures
are preserved unchanged in this file.
===========================================================
"""

import random
from math import exp

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm
from scipy.special import gamma
from sklearn.linear_model import LinearRegression

# `color_list` is an externally defined palette (e.g. a list of matplotlib
# colour strings) used to randomly assign a distinct colour per moment-order
# curve in MomentIncrementsRepresentationSimulatedVSTheoretical. It is
# expected to be imported from the project's plotting-utilities module.
from Utils import color_list


# ===========================================================================
# Section 1 — Increment generation
# ===========================================================================

def Increments(
    model,
    tau: int,
    size: int = 4096,
    subsample: int = 4,
    logincr_flag: bool = False,
    type_inc: str = "increments MRW",
    marketlogvol: np.ndarray = np.array([])
):
    """
    Compute increments of log-volatility or multifractal processes.

    Mathematical definition:

        Δ_τ X_t = X_{t+τ} − X_t

    and optionally:

        log|Δ_τ X_t|

    where X_t is chosen depending on the model:

        - MRW:     X_t = ω_t (log-price volatility)
        - MRM:     X_t = M_t (integrated variance)
        - S-fBM:   X_t = ω_t (stationary log-volatility)
        - log MRM: X_t = log M_t proxy

    Parameters
    ----------
    model : LogSfBM_Class
        Model instance providing simulation methods.
    tau : int
        Time lag for increments.
    size : int
        Simulation size.
    subsample : int
        Subsampling factor.
    logincr_flag : bool
        If True returns log|Δ_τ X_t|.
    type_inc : str
        Type of process:
            - "increments MRW"
            - "increments MRM"
            - "S-fbm"
            - "log MRM"
    marketlogvol : np.ndarray
        If provided, uses empirical data instead of simulation.

    Returns
    -------
    np.ndarray
        Increment series Δ_τ X_t or log|Δ_τ X_t|.
    """

    # ----------------------------------------------------------
    # Empirical case
    # ----------------------------------------------------------
    # If the caller supplied a non-empty market series, bypass simulation
    # entirely and compute increments directly from the observed data.
    # This lets the same increment / covariance machinery below be reused
    # for both simulated paths and real (e.g. realised-volatility) series.
    if marketlogvol.size != 0:
        # Standard forward difference at lag τ:  Δ_τX_t = X_{t+τ} − X_t
        inc = marketlogvol[tau:] - marketlogvol[:-tau]
        # Optionally return the log-absolute transform log|Δ_τX_t|,
        # used by the log-covariance diagnostics further below.
        return np.log(np.abs(inc)) if logincr_flag else inc

    # ----------------------------------------------------------
    # Simulation case
    # ----------------------------------------------------------
    # Dispatch on `type_inc` to select which underlying process X_t the
    # increments are computed on. Each branch calls into the model's own
    # simulation routine and extracts the relevant output component.
    if type_inc == "increments MRW":
        # X_t = MRW price path. LogSfBM_Simulation(flagm=False) returns the
        # tuple (mrw, mrm); index [0] selects the price path.
        X = model.LogSfBM_Simulation(size=size, subsample=subsample, flagm=False)[0]

    elif type_inc == "increments MRM":
        # X_t = integrated-variance (MRM) path; index [1] of the same tuple.
        X = model.LogSfBM_Simulation(size=size, subsample=subsample, flagm=False)[1]

    elif type_inc == "S-fbm":
        # X_t = the raw stationary log-volatility field ω_t itself, obtained
        # by requesting returnomega=True (bypasses the price/variance
        # construction and returns the fine-grid Gaussian driving process).
        X = model.LogSfBM_Simulation(
            size=size, subsample=subsample, flagm=False, returnomega=True
        )

    elif type_inc == "log MRM":
        # X_t = the log-realised-variance QV proxy. genlogVol returns
        # (log-QV series, log-integrated-variance series); index [1]
        # selects the log integrated-variance (log-MRM) series.
        # NOTE: M=32 is hard-coded here as the QV aggregation scale.
        X = model.genlogVol(size=size, subsample=subsample, M=32)[1]

    else:
        # Defensive programming: fail loudly on an unrecognised type_inc
        # string rather than silently returning garbage.
        raise ValueError(f"Unknown type_inc: {type_inc}")

    # ----------------------------------------------------------
    # Increment operator
    # ----------------------------------------------------------
    # Same forward-difference operator as in the empirical branch above,
    # now applied to the simulated process X.
    inc = X[tau:] - X[:-tau]

    if logincr_flag:
        # log|Δ_τX_t| — used when studying multiplicative / log-Gaussian
        # increment structure rather than the raw additive increments.
        inc = np.log(np.abs(inc))

    return inc


# ===========================================================================
# Section 2 — Distributional diagnostics
# ===========================================================================

def LogIncrementsDistributionDensityAcrossScale(
    model,
    taulist,
    size: int = 4096,
    subsample: int = 4,
    logincr_flag: bool = False,
    savepath: str = None
):
    """
    Plot log-density of normalized increments across multiple scales.

    Mathematical object:

        Δ_τ X_t = X_{t+τ} − X_t

    We study the empirical PDF:

        p_τ(x) ≈ KDE({Δ_τ X_t / std})

    and plot:

        log p_τ(x)

    vertically shifted for visualization.

    A Gaussian reference is:

        p(x) = (2π)^{-1/2} exp(-x²/2)

    Parameters
    ----------
    model : LogSfBM_Class
    taulist : list[int]
        Set of scales τ.
    size : int
    subsample : int
    logincr_flag : bool
        If True uses log|Δ_τ X_t|.
    savepath : str or None
        Optional output path.

    Returns
    -------
    None
    """

    # Fixed evaluation grid for the kernel-density estimates, common to
    # all scales τ so the curves are directly comparable on one plot.
    x_values = np.linspace(-4, 4, 200)

    # Vertical offset applied to each curve so that successive scales τ
    # do not overlap on the shared log-density axis; purely cosmetic.
    vertical_shift = 2 * len(taulist)

    plt.figure(figsize=(7, 6))

    for tau in taulist:

        # Δ_τ ω_t (MRW increments)
        # Generate the raw increment series for this scale τ via the
        # Section-1 helper (always uses MRW-type increments here).
        increments = Increments(
            model=model,
            tau=tau,
            size=size,
            subsample=subsample,
            logincr_flag=logincr_flag,
            type_inc="increments MRW"
        )

        # Normalisation:
        # Δ̃_τ X_t = Δ_τ X_t / sqrt(Var(Δ_τ X_t))
        # Standardise to unit variance so that, under self-similarity, all
        # rescaled curves could in principle collapse onto a single shape;
        # deviations from the Gaussian benchmark reveal multifractality.
        increments = increments / np.std(increments)

        # KDE estimate:
        # p_hat(x) ≈ (1/Nh) Σ K((x - x_i)/h)
        # Non-parametric density estimate of the standardised increments,
        # using scipy's default (Scott's rule) bandwidth selection.
        kde = gaussian_kde(increments)
        kde_values = kde(x_values)

        # Plot on a log scale (so that heavy/fat tails appear as straighter
        # or curved deviations from the parabola of a Gaussian log-density),
        # offset vertically to separate the τ curves visually.
        plt.plot(
            x_values,
            np.log(kde_values) + vertical_shift,
            label=rf"$\tau={tau}$"
        )

        # Decrease the offset for the next (typically larger) τ so curves
        # stack from top (smallest τ) to bottom (largest τ).
        vertical_shift -= 2

    # Gaussian benchmark
    # Reference standard-normal log-density, plotted without vertical shift,
    # against which the empirical (possibly fat-tailed) curves are compared.
    plt.plot(
        x_values,
        np.log(norm.pdf(x_values)),
        color="black",
        linewidth=2,
        label="N(0,1)"
    )

    plt.xlabel("Normalized increments")
    plt.ylabel("log density (shifted)")
    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)

    plt.show()


# ===========================================================================
# Section 3 — Multifractal moment-scaling diagnostics
# ===========================================================================

def MomentIncrementsRepresentation(
    logsfbm,
    q_list,
    tau_list,
    size=4096,
    subsample=4,
    type_inc="increments MRW"
):
    """
    Estimate scaling of q-th order moments of MRW/SfBM increments
    across multiple time scales τ using Monte Carlo simulation.

    This function studies the multifractal scaling law:

        M_q(τ) = E[ |δ_τ X_t|^q ],

    where the increment process is defined as:

        δ_τ X_t = X_{t+τ} − X_t.

    The goal is to empirically verify the power-law scaling:

        M_q(τ) ~ τ^{ζ(q)},

    where ζ(q) is the structure function (multifractal spectrum).

    ------------------------------------------------------------------------
    Monte Carlo estimator
    ------------------------------------------------------------------------

    For each q and τ, we estimate:

        M_q(τ)
        ≈
        (1/N_τ) Σ_{k=1}^{N_τ} |δ_τ X_{kτ}|^q,

    where N_τ = ⌊N / τ⌋.

    ------------------------------------------------------------------------
    Log-log regression (scaling law identification)
    ------------------------------------------------------------------------

    A linear regression is performed:

        log M_q(τ) = ζ(q) log τ + c_q,

    where ζ(q) is estimated as the slope.

    ------------------------------------------------------------------------
    Parameters
    ----------
    logsfbm : LogS_fBM
        Model instance providing increment simulation.

    q_list : list of float
        Orders of moments.

    tau_list : list of int
        Time scales.

    size : int
        Simulation size.

    subsample : int
        Subsampling factor.

    type_inc : str
        Type of increments:
            - "increments MRW"
            - "increments MRM"
            - "S-fbm"
            - "log MRM"

    Returns
    -------
    list of lists
        Empirical moment curves M_q(τ) for each q.
    """

    moment_increment_curve = []

    for q in q_list:
        moment_curve_q = []

        for tau in tau_list:

            # NOTE: this calls logsfbm.Increments(...) as a bound method of
            # the model object; see the module docstring's remark on
            # exposing Increments() as a method of LogS_fBM.
            increments = logsfbm.Increments(
                tau=tau,
                size=size,
                subsample=subsample,
                logincr_flag=False,
                type_inc=type_inc
            )

            # Number of non-overlapping τ-spaced sampling points available
            # in the increment series, used to average |δ_τX|^q below.
            N_tau = int(len(increments) // tau)

            # Sample mean of |δ_τX_{kτ}|^q over k = 0, …, N_τ−1 — i.e. the
            # q-th absolute moment of the increment process at scale τ,
            # sampled at non-overlapping τ-spaced points.
            Mq_tau = np.mean([
                np.abs(increments[k * tau]) ** q
                for k in range(N_tau)
            ])

            moment_curve_q.append(Mq_tau)

        # --- scaling regression ---
        # Fit log M_q(τ) = ζ(q)·log τ + c_q via ordinary least squares;
        # the slope is the empirical structure-function exponent ζ(q).
        slope, intercept = np.polyfit(
            np.log(tau_list),
            np.log(moment_curve_q),
            1
        )

        regression_line = slope * np.log(tau_list) + intercept

        plt.scatter(
            np.log(tau_list),
            np.log(moment_curve_q),
            label=rf"$q={q}$"
        )
        plt.plot(np.log(tau_list), regression_line, color="black")

        moment_increment_curve.append(moment_curve_q)

    plt.title("Multifractal scaling of increments")
    plt.xlabel(r"$\log(\tau)$")
    plt.ylabel(r"$\log(M_q(\tau))$")
    plt.legend()
    plt.show()

    return moment_increment_curve


def MomentIncrementsRepresentationSimulatedVSTheoretical(
    logsfbm,
    q_list,
    tau_list,
    size=4096,
    subsample=4,
    type_inc="increments MRW"
):
    """
    Compare empirical and theoretical scaling of q-th order increment moments
    for MRW / SfBM / MRM models.

    This function studies:

        M_q(τ) = E[ |δ_τ X_t|^q ],

    and compares:

        (i) empirical Monte Carlo estimates
        (ii) theoretical asymptotic or exact formulas

    ------------------------------------------------------------------------
    General scaling principle
    ------------------------------------------------------------------------

        M_q(τ) ~ τ^{ζ(q)} exp( correction terms ),

    where the correction depends on:

        - fractional structure (H)
        - intermittency (λ²)
        - log-volatility covariance structure

    ------------------------------------------------------------------------
    Empirical estimator
    ------------------------------------------------------------------------

        M_q(τ)
        ≈
        (1/N_τ) Σ |δ_τ X_{kτ}|^q

    ------------------------------------------------------------------------
    Returns
    -------
    list
        Empirical moment curves for each q.
    """

    moment_increment_curve = []

    # ---------------------------------------------------------------------
    # 1. Empirical estimation
    # ---------------------------------------------------------------------
    # Identical Monte-Carlo moment-estimation loop as in
    # MomentIncrementsRepresentation, repeated here so this function is
    # self-contained and can be directly compared against the theoretical
    # curves computed in step 2 below.
    for q in q_list:
        moment_curve_q = []

        for tau in tau_list:

            increments = logsfbm.Increments(
                tau=tau,
                size=size,
                subsample=subsample,
                logincr_flag=False,
                type_inc=type_inc
            )

            N_tau = int(len(increments) // tau)

            Mq_tau = np.mean([
                np.abs(increments[k * tau]) ** q
                for k in range(N_tau)
            ])

            moment_curve_q.append(Mq_tau)

        moment_increment_curve.append(moment_curve_q)

    # Assign one random colour per moment order q purely for plot clarity;
    # `color_list` is an externally defined palette of plotting colours.
    colors = [random.choice(color_list) for _ in q_list]

    # ---------------------------------------------------------------------
    # 2. Theoretical comparisons
    # ---------------------------------------------------------------------
    if type_inc == "log MRM":

        # No closed-form theoretical moment formula is available for the
        # log-MRM increment type in this implementation; instead, a
        # log-log linear regression of the *empirical* curve itself is
        # used as a reference/benchmark line (i.e. the "theory" here is
        # simply the best-fit power law to the data, not an independent
        # prediction).
        theoretical_curves = []

        for mc in moment_increment_curve:
            X = np.log(tau_list).reshape(-1, 1)
            y = np.log(mc)

            model = LinearRegression().fit(X, y)

            slope, intercept = model.coef_[0], model.intercept_
            theoretical_curves.append(
                slope * np.log(tau_list) + intercept
            )

        for mc, tc, c, q in zip(moment_increment_curve, theoretical_curves, colors, q_list):
            plt.scatter(np.log(tau_list), np.log(mc), label=f"q={q}", color=c)
            plt.plot(np.log(tau_list), tc, color="gray")

    # ---------------------------------------------------------------------
    elif type_inc == "increments MRW":

        # Closed-form theoretical log-moment for MRW increments, derived
        # from the Gaussian multiplicative-noise representation of the MRW:
        # a deterministic τ-scaling term (p/2)log τ, an intermittency
        # correction from the variance of the log-vol field (proportional
        # to p+p² and the second-order increment kernel scaled by T^{2H}),
        # and a Gaussian-moment normalisation term involving the Gamma
        # function (since |Z|^p for Z~N(0,1) has a known moment formula).
        def theory(tau, p):
            return (
                (p / 2) * np.log(tau)
                + logsfbm.lambda2 / (4 * tau ** 2) * (p + p**2)
                * (
                    (tau**2) / (2 * logsfbm.H * (1 - 2 * logsfbm.H))
                    - (2 * abs(tau) ** (2 * logsfbm.H + 2))
                    / (4 * logsfbm.H * logsfbm.T ** (2 * logsfbm.H)
                       * (1 - 4 * logsfbm.H**2) * (1 + logsfbm.H))
                )
                + (p / 2) * np.log(2)
                + np.log(gamma((p + 1) / 2) / np.sqrt(np.pi))
            )

        theoretical_curves = [
            [theory(tau, q) for tau in tau_list]
            for q in q_list
        ]

        for mc, tc, c, q in zip(moment_increment_curve, theoretical_curves, colors, q_list):
            plt.scatter(np.log(tau_list), np.log(mc), label=f"q={q}", color=c)
            plt.plot(np.log(tau_list), tc, color="black")

    # ---------------------------------------------------------------------
    elif type_inc == "increments MRM":

        # Analogous closed-form formula for MRM (integrated-variance)
        # increments: leading term p·log τ (note: not p/2, reflecting the
        # MRM's exp(ω) rather than exp(ω/2) scaling), plus the same
        # intermittency-correction structure as above but with the
        # prefactor halved (2τ² instead of 4τ²).
        def theory(tau, p):
            return (
                p * np.log(tau)
                + logsfbm.lambda2 / (2 * tau**2) * (p + p**2)
                * (
                    (tau**2) / (2 * logsfbm.H * (1 - 2 * logsfbm.H))
                    - (2 * abs(tau) ** (2 * logsfbm.H + 2))
                    / (4 * logsfbm.H * logsfbm.T**(2 * logsfbm.H)
                       * (1 - 4 * logsfbm.H**2) * (1 + logsfbm.H))
                )
            )

        theoretical_curves = [
            [theory(tau, q) for tau in tau_list]
            for q in q_list
        ]

        for mc, tc, c, q in zip(moment_increment_curve, theoretical_curves, colors, q_list):
            plt.scatter(np.log(tau_list), np.log(mc), label=f"q={q}", color=c)
            plt.plot(np.log(tau_list), tc, color="black")

    # ---------------------------------------------------------------------
    elif type_inc == "S-fbm":

        # For the raw log-vol field ω_t (S-fbm), the moment formula follows
        # directly from ω_t being Gaussian with variance
        # λ²/(H(1−2H))·(τ/T)^{2H}: the p-th absolute moment of a centred
        # Gaussian with that variance has a closed-form expression
        # involving the Gamma function, exactly as coded here.
        def theory(tau, p):
            return np.log(
                (
                    (logsfbm.lambda2 / (logsfbm.H * (1 - 2 * logsfbm.H))
                     * (tau / logsfbm.T) ** (2 * logsfbm.H))
                    ** (p / 2)
                )
                * 2 ** (p / 2)
                * gamma((p + 1) / 2)
                / np.sqrt(np.pi)
            )

        theoretical_curves = [
            [theory(tau, q) for tau in tau_list]
            for q in q_list
        ]

        for mc, tc, c, q in zip(moment_increment_curve, theoretical_curves, colors, q_list):
            plt.scatter(np.log(tau_list), np.log(mc), label=f"q={q}", color=c)
            plt.plot(np.log(tau_list), tc, color="black")

    # ---------------------------------------------------------------------
    plt.xlabel(r"$\log(\tau)$")
    plt.ylabel(r"$\log(M_q(\tau))$")
    plt.legend()
    plt.title("Simulated vs Theoretical moment scaling")
    plt.show()

    return moment_increment_curve

def MomentIncrementsRepresentationSimulatedVSTheoretical(
    logsfbm,
    q_list,
    tau_list,
    size=4096,
    subsample=4,
    type_inc="increments MRW"
):
    """
    Compare empirical and theoretical scaling of q-th order increment moments
    for MRW / SfBM / MRM models.

    This function studies:

        M_q(τ) = E[ |δ_τ X_t|^q ],

    and compares:

        (i) empirical Monte Carlo estimates
        (ii) theoretical asymptotic or exact formulas

    ------------------------------------------------------------------------
    General scaling principle
    ------------------------------------------------------------------------

        M_q(τ) ~ τ^{ζ(q)} exp( correction terms ),

    where the correction depends on:

        - fractional structure (H)
        - intermittency (λ²)
        - log-volatility covariance structure

    ------------------------------------------------------------------------
    Empirical estimator
    ------------------------------------------------------------------------

        M_q(τ)
        ≈
        (1/N_τ) Σ |δ_τ X_{kτ}|^q

    ------------------------------------------------------------------------
    Returns
    -------
    list
        Empirical moment curves for each q.
    """

    moment_increment_curve = []

    # ---------------------------------------------------------------------
    # 1. Empirical estimation
    # ---------------------------------------------------------------------
    # Identical Monte-Carlo moment-estimation loop as in
    # MomentIncrementsRepresentation, repeated here so this function is
    # self-contained and can be directly compared against the theoretical
    # curves computed in step 2 below.
    for q in q_list:
        moment_curve_q = []

        for tau in tau_list:

            increments = logsfbm.Increments(
                tau=tau,
                size=size,
                subsample=subsample,
                logincr_flag=False,
                type_inc=type_inc
            )

            N_tau = int(len(increments) // tau)

            Mq_tau = np.mean([
                np.abs(increments[k * tau]) ** q
                for k in range(N_tau)
            ])

            moment_curve_q.append(Mq_tau)

        moment_increment_curve.append(moment_curve_q)

    # Assign one random colour per moment order q purely for plot clarity;
    # `color_list` is an externally defined palette of plotting colours.
    colors = [random.choice(color_list) for _ in q_list]

    # ---------------------------------------------------------------------
    # 2. Theoretical comparisons
    # ---------------------------------------------------------------------
    if type_inc == "log MRM":

        # No closed-form theoretical moment formula is available for the
        # log-MRM increment type in this implementation; instead, a
        # log-log linear regression of the *empirical* curve itself is
        # used as a reference/benchmark line (i.e. the "theory" here is
        # simply the best-fit power law to the data, not an independent
        # prediction).
        theoretical_curves = []

        for mc in moment_increment_curve:
            X = np.log(tau_list).reshape(-1, 1)
            y = np.log(mc)

            model = LinearRegression().fit(X, y)

            slope, intercept = model.coef_[0], model.intercept_
            theoretical_curves.append(
                slope * np.log(tau_list) + intercept
            )

        for mc, tc, c, q in zip(moment_increment_curve, theoretical_curves, colors, q_list):
            plt.scatter(np.log(tau_list), np.log(mc), label=f"q={q}", color=c)
            plt.plot(np.log(tau_list), tc, color="gray")

    # ---------------------------------------------------------------------
    elif type_inc == "increments MRW":

        # Closed-form theoretical log-moment for MRW increments, derived
        # from the Gaussian multiplicative-noise representation of the MRW:
        # a deterministic τ-scaling term (p/2)log τ, an intermittency
        # correction from the variance of the log-vol field with prefactor
        # (p²/4 − p/2) — scaling the second-order increment
        # kernel, and a Gaussian-moment normalisation term involving the
        # Gamma function (since |Z|^p for Z~N(0,1) has a known moment
        # formula). The (p²/4 − p/2) coefficient comes from expanding
        # E[exp(p·ω/2)] for ω ~ N(μ, σ²): the cumulant-generating-function
        # quadratic term contributes p²/4 from σ², while the cross-term
        # with the compensating mean correction −σ²/4 contributes −p/2 —
        # together giving p²/4 − p/2.
        def theory(tau, p):
            return (
                (p / 2) * np.log(tau)
                + logsfbm.lambda2 / tau ** 2 * (p**2 / 4 - p / 2)
                * (
                    (tau**2) / (2 * logsfbm.H * (1 - 2 * logsfbm.H))
                    - (2 * abs(tau) ** (2 * logsfbm.H + 2))
                    / (4 * logsfbm.H * logsfbm.T ** (2 * logsfbm.H)
                       * (1 - 4 * logsfbm.H**2) * (1 + logsfbm.H))
                )
                + (p / 2) * np.log(2)
                + np.log(gamma((p + 1) / 2) / np.sqrt(np.pi))
            )

        theoretical_curves = [
            [theory(tau, q) for tau in tau_list]
            for q in q_list
        ]

        for mc, tc, c, q in zip(moment_increment_curve, theoretical_curves, colors, q_list):
            plt.scatter(np.log(tau_list), np.log(mc), label=f"q={q}", color=c)
            plt.plot(np.log(tau_list), tc, color="black")

    # ---------------------------------------------------------------------
    elif type_inc == "increments MRM":

        # Analogous closed-form formula for MRM (integrated-variance)
        # increments: leading term p·log τ (note: not p/2, reflecting the
        # MRM's exp(ω) rather than exp(ω/2) scaling), plus the same
        # second-order increment kernel as the MRW case but now with
        # prefactor (p² − p) — analogous to the MRW
        # correction above but without the extra 1/4 factor (since the
        # MRM uses the full ω rather than ω/2, the cumulant expansion of
        # E[exp(p·ω)] yields p² for the quadratic term and a compensating
        # −p from the mean correction, giving p² − p rather than p²/4 − p/2).
        def theory(tau, p):
            return (
                p * np.log(tau)
                + logsfbm.lambda2 / (2 * tau**2) * (p**2 - p)
                * (
                    (tau**2) / (2 * logsfbm.H * (1 - 2 * logsfbm.H))
                    - (2 * abs(tau) ** (2 * logsfbm.H + 2))
                    / (4 * logsfbm.H * logsfbm.T**(2 * logsfbm.H)
                       * (1 - 4 * logsfbm.H**2) * (1 + logsfbm.H))
                )
            )

        theoretical_curves = [
            [theory(tau, q) for tau in tau_list]
            for q in q_list
        ]

        for mc, tc, c, q in zip(moment_increment_curve, theoretical_curves, colors, q_list):
            plt.scatter(np.log(tau_list), np.log(mc), label=f"q={q}", color=c)
            plt.plot(np.log(tau_list), tc, color="black")

    # ---------------------------------------------------------------------
    elif type_inc == "S-fbm":

        # For the raw log-vol field ω_t (S-fbm), the moment formula follows
        # directly from ω_t being Gaussian with variance
        # λ²/(H(1−2H))·(τ/T)^{2H}: the p-th absolute moment of a centred
        # Gaussian with that variance has a closed-form expression
        # involving the Gamma function, exactly as coded here.
        def theory(tau, p):
            return np.log(
                (
                    (logsfbm.lambda2 / (logsfbm.H * (1 - 2 * logsfbm.H))
                     * (tau / logsfbm.T) ** (2 * logsfbm.H))
                    ** (p / 2)
                )
                * 2 ** (p / 2)
                * gamma((p + 1) / 2)
                / np.sqrt(np.pi)
            )

        theoretical_curves = [
            [theory(tau, q) for tau in tau_list]
            for q in q_list
        ]

        for mc, tc, c, q in zip(moment_increment_curve, theoretical_curves, colors, q_list):
            plt.scatter(np.log(tau_list), np.log(mc), label=f"q={q}", color=c)
            plt.plot(np.log(tau_list), tc, color="black")

    # ---------------------------------------------------------------------
    plt.xlabel(r"$\log(\tau)$")
    plt.ylabel(r"$\log(M_q(\tau))$")
    plt.legend()
    plt.title("Simulated vs Theoretical moment scaling")
    plt.show()

    return moment_increment_curve

def MomentIncrementsEvolutionWrtParameter(
    logsfbm,
    q,
    tau_list,
    size=4096,
    subsample=4,
    type_inc="Distribution",
    parameter=None
):
    """
    Study the evolution of q-th order structure functions of increments
    with respect to model parameters (H or λ²) in a Log-SfBM / MRW framework.

    ------------------------------------------------------------------------
    OBJECTIVE
    ------------------------------------------------------------------------

    This function investigates how the scaling law

        M_q(τ) = E[ |δ_τ X_t|^q ]

    depends on model parameters:

        - H      (Hurst exponent)
        - λ²     (intermittency / log-volatility strength)

    where increments are defined as:

        δ_τ X_t = X_{t+τ} − X_t.

    The goal is to quantify how the multifractal spectrum ζ(q)
    changes when varying model parameters.

    ------------------------------------------------------------------------
    EMPIRICAL ESTIMATOR
    ------------------------------------------------------------------------

    For each parameter value θ and scale τ:

        M_q^(θ)(τ)
        ≈
        (1/N_τ) Σ_{k=1}^{N_τ} |δ_τ X_{kτ}^{(θ)}|^q,

    where X^{(θ)} is the simulated process under parameter θ.

    ------------------------------------------------------------------------
    SCALING ANALYSIS
    ------------------------------------------------------------------------

    A log-log regression is performed:

        log M_q^(θ)(τ) = ζ_θ(q) log τ + c_θ(q),

    allowing comparison of scaling exponents across parameter regimes.

    ------------------------------------------------------------------------
    PARAMETERS
    ------------------------------------------------------------------------

    logsfbm : LogS_fBM
        Model instance.

    q : float
        Moment order.

    tau_list : list of int
        Set of time scales.

    size : int
        Simulation size.

    subsample : int
        Subsampling factor.

    type_inc : str
        Increment type passed to simulation.

    parameter : dict
        Dictionary specifying parameter variation:

            - {"H": list of H values}
            - {"lambdasquare": list of λ² values}

        Exactly one parameter is fixed, the other varies.

    ------------------------------------------------------------------------
    RETURNS
    ------------------------------------------------------------------------

    list of lists
        Each list contains M_q(τ) for a given parameter value.
    """

    # Guard against missing or malformed `parameter` argument: the caller
    # must specify which model parameter to sweep via a dict.
    if parameter is None or not isinstance(parameter, dict):
        raise TypeError(
            "MomentIncrementsEvolutionWrtParameter: "
            "parameter must be a dict with keys 'H' or 'lambdasquare'."
        )

    moment_increment_curve = []

    # If a key is absent, default to the model's *current* value as a
    # single-element list, effectively "fixing" that parameter at its
    # current setting while the other one is swept.
    H_values = parameter.get("H", [logsfbm.H])
    lambda2_values = parameter.get("lambdasquare", [logsfbm.lambda2])

    # ================================================================
    # CASE 1 — VARY λ² (H fixed)
    # ================================================================
    # Triggered when exactly one H value was supplied (i.e. H is held
    # fixed) — λ² is then swept over `lambda2_values`.
    if len(H_values) == 1:

        # Mutate the model's H attribute in place to the (single) fixed
        # value before sweeping λ².
        logsfbm.H = H_values[0]

        for lambda2 in lambda2_values:

            # Mutate the model's λ² attribute in place for this sweep
            # iteration; subsequent calls to logsfbm.Increments(...) will
            # simulate under this new parameter value.
            logsfbm.lambda2 = lambda2
            moment_curve_q = []

            for tau in tau_list:

                increments = logsfbm.Increments(
                    tau=tau,
                    size=size,
                    subsample=subsample,
                    logincr_flag=False,
                    type_inc=type_inc
                )

                N_tau = len(increments) // tau

                Mq_tau = np.mean([
                    abs(increments[k * tau]) ** q
                    for k in range(N_tau)
                ])

                moment_curve_q.append(Mq_tau)

            # --- scaling regression ---
            # Empirical structure-function exponent ζ_θ(q) for this λ²
            # value, fitted exactly as in MomentIncrementsRepresentation.
            slope, intercept = np.polyfit(np.log(tau_list),np.log(moment_curve_q),1)

            regression_line = slope * np.log(tau_list) + intercept

            plt.scatter(
                np.log(tau_list),
                np.log(moment_curve_q),
                label=rf"$\lambda^2 = {lambda2:.3f}$"
            )
            plt.plot(np.log(tau_list), regression_line, color="gray")

            moment_increment_curve.append(moment_curve_q)

        plt.title(f"Moment scaling vs λ² ({type_inc})")
        plt.xlabel(r"$\log(\tau)$")
        plt.ylabel(r"$\log(M_q(\tau))$")
        plt.legend()
        plt.show()

    # ================================================================
    # CASE 2 — VARY H (λ² fixed)
    # ================================================================
    # Triggered when exactly one λ² value was supplied — H is then swept
    # over `H_values`. NOTE: both blocks can execute sequentially if the
    # caller passes single-element lists for *both* parameters (i.e. this
    # is not a mutually exclusive if/elif), which would re-run/append a
    # second sweep on top of the first; this mirrors the original logic.
    if len(lambda2_values) == 1:

        # Fix λ² at its single supplied (or default) value before
        # sweeping H.
        logsfbm.lambda2 = lambda2_values[0]

        for H in H_values:

            # Mutate the model's H attribute in place for this sweep
            # iteration.
            logsfbm.H = H
            moment_curve_q = []

            for tau in tau_list:

                increments = logsfbm.Increments(
                    tau=tau,
                    size=size,
                    subsample=subsample,
                    logincr_flag=False,
                    type_inc=type_inc
                )

                N_tau = len(increments) // tau

                Mq_tau = np.mean([
                    abs(increments[k * tau]) ** q
                    for k in range(N_tau)
                ])

                moment_curve_q.append(Mq_tau)

            # --- scaling regression ---
            # Empirical structure-function exponent ζ_θ(q) for this H value.
            slope, intercept = np.polyfit(np.log(tau_list),np.log(moment_curve_q),1)

            regression_line = slope * np.log(tau_list) + intercept

            plt.scatter(
                np.log(tau_list),
                np.log(moment_curve_q),
                label=rf"$H = {H:.3f}$"
            )
            plt.plot(np.log(tau_list), regression_line, color="gray")

            moment_increment_curve.append(moment_curve_q)

        plt.title(f"Moment scaling vs H ({type_inc})")
        plt.xlabel(r"$\log(\tau)$")
        plt.ylabel(r"$\log(M_q(\tau))$")
        plt.legend()
        plt.show()

    return moment_increment_curve
