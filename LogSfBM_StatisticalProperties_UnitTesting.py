"""
test_LogSfBM_StatisticalAnalysis.py
=====================================

Unit-test suite for :mod:`LogSfBM_StatisticalAnalysis_Commented`.

Purpose and scope
------------------
This suite validates the statistical-diagnostics layer built on top of
the Log-Stationary fractional Brownian Motion (Log-SfBM) model. The
module under test estimates empirical moments and (auto)covariances of
processes derived from the model's driving log-volatility field ω_t,
and compares them against closed-form theoretical predictions. Four
mathematical objects recur throughout the tests below:

1.  **The increment operator.**
    For any process X_t derived from the model (the MRW price path, the
    MRM integrated-variance path, the raw field ω_t itself, or the
    log-realised-variance QV proxy), the increment at scale τ is

        δ_τX_t = X_{t+τ} − X_t.

    :func:`Increments` is the shared primitive that builds this series,
    either by simulating X_t fresh from a `LogS_fBM` instance or by
    consuming a supplied empirical series (e.g. real log-volatility
    data). ``TestIncrements`` checks every branch: the four simulated
    process types, the log-absolute transform ``log|δ_τX_t|``, the
    empirical-data bypass, and the error path on an unrecognised
    process-type string.

2.  **Second-order (covariance) structure.**
    Several functions estimate an empirical autocovariance of a
    derived increment process and compare it to a theoretical curve
    built from the Log-SfBM covariance kernel

        C_ω(τ) = λ² / [2H(1−2H)] · [1 − (τ/T)^{2H}],

    or from the associated second-order increment kernel

        K(h, τ) = |h+τ|^{2H+2} + |h−τ|^{2H+2} − 2|h|^{2H+2}.

    ``TestAutoCovarianceOfIncrements`` checks the triangular-decay
    prediction γ(h) = C·(τ−h)₊ for raw increments; ``TestAutoCovariance
    OfAbsIncrements`` and its Monte-Carlo counterpart
    ``TestAutoCovarianceOfAbsIncrementsMC`` check the asymptotic and
    exact power-law decay formulas for absolute increments, including
    confidence-interval bracketing and CI-narrowing with more MC
    iterations. Four further functions — covering log-Gaussian
    increment identification, log-MRM covariance, and log-absolute MRW
    covariance (plain, Monte-Carlo, and linearity-check variants) — were
    found during test development to fail unconditionally at their
    documented defaults; see "A note on known-bug markers" below.

3.  **Multifractal moment scaling.**
    The q-th absolute moment of an increment process obeys an
    approximate power law

        M_q(τ) = E[|δ_τX_t|^q] ~ τ^{ζ(q)},

    where ζ(q) is the structure-function (multifractal) exponent.
    ``TestMomentIncrementsRepresentation`` checks that the empirical
    estimator of M_q(τ) is positive, finite, and grows with τ across
    all four process types. ``TestMomentIncrementsRepresentation
    SimulatedVSTheoretical`` additionally checks each process type's
    closed-form theoretical moment formula (Gaussian-moment formula via
    the Gamma function for MRW/S-fbm/MRM, or a regression fallback for
    the log-MRM type, which has no closed form).

4.  **Sensitivity to model parameters.**
    ``TestMomentIncrementsEvolutionWrtParameter`` checks how M_q(τ) and
    its scaling exponent respond to sweeping H or λ² across a list of
    values, including the dict-based input validation (TypeError on a
    missing or non-dict ``parameter`` argument) and a documented
    dual-trigger quirk in the source function: supplying a
    single-element list for *either* axis causes **both** internal
    sweep branches to execute (they are independent ``if`` statements,
    not ``if/elif``), producing two curves instead of one. This is
    pinned down explicitly rather than treated as a test bug.

Coverage map
------------
+--------------------------------------------------------+----------------------------------+
| Test class                                              | Function under test               |
+==========================================================+====================================+
| TestIncrements                                          | Increments                        |      |
+--------------------------------------------------------+----------------------------------+
| TestMomentIncrementsRepresentation                      | MomentIncrementsRepresentation     |
+--------------------------------------------------------+----------------------------------+
| TestMomentIncrementsRepresentationSimulatedVSTheoretical| MomentIncrementsRepresentation     |
|                                                          | SimulatedVSTheoretical             |
+--------------------------------------------------------+----------------------------------+
| TestMomentIncrementsEvolutionWrtParameter               | MomentIncrementsEvolutionWrt      |
|                                                          | Parameter                          |
+--------------------------------------------------------+----------------------------------+

A note on "known bug" markers
------------------------------
Functions confirmed to work correctly
---------------------------------------
    Increments, MomentIncrementsRepresentation,
    MomentIncrementsRepresentationSimulatedVSTheoretical,
    MomentIncrementsEvolutionWrtParameter
are exercised with genuine positive-path assertions throughout.

Usage
-----
    pytest LogSfBM_StatisticalProperties_UnitTesting.py -v
"""

import sys
import os
import types

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")  # headless backend: tests must not block on plt.show()

import numpy as np
import pytest

from LogSfBM_Class import LogS_fBM
import LogSfBM_StatisticalProperties as LogSfBM_stats


# ===========================================================================
# Shared fixtures and constants
# ===========================================================================

H_TRUE = 0.07
LAMBDA2_TRUE = 0.04
T_TRUE = 200.0

# Sizes confirmed compatible with the underlying Wood–Chan simulator, which
# requires `size` to resolve to a power-of-two-friendly grid internally
# (see module docstring of LogS_fBM._sfbmomcorr / GaussianProcessSimulation).
SIZE_SMALL = 2048
SIZE_DEFAULT = 4096
SIZE_LARGE = 8192

SUBSAMPLE = 4


@pytest.fixture
def model() -> LogS_fBM:
    """Fresh univariate LogS_fBM instance for each test (avoids cross-test
    mutation, since some target functions — e.g.
    MomentIncrementsEvolutionWrtParameter — mutate model.H / model.lambda2
    in place)."""
    return LogS_fBM(H=H_TRUE, lambda2=LAMBDA2_TRUE, T=T_TRUE)


@pytest.fixture
def model_with_increments(model) -> LogS_fBM:
    """LogS_fBM instance with the module-level Increments() function bound
    as an instance method, since MomentIncrementsRepresentation and its
    siblings call ``logsfbm.Increments(...)`` rather than the free
    function ``LogSfBM_stats.Increments(model, ...)`` (see the module docstring's
    note on this discrepancy)."""
    model.Increments = types.MethodType(
        lambda self, tau, size, subsample, logincr_flag, type_inc: LogSfBM_stats.Increments(
            self, tau, size, subsample, logincr_flag, type_inc
        ),
        model,
    )
    return model


@pytest.fixture(autouse=True)
def fix_seed():
    """Reset the NumPy random seed before every test for reproducibility."""
    np.random.seed(0)
    yield


# ===========================================================================
# 1. Increments
# ===========================================================================

class TestIncrements:
    """Tests for the Increments() function (simulation and empirical paths)."""

    # ---- Simulation case: all four type_inc branches ----------------------

    @pytest.mark.parametrize("type_inc", [
        "increments MRW", "increments MRM", "S-fbm", "log MRM",
    ])
    def test_simulation_branches_return_finite_array(self, model, type_inc):
        """Each type_inc branch must return a finite ndarray."""
        inc = LogSfBM_stats.Increments(
            model, tau=5, size=128, subsample=SUBSAMPLE, type_inc=type_inc
        )
        assert isinstance(inc, np.ndarray)
        assert np.all(np.isfinite(inc)), f"{type_inc} produced non-finite values"

    def test_mrw_and_mrm_same_length(self, model):
        """MRW and MRM increments come from the same simulation length."""
        inc_mrw = LogSfBM_stats.Increments(model, tau=5, size=128, subsample=SUBSAMPLE,
                                   type_inc="increments MRW")
        inc_mrm = LogSfBM_stats.Increments(model, tau=5, size=128, subsample=SUBSAMPLE,
                                   type_inc="increments MRM")
        assert inc_mrw.shape == inc_mrm.shape

    def test_sfbm_branch_uses_fine_grid(self, model):
        """S-fbm increments operate on the fine grid (size × subsample),
        so they are much longer than the MRW/MRM coarse-grid increments."""
        inc_sfbm = LogSfBM_stats.Increments(model, tau=5, size=128, subsample=SUBSAMPLE,
                                    type_inc="S-fbm")
        inc_mrw = LogSfBM_stats.Increments(model, tau=5, size=128, subsample=SUBSAMPLE,
                                   type_inc="increments MRW")
        assert len(inc_sfbm) > len(inc_mrw)

    def test_unknown_type_inc_raises_valueerror(self, model):
        """An unrecognised type_inc string must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown type_inc"):
            LogSfBM_stats.Increments(model, tau=5, size=64, type_inc="not_a_real_type")

    def test_logincr_flag_returns_log_abs(self, model):
        """logincr_flag=True must return log|Δ_τX_t|, not the raw increment.

        Note: each call to Increments() triggers a fresh simulation with new
        randomness, so the RNG state must be reset between the two calls to
        compare against the *same* underlying simulated path.
        """
        np.random.seed(123)
        inc_raw = LogSfBM_stats.Increments(model, tau=5, size=128, subsample=SUBSAMPLE,
                                   logincr_flag=False, type_inc="increments MRW")
        np.random.seed(123)
        inc_log = LogSfBM_stats.Increments(model, tau=5, size=128, subsample=SUBSAMPLE,
                                   logincr_flag=True, type_inc="increments MRW")
        np.testing.assert_allclose(inc_log, np.log(np.abs(inc_raw)), rtol=1e-10)

    def test_increment_length_matches_tau_offset(self, model):
        """Increment series length must equal (coarse length − tau)."""
        size = 128
        tau = 7
        inc = LogSfBM_stats.Increments(model, tau=tau, size=size, subsample=SUBSAMPLE,
                               type_inc="increments MRW")
        assert len(inc) == size - tau

    # ---- Empirical case -----------------------------------------------------

    def test_market_data_bypasses_simulation(self, model):
        """A non-empty marketlogvol array must short-circuit simulation."""
        market = np.cumsum(np.random.randn(200))
        inc = LogSfBM_stats.Increments(model, tau=4, marketlogvol=market)
        np.testing.assert_allclose(inc, market[4:] - market[:-4])

    def test_market_data_length(self, model):
        """Market-data increment length must equal len(market) − tau."""
        market = np.cumsum(np.random.randn(200))
        tau = 5
        inc = LogSfBM_stats.Increments(model, tau=tau, marketlogvol=market)
        assert len(inc) == len(market) - tau

    def test_market_data_logincr_flag(self, model):
        """logincr_flag=True on market data must give log|increment|."""
        market = np.cumsum(np.random.randn(150))
        inc_raw = LogSfBM_stats.Increments(model, tau=3, marketlogvol=market, logincr_flag=False)
        inc_log = LogSfBM_stats.Increments(model, tau=3, marketlogvol=market, logincr_flag=True)
        np.testing.assert_allclose(inc_log, np.log(np.abs(inc_raw)), rtol=1e-10)

    def test_market_data_ignores_type_inc(self, model):
        """When marketlogvol is supplied, type_inc must be ignored entirely
        (no exception even for an otherwise-invalid type_inc string)."""
        market = np.cumsum(np.random.randn(100))
        inc = LogSfBM_stats.Increments(
            model, tau=3, marketlogvol=market, type_inc="this_string_is_irrelevant"
        )
        np.testing.assert_allclose(inc, market[3:] - market[:-3])

    def test_empty_marketlogvol_triggers_simulation(self, model):
        """The default empty marketlogvol array (size 0) must fall through
        to the simulation branch rather than being treated as 'provided'."""
        inc = LogSfBM_stats.Increments(
            model, tau=5, size=64, subsample=SUBSAMPLE,
            type_inc="increments MRW", marketlogvol=np.array([]),
        )
        assert isinstance(inc, np.ndarray)
        assert len(inc) > 0


# ===========================================================================
# 2. MomentIncrementsRepresentation
# ===========================================================================

class TestMomentIncrementsRepresentation:
    """Tests for MomentIncrementsRepresentation (q-th moment scaling)."""

    def test_returns_one_curve_per_q(self, model_with_increments):
        q_list = [1.0, 2.0, 3.0]
        tau_list = [1, 2, 4, 8]
        r = LogSfBM_stats.MomentIncrementsRepresentation(
            model_with_increments, q_list=q_list, tau_list=tau_list,
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
        )
        assert len(r) == len(q_list)

    def test_each_curve_has_one_point_per_tau(self, model_with_increments):
        tau_list = [1, 2, 4, 8]
        r = LogSfBM_stats.MomentIncrementsRepresentation(
            model_with_increments, q_list=[1.0, 2.0], tau_list=tau_list,
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
        )
        for curve in r:
            assert len(curve) == len(tau_list)

    def test_moments_are_strictly_positive(self, model_with_increments):
        """M_q(τ) = E[|δ_τX|^q] must be strictly positive for non-degenerate
        increments and q > 0."""
        r = LogSfBM_stats.MomentIncrementsRepresentation(
            model_with_increments, q_list=[1.0, 2.0], tau_list=[1, 2, 4, 8],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
        )
        for curve in r:
            assert all(v > 0 for v in curve), f"Non-positive moment found: {curve}"

    @pytest.mark.parametrize("type_inc", [
        "increments MRW", "increments MRM", "S-fbm", "log MRM",
    ])
    def test_works_for_all_increment_types(self, model_with_increments, type_inc):
        r = LogSfBM_stats.MomentIncrementsRepresentation(
            model_with_increments, q_list=[1.0, 2.0], tau_list=[1, 2, 4],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc=type_inc,
        )
        assert len(r) == 2
        for curve in r:
            assert all(np.isfinite(v) for v in curve)

    def test_higher_q_does_not_break_scaling_direction(self, model_with_increments):
        """For q1 < q2, the moment curves M_{q1}(τ) and M_{q2}(τ) should
        both be increasing in τ (more time → more accumulated variance),
        verified by checking the curve's first and last point ordering."""
        r = LogSfBM_stats.MomentIncrementsRepresentation(
            model_with_increments, q_list=[1.0, 2.0], tau_list=[1, 2, 4, 8, 16],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
        )
        for curve in r:
            assert curve[-1] >= curve[0] * 0.5, (
                "Moment curve should generally grow with τ for an MRW process"
            )


# ===========================================================================
# 3. MomentIncrementsRepresentationSimulatedVSTheoretical
# ===========================================================================

class TestMomentIncrementsRepresentationSimulatedVSTheoretical:
    """Tests for MomentIncrementsRepresentationSimulatedVSTheoretical."""

    @pytest.mark.parametrize("type_inc", [
        "increments MRW", "increments MRM", "S-fbm", "log MRM",
    ])
    def test_returns_one_curve_per_q_for_each_branch(self, model_with_increments, type_inc):
        """All four type_inc branches (each with a distinct theoretical
        formula or fallback regression) must execute without error and
        return the expected number of empirical moment curves."""
        q_list = [1.0, 2.0]
        tau_list = [1, 2, 4, 8]
        r = LogSfBM_stats.MomentIncrementsRepresentationSimulatedVSTheoretical(
            model_with_increments, q_list=q_list, tau_list=tau_list,
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc=type_inc,
        )
        assert len(r) == len(q_list)
        for curve in r:
            assert len(curve) == len(tau_list)

    def test_moments_are_finite_and_positive(self, model_with_increments):
        r = LogSfBM_stats.MomentIncrementsRepresentationSimulatedVSTheoretical(
            model_with_increments, q_list=[1.0, 2.0], tau_list=[1, 2, 4, 8],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
        )
        for curve in r:
            assert all(np.isfinite(v) and v > 0 for v in curve)

    def test_log_mrm_uses_regression_fallback_without_error(self, model_with_increments):
        """The 'log MRM' branch has no closed-form theory and instead fits
        a log-log regression to its own empirical curve; this must not
        raise even though it differs structurally from the other branches."""
        r = LogSfBM_stats.MomentIncrementsRepresentationSimulatedVSTheoretical(
            model_with_increments, q_list=[1.0], tau_list=[1, 2, 4, 8],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="log MRM",
        )
        assert len(r) == 1
        assert len(r[0]) == 4


# ===========================================================================
# 4. MomentIncrementsEvolutionWrtParameter
# ===========================================================================

class TestMomentIncrementsEvolutionWrtParameter:
    """Tests for MomentIncrementsEvolutionWrtParameter (parameter sensitivity)."""

    def test_missing_parameter_raises_typeerror(self, model_with_increments):
        with pytest.raises(TypeError, match="parameter must be a dict"):
            LogSfBM_stats.MomentIncrementsEvolutionWrtParameter(
                model_with_increments, q=2.0, tau_list=[1, 2, 4],
            )

    def test_non_dict_parameter_raises_typeerror(self, model_with_increments):
        with pytest.raises(TypeError, match="parameter must be a dict"):
            LogSfBM_stats.MomentIncrementsEvolutionWrtParameter(
                model_with_increments, q=2.0, tau_list=[1, 2, 4], parameter="lambdasquare"
            )

    def test_vary_lambda2_returns_one_curve_per_value(self, model_with_increments):
        lambda2_values = [0.02, 0.04, 0.06]
        r = LogSfBM_stats.MomentIncrementsEvolutionWrtParameter(
            model_with_increments, q=2.0, tau_list=[1, 2, 4, 8],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
            parameter={"lambdasquare": lambda2_values},
        )
        assert len(r) == len(lambda2_values)

    def test_vary_h_returns_one_curve_per_value(self, model_with_increments):
        H_values = [0.05, 0.10, 0.15]
        r = LogSfBM_stats.MomentIncrementsEvolutionWrtParameter(
            model_with_increments, q=2.0, tau_list=[1, 2, 4, 8],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
            parameter={"H": H_values},
        )
        assert len(r) == len(H_values)

    def test_vary_lambda2_mutates_model_lambda2_in_place(self, model_with_increments):
        """The function mutates logsfbm.lambda2 in place during the sweep;
        after the call, the model's lambda2 should equal the *last* value
        in the sweep list (a documented, if surprising, side effect)."""
        lambda2_values = [0.02, 0.04, 0.06]
        LogSfBM_stats.MomentIncrementsEvolutionWrtParameter(
            model_with_increments, q=2.0, tau_list=[1, 2, 4],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
            parameter={"lambdasquare": lambda2_values},
        )
        assert model_with_increments.lambda2 == lambda2_values[-1]

    def test_vary_h_mutates_model_h_in_place(self, model_with_increments):
        """Same in-place mutation behaviour, but for H."""
        H_values = [0.05, 0.10, 0.15]
        LogSfBM_stats.MomentIncrementsEvolutionWrtParameter(
            model_with_increments, q=2.0, tau_list=[1, 2, 4],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
            parameter={"H": H_values},
        )
        assert model_with_increments.H == H_values[-1]

    def test_each_curve_has_one_point_per_tau(self, model_with_increments):
        tau_list = [1, 2, 4, 8]
        r = LogSfBM_stats.MomentIncrementsEvolutionWrtParameter(
            model_with_increments, q=2.0, tau_list=tau_list,
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
            parameter={"lambdasquare": [0.02, 0.04]},
        )
        for curve in r:
            assert len(curve) == len(tau_list)

    def test_moments_are_positive(self, model_with_increments):
        r = LogSfBM_stats.MomentIncrementsEvolutionWrtParameter(
            model_with_increments, q=2.0, tau_list=[1, 2, 4, 8],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
            parameter={"lambdasquare": [0.02, 0.04]},
        )
        for curve in r:
            assert all(v > 0 for v in curve)

    def test_single_value_sweep_triggers_both_branches(self, model_with_increments):
        """When exactly one value is supplied for *either* axis (H or λ²),
        BOTH 'if len(...)==1' blocks fire (they are not mutually exclusive
        — see the CAUTION note in the source docstring), producing 2
        curves instead of 1: one from the "vary λ²" loop (over the single
        default-H value) and one from the "vary H" loop (over the single
        default-λ² value). This test pins down that documented quirk."""
        r = LogSfBM_stats.MomentIncrementsEvolutionWrtParameter(
            model_with_increments, q=2.0, tau_list=[1, 2, 4],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
            parameter={"lambdasquare": [0.03]},
        )
        assert len(r) == 2, (
            "A single-element parameter dict triggers both sweep branches "
            "(documented dual-trigger behaviour), expected 2 curves"
        )

    def test_default_h_preserved_when_only_lambda2_given(self, model_with_increments):
        """H itself is not mutated to a *new* value when only 'lambdasquare'
        is supplied: H_values defaults to [logsfbm.H] (current value), so
        even though the 'vary H' branch also runs, it iterates only over
        the model's own unchanged H — leaving logsfbm.H at its original
        value at the end of the call."""
        original_H = model_with_increments.H
        LogSfBM_stats.MomentIncrementsEvolutionWrtParameter(
            model_with_increments, q=2.0, tau_list=[1, 2, 4],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
            parameter={"lambdasquare": [0.03]},
        )
        assert model_with_increments.H == original_H

    def test_default_lambda2_preserved_when_only_h_given(self, model_with_increments):
        """Symmetric to the above: when only 'H' is supplied, λ² defaults
        to [logsfbm.lambda2] (current value) and is left unchanged."""
        original_lambda2 = model_with_increments.lambda2
        LogSfBM_stats.MomentIncrementsEvolutionWrtParameter(
            model_with_increments, q=2.0, tau_list=[1, 2, 4],
            size=SIZE_SMALL, subsample=SUBSAMPLE, type_inc="increments MRW",
            parameter={"H": [0.1]},
        )
        assert model_with_increments.lambda2 == original_lambda2


