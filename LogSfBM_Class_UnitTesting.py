import numpy as np
import pytest

from LogSfBM_Class import *


# ============================================================
# Helpers
# ============================================================

def make_univariate_model():
    return LogS_fBM(H=0.1, lambda2=0.02, T=50.0)


def make_bivariate_model():
    coH = np.array([[0.1, 0.08],
                    [0.08, 0.12]])
    coint = np.array([[0.02, 0.01],
                      [0.01, 0.03]])
    return LogS_fBM(H=None, lambda2=None, T=50.0,
                    coHurst_matrix=coH,
                    cointermittency_matrix=coint)


# ============================================================
# Constructor tests
# ============================================================

def test_univariate_init():
    model = make_univariate_model()
    assert model.dimension == 1
    assert model.H > 0
    assert model.lambda2 > 0


def test_multivariate_init():
    model = make_bivariate_model()
    assert model.dimension == 2
    assert model.coHurst_matrix.shape == (2, 2)


def test_multivariate_shape_error():
    with pytest.raises(ValueError):
        LogS_fBM(H=None, lambda2=None, T=1.0,
                 coHurst_matrix=np.ones((2, 3)),
                 cointermittency_matrix=np.ones((2, 2)))


def test_asymmetry_rejection():
    with pytest.raises(ValueError):
        LogS_fBM(
            H=None,
            lambda2=None,
            T=1.0,
            coHurst_matrix=np.array([[0.1, 0.2],
                                     [0.9, 0.1]]),  # not symmetric
            cointermittency_matrix=np.eye(2)
        )


# ============================================================
# Covariance tests
# ============================================================

def test_covariance_symmetry_univariate():
    model = make_univariate_model()
    t, s = 1.0, 2.0
    c1 = model.CovarianceFunction_SfBM(t, s)
    c2 = model.CovarianceFunction_SfBM(s, t)
    assert np.isclose(c1, c2)


def test_covariance_zero_outside_T():
    model = make_univariate_model()
    t, s = 0.0, model.T + 10
    assert model.CovarianceFunction_SfBM(t, s) == 0.0


def test_cross_covariance_bivariate_shape():
    model = make_bivariate_model()
    tau = np.linspace(-5, 5, 20)
    c = model.CrossAutocovariance_mSfBM(tau)
    assert c.shape == tau.shape


def test_cross_covariance_requires_bivariate():
    model = make_univariate_model()
    with pytest.raises(ValueError):
        model.CrossAutocovariance_mSfBM(np.array([0.0]))


# ============================================================
# Gaussian simulator
# ============================================================

def test_gaussian_process_shape():
    cov = np.exp(-np.arange(50) / 10)
    path = GaussianProcessSimulation(cov, size=64)
    assert len(path) == 64
    assert np.isfinite(path).all()


# ============================================================
# Simulation tests
# ============================================================

def test_log_sfbm_univariate_shapes():
    model = make_univariate_model()
    mrw, mrm = model.LogSfBM_Simulation(size=128, subsample=2)
    assert len(mrw) == 128
    assert len(mrm) == 128


def test_returnomega_flag():
    model = make_univariate_model()
    omega = model.LogSfBM_Simulation(size=64, subsample=2, returnomega=True)
    assert omega.shape[0] == 64 * 2


def test_returnOmega_flag():
    model = make_univariate_model()
    Omega = model.LogSfBM_Simulation(size=64, subsample=2, returnOmega=True)
    assert Omega.shape[0] == 64


def test_flagm_only_mrm():
    model = make_univariate_model()
    mrm = model.LogSfBM_Simulation(size=64, subsample=2, flagm=True)
    assert len(mrm) == 64


# ============================================================
# Multivariate simulation
# ============================================================

def test_multivariate_fft_shapes():
    model = make_bivariate_model()
    mrw, mrm = model.LogSfBM_Simulation(
        size=64,
        subsample=2,
        simulation_method="fft"
    )
    assert mrw.shape[0] == 2
    assert mrm.shape[0] == 2
    assert mrw.shape[1] == 64


def test_multivariate_cholesky_shapes():
    model = make_bivariate_model()
    mrw, mrm = model.LogSfBM_Simulation(
        size=32,
        subsample=2,
        simulation_method="Cholesky"
    )
    assert mrw.shape[0] == 2


# ============================================================
# Nested model sanity
# ============================================================

def test_nested_simulation_shape():
    model = make_univariate_model()
    x, m = model.LogSfBM_Nested_Simulation(size=64, subsample=2)
    assert len(x) == 64
    assert len(m) == 64


# ============================================================
# Log-volatility estimators
# ============================================================

def test_genlogvol_shapes():
    model = make_univariate_model()
    qv, mm = model.genlogVol(size=64, subsample=2, M=4)
    assert len(qv) == len(mm)


def test_genlogvol_new_scale():
    model = make_univariate_model()
    qv, mm = model.genlogVol_New_perscale(
        size=64,
        subsample=2,
        M=4,
        scale=2
    )
    assert len(qv) == len(mm)


# ============================================================
# Numerical sanity checks (non-regression)
# ============================================================

def test_no_nan_in_simulation():
    model = make_univariate_model()
    mrw, mrm = model.LogSfBM_Simulation(size=64, subsample=2)
    assert np.isfinite(mrw).all()
    assert np.isfinite(mrm).all()