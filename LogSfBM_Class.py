"""
===========================================================
File        : LogS_fBM.py
Project     : LogSfBMCore
Authors     : Othmane Zarhali, Jean-François Muzy
Created     : 2023
Description :
    Implementation of the Log-Stationary Fractional Brownian
    Motion (Log-SfBM) model, both in the univariate and
    multivariate settings.

    This module defines:
      - The covariance structure of the Log-SfBM driving process ω.
      - Gaussian process simulation via the Wood–Chan circulant-
        embedding (FFT) method.
      - Simulation of the associated Multifractal Random Walk (MRW)
        and Multifractal Random Measure (MRM) paths.
      - Generation of the log-realised-volatility proxy via
        quadratic-variation aggregation.

Mathematical background
-----------------------
The Log-SfBM is a continuous-time stochastic process whose log-volatility
ω_t is a centred stationary Gaussian process with covariance

    C_ω(t, s) = λ² / [2H(1 − 2H)]
                × [1 − |t − s|^{2H} / T^{2H}]  for |t − s| < T

where:
  H ∈ (0, 1/2)  — Hurst roughness exponent (smaller H → rougher paths),
  λ²             — intermittency coefficient (controls log-vol variance),
  T              — integral (decorrelation) scale.

The associated price process X_t and integrated-variance M_t are:

    dX_t = exp(ω_t / 2) dW_t        (MRW, Multifractal Random Walk)
    dM_t = σ² exp(ω_t) dt           (MRM, Multifractal Random Measure)

The multivariate extension replaces (H, λ²) with a co-Hurst matrix H_{ij}
and a cointermittency matrix Λ_{ij}, so that the cross-covariance between
components i and j depends on their joint roughness exponent H_{ij}.

References
----------
Muzy, J.-F., Bacry, E. (2002). "Multifractal stationary random measures and
    multifractal random walks."  Phys. Rev. E, 66, 056121.
Zarhali, O., Muzy, J.-F. (2023). "Log-SfBM core implementation."
    Internal report.
Bolko, A. E. et al. (2020). "A GMM approach to estimate the roughness of
    stochastic volatility." arXiv:2010.04610.
===========================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from math import sqrt
from scipy.fft import fft, ifft, ifftshift


# ---------------------------------------------------------------------------
# Utility: Wood–Chan circulant-embedding Gaussian process simulator
# ---------------------------------------------------------------------------

def GaussianProcessSimulation(covariance: np.ndarray, size: int) -> np.ndarray:
    """Simulate a stationary Gaussian process via the Wood–Chan FFT method.

    The Wood–Chan (circulant-embedding) algorithm embeds the target
    Toeplitz covariance matrix into a larger circulant matrix of size
    M = 2·2^⌊log₂(size)⌋.  Drawing a complex-Gaussian random vector in the
    frequency domain and applying the inverse FFT yields a path whose
    time-domain covariance matches the prescribed sequence exactly (up to
    machine precision), provided the circulant's eigenvalues are non-negative.

    Algorithm
    ---------
    Given the auto-covariance sequence c[0], c[1], …, c[K−1]:

    1.  Pad to length m = 2^⌊log₂(size)⌋ + 1 and build the circulant row
            r = [c₀, c₁, …, c_{m−1}, c_{m−2}, …, c₁]   (length M = 2m)

    2.  Compute the eigenvalues: λ_k = FFT(r)[k]  (must be ≥ 0).

    3.  Draw i.i.d. standard normals u_k, v_k ∈ ℝ and set
            Z_k = √λ_k · (u_k + i·v_k) / √2

    4.  Return  x = Re(IFFT(Z)) · √M   (properly scaled output path).

    Parameters
    ----------
    covariance : array_like, shape (K,)
        Auto-covariance sequence c[0], c[1], …, c[K−1].  Must satisfy
        c[k] ≥ |c[0]| for all k (valid p.s.d. Toeplitz matrix).
        Automatically zero-padded when K < m + 1.
    size : int
        Desired number of time steps in the output path.

    Returns
    -------
    path : ndarray, shape (size,)
        One realisation of the stationary Gaussian process.

    Notes
    -----
    Small negative eigenvalues arising from floating-point rounding are
    handled implicitly by the ``np.sqrt(fftcorr + 0j)`` call (complex
    square root yields a negligible imaginary part that is discarded by
    the ``np.real`` in step 4).
    """
    # Step 1 – choose the circulant block length (nearest lower power of 2)
    m = int(2 ** np.floor(np.log2(size)))
    M = 2 * m

    # Zero-pad or truncate the covariance to length m + 1
    covariance = np.concatenate(
        [covariance, np.zeros(m + 1 - len(covariance))]
    )

    # Build the circulant row: [c₀, …, c_m, c_{m-1}, …, c₁]
    circulant_row = np.concatenate([covariance, np.flip(covariance[1:-1])])

    # Step 2 – eigenvalues of the circulant (= DFT of the circulant row)
    fft_cov = np.real(np.fft.fft(circulant_row))

    # Step 3 – complex-Gaussian innovation in the frequency domain
    u = np.random.normal(size=M)
    v = np.random.normal(size=M) * 1j
    fft_cov = np.sqrt(fft_cov + 0j) * (u + v) / np.sqrt(2)

    # Step 4 – back to the time domain and rescale
    path = np.real(np.fft.ifft(fft_cov))
    return path[:size] * np.sqrt(2 * M)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LogS_fBM:
    """Log-Stationary Fractional Brownian Motion (Log-SfBM) model.

    Supports both **univariate** and **multivariate** settings.

    Univariate model
    ~~~~~~~~~~~~~~~~
    The log-volatility driving process ω_t is a centred stationary Gaussian
    process with auto-covariance

    .. math::

        C_\\omega(\\tau) = \\frac{\\lambda^2}{2H(1-2H)}
            \\left[1 - \\left(\\frac{|\\tau|}{T}\\right)^{2H}\\right]
            \\mathbf{1}_{|\\tau| < T}

    The associated price process (MRW) and integrated variance (MRM) are

    .. math::

        X_t &= \\int_0^t e^{\\omega_s/2}\\, dW_s \\quad \\text{(MRW)}\\\\
        M_t &= \\sigma^2 \\int_0^t e^{\\omega_s}\\, ds  \\quad \\text{(MRM)}

    Multivariate model
    ~~~~~~~~~~~~~~~~~~
    The dimension-d generalisation replaces the scalar pair (H, λ²) with:

    * **co-Hurst matrix** ``coHurst_matrix`` Hᵢⱼ (symmetric, d × d):
      the cross-scaling exponent between components i and j.
    * **cointermittency matrix** ``cointermittency_matrix`` Λᵢⱼ (symmetric,
      d × d): the cross-intermittency between components i and j.

    The marginal covariance of component i is recovered by setting i = j.

    Parameters
    ----------
    H : float
        Hurst exponent for the univariate model.
        Must satisfy 0 < H < 1/2.  Ignored in the multivariate case.
    lambda2 : float
        Intermittency coefficient λ² for the univariate model.
        Ignored in the multivariate case.
    T : float
        Integral (decorrelation) scale.  The covariance of ω vanishes for
        |τ| ≥ T.
    coHurst_matrix : array_like of shape (d, d) or None
        Co-Hurst exponent matrix for the multivariate model.
        Must be symmetric and square.  ``None`` triggers the univariate path.
    cointermittency_matrix : array_like of shape (d, d) or None
        Cointermittency matrix for the multivariate model.
        Must be symmetric and square.  ``None`` triggers the univariate path.

    Attributes
    ----------
    H : float or None
        Hurst exponent (univariate) or None (multivariate).
    lambda2 : float or None
        Intermittency coefficient (univariate) or None (multivariate).
    T : float
        Integral scale.
    coHurst_matrix : ndarray or None
        Co-Hurst matrix (multivariate) or None (univariate).
    cointermittency_matrix : ndarray or None
        Cointermittency matrix (multivariate) or None (univariate).
    dimension : int
        Number of components (1 for univariate).

    Raises
    ------
    TypeError
        If scalar H / lambda2 are provided in the multivariate case, or
        if matrix-valued H / lambda2 are provided in the univariate case.
    ValueError
        If the co-Hurst or cointermittency matrices are not square and
        symmetric, or if they have mismatched dimensions.

    Examples
    --------
    Univariate:

    >>> model = LogS_fBM(H=0.07, lambda2=0.04, T=200)
    >>> mrw, mrm = model.LogSfBM_Simulation(size=4096)

    Bivariate:

    >>> import numpy as np
    >>> coH = np.array([[0.07, 0.05], [0.05, 0.09]])
    >>> coint = np.array([[0.04, 0.02], [0.02, 0.05]])
    >>> model = LogS_fBM(H=None, lambda2=None, T=200,
    ...                   coHurst_matrix=coH,
    ...                   cointermittency_matrix=coint)
    >>> mrw, mrm = model.LogSfBM_Simulation(size=512, simulation_method="fft")
    """

    def __init__(
        self,
        H: float = 0.0,
        lambda2: float = 0.02,
        T: float = 200.0,
        coHurst_matrix: np.ndarray | None = None,
        cointermittency_matrix: np.ndarray | None = None,
    ) -> None:
        # Store the integral scale (shared by univariate and multivariate)
        self.T = T

        # ------------------------------------------------------------------
        # Case 1 – univariate model (both matrices absent)
        # ------------------------------------------------------------------
        if coHurst_matrix is None and cointermittency_matrix is None:
            if not np.isscalar(H):
                raise TypeError(
                    "Univariate case: H must be a scalar, "
                    f"but got type {type(H).__name__}."
                )
            if not np.isscalar(lambda2):
                raise TypeError(
                    "Univariate case: lambda2 must be a scalar, "
                    f"but got type {type(lambda2).__name__}."
                )
            self.H = H
            self.lambda2 = lambda2
            self.coHurst_matrix = None
            self.cointermittency_matrix = None
            self.dimension = 1

        # ------------------------------------------------------------------
        # Case 2 – multivariate model (at least one matrix provided)
        # ------------------------------------------------------------------
        else:
            if np.isscalar(H) and H != 0.0:
                raise TypeError(
                    "Multivariate case: pass H via coHurst_matrix, "
                    "not as a scalar."
                )
            if np.isscalar(lambda2) and lambda2 != 0.02:
                raise TypeError(
                    "Multivariate case: pass lambda2 via "
                    "cointermittency_matrix, not as a scalar."
                )

            # Validate and store the co-Hurst matrix
            if coHurst_matrix is not None:
                coHurst_matrix = np.asarray(coHurst_matrix, dtype=float)
                if coHurst_matrix.ndim != 2 or (
                    coHurst_matrix.shape[0] != coHurst_matrix.shape[1]
                ):
                    raise ValueError(
                        "coHurst_matrix must be a square 2-D matrix; "
                        f"got shape {coHurst_matrix.shape}."
                    )
                if not np.allclose(coHurst_matrix, coHurst_matrix.T):
                    raise ValueError(
                        "coHurst_matrix must be symmetric."
                    )
                self.dimension = coHurst_matrix.shape[0]

            # Validate and store the cointermittency matrix
            if cointermittency_matrix is not None:
                cointermittency_matrix = np.asarray(
                    cointermittency_matrix, dtype=float
                )
                if cointermittency_matrix.ndim != 2 or (
                    cointermittency_matrix.shape[0]
                    != cointermittency_matrix.shape[1]
                ):
                    raise ValueError(
                        "cointermittency_matrix must be a square 2-D matrix; "
                        f"got shape {cointermittency_matrix.shape}."
                    )
                if not np.allclose(
                    cointermittency_matrix, cointermittency_matrix.T
                ):
                    raise ValueError(
                        "cointermittency_matrix must be symmetric."
                    )
                # Cross-check dimensions when both matrices are supplied
                if hasattr(self, "dimension") and (
                    self.dimension != cointermittency_matrix.shape[0]
                ):
                    raise ValueError(
                        "coHurst_matrix and cointermittency_matrix must have "
                        f"the same dimension; got {self.dimension} vs "
                        f"{cointermittency_matrix.shape[0]}."
                    )
                self.dimension = cointermittency_matrix.shape[0]

            # In the multivariate case the scalar parameters are unused
            self.H = None
            self.lambda2 = None
            self.coHurst_matrix = coHurst_matrix
            self.cointermittency_matrix = cointermittency_matrix

    # -----------------------------------------------------------------------
    # Section 1 – Theoretical covariance functions
    # -----------------------------------------------------------------------

    def CovarianceFunction_SfBM(
        self,
        t: float,
        s: float,
        marginal: int | None = None,
    ) -> float:
        """Theoretical auto-covariance of the Log-SfBM driving process ω.

        For the **univariate** model the covariance between times t and s is

        .. math::

            C_\\omega(t, s) = \\frac{\\lambda^2}{2H(1-2H)}
                \\left[1 - \\frac{|t-s|^{2H}}{T^{2H}}\\right]
                \\mathbf{1}_{|t-s| < T}

        This is a stationary function of the lag τ = |t − s| only.

        For the **multivariate** model the marginal covariance of component
        ``marginal`` (i.e. the diagonal block Cᵢᵢ) is obtained by replacing
        (H, λ²) with (Hᵢᵢ, Λᵢᵢ) from the corresponding matrices.

        Parameters
        ----------
        t : float or ndarray
            First time argument.
        s : float or ndarray
            Second time argument.
        marginal : int or None, optional
            Index of the marginal component (0-based) to use in the
            multivariate setting.  When ``None`` the method uses
            ``self.H`` and ``self.lambda2`` directly (univariate case).

        Returns
        -------
        cov : float or ndarray
            Covariance value C_ω(t, s).
        """
        if marginal is None:
            # --- Univariate path: use self.H and self.lambda2 ---------------
            H = self.H
            lam2 = self.lambda2
        else:
            # --- Multivariate path: read the marginal from the matrices -----
            H = self.coHurst_matrix[marginal, marginal]
            lam2 = self.cointermittency_matrix[marginal, marginal]

        # Stationary covariance kernel:  K · [1 − (|τ|/T)^{2H}] · 1_{|τ| < T}
        K = lam2 / (2 * H * (1 - 2 * H))
        tau = np.abs(t - s)
        return K * (1 - tau ** (2 * H) / self.T ** (2 * H)) * (tau < self.T)

    def CrossAutocovariance_mSfBM(
        self,
        tau: float | np.ndarray,
        marginal: int | None = None,
    ) -> float | np.ndarray:
        """Cross-autocovariance of the bivariate multifractal Log-SfBM at lag τ.

        For the **off-diagonal** (cross) covariance between component 1 and
        component 2 (``marginal=None``) the formula is

        .. math::

            C_{12}(\\tau) = \\lambda_{12} \\xi_{12}
                \\Bigl[
                    \\frac{1}{2H_{12}} - \\frac{1}{2\\bar{H}-1}
                    + \\frac{|\\tau|}{T}\\left(\\frac{-1}{2H_{12}-1}
                      + \\frac{1}{2\\bar{H}-1}\\right)
                    + \\left(\\frac{|\\tau|}{T}\\right)^{2H_{12}}
                      \\left(\\frac{1}{2H_{12}-1} - \\frac{1}{2H_{12}}\\right)
                \\Bigr]
                \\mathbf{1}_{|\\tau| \\leq T}

        where:
          - :math:`\\bar{H} = (H_{11} + H_{22}) / 2` is the mean Hurst exponent,
          - :math:`H_{12}` is the off-diagonal co-Hurst exponent,
          - :math:`\\lambda_{12} = \\sqrt{\\Lambda_{11} \\Lambda_{22}}` is the
            geometric mean of the marginal intermittency coefficients,
          - :math:`\\xi_{12} = \\Lambda_{12}` is the off-diagonal cointermittency
            coefficient.

        When ``marginal`` is an integer index k, the method instead returns the
        **diagonal** covariance of component k (equivalent to calling
        :meth:`CovarianceFunction_SfBM` with ``marginal=k``).

        Parameters
        ----------
        tau : float or ndarray
            Lag value(s) at which the cross-covariance is evaluated.
        marginal : int or None, optional
            If None (default), returns the cross-covariance C₁₂(τ).
            If an integer k, returns the marginal auto-covariance Cₖₖ(τ).

        Returns
        -------
        float or ndarray
            Cross-autocovariance C₁₂(τ) (or the requested marginal Cₖₖ(τ)).

        Raises
        ------
        ValueError
            If the model dimension is not 2, or if the matrices are not 2 × 2.
        """
        if self.dimension != 2:
            raise ValueError(
                "CrossAutocovariance_mSfBM supports only the bivariate "
                f"(d=2) case; got dimension={self.dimension}."
            )
        if (
            self.coHurst_matrix.shape != (2, 2)
            or self.cointermittency_matrix.shape != (2, 2)
        ):
            raise ValueError(
                "coHurst_matrix and cointermittency_matrix must be 2×2."
            )

        if marginal is None:
            # ---- Off-diagonal (cross) covariance C₁₂(τ) -------------------
            # Geometric mean of marginal intermittency coefficients:
            # λ₁₂ = √(Λ₁₁ · Λ₂₂)
            l2 = np.sqrt(
                self.cointermittency_matrix[0, 0]
                * self.cointermittency_matrix[1, 1]
            )
            # Mean Hurst exponent  H̄ = (H₁₁ + H₂₂) / 2
            barH = (
                self.coHurst_matrix[0, 0] + self.coHurst_matrix[1, 1]
            ) / 2
            # Off-diagonal co-Hurst exponent H₁₂
            H_12 = self.coHurst_matrix[0, 1]
            # Off-diagonal cointermittency ξ₁₂ = Λ₁₂
            xi_12 = self.cointermittency_matrix[0, 1]
        else:
            # ---- Diagonal (marginal) covariance Cₖₖ(τ) --------------------
            # In this branch all parameters are drawn from the diagonal of
            # the matrices, effectively reducing to the univariate formula.
            l2 = np.sqrt(
                self.cointermittency_matrix[0, 0]
                * self.cointermittency_matrix[1, 1]
            )
            barH = (
                self.coHurst_matrix[0, 0] + self.coHurst_matrix[1, 1]
            ) / 2
            H_12 = self.coHurst_matrix[marginal, marginal]
            xi_12 = self.cointermittency_matrix[marginal, marginal]

        # ---- Three-term structural formula ---------------------------------
        # term1 = 1/(2H₁₂) − 1/(2H̄ − 1)   [constant term]
        term1 = (1 / (2 * H_12)) - (1 / (2 * barH - 1))

        # term2 = |τ|/T · [−1/(2H₁₂−1) + 1/(2H̄−1)]   [linear term in |τ|]
        term2 = (np.abs(tau) / self.T) * (
            -(1 / (2 * H_12 - 1)) + (1 / (2 * barH - 1))
        )

        # term3 = (|τ|/T)^{2H₁₂} · [1/(2H₁₂−1) − 1/(2H₁₂)]  [power-law term]
        term3 = ((np.abs(tau) / self.T) ** (2 * H_12)) * (
            1 / (2 * H_12 - 1) - (1 / (2 * H_12))
        )

        # Compact support: covariance is zero for |τ| > T
        return l2 * xi_12 * (term1 + term2 + term3) * (np.abs(tau) <= self.T)

    # -----------------------------------------------------------------------
    # Section 2 – Internal covariance sequence builder
    # -----------------------------------------------------------------------

    def _sfbmomcorr(
        self,
        size: int,
        dt: float = 0.1,
        T: float | None = None,
        lambda2: float | None = None,
        H: float | None = None,
    ) -> tuple[float, np.ndarray]:
        """Build the auto-covariance sequence of the log-vol Gaussian process.

        The Log-SfBM driving process ω_t has a stationary auto-covariance
        whose discrete approximation at time step dt is

        .. math::

            c[k] = \\frac{\\lambda^2}{2H(1-2H)}
                   \\left[1 - \\frac{(k \\cdot dt)^{2H}}{(T \\cdot dt)^{2H}}\\right]
                   \\mathbf{1}_{k \\cdot dt < T \\cdot dt}

        for k = 0, 1, …, size − 1.

        The special case H = 0 (MRW / log-correlated process) uses instead:

        .. math::

            c[k] = \\lambda^2 \\log\\!\\left(\\frac{T}{k \\cdot dt}\\right)
                   \\mathbf{1}_{k \\cdot dt < T}

        After building the sequence the method also returns the mean-correction
        term

        .. math::

            m = -c[0]

        which is added to the simulated Gaussian path so that
        E[exp(ω_t)] = 1 (i.e. the process is a proper martingale measure).

        Parameters
        ----------
        size : int
            Number of lags for which to compute the covariance sequence.
            The output array has length ``size``.
        dt : float, optional
            Time step size (default 0.1).
        T : float or None, optional
            Integral scale.  If None, ``self.T`` is used.
        lambda2 : float or None, optional
            Intermittency coefficient λ².  If None, ``self.lambda2`` is used.
        H : float or None, optional
            Hurst exponent.  If None, ``self.H`` is used.

        Returns
        -------
        m : float
            Mean correction term m = −c[0].
        cc : ndarray, shape (size,)
            Auto-covariance sequence of the log-vol Gaussian process ω.

        Notes
        -----
        The optional parameters ``T``, ``lambda2``, ``H`` allow computing the
        covariance for hypothetical parameter values without modifying the
        object's state (useful in multi-scale simulation loops).
        """
        # Lag vector τ_k = k · dt  for k = 1, …, size − 1
        xx = dt * np.arange(1, size)

        # Resolve parameters: use object attributes when overrides are absent
        use_self = (T is None) and (lambda2 is None) and (H is None)
        _H = self.H if use_self else H
        _T = self.T if use_self else T
        _l2 = self.lambda2 if use_self else lambda2

        if _H == 0:
            # ---- MRW / log-correlated case ---------------------------------
            # c[k] = λ² · log(T / (k·dt)) · 1_{k·dt < T}
            cc = _l2 * np.log(_T / xx) * (xx < _T)
            # Prepend the lag-0 variance: c[0] = λ²(1 + log(T/dt))
            cc = np.insert(cc, 0, _l2 * (1 + np.log(_T / dt)))
        else:
            # ---- Stationary S-fBM case  ------------------------------------
            # c[k] = K · [1 − (k·dt / T·dt)^{2H}] · 1_{k·dt < T·dt}
            # where K = λ² / [2H(1−2H)]
            xx = np.insert(xx, 0, 0)        # prepend lag-0  (τ = 0)
            K = _l2 / (2 * _H * (1 - 2 * _H))
            cc = K * (
                1 - xx ** (2 * _H) / (_T * dt) ** (2 * _H)
            ) * (xx < _T * dt)

        # Mean correction: shifts ω so that E[exp(ω)] = 1
        m = -cc[0]
        return m, cc

    # -----------------------------------------------------------------------
    # Section 3 – Path generators
    # -----------------------------------------------------------------------

    def GenerateSfBM_btwtimebounds(
        self,
        t_min: float,
        t_max: float,
        size: int = 4096,
        Msubsample: int = 32,
    ) -> tuple[np.ndarray]:
        """Generate ω on a time window [t_min, t_max] using the class covariance.

        This method builds the covariance vector of the log-vol driving process
        ω at the ``size`` equally-spaced points of [t_min, t_max] and simulates
        a Gaussian path via the Wood–Chan FFT method.

        The covariance used is the full non-stationary form evaluated against
        the left boundary:

        .. math::

            C_\\omega(t_{\\min}, t_k)
            = \\frac{\\lambda^2}{2H(1-2H)}
              \\left[1 - \\frac{|t_{\\min} - t_k|^{2H}}{T^{2H}}\\right]
              \\mathbf{1}_{|t_{\\min}-t_k| < T}

        A constant drift term is subtracted from the path:

        .. math::

            m = -\\frac{\\lambda^2 T^{2H}}{4H(1-2H)}

        so that E[exp(ω_t)] = 1 for all t.

        Parameters
        ----------
        t_min : float
            Left boundary of the time window.
        t_max : float
            Right boundary of the time window.
        size : int, optional
            Number of equally-spaced time points (default 4096).
        Msubsample : int, optional
            Sub-sampling resolution factor (reserved for future use;
            currently unused in this method).

        Returns
        -------
        complete_path : tuple of ndarray
            Single-element tuple containing the simulated ω path of length
            ``size``.
        """
        # Uniformly space time points over [t_min, t_max]
        timesteps = np.linspace(t_min, t_max, int(size))

        # Build covariance between the left boundary and all time points.
        # Each entry: C_ω(t_min, t_k)  using the class covariance kernel.
        covariance_list = np.array(
            [self.CovarianceFunction_SfBM(t_min, t) for t in timesteps]
        )

        # Simulate the Gaussian process ω via Wood–Chan FFT
        complete_path = GaussianProcessSimulation(covariance_list, size)

        # Apply the mean correction m = −λ²T^{2H} / [4H(1−2H)]
        # This ensures E[exp(ω_t)] = 1 (proper normalisation)
        mean_correction = (
            -self.lambda2 * self.T ** (2 * self.H)
            / (4 * self.H * (1 - 2 * self.H))
        )
        complete_path = complete_path + mean_correction

        return (complete_path,)

    def Generate_IntegratedSfBM(
        self,
        size: int = 4096,
        dt: float = 0.01,
    ) -> tuple[np.ndarray]:
        """Simulate the integrated log-vol process Ω.

        The integrated process is defined as the discrete cumulative sum of ω
        over blocks of length n_integ:

        .. math::

            \\Omega_j = \\frac{1}{\\sqrt{\\lambda^2}}
                        \\sum_{k \\in \\text{block}_j} \\omega_k \\cdot dt

        Its autocovariance is controlled by the second-order increment
        structure of ω through

        .. math::

            C_\\Omega(\\tau) = \\gamma_1 - \\gamma_2 \\left[
                |\\tau+1|^{2H+2} + |\\tau-1|^{2H+2} - 2|\\tau|^{2H+2}
            \\right]

        where

        .. math::

            \\gamma_1 = \\frac{T^{2H}}{2H(1-2H)}, \\qquad
            \\gamma_2 = \\frac{1}{2H\\,(1-(2H)^2)\\,(2+2H)}

        Parameters
        ----------
        size : int, optional
            Number of time steps for the simulated Ω path (default 4096).
        dt : float, optional
            Fine-grid time step size (default 0.01).

        Returns
        -------
        complete_path : tuple of ndarray
            Single-element tuple containing the simulated Ω path of length
            ``size``.

        Notes
        -----
        The output path is centred (zero mean) by construction of the
        covariance function, since C_Ω(τ) → 0 as τ → ∞.
        """
        # Fine-grid lag vector: τ_k = k · dt  for k = 1, …, size − 1
        timesteps = dt * np.arange(1, size)

        # ---- Theoretical autocovariance constants --------------------------
        # γ₁ = T^{2H} / [2H(1−2H)]   (long-range variance level)
        gamma_1 = self.T ** (2 * self.H) / (2 * self.H * (1 - 2 * self.H))

        # γ₂ = 1 / [2H·(1−(2H)²)·(2+2H)]   (second-difference prefactor)
        gamma_2 = 1 / (
            2 * self.H * (1 - (2 * self.H) ** 2) * (2 + 2 * self.H)
        )

        def _theoretical_autocov(tau: float) -> float:
            """C_Ω(τ) = γ₁ − γ₂·[|τ+1|^{2H+2} + |τ−1|^{2H+2} − 2|τ|^{2H+2}]"""
            return gamma_1 - gamma_2 * (
                np.abs(tau + 1) ** (2 * self.H + 2)
                + np.abs(tau - 1) ** (2 * self.H + 2)
                - 2 * np.abs(tau) ** (2 * self.H + 2)
            )

        # Evaluate the covariance at each lag
        covariance_list = np.array(
            [_theoretical_autocov(tau) for tau in timesteps]
        )

        # Simulate via Wood–Chan FFT
        complete_path = GaussianProcessSimulation(covariance_list, size)
        return (complete_path,)

    # -----------------------------------------------------------------------
    # Section 4 – Log-SfBM simulation (univariate and multivariate)
    # -----------------------------------------------------------------------

    def LogSfBM_Simulation(
        self,
        size: int = 4096,
        subsample: int = 4,
        sigma: float = 1.0,
        flagm: bool = False,
        returnomega: bool = False,
        returnOmega: bool = False,
        simulation_method: str = "fft",
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Simulate Log-SfBM price and variance paths.

        **Univariate case** (``self.dimension == 1``)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The simulation proceeds in three steps:

        1.  Build the auto-covariance sequence of ω (see :meth:`_sfbmomcorr`).
        2.  Simulate the fine-grid Gaussian process
                ω_t ~ GP(0, C_ω)   of length ``size × subsample``
            via Wood–Chan FFT, then shift by the mean correction
                m = −λ²/[4H(1−2H)]
            so that E[exp(ω_t)] = 1.
        3.  Construct the price path or variance path from ω:

            .. math::

                X_t &= \\sum_{k=0}^{t/dt-1} e^{\\omega_k/2} \\, g_k \\, \\sigma \\sqrt{dt}
                        \\quad\\text{(MRW, cumulative)}\\\\
                M_t &= \\sigma^2 \\sum_{k=0}^{t/dt-1} e^{\\omega_k} \\, dt
                        \\quad\\text{(MRM, integrated variance)}

            where g_k ~ N(0,1) i.i.d. and dt = 1/subsample.

            Both X_t and M_t are sub-sampled at every ``subsample``-th point.

        **Multivariate case** (``self.dimension > 1``)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Two simulation methods are available:

        * ``"fft"`` (default):
            Constructs the joint spectral density matrix S(f) from the
            marginal covariances (diagonal blocks) and cross-covariances
            (off-diagonal blocks), factorises it via eigendecomposition at
            each frequency bin, and applies the inverse FFT to produce jointly
            correlated paths.  The central quarter of each path is returned
            to avoid circular boundary artefacts.

        * ``"Cholesky"``:
            Builds the full (d·size × d·size) covariance matrix from the
            analytic cross-covariance formula, draws a multivariate normal
            sample, and reshapes it into d paths of length ``size``.
            Exact but O((d·size)³) in cost; only feasible for small size.

        Parameters
        ----------
        size : int, optional
            Number of coarse-grid time steps (default 4096).
        subsample : int, optional
            Fine-grid oversampling factor; dt = 1/subsample (default 4).
        sigma : float, optional
            Volatility scale σ (default 1).
        flagm : bool, optional
            If True, return only the integrated-variance path M_t;
            if False (default), return the pair (X_t, M_t).
        returnomega : bool, optional
            If True, return the raw fine-grid log-vol process ω
            (univariate only).  Default False.
        returnOmega : bool, optional
            If True, return the integrated log-vol process Ω instead.
            Default False.
        simulation_method : str, optional
            Simulation backend for the multivariate case: ``"fft"``
            (default) or ``"Cholesky"``.

        Returns
        -------
        omega : ndarray, shape (size × subsample,)
            Returned when ``returnomega=True`` (univariate only).
        Omega : ndarray
            Returned when ``returnOmega=True``.
        mrm : ndarray, shape (size,)
            Integrated-variance path M_t (returned when ``flagm=True``).
        mrw : ndarray, shape (size,)
            MRW price path X_t (returned as first element of pair).
        mrm : ndarray, shape (size,)
            MRM variance path M_t (returned as second element of pair).

        Notes
        -----
        In the FFT multivariate method the spectral density matrix at each
        frequency bin k is regularised by adding 10⁻¹⁴·I before the
        eigendecomposition to ensure numerical stability.
        """
        # ==================================================================
        # UNIVARIATE CASE
        # ==================================================================
        if self.dimension == 1:
            dt = 1.0 / subsample

            # Step 1: build fine-grid covariance sequence for ω
            N = int(2 ** np.ceil(np.log2(size - 1)))
            m, corr = self._sfbmomcorr(size=N * subsample, dt=dt)

            # Step 2: apply mean correction  m = −λ²/[4H(1−2H)]
            # This ensures E[exp(ω_t)] = 1 and M_t is a proper measure.
            m = -self.lambda2 / (4 * self.H * (1 - 2 * self.H))

            # Step 3: simulate the fine-grid Gaussian path ω
            om = GaussianProcessSimulation(corr, size * subsample) + m

            # ---- Optional early returns ------------------------------------

            # Return the raw log-vol process (no price or variance)
            if returnomega:
                return om

            # Return the integrated log-vol Ω = (1/√λ²) Σ_j (ω_j − 0) · dt
            if returnOmega:
                # Split ω into `size` non-overlapping blocks and integrate
                blocks = np.array_split(om, size)
                Omega = np.array(
                    [
                        (1.0 / sqrt(self.lambda2))
                        * np.sum(np.array(blocks[j]) * dt)
                        for j in range(size)
                    ]
                )
                return Omega - Omega.mean()

            # ---- Price and variance paths -----------------------------------

            if flagm:
                # Integrated-variance (MRM):  M_t = σ² ∫ exp(ω_s) ds
                mrm = np.cumsum(sigma ** 2 * np.exp(om) * dt)
                return mrm[::subsample]
            else:
                # MRW price path:  dX_t = exp(ω_t/2) dW_t
                gg = np.random.normal(size=len(om))
                mrw = np.cumsum(np.exp(0.5 * om) * gg * sigma * np.sqrt(dt))
                # Integrated variance in parallel
                mrm = np.cumsum(sigma ** 2 * np.exp(om) * dt)
                # Sub-sample both paths back to the coarse grid
                return mrw[::subsample], mrm[::subsample]

        # ==================================================================
        # MULTIVARIATE CASE
        # ==================================================================
        else:
            dt = 1.0 / subsample

            # ------------------------------------------------------------------
            # Method A – Cholesky (exact, expensive for large size)
            # ------------------------------------------------------------------
            if simulation_method == "Cholesky":
                # Build the full (d·size × d·size) covariance matrix by filling
                # d² blocks of size (size × size), each computed from the
                # analytic cross-covariance at all pairwise time differences.
                time_lags = np.arange(size) * dt
                total_size = self.dimension * size
                cov_matrix = np.zeros((total_size, total_size))

                for i in range(self.dimension):
                    for j in range(self.dimension):
                        # Matrix of all pairwise lags |t_a − t_b|
                        time_diff = np.abs(
                            time_lags[:, None] - time_lags[None, :]
                        )
                        # Fill the (i,j) covariance block
                        cov_block = self.CrossAutocovariance_mSfBM(time_diff)
                        cov_matrix[
                            i * size : (i + 1) * size,
                            j * size : (j + 1) * size,
                        ] = cov_block

                # Draw a single multivariate normal sample and reshape
                sample = np.random.multivariate_normal(
                    np.zeros(total_size), cov_matrix
                )
                complete_paths = [
                    sample[k * size : (k + 1) * size]
                    for k in range(self.dimension)
                ]

                # Add the per-component mean correction
                means_paths = [
                    np.full(
                        size,
                        -self.cointermittency_matrix[i, i]
                        / (
                            4
                            * self.coHurst_matrix[i, i]
                            * (1 - 2 * self.coHurst_matrix[i, i])
                        ),
                    )
                    for i in range(self.dimension)
                ]
                om = [
                    cp + mp
                    for cp, mp in zip(complete_paths, means_paths)
                ]

            # ------------------------------------------------------------------
            # Method B – FFT spectral simulation (default, scalable)
            # ------------------------------------------------------------------
            elif simulation_method == "fft":
                # Pad the grid size to the next power of two and upsample
                N = 2 * (size - 1)
                NN = 2 ** np.ceil(np.log2(N))
                N = int(NN) * subsample

                # Symmetric lag grid: τ ∈ {−N/2 · dt, …, (N/2 − 1) · dt}
                tau = np.arange(-N // 2, N // 2) * dt

                # --- Evaluate covariance functions on the lag grid -----------
                # Diagonal spectral densities (marginal components)
                C11_vals = self.CovarianceFunction_SfBM(0, tau, marginal=0)
                C22_vals = self.CovarianceFunction_SfBM(0, tau, marginal=1)
                # Off-diagonal spectral density (cross-covariance)
                C12_vals = self.CrossAutocovariance_mSfBM(tau)
                C21_vals = self.CrossAutocovariance_mSfBM(-tau)

                # --- Compute spectral density matrices via FFT ---------------
                # S_ij(f) = FFT(C_ij) · dt   (Wiener–Khinchin theorem)
                S11 = fft(ifftshift(C11_vals)) * dt
                S22 = fft(ifftshift(C22_vals)) * dt
                S12 = fft(ifftshift(C12_vals)) * dt
                S21 = fft(ifftshift(C21_vals)) * dt

                # Assemble the 2 × 2 spectral density matrix S(f) for each bin
                S = np.zeros((N, 2, 2), dtype=complex)
                S[:, 0, 0] = S11
                S[:, 1, 1] = S22
                S[:, 0, 1] = S12
                S[:, 1, 0] = S21

                # Enforce Hermitian symmetry:  S[−k] = conj(S[k])
                for k in range(1, N // 2):
                    S[-k] = np.conj(S[k])

                # --- Draw a Hermitian-symmetric complex Gaussian vector ------
                # Z_k ~ CN(0, I) with Z[N/2+1:] = conj(flip(Z[1:N/2]))
                Z = (
                    np.random.randn(N, 2) + 1j * np.random.randn(N, 2)
                ) / np.sqrt(2)
                Z[N // 2 + 1 :] = np.conj(np.flipud(Z[1 : N // 2]))

                # --- Spectral factorisation  S(f) = H(f) H(f)* --------------
                # Use eigendecomposition for numerical stability.
                # At each frequency bin k: S_k = V Λ V*,  H_k = V √Λ
                H_mat = np.zeros_like(S, dtype=complex)
                for k in range(N):
                    # Symmetrise to suppress tiny numerical asymmetries
                    S_k = (S[k] + S[k].T.conj()) / 2
                    # Regularise: add 10⁻¹⁴ · I before decomposition
                    eigvals, eigvecs = np.linalg.eigh(
                        S_k + 1e-14 * np.eye(2)
                    )
                    # Clip negative eigenvalues (numerical artefacts)
                    eigvals = np.maximum(eigvals, 0.0)
                    H_mat[k] = eigvecs @ np.diag(np.sqrt(eigvals))

                # --- Generate correlated frequency-domain samples ------------
                # X̂_k = H_k · Z_k
                Xf = np.einsum("kij,kj->ki", H_mat, Z)

                # --- Inverse FFT back to the time domain --------------------
                # The √(N/dt) factor restores the correct variance scaling.
                x = np.real(ifft(Xf, axis=0)) * np.sqrt(N / dt)
                x1, x2 = x[:, 0], x[:, 1]

                # Crop the central quarter to remove circular boundary effects
                center = N // 2
                crop = N // 4
                om = [
                    x1[center - crop : center + crop],
                    x2[center - crop : center + crop],
                ]

            else:
                raise ValueError(
                    f"Unknown simulation_method {simulation_method!r}. "
                    "Choose 'fft' or 'Cholesky'."
                )

            # ------------------------------------------------------------------
            # Shared post-processing for multivariate paths
            # ------------------------------------------------------------------

            # Return the integrated log-vol Ω for each component
            if returnOmega:
                Omega_list = []
                for om_k in om:
                    blocks = np.array_split(om_k, size)
                    Omega_k = np.array(
                        [
                            (1.0 / np.sqrt(self.lambda2 if self.lambda2 else 1.0))
                            * np.sum(np.array(blocks[j]) * dt)
                            for j in range(size)
                        ]
                    )
                    Omega_list.append(Omega_k - Omega_k.mean())
                return np.array(Omega_list)

            if flagm:
                # Integrated variance M_t^{(k)} for each component k
                mrm_list = [
                    np.cumsum(sigma ** 2 * np.exp(om_k) * dt)[::subsample]
                    for om_k in om
                ]
                return np.array(mrm_list)

            else:
                # MRW and MRM for each component k
                mrw_list, mrm_list = [], []
                for om_k in om:
                    gg = np.random.normal(size=len(om_k))
                    mrw_k = np.cumsum(
                        np.exp(0.5 * om_k) * gg * sigma * np.sqrt(dt)
                    )
                    mrm_k = np.cumsum(sigma ** 2 * np.exp(om_k) * dt)
                    mrw_list.append(mrw_k[::subsample])
                    mrm_list.append(mrm_k[::subsample])
                return np.array(mrw_list), np.array(mrm_list)

    # -----------------------------------------------------------------------
    # Section 5 – Nested Log-SfBM
    # -----------------------------------------------------------------------

    def LogSfBM_Nested_Simulation(
        self,
        size: int = 4096,
        subsample: int = 4,
        sigma: float = 1.0,
        flagm: bool = False,
        modevol: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Simulate a nested Log-SfBM using an optional precomputed ω.

        In the **nested** variant the log-vol driving process ω can be
        supplied externally via ``modevol`` (e.g. from a first-layer
        Log-SfBM), allowing the construction of hierarchical multifractal
        models.  If ``modevol`` is None the internal ω is simulated from
        the class parameters in the standard way (identical to
        :meth:`LogSfBM_Simulation` with the original S-fBM covariance).

        The drift correction applied here follows the original S-fBM
        parametrisation:

        .. math::

            m = -\\frac{\\lambda^2 T^{2H}}{4H(1-2H)}

        After constructing ω the method produces the price/variance paths
        using squared exponential volatility:

        .. math::

            \\tilde{X}_t &= \\int_0^t e^{\\omega_s}\\, dW_s
                \\qquad \\text{(nested MRW)}\\\\
            \\tilde{M}_t &= \\sigma^2 \\int_0^t e^{2\\omega_s}\\, ds
                \\qquad \\text{(nested MRM)}

        Note the factor of 2 in the exponent of M compared to the standard
        MRM; this reflects the squared-volatility convention of the nested
        model.

        Parameters
        ----------
        size : int, optional
            Number of coarse-grid time steps (default 4096).
        subsample : int, optional
            Fine-grid oversampling factor; dt = 1/subsample (default 4).
        sigma : float, optional
            Volatility scale σ (default 1).
        flagm : bool, optional
            If True, return only M̃_t; if False (default), return (X̃_t, M̃_t).
        modevol : ndarray or None, optional
            Precomputed fine-grid log-vol path ω (length size × subsample).
            If None, ω is simulated internally.

        Returns
        -------
        mrm : ndarray, shape (size,)
            Nested integrated-variance path M̃_t  (returned when flagm=True).
        mrw : ndarray, shape (size,)
            Nested MRW price path X̃_t.
        mrm : ndarray, shape (size,)
            Nested MRM variance path M̃_t.
        """
        dt = 1.0 / subsample

        if modevol is None:
            # ---- Simulate ω internally from class parameters ---------------
            N = int(2 ** np.ceil(np.log2(size - 1)))
            m, corr = self._sfbmomcorr(size=N * subsample, dt=dt)

            # Mean correction:  m = −λ²T^{2H} / [4H(1−2H)]
            m = (
                -self.lambda2
                * self.T ** (2 * self.H)
                / (4 * self.H * (1 - 2 * self.H))
            )
            om = GaussianProcessSimulation(corr, size * subsample) + m
        else:
            # ---- Use a precomputed ω (nested / hierarchical model) ---------
            om = modevol

        if flagm:
            # Nested integrated variance:  M̃_t = σ² ∫ exp(2ω_s) ds
            # Note the factor of 2 in the exponent (squared-volatility convention)
            mrm = np.cumsum(sigma ** 2 * np.exp(2 * om) * dt)
            return mrm[::subsample]
        else:
            # Nested MRW:  dX̃_t = exp(ω_t) dW_t   (not exp(ω_t/2))
            gg = np.random.normal(size=len(om))
            mrw = np.cumsum(np.exp(om) * gg * sigma * np.sqrt(dt))
            mrm = np.cumsum(sigma ** 2 * np.exp(2 * om) * dt)
            return mrw[::subsample], mrm[::subsample]

    # -----------------------------------------------------------------------
    # Section 6 – Log-realised-volatility estimators
    # -----------------------------------------------------------------------

    def genlogVol(
        self,
        size: int = 4096,
        subsample: int = 4,
        M: int = 32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate log-realised-volatility series via QV aggregation.

        Simulates a fine-grid Log-SfBM path of length ``size × M``, computes
        the quadratic variation (QV) of the price process at aggregation scale
        M, and returns the log-QV as a proxy for log-realised variance.

        **Aggregation scheme**

        Given the price path X_t on the fine grid of step 1/M, the aggregated
        quadratic variation at coarse step k is

        .. math::

            \\text{QV}_k = \\sum_{j=(k-1)M}^{kM-1}
                           (X_{(j+1)/M} - X_{j/M})^2

        The log-volatility proxy is then

        .. math::

            \\ell_k = \\log(\\text{QV}_k) - \\overline{\\log(\\text{QV})}

        (mean-centred).  Numerical ±∞ values arising from QV = 0 are
        replaced by linear interpolation.

        Similarly, the integrated-variance proxy is

        .. math::

            \\ell^M_k = \\log(M_k) - \\overline{\\log(M)}

        where M_k = ∫_{(k-1)/M}^{k/M} σ² exp(ω_t) dt is the coarse-grid
        integrated variance increment.

        Parameters
        ----------
        size : int, optional
            Number of coarse-grid time steps (default 4096).
        subsample : int, optional
            Fine-grid oversampling factor within each QV block (default 8).
        M : int, optional
            Aggregation factor; each coarse step aggregates M fine steps
            (default 32).

        Returns
        -------
        logvol_qv : ndarray, shape (size − 1,)
            Mean-centred log-QV series (proxy for log-realised variance).
        logvol_mm : ndarray, shape (size − 1,)
            Mean-centred log integrated-variance series.

        Notes
        -----
        The intermittency coefficient is implicitly corrected by the factor
        M^{−2H}/4 when H > 0 to account for the sub-sampling scheme.
        This correction is baked into the simulation via :meth:`LogSfBM_Simulation`
        which uses the class's own λ² (no explicit rescaling in this method).
        """
        # Generate fine-grid price and variance paths of length size × M
        vv, mm = self.LogSfBM_Simulation(
            size=size * M, subsample=subsample, flagm=False
        )

        # ---- Compute QV increments at aggregation scale M ------------------
        # First-differences of the price path (fine grid)
        dvv = np.diff(vv)
        dvv = np.insert(dvv, 0, dvv[0])    # prepend to maintain length

        # First-differences of the integrated-variance path (fine grid)
        dmm = np.diff(mm)
        dmm = np.insert(dmm, 0, dmm[0])

        # Cumulative sums, then sub-sample at scale M
        qv_cs = np.cumsum(dvv ** 2)[::M]   # cumulative QV at coarse grid
        mm_cs = np.cumsum(dmm)[::M]         # cumulative MRM at coarse grid

        # Non-overlapping increments: QV_k = cum_QV[k] − cum_QV[k−1]
        qv = qv_cs[1:] - qv_cs[:-1]
        mm = mm_cs[1:] - mm_cs[:-1]

        # ---- Log-transform and clean ----------------------------------------
        # Replace ±∞ (from log(0)) by linear interpolation
        zz1 = (
            pd.Series(np.log(qv))
            .replace([np.inf, -np.inf], np.nan)
            .interpolate(limit_direction="both")
        )
        zz1 -= zz1.mean()                   # mean-centre

        zz2 = np.log(mm) - np.mean(np.log(mm))

        return zz1.values, zz2

    def genlogVol_New_perscale(
        self,
        size: int = 4096,
        subsample: int = 8,
        M: int = 32,
        scale: int = 1,
        sigma: float = 1.0,
        flagm: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate log-volatility series at a rescaled aggregation level.

        Same as :meth:`genlogVol` but allows testing the QV proxy at a
        rescaled aggregation factor ``M × scale``.  This is useful for
        studying the multi-scale behaviour of the log-volatility proxy and
        for bias analysis across different aggregation windows.

        The aggregation formula is identical to :meth:`genlogVol` but with
        M replaced by M_scaled = M × scale:

        .. math::

            \\text{QV}_k^{(s)} = \\sum_{j=(k-1) M_s}^{k M_s - 1}
                                  (X_{j+1} - X_j)^2,
            \\qquad M_s = M \\cdot \\text{scale}

        Parameters
        ----------
        size : int, optional
            Number of coarse-grid time steps before aggregation (default 4096).
        subsample : int, optional
            Fine-grid oversampling factor (default 8).
        M : int, optional
            Base aggregation factor (default 32).
        scale : int, optional
            Additional rescaling multiplier applied on top of M (default 1).
            Setting scale > 1 produces a coarser QV series.
        sigma : float, optional
            Volatility scale σ passed to the simulator (default 1).
        flagm : bool, optional
            If True, use the integrated variance path instead of MRW
            (passed directly to :meth:`LogSfBM_Simulation`).

        Returns
        -------
        logvol_qv : ndarray
            Mean-centred log-QV series at aggregation level M × scale.
        logvol_mm : ndarray
            Mean-centred log integrated-variance series at the same level.
        """
        # Generate fine-grid paths at resolution size × M
        vv, mm = self.LogSfBM_Simulation(
            size=size * M, subsample=subsample, flagm=False
        )

        # Effective (rescaled) aggregation factor
        M_scaled = M * scale

        # ---- QV at the rescaled aggregation level --------------------------
        dvv = np.diff(vv)
        dvv = np.insert(dvv, 0, dvv[0])
        dmm = np.diff(mm)
        dmm = np.insert(dmm, 0, dmm[0])

        # Sub-sample cumulative sums at the rescaled step M_scaled
        qv_cs = np.cumsum(dvv ** 2)[::M_scaled]
        mm_cs = np.cumsum(dmm)[::M_scaled]

        # Non-overlapping increments
        qv = qv_cs[1:] - qv_cs[:-1]
        mm = mm_cs[1:] - mm_cs[:-1]

        # ---- Log-transform and clean ----------------------------------------
        zz1 = (
            pd.Series(np.log(qv))
            .replace([np.inf, -np.inf], np.nan)
            .interpolate(limit_direction="both")
        )
        zz1 -= zz1.mean()

        zz2 = np.log(mm) - np.mean(np.log(mm))
        return zz1.values, zz2
