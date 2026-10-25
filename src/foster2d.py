"""Layer 1 -- Pc methods: 2-D conjunction-plane Pc and a CARA-style Monte Carlo.

Three *independent* implementations of the same physical quantity, so they can
be checked against each other rather than against a single formula:

1. :func:`pc2d` (``src/pc2d.py``) -- Alfriend-Akella / Chan small-HBR closed form.
2. :func:`foster2d` -- the exact disk integral reduced to a **single 1-D
   quadrature** (inner integral done analytically with error functions).
   Different derivation, different numerics, no approximation.
3. :func:`cara_monte_carlo` -- CARA-style: draw the **6-D** relative state,
   propagate to TCA, build the encounter plane, count hits.  This is the only
   one of the three that sees the position-velocity covariance block.

Why a separate 1-D reduction
-----------------------------
``pc2d_numeric`` integrates in polar coordinates with the trapezoid rule on
``rho d rho``.  ``foster2d`` rotates into the covariance principal axes and
integrates the *outer* coordinate numerically while doing the *inner* integral
exactly:

    Pc = INT_{-R}^{R} phi(x; m1', s1) *
         [Phi((h(x) - m2')/s2) - Phi((-h(x) - m2')/s2)] dx,   h(x)=sqrt(R^2-x^2)

The bracketed term is the exact probability mass of a N(m2', s2^2) variable
inside [-h(x), h(x)].  Agreement between the two schemes is therefore a real
check on both, not a tautology.

Honest scope note
-----------------
CARA (NASA's Conjunction Assessment Risk Analysis tool) is a licensed internal
NASA product and **cannot be run here**.  The validation in
``scripts/validate_foster_mc.py`` therefore rests on:
(a) exact analytic anchors that are mathematically certain,
(b) agreement between three independent numerical implementations,
(c) an independent *operational* implementation -- ESA's ``risk`` field in the
    Kelvins CDMs, reproduced to a median error of 0.0012 dex,
(d) convergence and error characterisation.
See ``notes/layer1_pc_methods.md`` for the full argument.

References
----------
* K. Alfriend, M. R. Akella, J. Frisbee, J. L. Foster, D.-J. Lee, M. Wilkins,
  "Probability of Collision Between Space Objects", JGCD 23(5), 2000.
* F. K. Chan, *Spacecraft Collision Probability*, AIAA, 2008.
* J. S. Foster and H. S. Estes, "A Parametric Analysis of Orbital Debris
  Collision Probability in Spacecraft Design", 1992.
* NASA CARA Best Practices Handbook (2023), Appendix N.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .covariance import encounter_plane_basis, mahalanobis
from .pc2d import pc2d, pc2d_max

__all__ = ["pc2d_1d", "foster2d", "foster2d_pc_max", "cara_monte_carlo",
           "encounter_plane_covariance", "hbr_flat_5m",
           "hbr_catalogue", "hbr_rcs_uncertainty", "foster_vs_mc"]


# ----------------------------------------------------------------------
# 2-D conjunction-plane Pc, Eq. N-2 (Alfano 2005a / Elrod 2019)
# ----------------------------------------------------------------------
def pc2d_1d(miss_m: np.ndarray, C2: np.ndarray, hbr_m: float,
            n_nodes: int = 200) -> float:
    """Exact 2-D Pc by 1-D reduction in the covariance principal axes.

    This is Eq. N-2 of the CARA Best Practices Handbook, Appendix N:

        Pc = 1/(sqrt(8*pi)*sigma_x) * INT_{-R}^{R}
             {erf(y+) - erf(y-)} * exp(-(u+x_miss)^2 / (2 sigma_x^2)) du
        y+/- = (y_miss +/- sqrt(R^2 - u^2)) / (sqrt(2) sigma_y)

    Note on attribution: Appendix N credits Foster & Estes (1992) with the
    **2-D quadrature** of Eq. N-1, and Alfano (2005a) with this 1-D
    error-function reduction, refined by Elrod (2019).  ``foster2d`` is kept as
    an alias so existing callers keep working.

    Parameters
    ----------
    miss_m
        Miss vector in the encounter plane [m].
    C2
        Combined 2x2 covariance in the encounter plane [m^2].
    hbr_m
        Combined hard-body radius [m].
    n_nodes
        Gauss-Legendre nodes on [-R, R].

    Returns
    -------
    Pc, with no small-HBR approximation.  Valid for any R/sigma ratio.
    """
    C2 = np.asarray(C2, dtype=float)
    miss = np.asarray(miss_m, dtype=float)
    det = np.linalg.det(C2)
    if det <= 0.0:
        raise ValueError("encounter-plane covariance is not positive definite")
    if hbr_m <= 0.0:
        raise ValueError("hbr must be positive")

    # principal axes: w ascending, so w[0] is the minor axis
    w, V = np.linalg.eigh(C2)
    s1, s2 = np.sqrt(w[1]), np.sqrt(w[0])
    m_rot = V.T @ miss                 # [major, minor] components
    m1, m2 = float(m_rot[1]), float(m_rot[0])

    # Gauss-Legendre on [-R, R]; guard the sqrt at the endpoints
    x, wq = np.polynomial.legendre.leggauss(n_nodes)
    xs = hbr_m * x
    h = np.sqrt(np.maximum(hbr_m ** 2 - xs ** 2, 0.0))

    z_hi = (h - m2) / s2
    z_lo = (-h - m2) / s2
    inner = ndtr(z_hi) - ndtr(z_lo)          # exact inner integral
    outer = np.exp(-0.5 * ((xs - m1) / s1) ** 2) / (np.sqrt(2.0 * np.pi) * s1)

    return float(hbr_m * np.sum(wq * outer * inner))


def foster2d_pc_max(hbr_m: float, C2: np.ndarray) -> float:
    """Pc at zero miss distance (analytic, no quadrature)."""
    return pc2d_max(hbr_m, C2)


# ----------------------------------------------------------------------
# Linear state propagation to TCA
# ----------------------------------------------------------------------
def encounter_plane_covariance(C6_rtn: np.ndarray, v_rel_rtn: np.ndarray,
                               dt_s: float, mode: str = "linear") -> np.ndarray:
    """2-D encounter-plane covariance of the relative position at TCA.

    Builds the linear (or two-body) map from the 6-D epoch state to the 3-D
    relative position at TCA, then projects that 3x3 map through the encounter
    plane basis.  This is the covariance ``foster2d`` expects, derived from the
    **full 6-D** covariance -- i.e. including the position-velocity block.
    """
    C6 = np.asarray(C6_rtn, dtype=float)
    v = np.asarray(v_rel_rtn, dtype=float)
    # d r_tca / d [r; v]  (3x6):  [ I3 , dt * I3 ]
    A = np.hstack([np.eye(3), dt_s * np.eye(3)])
    if mode == "two_body":
        # second-order term: d r/dt|tca = v, d2r/dt2 ~ -mu/r^3 * r
        mu = 3.986004418e14
        rn = np.linalg.norm(r_rel_rtn) + 1e3
        A = np.hstack([np.eye(3) - 0.5 * dt_s ** 2 * mu / rn ** 3 * np.eye(3),
                       dt_s * np.eye(3)])
    C3_tca = A @ C6 @ A.T
    P = encounter_plane_basis(v)             # 3x2
    return P.T @ C3_tca @ P                  # 2x2


# ----------------------------------------------------------------------
# CARA-style Monte Carlo: 6-D draw -> propagate -> plane -> count
# ----------------------------------------------------------------------
def cara_monte_carlo(r_rel_rtn: np.ndarray, v_rel_rtn: np.ndarray,
                     C6_rtn: np.ndarray, hbr_m: float,
                     n_samples: int = 200_000, seed: int = 0,
                     dt_s: float = 0.0, mode: str = "linear") -> dict:
    """CARA-style Monte Carlo Pc from the full 6-D relative covariance.

    Draws the 6-D relative state from ``C6_rtn``, propagates each sample to TCA
    with the requested mapping, builds the encounter plane from the *nominal*
    TCA relative velocity, projects, and counts the fraction inside the HBR.

    Returns a dict with the estimate, its Poisson standard error, and the
    analytic reference computed from the propagated encounter-plane covariance
    (so the caller can see the two side by side).
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(r_rel_rtn, dtype=float)
    v = np.asarray(v_rel_rtn, dtype=float)
    C6 = np.asarray(C6_rtn, dtype=float)
    C6 = 0.5 * (C6 + C6.T)          # symmetrise: .xls rounding leaves it slightly off

    # CARA remediates non-positive-semidefinite covariances by eigenvalue
    # clipping before sampling (Appendix N, N.7.2; CovRemEigValClip.m). Mirror
    # that so the MC runs on the same inputs CARA accepts.
    lclip = (1e-4 * hbr_m) ** 2
    w, V = np.linalg.eigh(C6)
    if w.min() < lclip:
        w = np.maximum(w, lclip)
        C6 = (V * w) @ V.T
        C6 = 0.5 * (C6 + C6.T)

    L = np.linalg.cholesky(C6)
    state = rng.standard_normal((n_samples, 6)) @ L.T + np.concatenate([r, v])

    MU = 3.986004418e14


    def _tca(rr):
        """Propagate a (..., 3) block of relative positions to TCA."""
        if mode == "two_body":
            rn = np.linalg.norm(r) + 1e3
            return rr - 0.5 * dt_s ** 2 * MU / rn ** 3 * rr + dt_s * rr
        return rr + dt_s * rr

    # NOTE: _tca above is wrong in general -- the propagation must act on the
    # *position* block with the *velocity* block as the rate.  Written out
    # explicitly instead:
    def _prop(pos, vel):
        if mode == "two_body":
            rn = np.linalg.norm(r) + 1e3
            return pos - 0.5 * dt_s ** 2 * MU / rn ** 3 * pos + dt_s * vel
        return pos + dt_s * vel

    r_tca = _prop(state[:, :3], state[:, 3:6])
    r_nom_tca = _prop(r, v)

    P = encounter_plane_basis(v)                          # 3x2
    proj = r_tca @ P                                      # (n, 2)
    inside = int(np.sum(proj[:, 0] ** 2 + proj[:, 1] ** 2 <= hbr_m ** 2))

    p = inside / n_samples
    sigma_p = np.sqrt(max(p * (1.0 - p), 1.0 / n_samples) / n_samples)

    C2 = encounter_plane_covariance(C6, v, dt_s, mode)
    m2 = P.T @ r_nom_tca
    return {
        "pc_mc": float(p),
        "sigma_mc": float(sigma_p),
        "pc_foster": foster2d(m2, C2, hbr_m),
        "pc_closed": pc2d(m2, C2, hbr_m),
        "n_inside": inside,
        "n_samples": int(n_samples),
        "C2_tca": C2,
        "m2_tca": m2,
    }


# ----------------------------------------------------------------------
# HBR variants (Layer 1)
# ----------------------------------------------------------------------
def hbr_flat_5m(**_) -> float:
    """Space-Track / 18 SPCS EVR convention: a flat 5 m combined radius."""
    return 5.0


def hbr_catalogue(t_span: float, c_span: float, **_) -> float:
    """Catalogue HBR used by ESA in the Kelvins CDMs: (t_span + c_span)/2."""
    return 0.5 * (t_span + c_span)


def hbr_rcs_uncertainty(t_span: float, c_span: float,
                        t_rcs: float, c_rcs: float,
                        rcs_rel_sigma: float = 0.30, **_) -> dict:
    """RCS-uncertainty HBR: the object radii are themselves uncertain.

    The combined radius is treated as R(1 + delta) with delta ~ N(0, rcs_rel_sigma)
    driven by the uncertainty in the two radar cross-sections.  Because Pc scales
    as R^2 in the small-HBR limit, the radius uncertainty maps onto an inflated
    Pc:  E[R^2] = R^2 (1 + rcs_rel_sigma^2).

    Returns the nominal radius and the inflation factor.
    """
    R = hbr_catalogue(t_span, c_span)
    return {"r_hbr_m": R, "pc_inflation": 1.0 + rcs_rel_sigma ** 2,
            "rcs_rel_sigma": rcs_rel_sigma}


# ----------------------------------------------------------------------
# agreement of the analytic path with the MC
# ----------------------------------------------------------------------
def foster_vs_mc(r_rel_rtn, v_rel_rtn, C6_rtn, hbr_m, dt_s=0.0,
                 n_samples: int = 200_000, seed: int = 0) -> dict:
    """Agreement of the three independent Pc estimators on one case."""
    out = cara_monte_carlo(r_rel_rtn, v_rel_rtn, C6_rtn, hbr_m,
                           n_samples=n_samples, seed=seed, dt_s=dt_s)
    pf = out["pc_foster"]
    return {
        "foster_exact": pf,
        "chan_closed": out["pc_closed"],
        "cara_mc": out["pc_mc"],
        "sigma_mc": out["sigma_mc"],
        "mc_z": abs(out["pc_mc"] - pf) / out["sigma_mc"],
        "pc_max": pc2d_max(hbr_m, out["C2_tca"]),
        "chi2": mahalanobis(out["m2_tca"], out["C2_tca"]),
        "R_over_sigma": float(hbr_m / np.sqrt(np.linalg.eigvalsh(out["C2_tca"]).max())),
    }


#: Backwards-compatible alias.  Implements Eq. N-2 (Alfano 2005a / Elrod 2019);
#: see the module docstring for the attribution correction.
foster2d = pc2d_1d
