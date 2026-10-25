"""Two-dimensional probability of collision (2-D Pc).

Layer 4 of the learning path.  The 2-D Pc is the probability that the relative
position at TCA, modelled as a 2-D Gaussian in the encounter plane, falls
inside the disk of radius R (the combined hard-body radius) centred on the
origin.

Derivation
----------
With the combined covariance projected into the encounter plane, C2 (2x2), and
the projected miss vector m (2-vector), the probability is

    Pc = (1 / (2*pi*sqrt(det C2))) * INT_{|x| <= R} exp(-1/2 x^T C2^-1 x) dx

The exact integral has no elementary closed form, so the universally used
operational form (Alfriend-Akella / Chan; the form behind the CDM ``risk``
field) is the **small-hard-body-radius approximation**, obtained by replacing
the integrand with its value at the centre of the disk:

    Pc ~= (R^2 / (2 * sigma_1 * sigma_2)) * exp(-1/2 * m^T C2^-1 m)

where sigma_1, sigma_2 are the principal standard deviations of C2, so that
sigma_1 * sigma_2 = sqrt(det C2).  This is exact to O(R^2) and is what
"2-D Pc" means in CARA practice.  :func:`pc2d_numeric` evaluates the true
integral by polar quadrature and :func:`pc2d_monte_carlo` by sampling, so the
size of the approximation error can be measured directly.

Note that with m = 0 the formula reduces to Pc_max = R^2 / (2 sigma_1 sigma_2),
the "theoretical maximum collision probability obtained by scaling the
combined covariance" used to filter the Kelvins database.

References
----------
* NASA CARA Best Practices Handbook (2023), Appendix N.
* K. Alfriend, M. R. Akella, J. Frisbee, J. L. Foster, D.-J. Lee, M. Wilkins,
  "Probability of Collision Between Space Objects", JGCD 23(5), 2000.
* F. K. Chan, *Spacecraft Collision Probability*, AIAA, 2008.
* D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.
"""
from __future__ import annotations

import numpy as np

from .covariance import encounter_plane_basis, mahalanobis, project_to_plane


def pc2d(miss_m: np.ndarray, C2: np.ndarray, hbr_m: float) -> float:
    """2-D Pc, small-hard-body-radius (Alfriend-Akella / Chan) closed form.

    Parameters
    ----------
    miss_m
        Miss vector projected into the encounter plane [m].
    C2
        Combined 2x2 covariance in the encounter plane [m^2].
    hbr_m
        Combined hard-body radius [m].
    """
    C2 = np.asarray(C2, dtype=float)
    det = np.linalg.det(C2)
    if det <= 0.0:
        raise ValueError("encounter-plane covariance is not positive definite")
    chi2 = mahalanobis(np.asarray(miss_m, dtype=float), C2)
    return float(hbr_m ** 2 / (2.0 * np.sqrt(det)) * np.exp(-0.5 * chi2))


def pc2d_max(hbr_m: float, C2: np.ndarray) -> float:
    """Theoretical maximum Pc (miss distance = 0): R^2 / (2 sigma_1 sigma_2)."""
    det = np.linalg.det(np.asarray(C2, dtype=float))
    if det <= 0.0:
        raise ValueError("encounter-plane covariance is not positive definite")
    return float(hbr_m ** 2 / (2.0 * np.sqrt(det)))


def pc2d_numeric(miss_m: np.ndarray, C2: np.ndarray, hbr_m: float,
                 n_r: int = 400, n_theta: int = 720) -> float:
    """Exact 2-D Pc by polar quadrature of the 2-D Gaussian over the HBR disk."""
    C2 = np.asarray(C2, dtype=float)
    Cinv = np.linalg.inv(C2)
    norm = 1.0 / (2.0 * np.pi * np.sqrt(np.linalg.det(C2)))

    rho = np.linspace(0.0, hbr_m, n_r)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    dtheta = 2.0 * np.pi / n_theta
    # integrate inward from the rim so the r=0 weight is handled by the
    # trapezoid on the midpoint rule rather than a singular weight
    rho_mid = 0.5 * (rho[:-1] + rho[1:])
    w_r = np.diff(rho) * rho_mid  # trapezoid of rho d rho

    miss = np.asarray(miss_m, dtype=float)
    acc = 0.0
    for wr, rm in zip(w_r, rho_mid):
        x = rm * np.cos(theta) - miss[0]
        y = rm * np.sin(theta) - miss[1]
        q = Cinv[0, 0] * x * x + 2.0 * Cinv[0, 1] * x * y + Cinv[1, 1] * y * y
        acc += wr * np.sum(np.exp(-0.5 * q))
    return float(norm * acc * dtheta)


def pc2d_monte_carlo(miss_m: np.ndarray, C2: np.ndarray, hbr_m: float,
                     n_samples: int = 2_000_000, seed: int = 0) -> float:
    """2-D Pc by Monte Carlo: fraction of Gaussian draws inside the HBR disk."""
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(np.asarray(C2, dtype=float))
    draws = rng.standard_normal((n_samples, 2)) @ L.T + np.asarray(miss_m, dtype=float)
    inside = np.sum(draws[:, 0] ** 2 + draws[:, 1] ** 2 <= hbr_m ** 2)
    return float(inside / n_samples)


def pc2d_from_rtn(miss_rtn_m: np.ndarray, v_rel_rtn: np.ndarray,
                  C_rtn_m2: np.ndarray, hbr_m: float,
                  method: str = "closed") -> float:
    """End-to-end 2-D Pc from RTN inputs.

    Projects the combined RTN covariance and the RTN miss vector into the
    encounter plane (perpendicular to the relative velocity) and evaluates Pc.
    """
    P = encounter_plane_basis(np.asarray(v_rel_rtn, dtype=float))
    C2 = project_to_plane(np.asarray(C_rtn_m2, dtype=float), P)
    m2 = P.T @ np.asarray(miss_rtn_m, dtype=float)
    if method == "closed":
        return pc2d(m2, C2, hbr_m)
    if method == "numeric":
        return pc2d_numeric(m2, C2, hbr_m)
    if method == "monte_carlo":
        return pc2d_monte_carlo(m2, C2, hbr_m)
    raise ValueError(f"unknown method {method!r}")
