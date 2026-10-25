"""Covariance utilities and frame conventions for conjunction assessment.

Layer 3 of the learning path: covariance basics -> 2-D Pc.

The two objects of a conjunction are each described by a 6x6 (or 3x3)
covariance in their own **RTN** frame:

    R  radial        -- along the position vector, away from the Earth
    T  transverse    -- in the orbit plane, perpendicular to R, along-track
    N  normal        -- completes the right-handed triad (orbit normal,
                        ~ cross-track)

Because the two objects are tracked independently, their errors are treated as
uncorrelated and the *combined* covariance is simply the sum (CARA Best
Practices, Appendix N):

    C = C_primary + C_secondary

The 2-D Pc is then evaluated in the **encounter plane**: the plane through the
relative position that is perpendicular to the relative velocity at TCA.  The
component of the miss vector along the relative velocity is irrelevant on the
timescale of the encounter (rectilinear-motion assumption), so the problem
reduces to a 2-D Gaussian over a disk of radius = hard-body radius.

References
----------
* NASA CARA Best Practices Handbook (2023), Appendix N "Pc Calculation
  Approaches", eqs. (N-17)/(N-18) for the combined and cross-correlated
  covariances.
* D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th ed.,
  ch. 11 (coordinate transformations and covariance transformations).
* K. Alfriend, M. R. Akella et al., "Probability of Collision Between Space
  Objects", JGCD 23(5), 2000.
"""
from __future__ import annotations

import numpy as np


def rtn_basis(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """3x3 matrix whose rows are the R, T, N unit vectors, given an ECI state."""
    R = r_eci / np.linalg.norm(r_eci)
    h = np.cross(r_eci, v_eci)
    N = h / np.linalg.norm(h)
    T = np.cross(N, R)
    return np.vstack([R, T, N])


def encounter_plane_basis(v_rel: np.ndarray) -> np.ndarray:
    """3x2 matrix [e1 e2] spanning the plane perpendicular to ``v_rel``.

    ``e1`` lies in the horizontal plane (perpendicular to the orbit normal)
    whenever possible, which keeps the projection numerically well conditioned;
    ``e2 = e3 x e1`` completes the right-handed triad.
    """
    e3 = v_rel / np.linalg.norm(v_rel)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, e3)) > 0.99:
        ref = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, e3)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    return np.column_stack([e1, e2])


def project_to_plane(C3: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project a 3x3 covariance onto the 2-D subspace spanned by columns of P."""
    return P.T @ C3 @ P


def mahalanobis(m: np.ndarray, C: np.ndarray) -> float:
    """Squared Mahalanobis distance  m^T C^-1 m  (the 'miss distance scaled
    with uncertainty' column of the Kelvins dataset)."""
    return float(m @ np.linalg.solve(C, m))


def combine_covariances(*Cs: np.ndarray) -> np.ndarray:
    """Sum of independent covariances (element-wise, no cross-correlation)."""
    return np.sum(Cs, axis=0)


def principal_sigmas(C2: np.ndarray) -> tuple[float, float]:
    """The two 1-sigma values (square roots of the eigenvalues) of a 2x2 matrix."""
    w = np.linalg.eigvalsh(C2)
    return float(np.sqrt(max(w[0], 0.0))), float(np.sqrt(max(w[1], 0.0)))


def covariance_ellipse(C2: np.ndarray, nsigma: float = 1.0, n: int = 200) -> np.ndarray:
    """Points on the n-sigma error ellipse of a 2x2 covariance, for plotting."""
    w, V = np.linalg.eigh(C2)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.vstack([np.cos(t), np.sin(t)])
    return V @ np.diag(nsigma * np.sqrt(np.maximum(w, 0))) @ circle
