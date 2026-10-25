"""Rebuild the full 6x6 RTN covariance of a Kelvins CDM from the raw columns.

The flattened competition CSVs carry no covariance at all, and the raw CDM dump
stores it element-by-element under ESA's naming convention rather than as a
matrix.  This module assembles it, so the raw 199,082-CDM corpus can be scored
with a real covariance instead of the diagonal-only approximation.

Column naming (per object prefix ``t_`` / ``c_``), in RTN order [R T N Rdot
Tdot Ndot]::

    sigma_r/t/n            -> position variances (diagonal)
    ct_r, cn_r, cn_t       -> position covariances
    crdot_{r,t,n}          -> cov(Rdot, {R,T,N})
    ctdot_{r,t,n}          -> cov(Tdot, {R,T,N})
    cndot_{r,t,n}          -> cov(Ndot, {R,T,N})
    ctdot_rdot             -> cov(Tdot, Rdot)
    cndot_rdot             -> cov(Ndot, Rdot)
    cndot_tdot             -> cov(Ndot, Tdot)
    sigma_rdot/tdot/ndot   -> velocity variances (diagonal)

15 unique elements = 6*7/2, i.e. the complete symmetric 6x6.

The reconstruction is verified against ESA's own ``risk`` field: with
C2 the 2-D encounter-plane projection and chi^2 the Mahalanobis distance of the
miss vector, ESA's published functional form

    risk = log10( R^2 / (2 sqrt(det C2)) ) - chi^2 / (2 ln 10)

must be reproduced.  If the element mapping is wrong this fails loudly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: (row, col) -> column suffix, RTN order [R T N Rdot Tdot Ndot].
_ELEMENTS: tuple[tuple[int, int, str], ...] = (
    # position block
    (0, 1, "ct_r"), (0, 2, "cn_r"), (1, 2, "cn_t"),
    # position-velocity block
    (0, 3, "crdot_r"), (0, 4, "ctdot_r"), (0, 5, "cndot_r"),
    (1, 3, "crdot_t"), (1, 4, "ctdot_t"), (1, 5, "cndot_t"),
    (2, 3, "crdot_n"), (2, 4, "ctdot_n"), (2, 5, "cndot_n"),
    # velocity block
    (3, 4, "ctdot_rdot"), (3, 5, "cndot_rdot"), (4, 5, "cndot_tdot"),
)

#: diagonal entries, (index, sigma column suffix).
_DIAGONAL: tuple[tuple[int, str], ...] = (
    (0, "sigma_r"), (1, "sigma_t"), (2, "sigma_n"),
    (3, "sigma_rdot"), (4, "sigma_tdot"), (5, "sigma_ndot"),
)

NEEDED = ([f"t_{s}" for _, _, s in _ELEMENTS] + [f"t_{s}" for _, s in _DIAGONAL]
          + [f"c_{s}" for _, _, s in _ELEMENTS] + [f"c_{s}" for _, s in _DIAGONAL])


def covariance_6x6(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """Assemble the (N, 6, 6) RTN covariance for one object.

    ``prefix`` is ``"t"`` or ``"c"`` (target / chaser).  The matrix is forced
    symmetric and any negative eigenvalue from ESA rounding is clipped, which is
    the same remediation CARA applies (Appendix N N.7.2).
    """
    n = len(df)
    C = np.zeros((n, 6, 6))
    for i, s in _DIAGONAL:
        v = df[f"{prefix}_{s}"].to_numpy(dtype=float)
        C[:, i, i] = np.where(np.isfinite(v), v, 0.0) ** 2
    for i, j, s in _ELEMENTS:
        v = df[f"{prefix}_{s}"].to_numpy(dtype=float)
        v = np.where(np.isfinite(v), v, 0.0)
        C[:, i, j] = v
        C[:, j, i] = v
    return C


def combined_position_covariance(df: pd.DataFrame) -> np.ndarray:
    """(N, 3, 3) combined position covariance C = C_t[:3,:3] + C_c[:3,:3]."""
    return covariance_6x6(df, "t")[:, :3, :3] + covariance_6x6(df, "c")[:, :3, :3]


def plane_basis(v: np.ndarray) -> np.ndarray:
    """Orthonormal (e1, e2) spanning the plane perpendicular to each row of v.

    e1 is the cross product of v with the coordinate axis it is *least* aligned
    with, which is well defined for every direction.  Any basis of the plane is
    equivalent: the 2-D disk integral is rotation-invariant within it.
    """
    nrm = np.linalg.norm(v, axis=1, keepdims=True)
    vh = v / np.where(nrm > 0, nrm, 1.0)
    ax = np.zeros_like(vh)
    ax[np.arange(len(vh)), np.argmin(np.abs(vh), axis=1)] = 1.0
    e1 = np.cross(vh, ax)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(vh, e1)
    return np.stack([e1, e2], axis=-1)          # (N, 3, 2)


def project(C3: np.ndarray, E: np.ndarray) -> np.ndarray:
    """(N,2,2) = E^T C3 E, the covariance restricted to the encounter plane."""
    return np.einsum("nia,nij,njb->nab", E, C3, E)


def clip_psd(C: np.ndarray, floor_rel: float = 1e-12) -> np.ndarray:
    """Clip negative eigenvalues, mirroring CARA's CovRemEigValClip."""
    w, V = np.linalg.eigh(C)
    floor = floor_rel * np.maximum(np.abs(w).max(axis=1, keepdims=True), 1e-30)
    w = np.maximum(w, floor)
    return np.einsum("nij,nj,nkj->nik", V, w, V)
