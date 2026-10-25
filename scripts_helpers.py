"""Shared vectorised 2-D Pc kernels used by the Layer 2 corpus scripts.

Kept in one place so `attribution_method.py` and `interpretability_scorecard.py`
cannot drift apart: both must compute the same quantity the same way.

`pc2d_1d` in `src/foster2d.py` is the scalar reference and the authority; these
are chunked batch equivalents, and `attribution_method.py` asserts they agree
with it to 1e-9 before trusting any corpus-level number.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

N_NODES = 200
_GLX, _GLW = np.polynomial.legendre.leggauss(N_NODES)


def pc2d_exact_batch(C2: np.ndarray, miss: np.ndarray, hbr: np.ndarray,
                     chunk: int = 8192) -> np.ndarray:
    """Vectorised Eq. N-2 (Alfano 2005a / Elrod 2019), chunked for memory."""
    out = np.empty(len(C2))
    for s in range(0, len(C2), chunk):
        c = slice(s, min(s + chunk, len(C2)))
        w, V = np.linalg.eigh(C2[c])            # w ascending -> w[:,0] minor
        s1, s2 = np.sqrt(w[:, 1]), np.sqrt(w[:, 0])
        m_rot = np.einsum("nji,ni->nj", V, miss[c])
        m1, m2 = m_rot[:, 1], m_rot[:, 0]
        R = hbr[c][:, None]
        xs = R * _GLX[None, :]
        h = np.sqrt(np.maximum(R ** 2 - xs ** 2, 0.0))
        inner = (ndtr((h - m2[:, None]) / s2[:, None])
                 - ndtr((-h - m2[:, None]) / s2[:, None]))
        outer = np.exp(-0.5 * ((xs - m1[:, None]) / s1[:, None]) ** 2) \
            / (np.sqrt(2.0 * np.pi) * s1[:, None])
        out[c] = R[:, 0] * np.sum(_GLW[None, :] * outer * inner, axis=1)
    return out


def pc2d_closed_batch(C2: np.ndarray, miss: np.ndarray,
                      hbr: np.ndarray) -> np.ndarray:
    """The small-HBR closed form: R^2/(2 sqrt(det C)) * exp(-chi^2/2)."""
    w = np.linalg.eigvalsh(C2)
    det = np.clip(w[:, 0] * w[:, 1], 1e-300, None)
    chi2 = np.einsum("ni,nij,nj->n", miss, np.linalg.inv(C2), miss)
    return hbr ** 2 / (2.0 * np.sqrt(det)) * np.exp(-0.5 * chi2)
