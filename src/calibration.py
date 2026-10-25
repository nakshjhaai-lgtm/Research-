"""Layer 2 -- the evaluation core.

Calibration, precision-recall, and the FP/FN bookkeeping that Layer 0 demands.

The central question this module answers: **does the predicted Pc mean anything?**
A Pc of 1e-4 is a claim that 1 in 10,000 such conjunctions collides. If, over
the set of events forecast at 1e-4, the observed collision frequency is 1 in
100, the forecast is miscalibrated by two orders of magnitude -- and no amount
of threshold tuning fixes that.

Because the binary label depends on the threshold, **every reliability curve is
computed per threshold** (Layer 0). A single "the model is well calibrated"
statement is as meaningless as a single F2.

Functions
---------
reliability_curve   binned predicted Pc vs observed high-risk frequency
calibration_regression  logistic fit of outcome on log-risk (slope/intercept)
pr_curve            precision-recall over a continuous decision threshold
expected_calibration_error, brier_score, log_loss
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, log_expit

__all__ = ["reliability_curve", "calibration_regression", "pr_curve",
           "expected_calibration_error", "brier_score", "log_loss",
           "reliability_summary", "platt_calibrate"]

LOG10_FLOOR = -30.0


def _probs(r_pred: np.ndarray) -> np.ndarray:
    """Predicted collision probability, floored to avoid exact zeros."""
    return np.power(10.0, np.maximum(np.asarray(r_pred, dtype=float), LOG10_FLOOR))


# ----------------------------------------------------------------------
# reliability
# ----------------------------------------------------------------------
def reliability_curve(r_true: np.ndarray, r_pred: np.ndarray,
                      threshold: float, n_bins: int = 10,
                      strategy: str = "quantile") -> dict:
    """Binned reliability of the Pc forecast at one threshold.

    Parameters
    ----------
    r_true, r_pred
        True final risk and predicted risk, log10(Pc).
    threshold
        The Layer 0 threshold defining "high risk" (log10 Pc).
    n_bins, strategy
        ``"quantile"`` gives equal-count bins (recommended for sparse tails);
        ``"uniform"`` gives equal-width log10 bins.

    Returns
    -------
    dict with per-bin mean predicted probability, observed frequency, count,
    and the gap between them.
    """
    p = _probs(r_pred)
    y = (np.asarray(r_true, dtype=float) >= threshold).astype(float)

    if strategy == "quantile":
        edges = np.unique(np.quantile(r_pred, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        lo, hi = float(np.min(r_pred)), float(np.max(r_pred))
        edges = np.unique(np.linspace(lo, hi, n_bins + 1))
    if edges.size < 2:
        # degenerate (e.g. a constant predictor): everything is one bin
        edges = np.array([edges[0] - 1.0, edges[0] + 1.0]) if edges.size else \
            np.array([0.0, 1.0])
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(r_pred, edges[1:-1])

    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        mean_p = float(p[m].mean())
        obs = float(y[m].mean())
        rows.append({"bin": b, "n": n, "mean_pred_p": mean_p,
                     "observed_freq": obs, "abs_gap": abs(obs - mean_p),
                     "rel_gap": (obs - mean_p) / mean_p if mean_p > 0 else np.nan,
                     "log10_mean_pred": float(np.log10(mean_p)),
                     "n_high_risk": int(y[m].sum())})
    n_tot = sum(r["n"] for r in rows)
    ece = float(sum(r["n"] * r["abs_gap"] for r in rows) / n_tot) if n_tot else np.nan
    return {"threshold": float(threshold), "n_events": int(len(y)),
            "n_high_risk": int(y.sum()), "bins": rows, "ece": ece}


def calibration_regression(r_true: np.ndarray, r_pred: np.ndarray,
                           threshold: float, C: float = 1.0) -> dict:
    """Logistic regression of the outcome on the log-risk, L2-penalised.

    A perfectly calibrated forecaster has slope 1 and intercept 0 in
    logit(observed) = intercept + slope * log10(p).  Slope > 1 means the
    forecast is under-confident (too compressed); slope < 1 means
    over-confident.

    The penalty is **essential**, not cosmetic.  The unpenalised MLE diverges
    to +/-infinity whenever the predictor perfectly separates the classes --
    which happens routinely here, because at the low thresholds a bin of
    predicted-Pc values contains no collisions at all.  An earlier version of
    this function returned slopes of ~1e10 from exactly that separation; those
    numbers are not estimates, they are a symptom.

    The fit is done on standardised log-risk so the penalty is scale-free, then
    mapped back to the original units.
    """
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(r_pred, dtype=float)
    y = (np.asarray(r_true, dtype=float) >= threshold).astype(int)
    mu_x, sd_x = float(x.mean()), float(x.std())
    xs = (x - mu_x) / (sd_x if sd_x > 0 else 1.0)

    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=5000)
    clf.fit(xs.reshape(-1, 1), y)
    slope_std = float(clf.coef_[0, 0])
    intercept_std = float(clf.intercept_[0])
    slope = slope_std / (sd_x if sd_x > 0 else 1.0)
    intercept = intercept_std - slope_std * mu_x / (sd_x if sd_x > 0 else 1.0)

    ll = float(np.sum(y * log_expit(intercept + slope * x)
                      + (1 - y) * log_expit(-(intercept + slope * x))))
    # separation diagnostic: is the unpenalised MLE unbounded?  True when the
    # two classes are perfectly separable by a threshold on x, in either order.
    if y.sum() and (1 - y).sum():
        x1, x0 = x[y == 1], x[y == 0]
        sep = bool(x1.min() > x0.max() or x0.min() > x1.max())
    else:
        sep = False
    return {"threshold": float(threshold), "intercept": intercept, "slope": slope,
            "slope_standardised": slope_std, "log_likelihood": ll,
            "n_high_risk": int(y.sum()), "penalty_C": C,
            "perfect_separation_risk": sep}


# ----------------------------------------------------------------------
# precision-recall
# ----------------------------------------------------------------------
def pr_curve(r_true: np.ndarray, r_pred: np.ndarray,
             threshold: float, n_points: int = 200) -> dict:
    """Precision-recall over a *continuous* decision threshold.

    Distinct from the Layer 0 provider sweep: here the decision threshold is
    swept continuously across the predicted risk range, so the curve shows the
    whole achievable operating characteristic rather than one point on it.
    """
    y = (np.asarray(r_true, dtype=float) >= threshold).astype(float)
    rp = np.asarray(r_pred, dtype=float)
    grid = np.linspace(float(rp.min()), float(rp.max()), n_points)

    if np.ptp(rp) == 0.0:
        # constant predictor: there is no curve, only a point
        return {"threshold": float(threshold), "decision_grid": grid,
                "precision": np.full(n_points, np.nan),
                "recall": np.full(n_points, np.nan),
                "f2": np.full(n_points, np.nan), "pr_auc": np.nan,
                "n_high_risk": int(y.sum()), "degenerate": True}

    prec, rec, f2 = [], [], []
    for t in grid:
        yp = (rp >= t).astype(float)
        tp = int(np.sum(yp * y))
        fp = int(np.sum(yp * (1 - y)))
        fn = int(np.sum((1 - yp) * y))
        prec.append(tp / (tp + fp) if tp + fp else np.nan)
        rec.append(tp / (tp + fn) if tp + fn else np.nan)
        d = (1 + 4) * tp
        f2.append(d / (d + 4 * fn + fp) if d + 4 * fn + fp else np.nan)

    prec_a = np.asarray(prec, dtype=float)
    rec_a = np.asarray(rec, dtype=float)
    ok = ~np.isnan(prec_a) & ~np.isnan(rec_a)
    # PR-AUC by trapezoid on recall-sorted points
    if ok.sum() >= 2:
        o = np.argsort(rec_a[ok])
        auc = float(np.trapezoid(prec_a[ok][o], rec_a[ok][o]))
    else:
        auc = np.nan
    return {"threshold": float(threshold), "decision_grid": grid,
            "precision": prec_a, "recall": rec_a, "f2": np.asarray(f2, dtype=float),
            "pr_auc": auc, "n_high_risk": int(y.sum())}


# ----------------------------------------------------------------------
# scalar scores
# ----------------------------------------------------------------------
def brier_score(r_true: np.ndarray, r_pred: np.ndarray, threshold: float) -> float:
    """Brier score of the Pc forecast against the thresholded outcome."""
    p = _probs(r_pred)
    y = (np.asarray(r_true, dtype=float) >= threshold).astype(float)
    return float(np.mean((p - y) ** 2))


def log_loss(r_true: np.ndarray, r_pred: np.ndarray, threshold: float) -> float:
    """Log loss (cross-entropy) of the Pc forecast."""
    x = np.asarray(r_pred, dtype=float)
    y = (np.asarray(r_true, dtype=float) >= threshold).astype(float)
    return float(-np.mean(y * np.log10(np.maximum(_probs(x), 1e-300))
                          + (1 - y) * np.log10(np.maximum(1 - _probs(x), 1e-300))))


def expected_calibration_error(curve: dict) -> float:
    """ECE from a :func:`reliability_curve` result."""
    return float(curve["ece"])


def reliability_summary(curve: dict) -> dict:
    """One-line summary of a reliability curve for a report table."""
    rows = curve["bins"]
    return {"threshold": curve["threshold"], "n_events": curve["n_events"],
            "n_high_risk": curve["n_high_risk"], "ece": curve["ece"],
            "worst_bin_rel_gap": float(np.nanmax([abs(r["rel_gap"]) for r in rows]))
            if rows else np.nan,
            "n_bins_populated": len(rows)}


# ----------------------------------------------------------------------
# recalibration
# ----------------------------------------------------------------------
def platt_calibrate(r_fit: np.ndarray, y_fit: np.ndarray, r_apply: np.ndarray,
                    threshold: float, C: float = 1.0) -> tuple[np.ndarray, dict]:
    """Platt-scale a log-risk forecaster so its probabilities mean something.

    Fits ``logit(p) = a + b * r`` on one split and applies it to another.  The
    returned risk is ``log10(sigmoid(a + b*r))`` -- same units and same
    threshold semantics as the input, so it drops straight into the Layer 0
    sweep.

    Fit on train, apply to test: never fit and evaluate on the same events.
    """
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(r_fit, dtype=float)
    y = (np.asarray(y_fit, dtype=float) >= threshold).astype(int)
    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=5000)
    clf.fit(x.reshape(-1, 1), y)
    a, b = float(clf.intercept_[0]), float(clf.coef_[0, 0])
    z = a + b * np.asarray(r_apply, dtype=float)
    # log10(sigmoid(z)), numerically stable
    out = np.where(z >= 0, -np.logaddexp(0.0, -z), z - np.logaddexp(0.0, z))
    out = out / np.log(10.0)
    return out, {"a": a, "b": b, "threshold": float(threshold)}
