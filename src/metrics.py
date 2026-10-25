"""Competition metric for the ESA Kelvins Spacecraft Collision Avoidance Challenge.

Reference: Uriot et al. (2020/2021), "Spacecraft Collision Avoidance Challenge:
design and results of a machine learning competition", Astrodynamics 5:1-17
(Sec. 4.3, Eq. 1).

    L(r_hat) = (1 / F2) * MSE_HR(r, r_hat)

where
    r        : true final risk, log10 of the collision probability
    r_hat    : predicted final risk (log10)
    F2       : F-beta score (beta = 2) over the whole test set using two
               classes  ->  high final risk  r >= -6,  low final risk  r < -6
    MSE_HR   : mean squared error computed ONLY over high-risk events
               (1/N*) * sum_{i: r_i >= -6} (r_i - r_hat_i)^2

beta = 2 weights recall twice as heavily as precision, i.e. false negatives
(a missed high-risk conjunction) are penalised ~4x more than false positives.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

#: log10 risk threshold defining a "high risk" event (i.e. Pc >= 1e-6).
HIGH_RISK_THRESHOLD = -6.0

#: LRP / metric clipping floor for predicted risk (log10).
RISK_CLIP_FLOOR = -6.001


def fbeta(y_true_high: np.ndarray, y_pred_high: np.ndarray, beta: float = 2.0) -> dict:
    """F-beta score plus the underlying precision / recall / confusion counts."""
    y_true_high = np.asarray(y_true_high, dtype=bool)
    y_pred_high = np.asarray(y_pred_high, dtype=bool)

    tp = int(np.sum(y_true_high & y_pred_high))
    fp = int(np.sum(~y_true_high & y_pred_high))
    fn = int(np.sum(y_true_high & ~y_pred_high))
    tn = int(np.sum(~y_true_high & ~y_pred_high))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2 = beta ** 2
    denom = (b2 * precision) + recall
    fbeta = (1.0 + b2) * precision * recall / denom if denom > 0 else 0.0

    return {
        f"F{beta:g}": fbeta,
        "precision": precision,
        "recall": recall,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


@dataclass
class ChallengeResult:
    n_events: int
    n_high_risk: int
    F2: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int
    MSE_HR: float
    L: float

    def as_dict(self) -> dict:
        return asdict(self)

    def report(self) -> str:
        return (
            f"  events            : {self.n_events}  ({self.n_high_risk} high-risk)\n"
            f"  confusion matrix  : TP={self.tp}  FP={self.fp}  FN={self.fn}  TN={self.tn}\n"
            f"  precision         : {self.precision:.6f}\n"
            f"  recall            : {self.recall:.6f}\n"
            f"  F2                : {self.F2:.6f}\n"
            f"  MSE_HR            : {self.MSE_HR:.6f}\n"
            f"  L = (1/F2)*MSE_HR : {self.L:.6f}"
        )


def challenge_metrics(r_true, r_pred, threshold: float = HIGH_RISK_THRESHOLD) -> ChallengeResult:
    """Full challenge metric from arrays of log10 risk values."""
    r_true = np.asarray(r_true, dtype=float)
    r_pred = np.asarray(r_pred, dtype=float)

    y_true_high = r_true >= threshold
    y_pred_high = r_pred >= threshold

    fb = fbeta(y_true_high, y_pred_high, beta=2.0)

    n_star = int(np.sum(y_true_high))
    if n_star > 0:
        mse_hr = float(np.mean((r_true[y_true_high] - r_pred[y_true_high]) ** 2))
    else:
        mse_hr = float("nan")

    f2 = fb["F2"]
    L = (1.0 / f2) * mse_hr if f2 > 0 else float("inf")

    return ChallengeResult(
        n_events=len(r_true),
        n_high_risk=n_star,
        F2=f2,
        precision=fb["precision"],
        recall=fb["recall"],
        tp=fb["tp"], fp=fb["fp"], fn=fb["fn"], tn=fb["tn"],
        MSE_HR=mse_hr,
        L=L,
    )


def clip_low_risk(r_pred, floor: float = RISK_CLIP_FLOOR, threshold: float = HIGH_RISK_THRESHOLD):
    """Eq. 6 of the paper: r_hat = r if r > -6, else -6.001."""
    r_pred = np.asarray(r_pred, dtype=float)
    return np.where(r_pred > threshold, r_pred, floor)
