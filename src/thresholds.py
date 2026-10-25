"""Layer 0 -- operational risk thresholds and threshold-sweep reporting.

Motivation
----------
"High risk" is not a physical quantity, it is a **policy choice**.  Different
operators draw the line in different places, and the line they draw defines
what a false positive and a false negative even *are*:

    JAXA   manoeuvres at Pc >= 1e-3            (JAXA-JMR-016(E), Level 3)
    ESA    manoeuvres at Pc >= 1e-4 at the final decision point
    DLR / CSA / 18 SPCS  alert at Pc > 1e-4
    ESA lower bound      1e-5                  (mission-dependent, 1e-5..1e-4)
    SpaceX Starlink      manoeuvres autonomously at Pc > 3e-7

Two consequences that this module makes explicit:

1. **Every metric is threshold-relative.**  A model's F2, precision and recall
   are only defined *given* a threshold, because the threshold defines the
   positive class.  :func:`sweep` therefore returns a whole table, never a
   single number.
2. **Cross-provider disagreement is itself a metric.**  The same predicted
   risk can be "act" for one operator and "ignore" for another.
   :func:`disagreement` quantifies that.

References
----------
* JAXA, *JMR-016(E) Standard for Spacecraft Collision Risk Management* (2022),
  sec. 6.2.2 -- risk levels 1/2/3 at Pc >= 1e-5 / 1e-4 / 1e-3.
  <https://sma.jaxa.jp/TechDoc/Docs/E_JAXA-JMR-016.pdf>
* ESA, *State of the art and future needs in conjunction assessment*, 8th
  European Conference on Space Debris (2021) -- "ESA set a threshold of
  PoC>10^-4 for HIEs, as does DLR and CSA, while JAXA uses a threshold of
  10^-3".
* NASA CARA Best Practices Handbook -- mitigate when Pc > 1e-4 or the miss
  distance is below the hard-body radius.
* SpaceX, Starlink autonomous collision avoidance: manoeuvre when the
  probability of collision exceeds ~3e-7.
* T. Uriot et al., Astrodynamics 5:1-17 (2021) -- "ESA missions in LEO adopt
  reaction thresholds ranging between 1e-5 and 1e-4".
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .metrics import fbeta

#: Default clipping epsilon, as used by the Kelvins competition (eps = 0.001).
DEFAULT_EPSILON = 0.001


@dataclass(frozen=True)
class RiskThreshold:
    """One operator's definition of "high risk".

    ``log10_risk`` is the threshold on log10(Pc); an event is high risk for
    this provider when ``log10(Pc) >= log10_risk``.
    """

    key: str
    provider: str
    log10_risk: float
    source: str
    role: str = "reaction"

    @property
    def pc(self) -> float:
        return 10.0 ** self.log10_risk

    def __str__(self) -> str:
        return f"{self.provider} @ 1e{self.log10_risk:.4f}"


#: The provider thresholds used by default, most conservative first.
PROVIDER_THRESHOLDS: tuple[RiskThreshold, ...] = (
    RiskThreshold(
        key="spacex_auto",
        provider="SpaceX Starlink (autonomous)",
        log10_risk=np.log10(3e-7),
        source="SpaceX published autonomous avoidance policy",
        role="autonomous manoeuvre",
    ),
    RiskThreshold(
        key="esa_lower",
        provider="ESA (lower bound of mission range)",
        log10_risk=-5.0,
        source="Uriot et al. 2021: ESA LEO missions use 1e-5..1e-4",
        role="reaction (lower bound)",
    ),
    RiskThreshold(
        key="esa_dlr_18spcs",
        provider="ESA / DLR / CSA / 18 SPCS / NASA CARA",
        log10_risk=-4.0,
        source="ESA SDC8 (2021); CARA Best Practices Handbook",
        role="reaction",
    ),
    RiskThreshold(
        key="jaxa",
        provider="JAXA",
        log10_risk=-3.0,
        source="JAXA-JMR-016(E) sec. 6.2.2, risk level 3",
        role="reaction (urgent)",
    ),
)

#: The threshold the Kelvins competition itself used, kept for continuity with
#: the LRP baseline reproduction.
COMPETITION_THRESHOLD = RiskThreshold(
    key="kelvins",
    provider="Kelvins competition",
    log10_risk=-6.0,
    source="Uriot et al. 2021, notification threshold 1e-6",
    role="notification",
)


def all_thresholds() -> list[RiskThreshold]:
    """Competition threshold plus the provider thresholds, ascending in risk."""
    return sorted((*PROVIDER_THRESHOLDS, COMPETITION_THRESHOLD),
                  key=lambda t: t.log10_risk)


def threshold_from_log10(values: list[float]) -> list[RiskThreshold]:
    """Build a custom threshold set (the 'configurable' part of Layer 0)."""
    return [
        RiskThreshold(key=f"custom_{v:+.4f}", provider=f"custom @ 1e{v:.4f}",
                      log10_risk=float(v), source="user configuration")
        for v in sorted(values)
    ]


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def sweep(r_true, r_pred, thresholds: list[RiskThreshold] | None = None,
          margin: float = 0.0, epsilon: float = DEFAULT_EPSILON,
          clip: bool = True) -> pd.DataFrame:
    """Evaluate a predictor across a whole threshold sweep.

    Parameters
    ----------
    r_true, r_pred
        True and predicted final risk, log10(Pc).  ``r_pred`` should be the
        **raw** model output; the threshold-relative clipping is applied here.
    thresholds
        Threshold set to sweep.  Defaults to :data:`all_thresholds`.
    margin
        Shift applied to the *prediction* threshold only.  ``margin > 0``
        makes the predictor more reluctant to call high risk (fewer FP, more
        FN); ``margin < 0`` the opposite.  This is the knob that trades false
        positives against false negatives *at fixed model output*.
    epsilon
        Clipping epsilon for low-risk predictions.
    clip
        If True (default), low-risk predictions are clipped to ``T - epsilon``
        **per threshold**, as the Kelvins paper prescribes: "as soon as an
        event is predicted to be low-risk the optimal prediction is
        -6 - epsilon".  That floor is *threshold-relative*, so it moves with
        the sweep.  Re-using a single fixed floor (e.g. -6.001) across
        thresholds silently reclassifies every low-risk event as high risk at
        any threshold below the floor.

    Returns
    -------
    DataFrame, one row per threshold, sorted ascending in risk.
    """
    r_true = np.asarray(r_true, dtype=float)
    r_pred = np.asarray(r_pred, dtype=float)
    thresholds = all_thresholds() if thresholds is None else sorted(
        thresholds, key=lambda t: t.log10_risk)

    rows = []
    for T in thresholds:
        y_true = r_true >= T.log10_risk
        # threshold-relative clip floor, then the (optional) margin shift
        r_hat = np.where(r_pred >= T.log10_risk, r_pred, T.log10_risk - epsilon) \
            if clip else r_pred
        y_pred = r_hat >= (T.log10_risk + margin)

        tp = int(np.sum(y_true & y_pred))
        fp = int(np.sum(~y_true & y_pred))
        fn = int(np.sum(y_true & ~y_pred))
        tn = int(np.sum(~y_true & ~y_pred))

        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        b2 = 4.0
        f2 = ((1 + b2) * precision * recall / (b2 * precision + recall)
              if np.isfinite(precision) and np.isfinite(recall)
              and (b2 * precision + recall) > 0 else np.nan)

        n_star = int(np.sum(y_true))
        mse_hr = (float(np.mean((r_true[y_true] - r_hat[y_true]) ** 2))
                  if n_star else np.nan)

        rows.append({
            "key": T.key,
            "provider": T.provider,
            "log10_threshold": T.log10_risk,
            "pc_threshold": T.pc,
            "n_events": len(r_true),
            "n_true_high_risk": n_star,
            "flagged": tp + fp,          # manoeuvres the policy would order
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision,
            "recall": recall,
            "F2": f2,
            "F1": (2 * precision * recall / (precision + recall)
                   if np.isfinite(precision) and np.isfinite(recall)
                   and (precision + recall) > 0 else np.nan),
            "MSE_HR": mse_hr,
            "L": (1.0 / f2 * mse_hr
                  if np.isfinite(f2) and f2 > 0 and np.isfinite(mse_hr) else np.nan),
            "evaluable": n_star > 0 and (tp + fp) > 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-provider disagreement
# ---------------------------------------------------------------------------
def disagreement(values: np.ndarray,
                 thresholds: list[RiskThreshold] | None = None) -> dict:
    """Quantify how much the *decision* depends on whose rules you apply.

    Because the thresholds are ordered, the decision vector of an event is a
    staircase: provider p acts iff ``value >= T_p``.  Two providers therefore
    disagree exactly on the events whose value falls between their two
    thresholds.

    Returns overall disagreement, the pairwise matrix, and the distribution of
    "how many providers would act".
    """
    values = np.asarray(values, dtype=float)
    thresholds = all_thresholds() if thresholds is None else sorted(
        thresholds, key=lambda t: t.log10_risk)

    decisions = np.column_stack([values >= T.log10_risk for T in thresholds])
    n_prov = decisions.shape[1]

    # pairwise disagreement
    pair = pd.DataFrame(index=[T.key for T in thresholds],
                        columns=[T.key for T in thresholds], dtype=float)
    for i, a in enumerate(thresholds):
        for j, b in enumerate(thresholds):
            pair.iloc[i, j] = float(np.mean(decisions[:, i] != decisions[:, j]))

    unanimous = np.all(decisions == decisions[:, :1], axis=1)
    n_acting = decisions.sum(axis=1)

    return {
        "n_events": len(values),
        "n_providers": n_prov,
        "disagreement_rate": float(1.0 - unanimous.mean()),
        "all_act_rate": float(np.all(decisions, axis=1).mean()),
        "none_act_rate": float(np.all(~decisions, axis=1).mean()),
        "max_pairwise_disagreement": float(pair.values.max()),
        "most_divergent_pair": (
            thresholds[int(np.argmax(pair.values)) // n_prov].key,
            thresholds[int(np.argmax(pair.values)) % n_prov].key,
        ),
        "pairwise": pair,
        "n_acting_histogram": {
            int(k): int(v) for k, v in
            zip(*np.unique(n_acting, return_counts=True))
        },
        "mean_providers_acting": float(n_acting.mean()),
    }


def decision_instability(values: np.ndarray,
                         thresholds: list[RiskThreshold] | None = None) -> np.ndarray:
    """Per-event count of distinct decisions across providers (0 or 1 = stable)."""
    values = np.asarray(values, dtype=float)
    thresholds = all_thresholds() if thresholds is None else thresholds
    decisions = np.column_stack([values >= T.log10_risk for T in thresholds])
    return decisions.sum(axis=1)


# ---------------------------------------------------------------------------
# Operational cost of a threshold choice
# ---------------------------------------------------------------------------
def operational_cost(sweep_df: pd.DataFrame, cost_fp: float = 1.0,
                     cost_fn: float = 10.0) -> pd.DataFrame:
    """Attach an operational cost to every row of a sweep.

    ``cost_fp`` is the cost of one unnecessary avoidance manoeuvre (propellant,
    operations burden, downtime) and ``cost_fn`` the cost of one missed
    high-risk conjunction.  The ratio is a *policy* input, not a measurement;
    it is passed in explicitly so it can never be smuggled in as a default.

    The interesting output is not the absolute cost but how the *ranking* of
    predictors and thresholds changes as the ratio moves.
    """
    out = sweep_df.copy()
    out["cost_fp_weight"] = cost_fp
    out["cost_fn_weight"] = cost_fn
    out["expected_cost"] = cost_fp * out["fp"] + cost_fn * out["fn"]
    out["cost_per_event"] = out["expected_cost"] / out["n_events"]
    return out


def format_sweep(df: pd.DataFrame) -> str:
    """Human-readable sweep table."""
    lines = [f"{'provider':44s} {'log10T':>7s} {'nHR':>5s} {'flag':>5s} "
             f"{'TP':>4s} {'FP':>4s} {'FN':>4s} {'prec':>6s} {'rec':>6s} {'F2':>7s}"]
    for _, r in df.iterrows():
        prov = r["provider"][:43]
        prec = f"{r['precision']:.4f}" if np.isfinite(r["precision"]) else "  n/a"
        rec = f"{r['recall']:.4f}" if np.isfinite(r["recall"]) else "  n/a"
        f2 = f"{r['F2']:.4f}" if np.isfinite(r["F2"]) else "   n/a"
        lines.append(f"{prov:44s} {r['log10_threshold']:7.3f} "
                     f"{r['n_true_high_risk']:5d} {r['flagged']:5d} "
                     f"{r['tp']:4d} {r['fp']:4d} {r['fn']:4d} {prec:>6s} {rec:>6s} {f2:>7s}")
    return "\n".join(lines)
